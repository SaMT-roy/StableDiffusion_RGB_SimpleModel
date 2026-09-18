"""
The text encoder: OpenAI's CLIP, pretrained and frozen.

This is the one network we do not train. Stable Diffusion does exactly the same thing,
and for the same reason: a text encoder that has already read 400 million captions
knows that "puppy" and "dog" are related, that "two men" is plural, and that
"skateboard" and "surfboard" are different objects. There is no way to learn that from
28,000 COCO images, so we borrow it.

Two pieces, both small enough to read:
  1. a tokeniser   - splits a sentence into CLIP's 49,408 known word-pieces
  2. a transformer - turns those pieces into one 512-d vector each

Run `python clip_text.py` for a self-test. The weights (600 MB) download once.
"""

import json
import math
import os
import re
import urllib.request

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_DIR = os.path.join(HERE, "data", "clip")
BASE_URL = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/"
WEIGHTS = os.path.join(CLIP_DIR, "clip_text.pt")          # 126 MB: what we keep
DOWNLOAD = os.path.join(CLIP_DIR, "pytorch_model.bin")    # 592 MB: deleted after extraction

WIDTH = 512          # CLIP ViT-B/32's text width, and therefore our cross-attention width
LAYERS = 12
HEADS = 8
MAX_TOKENS = 20      # COCO captions are ~13 word-pieces; CLIP was trained with 77 slots,
                     # but attention here is causal, so using only the first 20 positions
                     # gives bit-identical results for those positions and is 4x cheaper.
START, END = 49406, 49407    # <|startoftext|>, <|endoftext|>


def _download():
    """Fetch the tokeniser files, and the text encoder's weights if we do not have them yet.

    The published checkpoint also contains CLIP's *image* tower, which is 58% of the file
    and which we never use. So we extract the text half once, store it at half precision
    (CLIP was trained in fp16, so this loses nothing), and delete the download:
    592 MB becomes 126 MB.
    """
    os.makedirs(CLIP_DIR, exist_ok=True)
    for name in ["vocab.json", "merges.txt"]:
        if not os.path.exists(os.path.join(CLIP_DIR, name)):
            print(f"downloading CLIP {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, os.path.join(CLIP_DIR, name))

    if os.path.exists(WEIGHTS):
        return
    if not os.path.exists(DOWNLOAD):
        print("downloading CLIP weights (592 MB, once) ...")
        urllib.request.urlretrieve(BASE_URL + "pytorch_model.bin", DOWNLOAD)

    full = torch.load(DOWNLOAD, map_location="cpu", weights_only=True)
    text_only = {}
    for key, value in full.items():
        if not key.startswith("text_model.") or "position_ids" in key:
            continue
        key = key[len("text_model."):]
        for old, new in [("encoder.layers", "layers"), ("self_attn.", ""),
                         ("mlp.", ""), ("embeddings.", "")]:
            key = key.replace(old, new)
        text_only[key] = value.half()
    torch.save(text_only, WEIGHTS)
    os.remove(DOWNLOAD)
    print(f"kept the text encoder ({os.path.getsize(WEIGHTS)/1e6:.0f} MB), deleted the rest")


# ----------------------------------------------------------------------------
# 1. Tokeniser (byte-pair encoding)
#
# BPE starts with single characters and repeatedly glues together whichever adjacent
# pair is highest in a ranked list of merges that was learned from text. So "skateboard"
# might become ["skate", "board</w>"] - common words stay whole, rare ones get split
# into familiar pieces. The </w> marker means "this piece ends a word".
# ----------------------------------------------------------------------------
class Tokenizer:
    def __init__(self):
        _download()
        self.vocab = json.load(open(os.path.join(CLIP_DIR, "vocab.json")))
        lines = open(os.path.join(CLIP_DIR, "merges.txt"), encoding="utf-8").read().split("\n")
        # line 0 is a version comment; the rest are "a b" pairs, best merge first
        self.ranks = {tuple(line.split()): i for i, line in enumerate(lines[1:]) if len(line.split()) == 2}
        self.cache = {}
        # CLIP's word splitter. The original uses unicode letter classes; our captions are
        # English, and we strip to ASCII first, so these character classes are equivalent.
        self.pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+")

    def _merge_word(self, word):
        """Apply BPE merges to one word. 'dog' -> ('dog</w>',), 'skateboarder' -> several."""
        if word in self.cache:
            return self.cache[word]
        pieces = tuple(word[:-1]) + (word[-1] + "</w>",)
        while len(pieces) > 1:
            pairs = list(zip(pieces[:-1], pieces[1:]))
            best = min(pairs, key=lambda pair: self.ranks.get(pair, math.inf))
            if best not in self.ranks:
                break
            merged, i = [], 0
            while i < len(pieces):
                if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == best:
                    merged.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    merged.append(pieces[i])
                    i += 1
            pieces = tuple(merged)
        self.cache[word] = pieces
        return pieces

    def encode(self, text):
        text = re.sub(r"\s+", " ", text.lower().encode("ascii", "ignore").decode()).strip()
        ids = [START]
        for word in self.pattern.findall(text):
            ids += [self.vocab[piece] for piece in self._merge_word(word) if piece in self.vocab]
        # always leave room for the end marker: the pooled vector is read from it
        return ids[:MAX_TOKENS - 1] + [END]

    def __call__(self, prompts):
        """['a dog on a beach', ...] -> int array (batch, MAX_TOKENS), padded with <|endoftext|>."""
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = np.full((len(prompts), MAX_TOKENS), END, dtype=np.int64)
        for row, prompt in enumerate(prompts):
            ids = self.encode(prompt)
            tokens[row, :len(ids)] = ids
        return tokens


# ----------------------------------------------------------------------------
# 2. The transformer
# ----------------------------------------------------------------------------
class CLIPLayer(nn.Module):
    """One pre-norm transformer layer: causal self-attention, then a feed-forward net."""

    def __init__(self, width=WIDTH, heads=HEADS):
        super().__init__()
        self.heads = heads
        self.layer_norm1 = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.layer_norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x, mask):
        h = self.layer_norm1(x)
        batch, positions, width = h.shape
        head_dim = width // self.heads

        def split_heads(t):
            return t.view(batch, positions, self.heads, head_dim).transpose(1, 2)

        q, k, v = split_heads(self.q_proj(h)), split_heads(self.k_proj(h)), split_heads(self.v_proj(h))
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim) + mask
        h = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, positions, width)
        x = x + self.out_proj(h)

        h = self.fc1(self.layer_norm2(x))
        h = h * torch.sigmoid(1.702 * h)          # QuickGELU, the activation CLIP was trained with
        return x + self.fc2(h)


class CLIPText(nn.Module):
    def __init__(self, width=WIDTH, layers=LAYERS, heads=HEADS):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, width)
        self.position_embedding = nn.Embedding(77, width)
        self.layers = nn.ModuleList([CLIPLayer(width, heads) for _ in range(layers)])
        self.final_layer_norm = nn.LayerNorm(width)

    def forward(self, tokens):
        """
        (batch, positions) ints -> two things:
          sequence (batch, positions, 512)  one vector per word-piece, for cross-attention
          pooled   (batch, 512)             one vector for the whole sentence
        """
        positions = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding.weight[:positions]
        # causal mask: each position may only look at itself and the ones before it
        mask = torch.full((positions, positions), float("-inf"),
                          device=tokens.device, dtype=x.dtype).triu(1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        # CLIP summarises a sentence with the vector at its <|endoftext|> position. That token
        # has the largest id in the vocabulary, so argmax finds the first one - the real end,
        # not the padding that follows it.
        pooled = x[torch.arange(len(tokens), device=tokens.device), tokens.argmax(dim=-1)]
        return x.float(), pooled.float()


def load_clip_text(device):
    """Build the text encoder and fill it with the pretrained weights. Runs in fp16,
    which is the precision CLIP was trained and published in, and costs 126 MB."""
    _download()
    model = CLIPText().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model = model.half().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


if __name__ == "__main__":
    tokenizer = Tokenizer()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip = load_clip_text(device)
    print(f"loaded CLIP text encoder: {sum(p.numel() for p in clip.parameters())/1e6:.1f}M frozen parameters")

    prompts = ["a man riding a horse next to a fence",
               "a man riding a horse next to a fence",
               "a plate of broccoli and carrots on a wooden table"]
    tokens = torch.from_numpy(tokenizer(prompts)).to(device)
    print("tokens  ", tokens[0].tolist())
    sequence, pooled = clip(tokens)
    print("sequence", tuple(sequence.shape), " pooled", tuple(pooled.shape))

    unit = torch.nn.functional.normalize(pooled, dim=-1)
    print(f"cosine similarity, identical sentences : {(unit[0]*unit[1]).sum():.4f}  (must be 1.000)")
    print(f"cosine similarity, different sentences : {(unit[0]*unit[2]).sum():.4f}")
