"""
LPIPS: perceptual distance between two images.

Pixel losses ask "is this pixel the right colour?". That question has no opinion about
*where* detail belongs, so a decoder can satisfy it - and satisfy an edge-matching term,
and satisfy a critic - by sprinkling plausible texture everywhere. The result looks
sharp and brittle at the same time.

LPIPS (Zhang et al. 2018) compares the two images through the eyes of a pretrained
AlexNet instead. Those features already encode that a face needs structure in particular
places and a painted wall does not, so detail has to be perceptually *correct*, not
merely present. It is the loss Stable Diffusion's own autoencoder is trained with.

We use only AlexNet's convolutional stack - 2.5M parameters, 10 MB - plus five tiny
per-channel calibration vectors fitted by the LPIPS authors on human similarity
judgements. The 233 MB torchvision checkpoint is stripped down on first use and deleted.

Run `python lpips.py` for a self-test.
"""

import os
import urllib.request

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
LPIPS_DIR = os.path.join(HERE, "data", "lpips")
WEIGHTS = os.path.join(LPIPS_DIR, "lpips_alex.pt")            # 10 MB: what we keep
ALEX_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
LINEAR_URL = ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
              "master/lpips/weights/v0.1/alex.pth")

# AlexNet's five ReLU outputs, and how many channels each has
TAPS = [(0, 2, 64), (2, 5, 192), (5, 8, 384), (8, 10, 256), (10, 12, 256)]


def _alexnet_features():
    """AlexNet's convolutional stack, written out. The classifier is never used."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(inplace=True),     # tap 1
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True),              # tap 2
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),             # tap 3
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),             # tap 4
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),             # tap 5
    )


def _prepare():
    """Download once, keep only what LPIPS actually uses, delete the rest."""
    if os.path.exists(WEIGHTS):
        return
    os.makedirs(LPIPS_DIR, exist_ok=True)
    alex_path = os.path.join(LPIPS_DIR, "alexnet_full.pth")
    print("downloading AlexNet (233 MB, once) ...")
    urllib.request.urlretrieve(ALEX_URL, alex_path)
    print("downloading LPIPS calibration weights (6 KB) ...")
    linear_path = os.path.join(LPIPS_DIR, "alex_linear.pth")
    urllib.request.urlretrieve(LINEAR_URL, linear_path)

    full = torch.load(alex_path, map_location="cpu", weights_only=True)
    linear = torch.load(linear_path, map_location="cpu", weights_only=True)
    kept = {k[len("features."):]: v for k, v in full.items() if k.startswith("features.")}
    kept.update({f"linear{i}": linear[f"lin{i}.model.1.weight"] for i in range(5)})
    torch.save(kept, WEIGHTS)
    os.remove(alex_path)
    os.remove(linear_path)
    print(f"kept the feature stack ({os.path.getsize(WEIGHTS)/1e6:.0f} MB), deleted the rest")


class LPIPS(nn.Module):
    """Perceptual distance. Inputs are images in [-1, 1]; output is one number per image."""

    def __init__(self):
        super().__init__()
        _prepare()
        weights = torch.load(WEIGHTS, map_location="cpu", weights_only=True)

        features = _alexnet_features()
        features.load_state_dict({k: v for k, v in weights.items() if not k.startswith("linear")})
        self.slices = nn.ModuleList([features[start:end] for start, end, _ in TAPS])
        for i in range(5):
            # one non-negative weight per channel, from the LPIPS authors' human-judgement fit
            self.register_buffer(f"linear{i}", weights[f"linear{i}"])

        # AlexNet's expected input statistics, as LPIPS applies them
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([.458, .448, .450]).view(1, 3, 1, 1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, a, b):
        a, b = (a - self.shift) / self.scale, (b - self.shift) / self.scale
        distance = 0.0
        for i, layer in enumerate(self.slices):
            a, b = layer(a), layer(b)
            # compare directions, not magnitudes: unit-normalise each position's channel vector
            unit_a = a / (a.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            unit_b = b / (b.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            weighted = (unit_a - unit_b).pow(2) * getattr(self, f"linear{i}")
            distance = distance + weighted.sum(1).mean((1, 2))
        return distance


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lpips = LPIPS().to(device)
    print(f"LPIPS-AlexNet: {sum(p.numel() for p in lpips.parameters())/1e6:.2f}M parameters, "
          f"{os.path.getsize(WEIGHTS)/1e6:.0f} MB on disk")

    # Real images, not random noise: perceptual distance is only meaningful on real
    # structure. (Blurring noise destroys everything, so noise makes a misleading test.)
    import numpy as np
    images = np.array(np.load(os.path.join(HERE, "data", "images.npy"), mmap_mode="r")[:8])
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    blurred = torch.nn.functional.avg_pool2d(x, 5, stride=1, padding=2)
    other = x.flip(0)                                   # a different photograph each time

    print(f"  a photo vs itself        {lpips(x, x).mean():.4f}   (must be 0.0000)")
    print(f"  a photo vs a blur of it  {lpips(x, blurred).mean():.4f}   (what our VAE does wrong)")
    print(f"  a photo vs another photo {lpips(x, other).mean():.4f}   (should be the largest)")
