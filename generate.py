"""
Generate images from text. Run:

    python generate.py                                      # the eight demo prompts
    python generate.py "a red bus on a city street"         # 8 samples of one prompt
    python generate.py "a red bus on a city street" 7.0     # ... at guidance scale 7

Images are written to out/ as PNG files.
"""

import re
import sys

import numpy as np
import torch

import clip_text
from diffusion import make_alpha_bar, ddim_sample
from model import VAE, UNet
from train_diffusion import MODEL_PATH, OUT_DIR, PREVIEW_PROMPTS
from train_vae import get_device, save_grid

SAMPLING_STEPS = 50    # DDIM steps: more is slower but slightly cleaner
GUIDANCE = 5.0         # how hard to follow the prompt (1.0 = ignore it entirely)


def load_models(device):
    """Load out/model_best.pt - the averaged weights at the best validation loss - plus CLIP."""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    vae = VAE().to(device).eval()
    vae.load_state_dict({k: v.float() for k, v in checkpoint["vae"].items()})
    unet = UNet().to(device).eval()
    unet.load_state_dict({k: v.float() for k, v in checkpoint["unet"].items()})

    clip = clip_text.load_clip_text(device)
    print(f"loaded a model trained for {checkpoint['step']} steps "
          f"(held-out validation loss {checkpoint['val_loss']:.4f})")
    return vae, unet, clip, clip_text.Tokenizer(), checkpoint["latent_scale"]


@torch.no_grad()
def generate(prompts, vae, unet, clip, tokenizer, latent_scale, device,
             guidance=GUIDANCE, steps=SAMPLING_STEPS, seed=0, snapshots=None):
    """Text -> images, as a float array (n,3,64,64) in [-1,1]."""
    context, pooled = clip(torch.from_numpy(tokenizer(prompts)).to(device))
    empty_context, empty_pooled = clip(torch.from_numpy(tokenizer([""])).to(device))

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = ddim_sample(unet, context, pooled, empty_context, empty_pooled,
                          make_alpha_bar(device), num_steps=steps, guidance=guidance,
                          generator=generator, snapshots=snapshots)
    # undo the training scale, then decode the latent back into pixels - once, at the end
    return vae.decode(latents / latent_scale).cpu().numpy()


def main():
    device = get_device()
    models = load_models(device)

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        guidance = float(sys.argv[2]) if len(sys.argv) > 2 else GUIDANCE
        seed = int(np.random.randint(100000))
        print(f'generating 8 samples of "{prompt}" at guidance {guidance}, seed {seed}')
        images = generate([prompt] * 8, *models, device, guidance=guidance, seed=seed)
        # name the file after the prompt, so successive runs do not overwrite each other
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60]
        path = f"{OUT_DIR}/{slug}_g{guidance:g}_s{seed}.png"
        save_grid(images, path, columns=4)
        print(f"saved {path}")
    else:
        print("generating 4 samples of each demo prompt")
        prompts = [p for p in PREVIEW_PROMPTS for _ in range(4)]
        images = generate(prompts, *models, device, seed=0)
        save_grid(images, f"{OUT_DIR}/samples.png", columns=4)
        print(f"saved {OUT_DIR}/samples.png  (one row per prompt, in the order:)")
        for prompt in PREVIEW_PROMPTS:
            print(f"    {prompt}")


if __name__ == "__main__":
    main()
