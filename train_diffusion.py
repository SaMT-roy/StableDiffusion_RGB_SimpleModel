"""
Stage 2: train the latent diffusion model.   python train.py

Run train_vae.py first. This loads that autoencoder, freezes it, and from here on the VAE
only turns images into latents and latents back into images - all the learning happens on
4x16x16 latents, which is the point of latent diffusion.

Data is streamed: each step fetches its own images, encodes them and tokenises their
captions, and keeps nothing. Memory is O(batch) rather than O(dataset), so the loop is
indifferent to how large the corpus is - point `dataset.load` at a sharded reader and
nothing below changes.

The recipe is a published one, not an invention:

  v-prediction + zero terminal SNR   Lin et al. 2024. The schedule is rescaled so the last
                                     step is exactly pure noise, matching what generation
                                     actually hands the model; v-prediction is what stays
                                     well-defined there.
  min-SNR-gamma loss weighting       Hang et al. 2023, gamma=5.
  cross-attention on frozen CLIP     Rombach et al. 2022, with 10% caption dropout so one
                                     network learns both conditional and unconditional.
  DDIM + classifier-free guidance    Song et al. 2020, Ho & Salimans 2022, with the
                                     guidance rescale of Lin et al. 2024.
  constant learning rate after warmup  as LDM, Stable Diffusion and DiT all use.

Two files come out: diffusion.pt is the rolling training state (resume, always current)
and model_best.pt is the model at its lowest held-out validation loss. Sample sheets are
written every SAVE_EVERY steps from fixed prompts and fixed noise, so successive sheets
show the model improving rather than showing different pictures.
"""

import os
import time

import numpy as np
import torch

import clip_text
import dataset
from diffusion import add_noise, ddim_sample, make_alpha_bar, snr_weight, training_loss, velocity
from model import UNet, VAE, count_parameters
from train_vae import OUT_DIR, get_device, save_atomically, save_grid, to_batch

# ----------------------------------------------------------------------------
# Settings - the only knobs.
# ----------------------------------------------------------------------------
DIFFUSION_STEPS = 200000
DIFFUSION_LR = 1e-4
WARMUP_STEPS = 500
BATCH_SIZE = 64
EMA_DECAY = 0.9995       # we generate with a slow-moving average of the weights
SAVE_EVERY = 2000        # validate, checkpoint, and write a sample sheet this often
ARCHIVE_EVERY = 10000    # keep a permanent weights-only snapshot this often. The rolling
                         # files overwrite themselves, and model_best.pt is chosen on
                         # validation loss - which is a weak proxy for how the samples
                         # actually look. Without these, the weights behind a sample sheet
                         # you liked are gone by the time you have seen it.
VALIDATION_LATENTS = 512

# Not vae_best.pt. Selection by LPIPS picked epoch 6, but epoch 9 carries 86% of the
# original fine detail against epoch 6's 78%, and - measured on held-out images - degrades
# more gracefully under exactly the kind of imperfect latent a diffusion model produces
# (LPIPS 0.0351 vs 0.0363 at a latent perturbation of 0.20; the two cross over at ~0.15).
# The archive check lives inside the `step % SAVE_EVERY` block, so this must be a
# multiple of SAVE_EVERY or it silently never fires.
assert ARCHIVE_EVERY % SAVE_EVERY == 0, "ARCHIVE_EVERY must be a multiple of SAVE_EVERY"

VAE_PATH = f"{OUT_DIR}/vae_epoch09.pt"
DIFFUSION_PATH = f"{OUT_DIR}/diffusion.pt"    # rolling training state: resume from this
MODEL_PATH = f"{OUT_DIR}/model_best.pt"       # the model, at its best validation loss

# Eight fixed prompts, drawn with the same fixed noise every time, so successive sheets
# show the model improving rather than showing different pictures.
PREVIEW_PROMPTS = [
    "a red double decker bus on a city street",
    "a giraffe standing in a green grassy field",
    "a man riding a wave on a surfboard",
    "a plate of food with broccoli and meat",
    "a bathroom with a white toilet and a sink",
    "a cat sitting on top of a laptop computer",
    "a large airplane flying in a blue sky",
    "a group of people sitting around a table",
]


def learning_rate(step):
    """
    Warm up, then hold constant.

    LDM, Stable Diffusion and DiT all train at a constant rate. Diffusion training is
    stable enough not to need decay, and holding it constant means the run can simply be
    extended if the model is still improving when it ends - which a decay-to-zero schedule
    would make impossible.
    """
    return DIFFUSION_LR * min(1.0, step / WARMUP_STEPS)


def save_model(vae, ema_unet, latent_scale, step, val_loss, path=None):
    """
    The model, at half precision because this file is only ever used to generate with.
    Measured on held-out images, fp16 and fp32 weights reconstruct identically to four
    decimals. diffusion.pt beside it is seven times larger because resuming also needs the
    raw weights and AdamW's two running averages - sixteen bytes per parameter against two.
    """
    save_atomically(path or MODEL_PATH, {
        "vae": {k: v.half() for k, v in vae.state_dict().items()},
        "unet": {k: v.half() for k, v in ema_unet.state_dict().items()},
        "latent_scale": latent_scale, "step": step, "val_loss": val_loss})


def rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    # .cpu() because loading with map_location puts these on the GPU, and the random
    # number generators only accept their state back on the CPU
    torch.set_rng_state(state["cpu"].cpu().to(torch.uint8))
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu().to(torch.uint8))


@torch.no_grad()
def stream_batch(images, captions, tokenizer, rng, device, batch_size=BATCH_SIZE):
    """
    One training batch, read and processed on demand. Nothing is cached.

    This is the shape every large-scale pipeline has: the loop asks for a batch, the batch
    is fetched and prepared, and it is thrown away at the end of the step. Memory is
    O(batch), not O(dataset), so the same loop runs unchanged whether the corpus is the
    116k images here or a hundred million - swap `images` for a sharded reader and the
    training code below does not change at all.

    Mirroring is decided per batch, and the caption is drawn fresh from the five each
    image has, so a given image is rarely seen twice the same way.
    """
    index = np.sort(rng.integers(0, len(images), batch_size))
    x = to_batch(images, index, device, flip=bool(rng.random() < 0.5))
    text = [captions[i][rng.integers(0, len(captions[i]))] for i in index]
    return x, torch.from_numpy(tokenizer(text)).to(device)


@torch.no_grad()
def measure_latent_scale(vae, images, device, sample=2048):
    """
    Stable Diffusion multiplies latents by a fixed 0.18215 so they have roughly unit
    variance, because the noise schedule assumes data at that scale. We measure it instead
    of hard-coding it - from a sample, because the standard deviation of a million latents
    is not meaningfully different from that of a few thousand.
    """
    means = torch.cat([vae.encode(to_batch(images, slice(i, i + 256), device))[0]
                       for i in range(0, min(sample, len(images)), 256)])
    return 1.0 / means.float().std().item()


@torch.no_grad()
def validation_loss(unet, latents, context, pooled, alpha_bar, timesteps, noise):
    """
    The training objective, on held-out images, with fixed noise and fixed timesteps.

    Being deterministic given the weights is the whole point: two checkpoints are then
    directly comparable, which is what makes this usable for choosing the best one. The
    timesteps are spread evenly across the schedule so every noise level is represented.
    """
    unet.eval()
    total = 0.0
    for i in range(0, len(latents), 128):
        piece = slice(i, i + 128)
        t, z, n = timesteps[piece], latents[piece], noise[piece]
        prediction = unet(add_noise(z, n, t, alpha_bar), t, context[piece], pooled[piece])
        target = velocity(z, n, t, alpha_bar)
        total += (snr_weight(t, alpha_bar) * (prediction - target) ** 2).mean().item() * len(z)
    unet.train()
    return total / len(latents)


@torch.no_grad()
def write_samples(ema_unet, vae, context, pooled, empty_context, empty_pooled,
                  latent_scale, alpha_bar, device, step):
    """Four samples of each preview prompt, from the averaged weights and fixed noise."""
    ema_unet.eval()
    generator = torch.Generator(device=device).manual_seed(1234)
    latents = ddim_sample(ema_unet, context, pooled, empty_context, empty_pooled,
                          alpha_bar, generator=generator)
    images = vae.decode(latents / latent_scale).cpu().numpy()
    save_grid(images, f"{OUT_DIR}/samples_{step:06d}.png", columns=4)
    save_grid(images, f"{OUT_DIR}/samples_latest.png", columns=4)
    ema_unet.train()


def train_diffusion(vae, train_images, train_captions, test_images, test_captions, device):
    """`vae` is frozen: it only turns images into latents and latents back into images."""
    tokenizer = clip_text.Tokenizer()
    clip = clip_text.load_clip_text(device)

    alpha_bar = make_alpha_bar(device)
    latent_scale = measure_latent_scale(vae, train_images, device)
    print(f"  latent scale {latent_scale:.4f} (measured on a sample)")

    with torch.no_grad():
        empty_context, empty_pooled = clip(torch.from_numpy(tokenizer([""])).to(device))

    # --- the held-out validation set: 512 latents, built once and never changed ---
    # This one *is* cached, deliberately. A validation number is only useful if two
    # checkpoints see exactly the same latents, captions, noise and timesteps - so this
    # set is fixed at startup. It is 2 MB, and it is evaluation, not training data.
    with torch.no_grad():
        val_latents = torch.cat([vae.encode(to_batch(test_images, slice(i, i + 128), device))[0]
                                 for i in range(0, VALIDATION_LATENTS, 128)]).float() * latent_scale
        val_context, val_pooled = clip(torch.from_numpy(
            tokenizer([c[0] for c in test_captions[:VALIDATION_LATENTS]])).to(device))
    val_timesteps = torch.linspace(0, len(alpha_bar) - 1, VALIDATION_LATENTS,
                                   device=device).long()
    val_noise = torch.randn(val_latents.shape, generator=torch.Generator(device=device)
                            .manual_seed(999), device=device)

    # --- the preview prompts, encoded once ---
    preview_tokens = torch.from_numpy(tokenizer([p for p in PREVIEW_PROMPTS for _ in range(4)]))
    with torch.no_grad():
        preview_context, preview_pooled = clip(preview_tokens.to(device))

    unet = UNet().to(device)
    ema_unet = UNet().to(device)
    ema_unet.load_state_dict(unet.state_dict())
    for parameter in ema_unet.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=DIFFUSION_LR, weight_decay=0.01)
    start_step, log, best_val = 0, [], float("inf")

    if os.path.exists(DIFFUSION_PATH):
        checkpoint = torch.load(DIFFUSION_PATH, map_location=device, weights_only=False)
        unet.load_state_dict(checkpoint["unet"])
        ema_unet.load_state_dict(checkpoint["ema_unet"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, log, best_val = checkpoint["step"], checkpoint["log"], checkpoint["best_val"]
        restore_rng(checkpoint["rng"])
        print(f"\nresuming from step {start_step} (best validation loss {best_val:.4f})")
        if start_step >= DIFFUSION_STEPS:
            print(f"  already finished. Raise DIFFUSION_STEPS to train further.")
            return
    else:
        print(f"\n{count_parameters(unet)/1e6:.2f}M UNet + {count_parameters(clip)/1e6:.1f}M "
              f"frozen CLIP, {DIFFUSION_STEPS} steps at batch {BATCH_SIZE} "
              f"= {DIFFUSION_STEPS*BATCH_SIZE/len(train_images):.0f} passes over the data")

    ema_parameters, unet_parameters = list(ema_unet.parameters()), list(unet.parameters())
    rng = np.random.default_rng(start_step)      # so a resumed run does not repeat batches
    running, start = 0.0, time.time()

    for step in range(start_step + 1, DIFFUSION_STEPS + 1):
        # fetch, encode and tokenise this batch, then throw it away
        x, tokens = stream_batch(train_images, train_captions, tokenizer, rng, device)
        with torch.no_grad():
            context, pooled = clip(tokens)
            mean, logvar = vae.encode(x)
            latents = vae.sample(mean, logvar) * latent_scale

        loss = training_loss(unet, latents, context, pooled,
                             empty_context, empty_pooled, alpha_bar)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            # the EMA is useless until it has seen enough steps, so let it track the raw
            # weights closely at first and slow down as it fills - standard practice, and
            # it makes the early sample sheets worth looking at
            decay = min(EMA_DECAY, (step + 1) / (step + 10))
            torch._foreach_lerp_(ema_parameters, unet_parameters, 1 - decay)

        running += loss.item()
        if step % 100 == 0:
            rate = (step - start_step) / (time.time() - start)
            print(f"  step {step:6d}/{DIFFUSION_STEPS}  loss {running/100:.4f}  "
                  f"({rate:.2f} steps/s, {(DIFFUSION_STEPS-step)/rate/3600:.1f}h left)", flush=True)
            running = 0.0

        if step % SAVE_EVERY == 0 or step == DIFFUSION_STEPS:
            val = validation_loss(ema_unet, val_latents, val_context, val_pooled,
                                  alpha_bar, val_timesteps, val_noise)
            improved = val < best_val
            if improved:
                best_val = val
                save_model(vae, ema_unet, latent_scale, step, val)
            log.append((step, val))
            if step % ARCHIVE_EVERY == 0:
                save_model(vae, ema_unet, latent_scale, step, val,
                           path=f"{OUT_DIR}/model_step{step:06d}.pt")

            save_atomically(DIFFUSION_PATH, {
                "unet": unet.state_dict(), "ema_unet": ema_unet.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "log": log,
                "latent_scale": latent_scale, "best_val": best_val, "rng": rng_state()})
            np.savetxt(f"{OUT_DIR}/diffusion_log.csv", np.array(log), header="step,val_loss")
            write_samples(ema_unet, vae, preview_context, preview_pooled,
                          empty_context, empty_pooled, latent_scale, alpha_bar, device, step)
            print(f"  step {step:6d}  held-out validation loss {val:.4f}"
                  f"{'  <- best' if improved else ''}  -> samples_{step:06d}.png", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    torch.manual_seed(0)
    print(f"device: {device}")

    if not os.path.exists(VAE_PATH):
        raise SystemExit(f"{VAE_PATH} not found - run  python train_vae.py  first.")

    images, captions = dataset.load()
    train_images, test_images = images[:-dataset.HOLDOUT], images[-dataset.HOLDOUT:]
    train_captions, test_captions = captions[:-dataset.HOLDOUT], captions[-dataset.HOLDOUT:]
    print(f"{len(train_images)} training images, {len(train_captions)*5} captions, "
          f"{len(test_images)} held out")

    vae = VAE().to(device).eval()
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device, weights_only=False)["vae"])
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"loaded {VAE_PATH}")

    train_diffusion(vae, train_images, train_captions, test_images, test_captions, device)
    print(f"\ndone. {MODEL_PATH} is the model; {DIFFUSION_PATH} is the training state "
          "and can be deleted once you stop training.")


if __name__ == "__main__":
    main()
