"""
Stage 1: train the autoencoder.   python train_vae.py

The VAE squeezes a 64x64x3 image into a 4x16x16 latent and back - 12x compression. Its
reconstruction quality is a hard ceiling on everything the diffusion model can ever
produce, so it is worth getting right before anything else runs.

The objective is the one Stable Diffusion's autoencoder uses (Esser et al. 2021,
Rombach et al. 2022):

  L1            per-pixel accuracy. L1 rather than L2 because, faced with uncertainty,
                L2's best answer is the average of the plausible textures - which is
                literally a blur - whereas L1's is to commit to one of them.
  LPIPS         perceptual accuracy: detail in the right *places*, not merely present.
                Without it a pixel loss and a critic both just push high-frequency energy,
                and the cheapest way to satisfy them is to sprinkle texture everywhere.
  KL (1e-6)     a whisper of pressure holding latents near N(0,1) so they cannot drift to
                huge magnitudes. Deliberately tiny, exactly as in SD.
  adversarial   a PatchGAN critic, weighted adaptively (see adaptive_weight).

Two phases. Up to DISC_START_EPOCH there is no critic, just reconstruction, with the
learning rate decaying from RECONSTRUCTION_LR to ADVERSARIAL_LR. After that the critic
joins - warming up alone for CRITIC_WARMUP_STEPS first, then voting - and the rate holds
low, because adversarial training is far less forgiving of a large step.

Two weight files come out: vae.pt is the rolling checkpoint (resume state, overwritten
every epoch) and vae_best.pt is the weights at the lowest held-out LPIPS, which is what
stage 2 loads. A reconstruction sheet is written every epoch.
"""

import os
import time

import numpy as np
import torch
from PIL import Image

import dataset
from lpips import LPIPS
from model import VAE, Discriminator, count_parameters, kl_divergence

# ----------------------------------------------------------------------------
# Settings - the only knobs. Every value here was measured, not guessed.
# ----------------------------------------------------------------------------
EPOCHS = 10                  # 116k images at batch 64 = 1,816 steps/epoch, so ~18,000 steps
BATCH_SIZE = 64
PROGRESS_EVERY = 100         # print a within-epoch progress line this often
KL_WEIGHT = 1e-6
PERCEPTUAL_WEIGHT = 1.0

RECONSTRUCTION_LR = 2e-4     # start here, decayed across the reconstruction phase
ADVERSARIAL_LR = 5e-5        # ... to here, which is also what LDM trains this model at
DISC_START_EPOCH = 0.6 * EPOCHS       # the critic joins 60% of the way in, as before
DISC_WEIGHT = 0.1            # the critic's gradient, as a fraction of reconstruction's.
                             # SD uses 0.5, tuned for an 84M autoencoder over 100k+
                             # adversarial steps. Measured at this size: 0.5 ran detail to
                             # 108% and wrecked LPIPS, 0.2 ran it to 103% and never beat
                             # the pre-critic checkpoint, 0.1 improved both (LPIPS
                             # 0.0247 -> 0.0245, detail 80% -> 83%). 0.1 is the only one
                             # of the three where the critic earned its place.
CRITIC_WARMUP_STEPS = 500    # the critic trains alone this long before it gets a vote.
                             # Without this it is still random when it starts voting, its
                             # gradient is near zero, and the adaptive weight below then
                             # amplifies pure noise to full strength.

OUT_DIR = "out"
VAE_PATH = f"{OUT_DIR}/vae.pt"            # rolling: resume state, overwritten each epoch
VAE_BEST_PATH = f"{OUT_DIR}/vae_best.pt"  # weights only, at the lowest held-out LPIPS


# ----------------------------------------------------------------------------
# Small shared helpers (train.py and generate.py import these)
# ----------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_batch(images, index, device, flip=False):
    """uint8 (N,64,64,3) -> float (N,3,64,64) in [-1,1], optionally mirrored."""
    x = torch.from_numpy(images[index]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x.flip(-1) if flip else x


def save_grid(images, path, columns):
    """images: (n,3,64,64) in [-1,1] -> one PNG, doubled in size so it is easy to look at."""
    images = np.clip((images + 1) * 127.5, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    rows = int(np.ceil(len(images) / columns))
    size = images.shape[1]
    canvas = np.zeros((rows * size, columns * size, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        row, column = divmod(i, columns)
        canvas[row * size:(row + 1) * size, column * size:(column + 1) * size] = image
    Image.fromarray(canvas).resize((columns * size * 2, rows * size * 2), Image.NEAREST).save(path)


def save_atomically(path, payload):
    """Write to a temporary file and rename, so a crash mid-save cannot leave a corrupt
    checkpoint behind - the previous one survives untouched."""
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def learning_rate(epoch):
    """
    Decay from RECONSTRUCTION_LR to ADVERSARIAL_LR across the reconstruction phase, then
    hold. Measured: dropping the rate by hand at epoch 13 of the previous run improved
    held-out LPIPS from 0.0284 to 0.0247 within two epochs, which says the high rate was
    already overshooting long before then.
    """
    if epoch >= DISC_START_EPOCH:
        return ADVERSARIAL_LR
    progress = epoch / DISC_START_EPOCH
    return ADVERSARIAL_LR + (RECONSTRUCTION_LR - ADVERSARIAL_LR) * 0.5 * (1 + np.cos(np.pi * progress))


# ----------------------------------------------------------------------------
# The adversarial part
# ----------------------------------------------------------------------------
def hinge_loss(real_scores, fake_scores):
    """
    The critic's objective: push real patches above +1 and reconstructed ones below -1.

    Watch this number. A critic emitting any constant between -1 and +1 - one that cannot
    tell real from reconstructed at all - scores exactly 1.0. If it sits there, the
    adversarial term is contributing nothing but noise.
    """
    return 0.5 * (torch.relu(1 - real_scores).mean() + torch.relu(1 + fake_scores).mean())


def adaptive_weight(reconstruction_loss, critic_loss, last_layer):
    """
    Scale the critic's gradient to a fixed fraction of the reconstruction gradient,
    measured at the decoder's final layer (Esser et al. 2021).

    Note what this does *not* do: it is one number per step, so it cannot decide where the
    critic applies. It makes the critic's contribution a defined fraction of
    reconstruction's, whatever the critic happens to be saying - which is only safe once
    the critic is saying something useful. Hence CRITIC_WARMUP_STEPS.
    """
    reconstruction_grad = torch.autograd.grad(reconstruction_loss, last_layer, retain_graph=True)[0]
    critic_grad = torch.autograd.grad(critic_loss, last_layer, retain_graph=True)[0]
    return (reconstruction_grad.norm() / (critic_grad.norm() + 1e-4)).clamp(max=1e4).detach()


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def detail(x):
    """Mean absolute difference between neighbouring pixels: how much fine structure an
    image carries. A diagnostic, not a quality score - blurred reconstructions fall below
    the original and over-textured ones rise above it, but pure noise would score highest
    of all, which is why the best checkpoint is chosen on LPIPS instead."""
    return ((x[..., 1:] - x[..., :-1]).abs().mean()
            + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()).item()


@torch.no_grad()
def validate(vae, perceptual, test_images, device):
    """Held-out quality. LPIPS is the one to trust: PSNR falls once the critic is active
    even as the pictures improve, because pixel accuracy is what buys sharpness."""
    vae.eval()
    batches = 8
    lpips_total = mse_total = real_detail = reconstructed_detail = 0.0
    for i in range(batches):
        x = to_batch(test_images, slice(i * 128, (i + 1) * 128), device)
        reconstruction = vae.decode(vae.encode(x)[0])
        lpips_total += perceptual(x, reconstruction).mean().item()
        mse_total += torch.nn.functional.mse_loss(reconstruction, x).item()
        real_detail += detail(x)
        reconstructed_detail += detail(reconstruction)
    vae.train()
    return (lpips_total / batches,
            10 * np.log10(4.0 / (mse_total / batches)),
            100 * reconstructed_detail / real_detail)


@torch.no_grad()
def write_reconstruction(vae, test_images, device, epoch):
    """Eight held-out images above their reconstructions - one sheet per epoch, kept."""
    vae.eval()
    x = to_batch(test_images, slice(0, 8), device)
    pairs = torch.cat([x, vae.decode(vae.encode(x)[0])]).cpu().numpy()
    save_grid(pairs, f"{OUT_DIR}/recon_epoch{epoch:02d}.png", columns=8)
    save_grid(pairs, f"{OUT_DIR}/recon_latest.png", columns=8)
    vae.train()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    torch.manual_seed(0)
    print(f"device: {device}")

    images, _ = dataset.load()
    train_images, test_images = images[:-dataset.HOLDOUT], images[-dataset.HOLDOUT:]
    print(f"{len(train_images)} training images, {len(test_images)} held out")

    vae = VAE().to(device)
    critic = Discriminator().to(device)
    perceptual = LPIPS().to(device)
    last_layer = vae.decoder[-1].weight        # where the adaptive weight is measured

    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=RECONSTRUCTION_LR, betas=(0.5, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=ADVERSARIAL_LR, betas=(0.5, 0.9))
    start_epoch, log, critic_steps, best_lpips = 0, [], 0, float("inf")

    if os.path.exists(VAE_PATH):
        checkpoint = torch.load(VAE_PATH, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint["vae"])
        critic.load_state_dict(checkpoint["critic"])
        vae_optimizer.load_state_dict(checkpoint["vae_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_epoch, log = checkpoint["epoch"], checkpoint["log"]
        critic_steps, best_lpips = checkpoint["critic_steps"], checkpoint["best_lpips"]
        torch.set_rng_state(checkpoint["rng"].cpu().to(torch.uint8))
        if start_epoch >= EPOCHS:
            print(f"already trained for {start_epoch} epochs. Delete {VAE_PATH} to retrain.")
            return
        print(f"resuming from epoch {start_epoch} (best held-out lpips so far {best_lpips:.4f})")
    else:
        print(f"VAE {count_parameters(vae)/1e6:.2f}M + critic {count_parameters(critic)/1e6:.2f}M "
              f"+ frozen LPIPS {count_parameters(perceptual)/1e6:.2f}M")
        print(f"{EPOCHS} epochs; the critic joins at epoch {DISC_START_EPOCH + 1} "
              f"at {DISC_WEIGHT:.0%} of the reconstruction gradient")

    steps_per_epoch = len(train_images) // BATCH_SIZE
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        order = np.random.default_rng(epoch).permutation(len(train_images))
        adversarial = epoch >= DISC_START_EPOCH
        for group in vae_optimizer.param_groups:
            group["lr"] = learning_rate(epoch)
        totals = np.zeros(4)                   # pixel, perceptual, critic, adaptive weight

        for i in range(steps_per_epoch):
            index = np.sort(order[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            real = to_batch(train_images, index, device, flip=bool(torch.rand(1) < 0.5))
            reconstruction, mean, logvar = vae(real)

            critic_loss = torch.zeros((), device=device)
            if adversarial:
                critic_loss = hinge_loss(critic(real), critic(reconstruction.detach()))
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_optimizer.step()
                critic_steps += 1

            pixel = (reconstruction - real).abs().mean()
            perceptual_loss = perceptual(real, reconstruction).mean()
            reconstruction_loss = pixel + PERCEPTUAL_WEIGHT * perceptual_loss
            loss = reconstruction_loss + KL_WEIGHT * kl_divergence(mean, logvar)

            weight = 0.0                       # the decoder hears the critic only once it knows something
            if adversarial and critic_steps > CRITIC_WARMUP_STEPS:
                fool_critic = -critic(reconstruction).mean()
                weight = adaptive_weight(reconstruction_loss, fool_critic, last_layer) * DISC_WEIGHT
                loss = loss + weight * fool_critic
                weight = float(weight)

            vae_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            vae_optimizer.step()
            totals += [pixel.item(), perceptual_loss.item(), float(critic_loss.detach()), weight]

            if (i + 1) % PROGRESS_EVERY == 0:
                done = i + 1
                rate = done / (time.time() - start)
                print(f"    step {done:5d}/{steps_per_epoch}  L1 {totals[0]/done:.4f}  "
                      f"lpips {totals[1]/done:.4f}  ({rate:.1f} steps/s, "
                      f"{(steps_per_epoch - done)/rate/60:.0f} min left in epoch)", flush=True)

        pixel, perceptual_mean, critic_mean, weight_mean = totals / steps_per_epoch
        val_lpips, psnr, detail_percent = validate(vae, perceptual, test_images, device)
        log.append((epoch + 1, pixel, perceptual_mean, val_lpips, psnr, detail_percent))

        improved = val_lpips < best_lpips
        if improved:
            best_lpips = val_lpips
            save_atomically(VAE_BEST_PATH, {"vae": vae.state_dict(), "epoch": epoch + 1,
                                            "val_lpips": val_lpips})

        print(f"  epoch {epoch+1:2d}/{EPOCHS}  lr {learning_rate(epoch):.1e}  "
              f"L1 {pixel:.4f} lpips {perceptual_mean:.4f}"
              f"{f'  critic {critic_mean:.3f} w {weight_mean:.4f}' if adversarial else '  (critic off)'}"
              f"   |  held-out lpips {val_lpips:.4f}  psnr {psnr:.2f} dB  detail {detail_percent:.0f}%"
              f"{'  <- best' if improved else ''}  ({time.time()-start:.0f}s)", flush=True)

        write_reconstruction(vae, test_images, device, epoch + 1)
        save_atomically(VAE_PATH, {
            "vae": vae.state_dict(), "critic": critic.state_dict(),
            "vae_optimizer": vae_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "epoch": epoch + 1, "log": log, "critic_steps": critic_steps,
            "best_lpips": best_lpips, "rng": torch.get_rng_state()})
        np.savetxt(f"{OUT_DIR}/vae_log.csv", np.array(log),
                   header="epoch,train_l1,train_lpips,val_lpips,psnr,detail_percent")

    best = torch.load(VAE_BEST_PATH, map_location="cpu", weights_only=False)
    print(f"\ndone. best held-out lpips {best['val_lpips']:.4f} at epoch {best['epoch']}, "
          f"saved to {VAE_BEST_PATH} - this is what stage 2 loads.")


if __name__ == "__main__":
    main()
