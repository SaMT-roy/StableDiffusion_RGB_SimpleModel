"""
The diffusion process: the mathematics, with no networks in it.

Forward (fixed, nothing learned). Mix noise into a clean latent until only static is
left. The whole chain collapses into one line, so we can jump straight to any step t:

    x_t = sqrt(ab_t) * x_0  +  sqrt(1 - ab_t) * noise

`ab_t` ("alpha bar") slides from 1 at t=0 down to 0 at t=T.

Reverse (learned). Show the network x_t and ask what has to be removed. We ask for the
"velocity" rather than the noise:

    v = sqrt(ab_t) * noise  -  sqrt(1 - ab_t) * x_0

which is just a rotation of the same information - given v you can recover both:

    x_0   = sqrt(ab_t) * x_t  -  sqrt(1 - ab_t) * v
    noise = sqrt(1 - ab_t) * x_t  +  sqrt(ab_t) * v

Why v and not the noise, which is what the original DDPM paper predicts? Because we
rescale the schedule so the final step is *exactly* pure noise (see below). At that
point recovering x_0 from a noise prediction means dividing by sqrt(ab) = 0. With v
there is no division anywhere: at ab = 0 the formula simply says x_0 = -v. This pairing
of zero terminal SNR with v-prediction is from Lin et al. 2024, and it is what lets the
model control overall brightness instead of making everything mid-grey.

Run `python diffusion.py` for a self-test.
"""

import math

import torch

from model import LATENT_CHANNELS, LATENT_SIZE

NUM_TRAIN_STEPS = 1000


def make_alpha_bar(device="cpu", num_steps=NUM_TRAIN_STEPS, s=0.008):
    """
    The noise schedule: alpha_bar[t] is the fraction of the original signal left at step t.

    Cosine (Nichol & Dhariwal 2021), which spreads the difficulty evenly, then rescaled so
    alpha_bar[T] is exactly zero (Lin et al. 2024). Without that rescale the last step
    still holds a faint trace of the image - so the model is never trained on pure static,
    yet pure static is exactly what we hand it when generating.
    """
    t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    # Slide and stretch sqrt(alpha_bar) so it ends at 0 and still starts where it did.
    signal = alpha_bar.sqrt()
    first, last = signal[0].clone(), signal[-1].clone()
    signal = (signal - last) * (first / (first - last))
    return (signal ** 2).to(device)


def add_noise(x0, noise, t, alpha_bar):
    """The forward process: mix `noise` into `x0` by the amount prescribed for step t."""
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def velocity(x0, noise, t, alpha_bar):
    """What we ask the network to predict."""
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * noise - (1 - ab).sqrt() * x0


def snr_weight(t, alpha_bar, gamma=5.0):
    """
    Min-SNR weighting (Hang et al. 2023), in its v-prediction form.

    Timesteps differ enormously in difficulty, and with uniform sampling the easy ones
    drown out the rest. This caps how much any single timestep can contribute. The +1 is
    what makes it correct for v-prediction: without it the weight would reach zero at the
    pure-noise end of the schedule, which is exactly the end zero-terminal-SNR exists to
    make meaningful.
    """
    snr = alpha_bar[t] / (1 - alpha_bar[t]).clamp(min=1e-8)
    return ((snr + 1).clamp(max=gamma) / (snr + 1)).view(-1, 1, 1, 1)


def training_loss(unet, latents, context, pooled, empty_context, empty_pooled,
                  alpha_bar, drop_prob=0.1, gamma=5.0):
    """
    One training step's loss.

      1. pick a random timestep for each latent in the batch
      2. add that much noise
      3. ask the UNet for the velocity, given the noisy latent and the caption
      4. weighted mean-squared error against the true velocity
    """
    batch, device = latents.shape[0], latents.device
    t = torch.randint(0, len(alpha_bar), (batch,), device=device)
    noise = torch.randn_like(latents)
    noisy = add_noise(latents, noise, t, alpha_bar)
    target = velocity(latents, noise, t, alpha_bar)

    # Classifier-free guidance training: throw the caption away 10% of the time, so one
    # network learns both "a picture matching this text" and "a picture in general".
    # The gap between those two answers is what we amplify at generation time.
    drop = torch.rand(batch, device=device) < drop_prob
    context = torch.where(drop[:, None, None], empty_context, context)
    pooled = torch.where(drop[:, None], empty_pooled, pooled)

    prediction = unet(noisy, t, context, pooled)
    return (snr_weight(t, alpha_bar, gamma) * (prediction - target) ** 2).mean()


@torch.no_grad()
def ddim_sample(unet, context, pooled, empty_context, empty_pooled, alpha_bar,
                num_steps=50, guidance=5.0, rescale=0.7, generator=None, snapshots=None):
    """
    Generate latents from pure noise (DDIM, Song et al. 2020).

    DDIM's observation is that if you trust the network's prediction you can hop straight
    from timestep t to any earlier one, so 50 hops replace all 1000 steps. Each hop:
    ask the network what is there, rebuild the guesses for x_0 and the noise, then mix
    them back together at the earlier noise level.
    """
    device = next(unet.parameters()).device
    batch = context.shape[0]
    z = torch.randn(batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
                    device=device, generator=generator)

    # the two halves of the guidance pair, stacked so a single UNet call does both
    both_context = torch.cat([empty_context.expand(batch, -1, -1), context])
    both_pooled = torch.cat([empty_pooled.expand(batch, -1), pooled])

    # "Trailing" spacing, so the very first step really is the terminal timestep - the one
    # the zero-SNR rescale above exists to make meaningful.
    timesteps = [int(t) for t in torch.linspace(len(alpha_bar) - 1, 0, num_steps).round()]
    for i, t in enumerate(timesteps):
        ab = alpha_bar[t]
        ab_prev = alpha_bar[timesteps[i + 1]] if i + 1 < num_steps else torch.ones((), device=device)

        t_batch = torch.full((2 * batch,), t, device=device, dtype=torch.long)
        v_uncond, v_cond = unet(torch.cat([z, z]), t_batch, both_context, both_pooled).chunk(2)

        # classifier-free guidance: exaggerate the difference the caption makes
        v = v_uncond + guidance * (v_cond - v_uncond)
        # That exaggeration also inflates the spread of the prediction, which is what
        # bleaches and blows out high-guidance images. Put the original spread back.
        spread = lambda a: a.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        v = rescale * v * (spread(v_cond) / spread(v)) + (1 - rescale) * v

        # Guess at the finished latent. The clamp is a safety net against a numerical
        # blow-up, and is set where it cannot touch real data: measured on held-out images,
        # 0.025% of genuine latent values exceed +-4 but none exceed +-8, so the +-4 this
        # used to be was quietly truncating real highlights and shadows.
        x0 = (ab.sqrt() * z - (1 - ab).sqrt() * v).clamp(-8, 8)
        noise = (1 - ab).sqrt() * z + ab.sqrt() * v               # ... and at the noise in it
        z = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * noise    # step back to a cleaner level

        if snapshots is not None:
            snapshots.append(x0.cpu())
    return z


if __name__ == "__main__":
    alpha_bar = make_alpha_bar()
    print("noise schedule (how much of the original signal survives):")
    for t in [0, 100, 300, 500, 700, 900, 999]:
        ab = alpha_bar[t].item()
        print(f"  t={t:4d}  alpha_bar={ab:.6f}   signal={ab**0.5:.4f}  noise={(1-ab)**0.5:.4f}")
    print(f"\n  terminal alpha_bar is exactly zero: {alpha_bar[-1].item() == 0.0}"
          "   <- the last step is pure noise, as generation assumes")

    # A correct forward process turns unit-variance data into unit-variance static.
    x0 = torch.randn(4096, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    print("\nforward process keeps the scale constant:")
    for t in [0, 500, 999]:
        xt = add_noise(x0, torch.randn_like(x0), torch.full((4096,), t), alpha_bar)
        print(f"  x_{t:<4d} mean={xt.mean():+.4f}  std={xt.std():.4f}")

    # The velocity has to be exactly invertible, or the sampler is solving the wrong problem.
    print("\nv-prediction round-trip (worst error over all 1000 timesteps):")
    worst_x0 = worst_noise = 0.0
    for t_value in range(NUM_TRAIN_STEPS):
        t = torch.full((64,), t_value)
        x0, noise = torch.randn(64, 4, 16, 16), torch.randn(64, 4, 16, 16)
        xt, v = add_noise(x0, noise, t, alpha_bar), velocity(x0, noise, t, alpha_bar)
        ab = alpha_bar[t].view(-1, 1, 1, 1)
        worst_x0 = max(worst_x0, (ab.sqrt() * xt - (1 - ab).sqrt() * v - x0).abs().max().item())
        worst_noise = max(worst_noise, ((1 - ab).sqrt() * xt + ab.sqrt() * v - noise).abs().max().item())
    print(f"  recovering x_0   : max error {worst_x0:.2e}")
    print(f"  recovering noise : max error {worst_noise:.2e}")

    print("\nmin-SNR loss weight by timestep (gamma=5):")
    for t in [0, 100, 300, 500, 700, 900, 999]:
        snr = alpha_bar[t] / (1 - alpha_bar[t]).clamp(min=1e-8)
        print(f"  t={t:4d}  snr={snr:10.2f}  weight={((snr+1).clamp(max=5.0)/(snr+1)).item():.4f}")
