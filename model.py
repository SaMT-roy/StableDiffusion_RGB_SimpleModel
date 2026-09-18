"""
The two networks we train.

  1. VAE   64x64x3 image  <->  4x16x16 latent        (12,288 numbers -> 1,024, a 12x squeeze)
  2. UNet  noisy latent + timestep + caption  ->  the "velocity" that has to be removed

The third network of a Stable Diffusion setup, the text encoder, is pretrained and
frozen - it lives in clip_text.py.

Why 12x and not Stable Diffusion's 48x? Because SD's autoencoder was trained on
hundreds of millions of images with a perceptual loss and a GAN discriminator. Ours
gets 116,000 images and an afternoon, so we ask it to throw away much less.

Run `python model.py` for a self-test.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_CHANNELS = 4
LATENT_SIZE = 16
IMAGE_SIZE = 64
CONTEXT_DIM = 512     # CLIP ViT-B/32's width: what the cross-attention layers read


def group_norm(channels):
    """Normalisation used throughout Stable Diffusion. Independent of batch size."""
    return nn.GroupNorm(min(32, channels // 4), channels)


def zero_init(module):
    """Start a layer at exactly zero so the block it ends begins life as a no-op.
    Training then starts from a stable identity function. Real SD does this too."""
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


# ============================================================================
# Shared building blocks
# ============================================================================
class ResBlock(nn.Module):
    """
    Two convolutions with a shortcut around them. In the UNet it also receives a
    conditioning vector (timestep + a summary of the caption), which is added to the
    feature maps so every layer knows how noisy the input is and what it should contain.
    """

    def __init__(self, in_channels, out_channels, time_dim=None):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels) if time_dim else None
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x, time_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_proj is not None:
            h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class Attention(nn.Module):
    """
    Multi-head attention. Every position builds a query, compares it against the keys
    of everything in `context`, and takes a weighted average of their values.

        context = None   -> self-attention  (the image looks at itself)
        context = words  -> cross-attention (the image looks at the caption)

    The second one is the mechanism that makes the picture follow the prompt.
    """

    def __init__(self, dim, context_dim=None, heads=8):
        super().__init__()
        context_dim = context_dim or dim
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        context = x if context is None else context
        batch, positions, dim = x.shape
        head_dim = dim // self.heads

        def split_heads(t):
            return t.view(batch, -1, self.heads, head_dim).transpose(1, 2)

        q, k, v = split_heads(self.to_q(x)), split_heads(self.to_k(context)), split_heads(self.to_v(context))
        out = F.scaled_dot_product_attention(q, k, v)      # the softmax(qk/sqrt(d))v formula
        out = out.transpose(1, 2).reshape(batch, positions, dim)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """self-attention -> cross-attention (optional) -> feed-forward, each with a shortcut."""

    def __init__(self, dim, context_dim=None, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, None, heads)
        self.norm2 = nn.LayerNorm(dim) if context_dim else None
        self.cross_attn = Attention(dim, context_dim, heads) if context_dim else None
        self.norm3 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, context=None):
        x = x + self.self_attn(self.norm1(x))
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        return x + self.feed_forward(self.norm3(x))


class SpatialTransformer(nn.Module):
    """
    Lets a TransformerBlock work on an image-shaped tensor: flatten the HxW grid into a
    sequence of H*W positions, attend, then fold it back into a grid.
    """

    def __init__(self, channels, context_dim=CONTEXT_DIM, heads=8):
        super().__init__()
        self.norm = group_norm(channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.block = TransformerBlock(channels, context_dim, heads)
        self.proj_out = zero_init(nn.Conv2d(channels, channels, 1))

    def forward(self, x, context=None):
        batch, channels, height, width = x.shape
        h = self.proj_in(self.norm(x))
        h = h.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        h = self.block(h, context)
        h = h.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return x + self.proj_out(h)


# ============================================================================
# 1. The VAE: image <-> latent
# ============================================================================
class VAE(nn.Module):
    """
    Squeezes a 3x64x64 image into a 4x16x16 latent and back.

    Diffusion on raw pixels means 12,288 numbers per image and 50 passes over all of
    them. Latent diffusion runs the expensive loop on 1,024 numbers instead, and only
    decodes once at the very end. That is the whole trick.

    It is *variational*: the encoder outputs a mean and a log-variance and we sample
    from that Gaussian. A very small penalty keeps those Gaussians near N(0,1), which
    stops latent values drifting to huge magnitudes.
    """

    def __init__(self, c1=48, c2=96, c3=192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),                       # 64x64
            ResBlock(c1, c1), ResBlock(c1, c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),            # -> 32x32
            ResBlock(c2, c2), ResBlock(c2, c2),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),            # -> 16x16
            ResBlock(c3, c3), ResBlock(c3, c3),
            SpatialTransformer(c3, context_dim=None),             # one global look, as in SD
            group_norm(c3), nn.SiLU(),
            nn.Conv2d(c3, 2 * LATENT_CHANNELS, 3, padding=1),     # 4 means + 4 log-variances
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, c3, 3, padding=1),         # 16x16
            ResBlock(c3, c3), ResBlock(c3, c3),
            SpatialTransformer(c3, context_dim=None),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c2, 3, padding=1),                      # -> 32x32
            ResBlock(c2, c2), ResBlock(c2, c2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, 3, padding=1),                      # -> 64x64
            ResBlock(c1, c1), ResBlock(c1, c1),
            group_norm(c1), nn.SiLU(),
            nn.Conv2d(c1, 3, 3, padding=1),
        )

    def encode(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)          # keeps exp(logvar) finite

    def sample(self, mean, logvar):
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decode(self.sample(mean, logvar)), mean, logvar


def kl_divergence(mean, logvar):
    """KL( N(mean, var) || N(0,1) ), summed over the latent and averaged over the batch."""
    return 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=(1, 2, 3)).mean()


# ============================================================================
# 2. The UNet: the denoiser
# ============================================================================
def timestep_embedding(timesteps, dim):
    """Turn the integer timestep into a vector of sines and cosines at many frequencies,
    giving the network a smooth notion of "how much noise is in this input"."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    angles = timesteps[:, None].float() * frequencies[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class UNet(nn.Module):
    r"""
    Walks the latent down to a coarse resolution and back up, with skip connections so
    fine detail survives the trip. Cross-attention at the two lower resolutions lets
    every position read the caption.

        16x16 ------------------------- skips --------------------------> 16x16
            \                                                            /
             8x8 --------------------- skips -------------------------> 8x8
                 \                                                     /
                  4x4 ------ skips ----> middle ------------------> 4x4

    There is deliberately no attention at 16x16: 256 positions attending to each other
    would dominate the cost for very little gain. Stable Diffusion leaves its own
    highest resolution un-attended for exactly the same reason.
    """

    def __init__(self, channels=128, context_dim=CONTEXT_DIM, heads=8):
        super().__init__()
        c1, c2 = channels, channels * 2          # 128 at 16x16, 256 at 8x8 and 4x4
        time_dim = channels * 4
        self.channels = channels

        self.time_mlp = nn.Sequential(nn.Linear(channels, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        # A summary of the whole caption, added to the timestep vector. Cross-attention
        # gives each position a detailed view of the words; this gives every layer a
        # cheap global one. SDXL does the same, and it helps a lot at small scale.
        self.text_pool = nn.Linear(context_dim, time_dim)

        self.conv_in = nn.Conv2d(LATENT_CHANNELS, c1, 3, padding=1)
        # ---- down, 16x16 ----
        self.d1 = ResBlock(c1, c1, time_dim)
        self.d2 = ResBlock(c1, c1, time_dim)
        self.down_a = nn.Conv2d(c1, c1, 3, stride=2, padding=1)                 # -> 8x8
        # ---- down, 8x8 ----
        self.d3, self.a3 = ResBlock(c1, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.d4, self.a4 = ResBlock(c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.down_b = nn.Conv2d(c2, c2, 3, stride=2, padding=1)                 # -> 4x4
        # ---- down, 4x4 ----
        self.d5, self.a5 = ResBlock(c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.d6, self.a6 = ResBlock(c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        # ---- middle ----
        self.m1 = ResBlock(c2, c2, time_dim)
        self.m_attn = SpatialTransformer(c2, context_dim, heads)
        self.m2 = ResBlock(c2, c2, time_dim)
        # ---- up, 4x4 (each input is concatenated with the matching skip) ----
        self.u6, self.b6 = ResBlock(c2 + c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.u5, self.b5 = ResBlock(c2 + c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.up_a = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                  nn.Conv2d(c2, c2, 3, padding=1))              # -> 8x8
        # ---- up, 8x8 ----
        self.u4, self.b4 = ResBlock(c2 + c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.u3, self.b3 = ResBlock(c2 + c2, c2, time_dim), SpatialTransformer(c2, context_dim, heads)
        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                  nn.Conv2d(c2, c1, 3, padding=1))              # -> 16x16
        # ---- up, 16x16 ----
        self.u2 = ResBlock(c1 + c1, c1, time_dim)
        self.u1 = ResBlock(c1 + c1, c1, time_dim)

        self.out_norm = group_norm(c1)
        self.out_conv = zero_init(nn.Conv2d(c1, LATENT_CHANNELS, 3, padding=1))

    def forward(self, z, timesteps, context, pooled):
        t = self.time_mlp(timestep_embedding(timesteps, self.channels)) + self.text_pool(pooled)

        h = self.conv_in(z)
        h = self.d1(h, t); skip1 = h
        h = self.d2(h, t); skip2 = h
        h = self.down_a(h)
        h = self.a3(self.d3(h, t), context); skip3 = h
        h = self.a4(self.d4(h, t), context); skip4 = h
        h = self.down_b(h)
        h = self.a5(self.d5(h, t), context); skip5 = h
        h = self.a6(self.d6(h, t), context); skip6 = h

        h = self.m2(self.m_attn(self.m1(h, t), context), t)

        h = self.b6(self.u6(torch.cat([h, skip6], 1), t), context)
        h = self.b5(self.u5(torch.cat([h, skip5], 1), t), context)
        h = self.up_a(h)
        h = self.b4(self.u4(torch.cat([h, skip4], 1), t), context)
        h = self.b3(self.u3(torch.cat([h, skip3], 1), t), context)
        h = self.up_b(h)
        h = self.u2(torch.cat([h, skip2], 1), t)
        h = self.u1(torch.cat([h, skip1], 1), t)
        return self.out_conv(F.silu(self.out_norm(h)))


# ============================================================================
# 3. The critic (training only - never shipped, never used to generate)
# ============================================================================
class Discriminator(nn.Module):
    """
    A PatchGAN critic, used only while training the VAE.

    A decoder trained on any per-pixel objective hedges: shown a patch that might be
    gravel or might be grass, the safest guess is the average of the two, which is a blur.
    This network never measures per-pixel accuracy. It asks one question - "does this look
    like a real photograph?" - and an average of two textures is a very poor answer to it.

    It scores overlapping patches rather than the whole image, so its verdict is about
    local texture, which is exactly what we want fixed. Thrown away once training ends.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2),      # 32x32
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2), nn.LeakyReLU(0.2),                       # 16x16
            nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, channels * 4), nn.LeakyReLU(0.2),                       # 8x8
            nn.Conv2d(channels * 4, 1, 3, padding=1),                               # a score per patch
        )
        # DCGAN initialisation - small normal weights. This is what taming-transformers
        # (and therefore Stable Diffusion) initialises this critic with; PyTorch's default
        # is tuned for classifiers, not for the delicate early balance of a GAN.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    vae, unet = VAE(), UNet()
    print(f"VAE           {count_parameters(vae)/1e6:6.2f}M parameters")
    print(f"UNet          {count_parameters(unet)/1e6:6.2f}M parameters")
    print(f"Discriminator {count_parameters(Discriminator())/1e6:6.2f}M parameters  (training only)")

    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    reconstruction, mean, logvar = vae(x)
    print(f"VAE   {tuple(x.shape)} -> {tuple(mean.shape)} -> {tuple(reconstruction.shape)}"
          f"   ({x[0].numel() / mean[0].numel():.0f}x compression)")

    z = torch.randn(2, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    context, pooled = torch.randn(2, 20, CONTEXT_DIM), torch.randn(2, CONTEXT_DIM)
    out = unet(z, torch.tensor([10, 999]), context, pooled)
    print(f"UNet  {tuple(z.shape)} -> {tuple(out.shape)}")
    print(f"output at initialisation is exactly zero: {bool((out == 0).all())}  "
          f"(the zero-initialised final conv)")
