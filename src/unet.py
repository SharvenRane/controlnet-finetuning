"""A tiny diffusion UNet backbone.

This is a small but architecturally real UNet of the kind used in diffusion
models. It has a downsampling encoder, a bottleneck, and an upsampling decoder
with skip connections. A sinusoidal timestep embedding is injected into every
residual block. The network is deliberately small so the unit tests run fast
on a CPU.

The block structure is designed so a ControlNet can hook into it. The backbone
exposes its encoder and bottleneck features through a forward pass that can
optionally accept additive control residuals, one per skip and one for the
bottleneck. When no residuals are passed the backbone behaves like an ordinary
UNet.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Build sinusoidal timestep embeddings.

    Args:
        timesteps: a 1D tensor of shape (batch,) holding integer or float steps.
        dim: the width of the embedding.

    Returns:
        A tensor of shape (batch, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """A residual block with group norm, SiLU, and a timestep bias."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 4):
        super().__init__()
        g_in = math.gcd(groups, in_ch) or 1
        g_out = math.gcd(groups, out_ch) or 1
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class TinyUNet(nn.Module):
    """A small UNet with two resolution levels.

    Args:
        in_ch: input image channels.
        base_ch: width of the first level.
        ch_mult: per level channel multipliers. Length sets the depth.
        time_dim: width of the timestep embedding.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        ch_mult: tuple = (1, 2),
        time_dim: int = 32,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder.
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        chans = [base_ch]
        cur = base_ch
        for i, mult in enumerate(ch_mult):
            out = base_ch * mult
            self.down_blocks.append(ResBlock(cur, out, time_dim))
            cur = out
            chans.append(cur)
            if i < len(ch_mult) - 1:
                self.downsamplers.append(Downsample(cur))
            else:
                self.downsamplers.append(None)

        # Bottleneck.
        self.mid_block = ResBlock(cur, cur, time_dim)

        # Decoder. Consumes skips in reverse.
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            skip_ch = chans.pop()
            out = base_ch * mult
            self.up_blocks.append(ResBlock(cur + skip_ch, out, time_dim))
            cur = out
            level = len(ch_mult) - 1 - i
            if level > 0:
                self.upsamplers.append(Upsample(cur))
            else:
                self.upsamplers.append(None)

        g_out = math.gcd(4, cur) or 1
        self.out_norm = nn.GroupNorm(g_out, cur)
        self.out_conv = nn.Conv2d(cur, in_ch, 3, padding=1)

    def encode(self, x: torch.Tensor, t_emb: torch.Tensor):
        """Run the encoder and bottleneck.

        Returns:
            mid: bottleneck feature.
            skips: list of encoder skip features, shallow to deep.
        """
        h = self.in_conv(x)
        skips: List[torch.Tensor] = [h]
        for block, down in zip(self.down_blocks, self.downsamplers):
            h = block(h, t_emb)
            skips.append(h)
            if down is not None:
                h = down(h)
        mid = self.mid_block(h, t_emb)
        return mid, skips

    def decode(
        self,
        mid: torch.Tensor,
        skips: List[torch.Tensor],
        t_emb: torch.Tensor,
        control_residuals: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run the decoder.

        Args:
            mid: bottleneck feature, possibly with a control residual already
                added by the caller.
            skips: encoder skip features. The caller may have already added
                control residuals to these.
            t_emb: timestep embedding.
            control_residuals: unused here. Residual injection is done by the
                ControlledUNet before calling decode so that the same decode
                path serves both controlled and uncontrolled runs.

        Returns:
            The predicted output image.
        """
        h = mid
        skips = list(skips)
        for block, up in zip(self.up_blocks, self.upsamplers):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            if up is not None:
                h = up(h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        mid, skips = self.encode(x, t_emb)
        return self.decode(mid, skips, t_emb)
