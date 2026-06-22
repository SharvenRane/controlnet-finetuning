"""ControlNet style conditional control for the tiny UNet.

The idea follows the original ControlNet. We freeze the backbone UNet and add a
trainable copy of its encoder, the control branch. A small conditioning network
turns the control image, for example edges or a mask, into a feature map that is
added to the input of the control branch. The control branch produces one
residual feature per skip connection and one for the bottleneck. Each residual
passes through a zero initialized convolution before it is added back into the
frozen backbone.

The zero initialized convolutions are the crucial detail. At initialization
every control residual is exactly zero, so the controlled model reproduces the
frozen backbone output bit for bit. As training proceeds the zero convolutions
learn nonzero weights and the control signal begins to steer the output. This
gives the network a safe starting point and is what the unit tests verify.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import TinyUNet, ResBlock, Downsample, timestep_embedding


def zero_conv(in_ch: int, out_ch: int) -> nn.Conv2d:
    """A 1x1 convolution with weights and bias initialized to zero."""
    conv = nn.Conv2d(in_ch, out_ch, 1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class ControlHint(nn.Module):
    """Encode a raw control image into a feature map matching the UNet input.

    A few small convolutions lift the conditioning image, such as a Canny edge
    map or a segmentation mask, into the same channel width the backbone uses
    right after its input convolution.
    """

    def __init__(self, hint_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hint_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )
        # A zero convolution so the hint contributes nothing at init.
        self.out = zero_conv(out_ch, out_ch)

    def forward(self, hint: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(hint))


class ControlNet(nn.Module):
    """A trainable copy of the backbone encoder that emits control residuals.

    The branch mirrors the backbone encoder layout exactly so its feature maps
    line up with the backbone skips and bottleneck. Every emitted residual goes
    through its own zero convolution.
    """

    def __init__(self, backbone: TinyUNet, hint_ch: int = 1):
        super().__init__()
        base_ch = backbone.base_ch
        time_dim = backbone.time_dim
        self.time_dim = time_dim

        # The control branch has its own timestep MLP and input convolution,
        # initialized from a structural copy of the backbone so shapes match.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(backbone.in_ch, base_ch, 3, padding=1)

        self.hint = ControlHint(hint_ch, base_ch)

        # Rebuild encoder blocks with the same channel plan as the backbone.
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.skip_zeros = nn.ModuleList()

        # First skip is the input convolution output.
        self.skip_zeros.append(zero_conv(base_ch, base_ch))
        cur = base_ch
        n_levels = len(backbone.down_blocks)
        for i in range(n_levels):
            out = backbone.down_blocks[i].conv2.out_channels
            self.down_blocks.append(ResBlock(cur, out, time_dim))
            cur = out
            self.skip_zeros.append(zero_conv(out, out))
            if i < n_levels - 1:
                self.downsamplers.append(Downsample(cur))
            else:
                self.downsamplers.append(None)

        self.mid_block = ResBlock(cur, cur, time_dim)
        self.mid_zero = zero_conv(cur, cur)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, hint: torch.Tensor
    ):
        """Compute control residuals.

        Returns:
            mid_res: residual for the bottleneck.
            skip_res: list of residuals, one per backbone skip, shallow to deep.
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.in_conv(x) + self.hint(hint)

        skip_res: List[torch.Tensor] = [self.skip_zeros[0](h)]
        for i, (block, down) in enumerate(
            zip(self.down_blocks, self.downsamplers)
        ):
            h = block(h, t_emb)
            skip_res.append(self.skip_zeros[i + 1](h))
            if down is not None:
                h = down(h)
        mid = self.mid_block(h, t_emb)
        mid_res = self.mid_zero(mid)
        return mid_res, skip_res


class ControlledUNet(nn.Module):
    """Wrap a frozen backbone and a ControlNet branch.

    Calling the module with a control hint injects the branch residuals into the
    backbone. Calling it with the hint omitted, or with a zero hint at
    initialization, reproduces the plain backbone output.
    """

    def __init__(self, backbone: TinyUNet, hint_ch: int = 1, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.control = ControlNet(backbone, hint_ch=hint_ch)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def base_forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The plain backbone output with no control."""
        return self.backbone(x, t)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hint: torch.Tensor = None,
    ) -> torch.Tensor:
        t_emb = self.backbone.time_mlp(
            timestep_embedding(t, self.backbone.time_dim)
        )
        mid, skips = self.backbone.encode(x, t_emb)

        if hint is not None:
            mid_res, skip_res = self.control(x, t, hint)
            mid = mid + mid_res
            skips = [s + r for s, r in zip(skips, skip_res)]

        return self.backbone.decode(mid, skips, t_emb)
