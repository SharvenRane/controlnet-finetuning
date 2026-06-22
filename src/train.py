"""A minimal training step for the controlled UNet.

The setup mirrors how ControlNet is trained in practice. The backbone is frozen
and only the control branch and the conditioning network receive gradients. We
use a plain mean squared error objective, the same loss family used for noise
prediction in diffusion training. Here the target is a fixed synthetic tensor so
the test can run offline and deterministically.
"""

from typing import Optional

import torch
import torch.nn as nn

from .controlnet import ControlledUNet


def trainable_parameters(model: ControlledUNet):
    """Yield only the parameters that require gradients.

    With a frozen backbone this returns the control branch parameters only.
    """
    return [p for p in model.parameters() if p.requires_grad]


def train_step(
    model: ControlledUNet,
    x: torch.Tensor,
    t: torch.Tensor,
    hint: torch.Tensor,
    target: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 1e-2,
) -> float:
    """Run one optimization step and return the scalar loss before the step.

    Args:
        model: the controlled UNet.
        x: noisy input image batch.
        t: timestep batch.
        hint: control conditioning batch.
        target: regression target batch.
        optimizer: optional optimizer. If omitted, an Adam optimizer over the
            trainable parameters is created.
        lr: learning rate used when an optimizer is created here.

    Returns:
        The loss value computed before the parameter update.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)

    model.train()
    optimizer.zero_grad()
    pred = model(x, t, hint)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    return loss.item()
