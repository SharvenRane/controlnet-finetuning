"""Behavior tests for the ControlNet style controlled UNet.

All tensors are tiny and synthetic so the suite runs in seconds on a CPU with
no downloads. The tests check three properties:

1. Shapes round trip through the backbone and the controlled UNet.
2. With zero initialized control convolutions the controlled output equals the
   base output, and after the control branch is perturbed the output shifts.
3. A single training step reduces the loss, and gradients flow only into the
   control branch while the frozen backbone stays put.
"""

import copy

import torch

from src.unet import TinyUNet, timestep_embedding
from src.controlnet import ControlledUNet
from src.train import train_step, trainable_parameters


def make_batch(batch=2, ch=3, size=16, hint_ch=1):
    x = torch.randn(batch, ch, size, size)
    t = torch.randint(0, 1000, (batch,))
    hint = torch.randn(batch, hint_ch, size, size)
    return x, t, hint


def test_timestep_embedding_shape():
    t = torch.arange(5)
    emb = timestep_embedding(t, 32)
    assert emb.shape == (5, 32)
    # Odd dimension is padded, not dropped.
    emb_odd = timestep_embedding(t, 31)
    assert emb_odd.shape == (5, 31)


def test_backbone_forward_shape():
    net = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    x, t, _ = make_batch()
    out = net(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_controlled_forward_shape():
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1)
    x, t, hint = make_batch()
    out = model(x, t, hint)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_zero_control_equals_base_output():
    """At init the zero convolutions make control a no op."""
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1)
    model.eval()
    x, t, hint = make_batch()

    with torch.no_grad():
        base = model.base_forward(x, t)
        controlled = model(x, t, hint)

    # Even with a nonzero hint, the zero initialized control convolutions
    # produce exactly zero residuals, so the outputs must match.
    assert torch.allclose(base, controlled, atol=1e-6)


def test_no_hint_equals_base_output():
    """Passing no hint skips the control branch entirely."""
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1)
    model.eval()
    x, t, _ = make_batch()

    with torch.no_grad():
        base = model.base_forward(x, t)
        controlled = model(x, t, hint=None)

    assert torch.allclose(base, controlled, atol=1e-6)


def test_nonzero_control_shifts_output():
    """Once the control branch is perturbed away from zero it changes output."""
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1)
    model.eval()
    x, t, hint = make_batch()

    with torch.no_grad():
        base = model.base_forward(x, t)

    # Perturb the zero convolutions so the control branch carries signal. This
    # covers the residual zero convs and the hint zero conv, so the control
    # path as a whole becomes active and hint dependent.
    with torch.no_grad():
        for name, p in model.control.named_parameters():
            if ("zero" in name or "hint.out" in name) and p.dim() > 1:
                p.add_(torch.randn_like(p) * 0.5)

    with torch.no_grad():
        controlled = model(x, t, hint)

    assert not torch.allclose(base, controlled, atol=1e-4)
    # The shift should depend on the hint: a different hint gives a different
    # output.
    with torch.no_grad():
        other_hint = hint + 1.0
        controlled_other = model(x, t, other_hint)
    assert not torch.allclose(controlled, controlled_other, atol=1e-4)


def test_backbone_is_frozen():
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1, freeze_backbone=True)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.control.parameters())
    # Trainable set is exactly the control branch.
    n_trainable = sum(p.numel() for p in trainable_parameters(model))
    n_control = sum(p.numel() for p in model.control.parameters())
    assert n_trainable == n_control


def test_train_step_reduces_loss():
    """A handful of steps on a fixed target must drive the loss down."""
    torch.manual_seed(123)
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1)

    x, t, hint = make_batch()
    # A fixed synthetic target that differs from the base output so the control
    # branch has something to learn.
    target = torch.randn_like(x)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=1e-2)
    first = train_step(model, x, t, hint, target, optimizer=optimizer)
    last = first
    for _ in range(20):
        last = train_step(model, x, t, hint, target, optimizer=optimizer)

    assert last < first, f"loss did not drop: {first} -> {last}"


def test_only_control_params_change_after_training():
    """The frozen backbone weights must be untouched by a training step."""
    torch.manual_seed(7)
    backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
    model = ControlledUNet(backbone, hint_ch=1)

    before = copy.deepcopy(model.backbone.state_dict())

    x, t, hint = make_batch()
    target = torch.randn_like(x)
    for _ in range(3):
        train_step(model, x, t, hint, target, lr=1e-2)

    after = model.backbone.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"backbone weight {k} changed"
