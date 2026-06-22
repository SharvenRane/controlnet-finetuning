# controlnet-finetuning

A compact, runnable implementation of ControlNet style conditional control on a
tiny diffusion UNet. The point of this repo is to show, in code small enough to
read in one sitting, how a control branch injects a conditioning signal such as
edges or a segmentation mask into a frozen diffusion backbone, and why the
trick of zero initialized convolutions makes that injection safe to train.

## The idea

A diffusion UNet learns to predict noise. ControlNet adds spatial control on top
of a model you already have. Rather than retrain the whole network, you freeze
the backbone and add a trainable copy of its encoder, the control branch. A
conditioning image, for example a Canny edge map or a binary mask, is fed into
the branch. The branch produces one residual feature per skip connection and one
for the bottleneck, and those residuals are added back into the frozen backbone
on the way down through the decoder.

The detail that makes this work is the zero convolution. Every residual the
control branch emits passes through a 1x1 convolution whose weights and bias
start at zero. At the very first step the control branch contributes exactly
nothing, so the controlled model reproduces the frozen backbone output bit for
bit. There is no shock to the pretrained weights. As training proceeds the zero
convolutions learn nonzero values and the control signal gradually takes effect.

## What is in here

`src/unet.py` holds `TinyUNet`, a small but real UNet with a downsampling
encoder, a bottleneck, an upsampling decoder, skip connections, and sinusoidal
timestep embeddings injected into every residual block. The encoder and decoder
are split into `encode` and `decode` so a control branch can hook into the skip
and bottleneck features.

`src/controlnet.py` holds three pieces. `ControlHint` lifts the raw conditioning
image into the backbone feature width and ends in a zero convolution.
`ControlNet` is the trainable encoder copy that emits one zero gated residual per
skip plus one for the bottleneck. `ControlledUNet` wraps a frozen backbone and a
control branch and adds the residuals into the backbone during the forward pass.
Call it without a hint and you get the plain backbone output.

`src/train.py` holds `train_step`, a single mean squared error optimization step
that updates only the trainable parameters. With the backbone frozen those are
exactly the control branch and the conditioning network.

## Why a tiny model

Everything is sized so the tests run in about a second on a CPU with no model
downloads and no network access. The tensors are small synthetic batches. The
architecture is the real thing; only the scale is reduced.

## Tests

The suite in `tests/` checks the properties that define correct ControlNet
behavior:

- Shapes round trip through the backbone and the controlled UNet.
- With zero initialized control convolutions the controlled output equals the
  base output, whether the hint is omitted or present.
- After the control branch is pushed away from zero the output shifts, and the
  shift depends on the hint, so different conditioning gives different output.
- The backbone is frozen, the trainable parameter set is exactly the control
  branch, a short run of training steps drives the loss down, and the frozen
  backbone weights are untouched by training.

## Running it

Install the dependencies and run the suite:

```
pip install -r requirements.txt
pytest tests/ -q
```

On the reference run all nine tests pass in under a second.

## Using the model

```python
import torch
from src import TinyUNet, ControlledUNet, train_step

backbone = TinyUNet(in_ch=3, base_ch=16, ch_mult=(1, 2))
model = ControlledUNet(backbone, hint_ch=1)   # backbone is frozen by default

x = torch.randn(2, 3, 16, 16)      # noisy image batch
t = torch.randint(0, 1000, (2,))   # diffusion timesteps
hint = torch.randn(2, 1, 16, 16)   # control conditioning, for example edges

out = model(x, t, hint)            # controlled prediction
base = model.base_forward(x, t)    # backbone only, no control
```

At initialization `out` and `base` are equal because of the zero convolutions.
After you train the control branch on a target they diverge as the conditioning
starts to steer the result.
