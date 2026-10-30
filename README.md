# Selfie Guard

Local research toolkit for evaluating protection of photos against unwanted AI
image editing.

**Status: early scaffold. No protection algorithm exists yet and no effectiveness
has been established.** The `doctor` command only reports the environment. It does
not change images and does not make them safe.

## Goal

Find out whether a photo can be processed locally so that recent image editing
models, such as Qwen-Image-2.1-Turbo and Flux, fail to produce a convincing edit
after the photo has been published. The main concern is non-consensual intimate
imagery; compromising scenes and face swaps are also in scope.

Processing is strictly local. Photos and anything derived from them are never sent
to an external service. The project has no telemetry and downloads nothing during
image processing.

## Adversaries

1. Edits the published photo with no cleanup. First target.
2. Applies ready-made cleanup such as upscaling or low-strength img2img. Next target.
3. Trains a dedicated purifier against this open-source code. Measured and
   reported; no protection is promised.

Published files cannot be updated later, so a file protected today stays exposed
to future models and cleanup methods.

## Planned measurement

The metric is the share of photos with no convincing edit within 50 attempts, taken
from the attacker's best result on each photo and compared with an unprotected
control. Digital screenshots, resizing and recompression are tested early, because
a protection that does not survive them is not useful. Robustness to screenshots
takes priority over invisibility. The first version targets PCs with a GPU.

See the [experiment protocol](experiments/README.md) and the
[research notes](docs/research.md), including published work on breaking such
protections.

## Run

Python 3.11 or newer. There are no runtime dependencies.

```sh
python -m selfie_guard --help
python -m selfie_guard doctor
python -m unittest discover -s tests -v
```

`doctor` prints JSON with the Python version, OS, installed ML packages and the
`nvidia-smi` response when available. It does not import PyTorch, download models or
use the network. A missing NVIDIA GPU is reported and is not an error.

## Data

Local images, model weights and results belong in `data/`, `models/` and `runs/`,
which are excluded from Git. See [data rules](data/README.md).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). The code is under the MIT license. Model
weights and any future third-party components have their own licenses; none are
included now.
