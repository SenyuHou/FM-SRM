# Third-Party Notices

FM-SRM is released under the MIT License, except for third-party components
and adaptations identified below. Those materials remain governed by their
upstream licenses.

## SimiFeat

The fixed-feature SimiFeat adaptation in
`lnl_foundation/partition/simifeat.py` is based on *Detecting Corrupted Labels
Without Training a Model to Predict* by Zhaowei Zhu, Zihao Dong, and Yang Liu.
The upstream project is licensed under Creative Commons
Attribution-NonCommercial 4.0. A copy is provided at
`docs/simifeat/LICENSE.md`.

Upstream project: <https://github.com/UCSC-REAL/SimiFeat>

## DeFT

Files under `lnl_foundation/baselines/vendor/deft/` contain adapted components
from DeFT, *Vision-Language Models are Strong Noisy Label Detectors*. The
upstream project is MIT licensed. A copy is provided at
`docs/baselines/DEFT_LICENSE.md`.

## CLIPCleaner

The adapter in `lnl_foundation/baselines/clipcleaner.py` follows the
CLIPCleaner sample-selection procedure from *CLIPCleaner: Cleaning Noisy
Labels with CLIP*. The upstream project is MIT licensed. A copy is provided at
`docs/baselines/CLIPCLEANER_LICENSE.md`.

## Other baseline adaptations

The frozen-feature Co-teaching, DivideMix, and DISC implementations are
independent protocol adaptations contained in
`lnl_foundation/training/baseline_linear_probe.py`. Attribution and algorithmic
details are listed in `docs/baselines/README.md`.
