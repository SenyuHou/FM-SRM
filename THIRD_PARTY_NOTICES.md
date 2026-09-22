# Third-Party Notices

FM-SRM is released under the MIT License, except for third-party components
and adaptations identified below. Those materials remain governed by their
upstream licenses.

## SimiFeat

The fixed-feature SimiFeat adaptation in
`lnl_foundation/partition/simifeat.py` is based on *Detecting Corrupted Labels
Without Training a Model to Predict* by Zhaowei Zhu, Zihao Dong, and Yang Liu.
The upstream project is licensed under Creative Commons
Attribution-NonCommercial 4.0.

Upstream project: <https://github.com/UCSC-REAL/SimiFeat>

## CLIPCleaner

The adapter in `lnl_foundation/baselines/clipcleaner.py` follows the
CLIPCleaner sample-selection procedure from *CLIPCleaner: Cleaning Noisy
Labels with CLIP*. The upstream project is MIT licensed.

## SSR

The frozen-feature SSR adaptation in
`lnl_foundation/training/baseline_linear_probe.py` follows *SSR: An Efficient
and Robust Framework for Learning with Unknown Label Noise*. The upstream
project is MIT licensed.

Upstream paper and implementation: <https://arxiv.org/abs/2111.11288>

## Other baseline adaptations

The frozen-feature Co-teaching, DivideMix, and DISC implementations are
independent protocol adaptations contained in
`lnl_foundation/training/baseline_linear_probe.py`. Please cite the respective
original papers when using these implementations.
