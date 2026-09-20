# Baseline implementations

This repository adapts the compared methods to a unified frozen-feature
protocol. CIFAR sample order, noisy-label vectors, random seeds, frozen
features, and test evaluation are shared across methods. Ground-truth clean
training labels and the FM-SRM Clean/Hard/Noisy partition are never supplied
to the baseline algorithms.

## Stage-1 detection baselines

### SimiFeat

`scripts/run_simifeat.py` retains SimiFeat's representation-only voting and
ranking mechanisms. The implementation is adapted from *Detecting Corrupted
Labels Without Training a Model to Predict* and is distributed for research
use under the upstream CC BY-NC 4.0 terms copied to
`docs/simifeat/LICENSE.md`.

The formal entry point is `scripts/run_stage1_baseline.py`. It consumes the
same CIFAR sample order and saved noisy labels as the current FM-SRM
Global-Local-GMM/SimiFeat comparison. Clean labels are not passed to either
detector and are used only by the shared binary evaluation step.

Source partitions do not need to share the evaluated feature hash or backbone.
The runner validates that all available formal candidates for each
dataset/noise/seed contain the same noisy-label vector, then prefers an exact
feature match, the same backbone, or another backbone in that order. The
evaluated feature hash remains part of every output path and metrics row.

## CLIPCleaner

The implementation ports `combined_selection` from the original CLIPCleaner
repository. It retains:

- the original class-specific CIFAR descriptor prompts;
- zero-shot probabilities with descriptor aggregation and temperature 0.07;
- balanced logistic regression on normalized visual CLIP features;
- per-class normalization followed by two-component GMM selection;
- zero-shot and visual label-consistency thresholds;
- intersection of all four selected sets and the empty-class safeguard.

The binary prediction is the original final selected set. For threshold-free
metrics, the noise score is the negative minimum margin among the four rules,
which is the continuous counterpart of their intersection. The original
source is MIT licensed; see `CLIPCLEANER_LICENSE.md`.

## DeFT

The implementation ports phase one of DeFT. It retains the authors' OpenAI
CLIP model, deep VPT length 20, learnable positive/negative prompts with 16
context tokens, SGD and cosine schedule, ten epochs, clean threshold 0.5, and
the original positive/negative losses. Synthetic settings follow
`main_phase1.py` with one warm-up epoch. Human noise follows
`main_real_phase1.py`, including SCE warm-up for five epochs and the additional
positive-class argmax selection condition. The final-epoch `1 - p_clean` is
used as the continuous noise score. The original source is MIT licensed; see
`DEFT_LICENSE.md`.

DeFT cannot use only a frozen feature cache: VPT changes intermediate visual
tokens and its learned dual text prompts are part of the detector. The cache
is still loaded and hashed to bind every run to the same formal FM-SRM source
partition and CLIP backbone.

## Stage-2 learning baselines

The formal entry point is `scripts/run_stage2_baseline.py`. All image encoders
remain frozen and only linear classifier heads are optimized. Baselines report
Accuracy, Macro-F1, and Raw ECE; FM-SRM's reliable-anchor calibration is not
applied to them.

### CE and GCE

CE trains one linear head on all noisy samples. GCE replaces cross entropy by
the generalized cross-entropy objective with `q=0.7`.

### Co-teaching

The adaptation retains two independently initialized heads, small-loss sample
selection, cross-update between heads, and the gradual forget-rate schedule
from *Co-teaching: Robust Training of Deep Neural Networks with Extremely
Noisy Labels* (NeurIPS 2018).

### DivideMix

The adaptation retains warm-up, per-sample-loss GMM estimation, two-head
co-division, labeled/unlabeled splitting, label co-refinement/co-guessing,
MixMatch-style mixup, sharpening, and unsupervised consistency from
*DivideMix: Learning with Noisy Labels as Semi-supervised Learning* (ICLR
2020). The original implementation is MIT licensed.

### DISC

The adaptation retains dynamic instance-specific confidence tracking,
two-view prediction, dynamic clean/hard/purified assignment, label correction,
mixup, and subset-specific losses from *DISC: Learning From Noisy Labels via
Dynamic Instance-Specific Selection and Correction* (CVPR 2023). The original
implementation is MIT licensed.

### CLIPCleaner subset training

The Stage-2 CLIPCleaner baseline reads the saved Stage-1 CLIPCleaner clean
selection and trains one linear head with CE on that subset only. It never
falls back to FM-SRM partitions when a selection file is missing.

## Upstream works

- SimiFeat: *Detecting Corrupted Labels Without Training a Model to Predict*
  (ICML 2022).
- CLIPCleaner: *CLIPCleaner: Cleaning Noisy Labels with CLIP* (ACM MM 2024),
  MIT License.
- DeFT: *Vision-Language Models are Strong Noisy Label Detectors* (NeurIPS
  2024), MIT License.
- Co-teaching: *Co-teaching: Robust Training of Deep Neural Networks with
  Extremely Noisy Labels* (NeurIPS 2018).
- DivideMix: *DivideMix: Learning with Noisy Labels as Semi-supervised
  Learning* (ICLR 2020), MIT License.
- DISC: *DISC: Learning From Noisy Labels via Dynamic Instance-Specific
  Selection and Correction* (CVPR 2023), MIT License.

The FM-SRM repository contains adaptations for a frozen-feature linear-probe
comparison, not drop-in copies of the original end-to-end training programs.
Please cite each upstream paper whose implementation is used.
