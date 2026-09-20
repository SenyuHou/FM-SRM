# Real-noise frozen-feature protocol

This extension uses the existing frozen backbones and Stage-1/Stage-2 algorithms.
It does not inject synthetic noise into Animal-10N or WebVision-50. Real-noise
runs do not perform RATC, temperature scaling, or ECE evaluation.

Real-noise Ours defaults to `--partition_mode classwise`. It computes the same
global prototype margin and full-training-set local KNN consistency, then fits
the two GMMs separately within each noisy-label class. A class with fewer than
5% Clean samples may promote only ambiguous samples with positive margin,
classwise clean posterior at least 0.5, and above-median local consistency.
The real-noise defaults are `--clean_threshold 0.7`,
`--global_noisy_threshold 0.95`, and `--local_noisy_threshold 0.8`:
Clean selection is looser and Noisy assignment is stricter than the earlier
0.8/0.8 global and 0.5 local thresholds. These values can be overridden on
both Stage-1 and Ours Stage-2 commands; use identical values for both phases.
The old classwise `0.8/0.8/0.5` combination still resolves to its original
partition and Stage-2 result directories for direct comparison.
Promotion counts are recorded per class; the rule does not guarantee a quota
when no trustworthy candidates exist. The earlier pooled-GMM results remain
available with `--partition_mode global` and keep their original paths. If a
matching pooled partition exists, classwise Stage-1 reuses its margin and local
consistency arrays without repeating KNN; pass `--refit_statistics` to recompute.

## Data layout

```text
Animal10N/
  training/<label-digit><image-name>
  testing/<label-digit><image-name>
WebVision/
  info/train_filelist_google.txt
  info/val_filelist.txt
  <train-image-path-from-list>
  val_images_256/<val-image-path-from-list>
ILSVRC12/
  ILSVRC2012_val_label.txt
  ILSVRC2012_img_val/<image-path-from-list>
```

WebVision annotation lists use `<relative-path> <class-id>`. ILSVRC12 accepts
both JYP's two-column format and `<relative-path> <ImageNet-class-id> <synset>`;
the latter checks IDs 0-49 against an explicit synset map.
Only IDs 0-49 are kept, in annotation order. The ILSVRC12 list explicitly
assigns those same classifier IDs to labeled ILSVRC2012 validation images.
No folder-order class mapping is used. Check that this annotation file matches
the JYP mapping before running the external evaluation.

## Example: Animal-10N

Run from the FM-SRM repository. Replace `/data/Animal10N` with the actual root.
Repeat Stage-1 and Stage-2 with `--seed 2` and `--seed 3` for three runs.

```bash
python scripts/extract_features.py --dataset animal10n --dataset_root /data/Animal10N --splits train test --set backbone=dinov2_vit_b14 --set device=cuda:0
python scripts/run_real_noise.py --phase stage1 --dataset animal10n --dataset_root /data/Animal10N --backbone dinov2_vit_b14 --seed 1 --device cuda:0
python scripts/run_real_noise.py --phase stage2 --method ours --dataset animal10n --dataset_root /data/Animal10N --backbone dinov2_vit_b14 --seed 1 --device cuda:0
python scripts/run_real_noise.py --phase stage2 --method ce --dataset animal10n --dataset_root /data/Animal10N --backbone dinov2_vit_b14 --seed 1 --device cuda:0
```

## Example: WebVision-50

Repeat Stage-1 and Stage-2 with `--seed 2` and `--seed 3` for three runs.
ILSVRC12 features are extracted once. The Stage-2 command uses WebVision
validation after each epoch, reloads its best checkpoint, then evaluates the
labeled ILSVRC2012 validation subset exactly once at the end.

```bash
python scripts/extract_features.py --dataset webvision --dataset_root /data/WebVision --splits train val --set backbone=dinov2_vit_b14 --set device=cuda:0
python scripts/extract_features.py --dataset ilsvrc12_50 --ilsvrc12_root /data/ILSVRC12 --splits test --set backbone=dinov2_vit_b14 --set device=cuda:0
python scripts/run_real_noise.py --phase stage1 --dataset webvision --dataset_root /data/WebVision --backbone dinov2_vit_b14 --seed 1 --device cuda:0
python scripts/run_real_noise.py --phase stage2 --method ours --dataset webvision --dataset_root /data/WebVision --ilsvrc12_root /data/ILSVRC12 --backbone dinov2_vit_b14 --seed 1 --device cuda:0
python scripts/run_real_noise.py --phase stage2 --method ce --dataset webvision --dataset_root /data/WebVision --ilsvrc12_root /data/ILSVRC12 --backbone dinov2_vit_b14 --seed 1 --device cuda:0
```

`--method` also accepts `gce`, `coteaching`, `dividemix`, `disc`, and
`clipcleaner`. SSR is not implemented in the current FM-SRM registry. For
CLIPCleaner, first extract CLIP features, then run the selection phase:

```bash
python scripts/run_real_noise.py --phase clipcleaner --dataset animal10n --dataset_root /data/Animal10N --backbone clip_vit_b16 --seed 1 --device cuda:0
python scripts/run_real_noise.py --phase stage2 --method clipcleaner --dataset animal10n --dataset_root /data/Animal10N --backbone clip_vit_b16 --clipcleaner_source_backbone clip_vit_b16 --seed 1 --device cuda:0
```

For a different classification backbone, extract both feature sets and pass
the CLIP selection backbone with `--clipcleaner_source_backbone`.

## Outputs

- Frozen features: `features/<dataset>/<backbone>/<split>_l2.pt`.
- Stage-1 partition: `outputs/generalization/<dataset>/<backbone>/real_classwise_c0.7_n0.95_l0.8_k.../seed_N/partition.csv` and `metrics.json`; older partitions remain in their original directories. No detection F1 is reported without noise ground truth.
- CLIPCleaner selection: `outputs/generalization/baselines/clipcleaner/<dataset>/<clip-backbone>/<feature-hash>/real/seed_N/predictions.csv`.
- Stage-2 run and summary: `outputs/real_noise/stage2/<method>/<dataset>/<backbone>/<feature-hash>/runs.csv` and `summary.csv`. Classwise Ours adds a threshold-named `partition_real_classwise_.../` directory to isolate each setting. Baselines add a `config_<hash>/` subdirectory; CLIPCleaner also adds `selection_<clip-backbone>_<selection-hash>/`.
- WebVision best checkpoint: `.../seed_N/best_val.pt`.

The Stage-2 CSV keeps WebVision best-validation and final ILSVRC12 metrics in
separate named columns. `accuracy`, `top5`, and `macro_f1` alias the final
external test only for cross-dataset summaries.

WebVision local consistency uses exact FAISS inner-product search when FAISS is
installed, otherwise chunked exact torch cosine top-k. The fallback avoids a
dense N-by-N matrix but can be slow at WebVision scale; FAISS is recommended on
the server, not required by the project.
