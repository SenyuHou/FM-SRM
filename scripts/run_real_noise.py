"""FM-SRM Stage-1 partitioning and Stage-2 training on real-noise datasets."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import canonical_backbone_name
from lnl_foundation.data.real_noise import REAL_DATASET_META, load_real_base
from lnl_foundation.features import feature_cache_path, load_real_features, real_manifest_sha256
from lnl_foundation.partition.global_local_gmm import CLEAN, HARD, NOISY, GlobalLocalGMMPartitioner
from lnl_foundation.partition.classwise_global_local_gmm import (
    ClasswiseGlobalLocalGMMPartitioner, DEFAULT_CLEAN_THRESHOLD,
    DEFAULT_GLOBAL_NOISY_THRESHOLD, DEFAULT_LOCAL_NOISY_THRESHOLD,
)
from lnl_foundation.training.real_evaluation import (
    BestValidationCheckpoint, evaluate_classification, linear_logits, load_best_heads,
)
from lnl_foundation.training.robust_linear_probe import train_robust_linear_probe
from lnl_foundation.utils import get_device, load_config, save_json


def _sha(features):
    return hashlib.sha256(features.numpy().tobytes()).hexdigest()


def _dataset(args, split):
    return load_real_base(args.dataset, args.dataset_root, split)


def _load_split(cfg, args, backbone, split):
    base = _dataset(args, split)
    path = feature_cache_path(cfg["features_root"], args.dataset, backbone, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. First run scripts/extract_features.py --dataset {args.dataset} "
            f"--dataset_root {args.dataset_root} --splits {split} --set backbone={backbone}"
        )
    features = load_real_features(path, args.dataset, split, base.targets, backbone,
                                  real_manifest_sha256(base))
    return features, torch.as_tensor(base.targets, dtype=torch.long)


def _partition_dir(cfg, dataset, backbone, feature_hash, seed, partition_mode="classwise",
                   classwise_thresholds=None):
    if partition_mode not in {"classwise", "global"}:
        raise ValueError(f"Unknown partition mode: {partition_mode}")
    if partition_mode == "classwise":
        clean, noisy, local = classwise_thresholds or (
            DEFAULT_CLEAN_THRESHOLD, DEFAULT_GLOBAL_NOISY_THRESHOLD,
            DEFAULT_LOCAL_NOISY_THRESHOLD,
        )
        if (clean, noisy, local) == (0.8, 0.8, 0.5):
            variant = f"real_classwise_g0.8_l0.5_k{cfg['knn_k']}_{feature_hash[:12]}"
        else:
            variant = f"real_classwise_c{clean:g}_n{noisy:g}_l{local:g}_k{cfg['knn_k']}_{feature_hash[:12]}"
    else:
        variant = (
            f"real_g{cfg['global_posterior_threshold']:g}_"
            f"l{cfg['local_posterior_threshold']:g}_k{cfg['knn_k']}_{feature_hash[:12]}"
        )
    return Path(cfg["output_dir"]) / dataset / backbone / variant / f"seed_{seed}"


def _read_partition(path, labels):
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage-1 partition: {path}. Run --phase stage1 first.")
    frame = pd.read_csv(path)
    if not np.array_equal(frame["index"].to_numpy(), np.arange(len(labels))):
        raise ValueError(f"Partition sample indices differ from feature manifest: {path}")
    if not np.array_equal(frame["noisy_label"].to_numpy(), labels.numpy()):
        raise ValueError(f"Partition noisy labels differ from dataset manifest: {path}")
    values = torch.as_tensor(frame["partition"].to_numpy(copy=True), dtype=torch.long)
    if not torch.isin(values, torch.tensor([CLEAN, HARD, NOISY])).all():
        raise ValueError(f"Invalid partition codes: {path}")
    return values, torch.as_tensor(frame["global_margin"].to_numpy(copy=True), dtype=torch.float32)


def _stage1(cfg, args, features, labels, backbone, feature_hash, num_classes):
    classwise_thresholds = (args.clean_threshold, args.global_noisy_threshold,
                            args.local_noisy_threshold)
    target = _partition_dir(cfg, args.dataset, backbone, feature_hash, args.seed,
                            args.partition_mode, classwise_thresholds)
    metrics_path = target / "metrics.json"
    if metrics_path.exists() and not args.force:
        _read_partition(target / "partition.csv", labels)
        print(f"Reusing Stage-1 partition: {target}")
        return
    partitioner_type = (ClasswiseGlobalLocalGMMPartitioner if args.partition_mode == "classwise"
                        else GlobalLocalGMMPartitioner)
    options = dict(num_classes=num_classes, tau_global=cfg["global_posterior_threshold"],
                   local_posterior_threshold=(args.local_noisy_threshold if args.partition_mode == "classwise"
                                              else cfg["local_posterior_threshold"]),
                   k=cfg["knn_k"], seed=args.seed)
    if args.partition_mode == "classwise":
        options.update(clean_threshold=args.clean_threshold,
                       global_noisy_threshold=args.global_noisy_threshold)
    model = partitioner_type(**options)
    source_statistics = None
    if args.partition_mode == "classwise" and not args.refit_statistics:
        pooled_dir = _partition_dir(cfg, args.dataset, backbone, feature_hash, args.seed, "global")
        pooled_csv = pooled_dir / "partition.csv"
        pooled_metadata = pooled_dir / "metrics.json"
        if pooled_csv.exists() and pooled_metadata.exists():
            metadata = json.loads(pooled_metadata.read_text(encoding="utf-8"))
            if (metadata.get("protocol") != "real_noise_stage1_v1"
                    or metadata.get("feature_sha256") != feature_hash
                    or metadata.get("num_samples") != len(labels)):
                raise ValueError(f"Old pooled partition metadata does not match current features: {pooled_metadata}")
            _read_partition(pooled_csv, labels)
            previous = pd.read_csv(pooled_csv, usecols=["global_margin", "local_consistency"])
            result = model.fit_from_statistics(
                previous["global_margin"], previous["local_consistency"], labels
            )
            source_statistics = pooled_csv.relative_to(Path(cfg["output_dir"])).as_posix()
            print(f"Reusing saved global margin and local KNN statistics: {pooled_csv}", flush=True)
    if source_statistics is None:
        result = model.fit_predict(features, labels)
    partition = result["partition"]
    counts = {
        "n_clean": int((partition == CLEAN).sum()),
        "n_hard": int((partition == HARD).sum()),
        "n_noisy": int((partition == NOISY).sum()),
    }
    if sum(counts.values()) != len(labels):
        raise ValueError("Stage-1 partitions do not cover all samples.")
    class_counts = {
        name: [int(((labels == cls) & (partition == state)).sum()) for cls in range(num_classes)]
        for name, state in (("clean", CLEAN), ("hard", HARD), ("noisy", NOISY))
    }
    frame = pd.DataFrame({
        "index": np.arange(len(labels)),
        "noisy_label": labels.numpy(),
        **{key: result[key].numpy() for key in (
            "partition", "global_margin", "global_clean_prob", "global_noisy_prob",
            "global_state", "local_consistency", "local_hard_prob", "local_noisy_prob",
        )},
    })
    if args.partition_mode == "classwise":
        frame["coverage_promoted"] = result["coverage_promoted"].numpy()
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "partition.csv", index=False)
    save_json(metrics_path, {
        "protocol": ("real_noise_stage1_classwise_v1" if args.partition_mode == "classwise"
                     else "real_noise_stage1_v1"),
        "partition_mode": args.partition_mode, "dataset": args.dataset, "backbone": backbone,
        "seed": args.seed, "num_samples": len(labels), "feature_sha256": feature_hash,
        "tau_global": (args.clean_threshold if args.partition_mode == "classwise"
                       else cfg["global_posterior_threshold"]),
        "tau_local": (args.local_noisy_threshold if args.partition_mode == "classwise"
                      else cfg["local_posterior_threshold"]),
        "knn_k": cfg["knn_k"],
        "local_gmm_fallback": bool(result["local_gmm_fallback"]),
        "global_gmm_fallback_classes": result.get("global_gmm_fallback_classes", []),
        "local_gmm_fallback_classes": result.get("local_gmm_fallback_classes", []),
        "class_counts": class_counts,
        "classes_without_clean": [cls for cls, count in enumerate(class_counts["clean"]) if count == 0],
        "coverage_promoted_per_class": result.get("coverage_promoted_per_class", []),
        "min_clean_fraction": result.get("min_clean_fraction"),
        "clean_threshold": result.get("clean_threshold"),
        "global_noisy_threshold": result.get("global_noisy_threshold"),
        "local_noisy_threshold": result.get("local_noisy_threshold"),
        "source_statistics": source_statistics,
        **counts,
    })
    print(f"Stage-1 {args.partition_mode} {counts}; "
          f"classes without Clean={sum(count == 0 for count in class_counts['clean'])} -> {target}", flush=True)


def _write_summary(runs, path):
    fields = ["accuracy", "top5", "macro_f1"]
    if runs["dataset"].iloc[0] == "webvision":
        fields.extend([
            "best_webvision_val_top1", "best_webvision_val_top5", "best_webvision_val_macro_f1",
            "ilsvrc12_test_top1", "ilsvrc12_test_top5", "ilsvrc12_test_macro_f1",
        ])
    summary = runs.groupby(["method", "dataset", "backbone"], dropna=False).agg(
        n_runs=("seed", "size"),
        **{f"{field}_mean": (field, "mean") for field in fields},
        **{f"{field}_std": (field, "std") for field in fields},
    ).reset_index()
    summary.to_csv(path, index=False)


def _stage2(cfg, args, features, labels, backbone, feature_hash, num_classes, device):
    method = "ours"
    output = (Path(cfg["output_dir"]).parent / "real_noise" / "stage2" /
              method / args.dataset / backbone / feature_hash[:12])
    if args.partition_mode == "classwise":
        variant = _partition_dir(
            cfg, args.dataset, backbone, feature_hash, args.seed, args.partition_mode,
            (args.clean_threshold, args.global_noisy_threshold, args.local_noisy_threshold),
        ).parent.name
        output = output / ("partition_classwise" if variant.startswith("real_classwise_g0.8_l0.5_")
                           else f"partition_{variant}")
    runs_path = output / "runs.csv"
    rows = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    prior = next((row for row in rows if int(row["seed"]) == args.seed), None)
    if prior is not None and not args.force:
        print(f"Reusing Stage-2 run: {runs_path}, seed={args.seed}")
        return
    if args.dataset == "webvision":
        external_root = Path(args.ilsvrc12_root)
        for required in (external_root / "ILSVRC2012_img_val",
                         external_root / "ILSVRC2012_val_label.txt",
                         feature_cache_path(cfg["features_root"], "ilsvrc12_50", backbone, "test")):
            if not required.exists():
                raise FileNotFoundError(f"Missing required final-test input before training: {required}")
        val_features, val_labels = _load_split(cfg, args, backbone, "val")
        print(
            f"Dataset: WebVision-50 | Train: WebVision Train ({len(labels)}) | "
            f"Validation: WebVision Validation ({len(val_labels)}) | "
            "External final test: ILSVRC12-50 | "
            f"Method: {method} | Backbone: {backbone} | Seed: {args.seed} | Classes: {num_classes}",
            flush=True,
        )
        evaluation_features, evaluation_labels = val_features, val_labels
        selector = BestValidationCheckpoint(
            val_features, val_labels, device, output / f"seed_{args.seed}" / "best_val.pt"
        )
    else:
        evaluation_features, evaluation_labels = _load_split(cfg, args, backbone, "test")
        selector = None
    partition_path = _partition_dir(
        cfg, args.dataset, backbone, feature_hash, args.seed, args.partition_mode,
        (args.clean_threshold, args.global_noisy_threshold, args.local_noisy_threshold),
    ) / "partition.csv"
    partition, margin = _read_partition(partition_path, labels)
    from lnl_foundation.training.robust_linear_probe import LinearProbeConfig
    result = train_robust_linear_probe(
        features, labels, partition, evaluation_features, evaluation_labels,
        num_classes, args.seed, device, config=LinearProbeConfig(),
        global_margin=margin, epoch_hook=selector, calibrate=False,
    )
    metrics = {key: value for key, value in result.items() if key != "_artifacts"}
    row = {
        "protocol": "real_noise_stage2_v1", "method": method, "dataset": args.dataset,
        "backbone": backbone, "feature_sha256": feature_hash, "seed": args.seed,
        "partition_mode": args.partition_mode,
        "clean_threshold": args.clean_threshold if args.partition_mode == "classwise" else None,
        "global_noisy_threshold": args.global_noisy_threshold if args.partition_mode == "classwise" else None,
        "local_noisy_threshold": args.local_noisy_threshold if args.partition_mode == "classwise" else None,
        "n_train": len(labels),
        "n_clean": int((partition == CLEAN).sum()),
        "n_hard": int((partition == HARD).sum()),
        "n_noisy": int((partition == NOISY).sum()),
    }
    if args.dataset == "webvision":
        if selector.best_epoch is None:
            raise RuntimeError("No validation checkpoint was saved.")
        heads, checkpoint = load_best_heads(
            selector.path, features.shape[1], num_classes, device
        )
        reduction = checkpoint["reduction"]
        val = evaluate_classification(
            linear_logits(heads, val_features, device, reduction), val_labels
        )
        if not np.isclose(val["top1"], checkpoint["validation_metrics"]["top1"], atol=1e-8):
            raise ValueError("Reloaded best checkpoint does not reproduce WebVision validation Top-1.")
        ilsvrc_base = load_real_base("ilsvrc12_50", args.ilsvrc12_root, "test")
        test_path = feature_cache_path(
            cfg["features_root"], "ilsvrc12_50", backbone, "test"
        )
        if not test_path.exists():
            raise FileNotFoundError(f"Missing ILSVRC12-50 frozen features: {test_path}")
        test_features = load_real_features(
            test_path, "ilsvrc12_50", "test", ilsvrc_base.targets, backbone,
            real_manifest_sha256(ilsvrc_base),
        )
        test_labels = torch.as_tensor(ilsvrc_base.targets)
        test = evaluate_classification(
            linear_logits(heads, test_features, device, reduction), test_labels
        )
        row.update(
            best_epoch=int(selector.best_epoch),
            best_checkpoint_path=str(selector.path),
            n_webvision_val=len(val_labels), n_ilsvrc12_test=len(test_labels),
            best_webvision_val_top1=val["top1"],
            best_webvision_val_top5=val["top5"],
            best_webvision_val_macro_f1=val["macro_f1"],
            ilsvrc12_test_top1=test["top1"],
            ilsvrc12_test_top5=test["top5"],
            ilsvrc12_test_macro_f1=test["macro_f1"],
            accuracy=test["top1"], top5=test["top5"], macro_f1=test["macro_f1"],
        )
        print(
            "========== Final External Test ==========\n"
            "ILSVRC12-50 (labeled validation split)\n"
            f"Top-1: {test['top1']:.6f}\nTop-5: {test['top5']:.6f}\n"
            f"Macro-F1: {test['macro_f1']:.6f}\n"
            "=========================================", flush=True,
        )
    else:
        row.update({key: metrics[key] for key in ("accuracy", "top5", "macro_f1")})
    output.mkdir(parents=True, exist_ok=True)
    rows = [old for old in rows if int(old["seed"]) != args.seed]
    rows.append(row)
    runs = pd.DataFrame(rows).sort_values("seed")
    runs.to_csv(runs_path, index=False)
    _write_summary(runs, output / "summary.csv")
    print(f"Stage-2 {method} {args.dataset}: {row} -> {runs_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Real-noise frozen-feature experiments; one phase per command.")
    parser.add_argument("--phase", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--dataset", required=True, choices=("animal10n", "webvision"))
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--ilsvrc12_root", help="Required for WebVision Stage-2 final evaluation.")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--partition_mode", choices=("classwise", "global"), default="classwise",
                        help="Ours Stage-1 mode; classwise is the real-noise default.")
    parser.add_argument("--clean_threshold", type=float, default=DEFAULT_CLEAN_THRESHOLD)
    parser.add_argument("--global_noisy_threshold", type=float, default=DEFAULT_GLOBAL_NOISY_THRESHOLD)
    parser.add_argument("--local_noisy_threshold", type=float, default=DEFAULT_LOCAL_NOISY_THRESHOLD)
    parser.add_argument("--refit_statistics", action="store_true",
                        help="Recompute margins/KNN instead of reusing matching pooled Stage-1 statistics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.partition_mode == "classwise":
        clean, noisy, local = (args.clean_threshold, args.global_noisy_threshold,
                               args.local_noisy_threshold)
        if (not all(map(math.isfinite, (clean, noisy, local)))
                or not 0.5 <= clean < 1 or not 0.5 <= noisy < 1
                or clean + noisy <= 1 or not 0.5 <= local < 1):
            parser.error("Classwise thresholds must be in [0.5, 1); clean/noisy global regions must not overlap.")
    if args.dataset == "webvision" and args.phase == "stage2" and not args.ilsvrc12_root:
        parser.error("WebVision Stage-2 requires --ilsvrc12_root for final test evaluation.")
    cfg = load_config(args.config, args.overrides)
    backbone = canonical_backbone_name(args.backbone)
    device = get_device(args.device or cfg["device"])
    features, labels = _load_split(cfg, args, backbone, "train")
    num_classes = REAL_DATASET_META[args.dataset]["num_classes"]
    feature_hash = _sha(features)
    print(f"{args.dataset}/{backbone}: train={tuple(features.shape)}, device={device}", flush=True)
    if args.phase == "stage1":
        _stage1(cfg, args, features, labels, backbone, feature_hash, num_classes)
    else:
        _stage2(cfg, args, features, labels, backbone, feature_hash, num_classes, device)


if __name__ == "__main__":
    main()
