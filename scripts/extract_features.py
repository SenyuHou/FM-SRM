import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lnl_foundation.backbones.frozen import FrozenBackbone
from lnl_foundation.data.datasets import load_cifar_base
from lnl_foundation.data.real_noise import REAL_DATASET_META, load_real_base
from lnl_foundation.features import extract_features, feature_cache_path, load_features, load_real_features, real_manifest_sha256
from lnl_foundation.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Extract frozen pretrained vision features once per dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "animal10n", "webvision", "ilsvrc12_50"], required=True)
    parser.add_argument("--dataset_root", help="Root containing the selected real-noise dataset")
    parser.add_argument("--ilsvrc12_root", help="Required for the ILSVRC12-50 validation images")
    parser.add_argument("--splits", nargs="+", choices=["train", "test", "val"], default=["train"])
    parser.add_argument("--force_extract", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if cfg["hf_endpoint"]:
        os.environ["HF_ENDPOINT"] = cfg["hf_endpoint"].rstrip("/")
    set_seed(1)
    backbone = None
    for split in dict.fromkeys(args.splits):
        is_real = args.dataset in REAL_DATASET_META
        if is_real:
            if split not in REAL_DATASET_META[args.dataset]["splits"]:
                parser.error(f"{args.dataset} supports splits {REAL_DATASET_META[args.dataset]['splits']}")
            root = args.ilsvrc12_root if args.dataset == "ilsvrc12_50" else args.dataset_root
            if not root:
                parser.error("Specify --ilsvrc12_root for ILSVRC12 or --dataset_root for Animal-10N/WebVision")
            base = load_real_base(args.dataset, root, split)
        else:
            if split not in {"train", "test"}:
                parser.error("CIFAR supports only train/test splits")
            base = load_cifar_base(args.dataset, cfg["data_root"], train=split == "train", download=True)
        path = feature_cache_path(
            cfg["features_root"], args.dataset, cfg["backbone"], split,
            prefer_existing=not args.force_extract,
        )
        if path.exists() and not args.force_extract:
            features = (load_real_features(path, args.dataset, split, base.targets, cfg["backbone"],
                                           real_manifest_sha256(base))
                        if is_real else load_features(path, base.targets, expected_backbone=cfg["backbone"]))
            print(f"Reusing {split}: {tuple(features.shape)} -> {path}", flush=True)
            continue
        if backbone is None:
            backbone = FrozenBackbone(cfg["backbone"], pretrained=True)
        dataset = (load_real_base(args.dataset, root, split, transform=backbone.preprocess)
                   if is_real else load_cifar_base(args.dataset, cfg["data_root"],
                                                   train=split == "train", transform=backbone.preprocess,
                                                   download=False))
        payload = extract_features(backbone, dataset, cfg["batch_size"], cfg["num_workers"], torch.device(cfg["device"]))
        if is_real:
            payload["labels"] = payload.pop("clean_labels")
            payload.update(dataset=args.dataset, split=split,
                           num_classes=REAL_DATASET_META[args.dataset]["num_classes"],
                           sample_count=len(dataset), manifest_sha256=real_manifest_sha256(dataset))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"Saved {split}: {tuple(payload['features'].shape)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
