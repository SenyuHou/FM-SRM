from pathlib import Path
import hashlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lnl_foundation.backbones.frozen import BACKBONE_ALIASES, canonical_backbone_name
from lnl_foundation.data.real_noise import REAL_DATASET_META


def feature_cache_candidates(root, dataset, backbone, split="train") -> tuple[Path, ...]:
    canonical = canonical_backbone_name(backbone)
    candidates = [Path(root) / dataset / canonical / f"{split}_l2.pt"]
    for alias, target in BACKBONE_ALIASES.items():
        if target == canonical:
            candidates.append(Path(root) / dataset / alias / f"{split}_l2.pt")
    return tuple(candidates)


def feature_cache_path(root, dataset, backbone, split="train", prefer_existing=True) -> Path:
    candidates = feature_cache_candidates(root, dataset, backbone, split)
    if prefer_existing:
        for path in candidates:
            if path.exists():
                return path
    return candidates[0]


def load_features(path, clean_labels, expected_backbone=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload["features"].float()
    labels = torch.as_tensor(clean_labels, dtype=torch.long)
    cached_dim = payload.get("feature_dim")
    cached_backbone = payload.get("backbone")
    valid = (
        features.ndim == 2
        and len(features) == len(labels)
        and torch.isfinite(features).all()
        and torch.equal(payload["indices"], torch.arange(len(labels)))
        and torch.equal(payload["clean_labels"], labels)
        and payload.get("normalized", False)
        and (cached_dim is None or int(cached_dim) == features.shape[-1])
    )
    if expected_backbone is not None:
        expected = canonical_backbone_name(expected_backbone)
        if cached_backbone is not None:
            valid = valid and canonical_backbone_name(cached_backbone) == expected
        else:
            legacy_name = Path(path).parent.name
            valid = valid and BACKBONE_ALIASES.get(legacy_name) == expected
    if not valid:
        raise ValueError(f"Invalid feature cache or CIFAR sample order: {path}")
    return features


def real_manifest_sha256(base):
    lines = [f"{image.relative_to(base.root).as_posix()}\t{label}\n"
             for image, label in base.samples]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def load_real_features(path, dataset, split, labels, expected_backbone=None, manifest_sha256=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload["features"].float()
    labels = torch.as_tensor(labels, dtype=torch.long)
    valid = (
        payload.get("dataset") == dataset
        and payload.get("split") == split
        and payload.get("num_classes") == REAL_DATASET_META[dataset]["num_classes"]
        and payload.get("sample_count") == len(labels)
        and features.ndim == 2
        and len(features) == len(labels)
        and torch.isfinite(features).all()
        and torch.equal(payload["indices"], torch.arange(len(labels)))
        and torch.equal(payload["labels"], labels)
        and payload.get("normalized", False)
    )
    if manifest_sha256 is not None:
        valid = valid and payload.get("manifest_sha256") == manifest_sha256
    if expected_backbone is not None:
        valid = valid and canonical_backbone_name(payload.get("backbone")) == canonical_backbone_name(expected_backbone)
    if not valid:
        raise ValueError(f"Invalid or misaligned real-noise feature cache: {path}")
    return features


@torch.no_grad()
def extract_features(backbone, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    backbone.to(device).eval()
    all_features, labels, indices = [], [], []
    for batch in tqdm(loader, desc="Extracting features"):
        images, clean = batch[:2]
        index = batch[2] if len(batch) >= 3 else None
        raw = backbone(images.to(device, non_blocking=True))
        if not torch.isfinite(raw).all():
            raise ValueError("Backbone output contains NaN or Inf values.")
        all_features.append(F.normalize(raw, dim=1).cpu())
        labels.append(clean)
        if index is not None:
            indices.append(index)
    features = torch.cat(all_features)
    ordered_indices = torch.cat(indices).long() if indices else torch.arange(len(dataset))
    if not torch.equal(ordered_indices, torch.arange(len(dataset))):
        raise ValueError("Feature extraction sample indices are not in manifest order.")
    return {
        "features": features,
        "clean_labels": torch.cat(labels).long(),
        "indices": ordered_indices,
        "normalized": True,
        "backbone": backbone.model_name,
        "feature_dim": int(features.shape[-1]),
        "checkpoint": backbone.spec.checkpoint,
        "preprocess_source": backbone.preprocessing_source,
        "preprocess_config": backbone.preprocess_config,
    }
