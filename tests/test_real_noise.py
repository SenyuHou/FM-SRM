import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd
from PIL import Image

from lnl_foundation.data.real_noise import Animal10N, ILSVRC12_50, WebVision50
from lnl_foundation.features import load_real_features, real_manifest_sha256
from lnl_foundation.partition.global_local_gmm import GlobalLocalGMMPartitioner
from lnl_foundation.partition.classwise_global_local_gmm import ClasswiseGlobalLocalGMMPartitioner
from lnl_foundation.training.baseline_linear_probe import train_stage2_baseline
from lnl_foundation.training.real_evaluation import (
    BestValidationCheckpoint, evaluate_classification, load_best_heads,
)
from lnl_foundation.training.robust_linear_probe import LinearProbeConfig, train_robust_linear_probe


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "red").save(path)


def _check_real_manifests_and_mapping(tmp_path):
    animal = tmp_path / "Animal10N"
    _image(animal / "training" / "1_b.jpg")
    _image(animal / "training" / "0_a.png")
    _image(animal / "testing" / "1_c.jpg")
    training = Animal10N(animal, "train")
    assert training.targets == [0, 1]
    assert training[1][1:] == (1, 1)
    assert Animal10N(animal, "test").targets == [1]

    web = tmp_path / "WebVision"
    _image(web / "google" / "first.jpg")
    _image(web / "google" / "second.jpg")
    _image(web / "val_images_256" / "valid.jpg")
    (web / "info").mkdir()
    (web / "info" / "train_filelist_google.txt").write_text(
        "google/second.jpg 49\ngoogle/ignored.jpg 50\ngoogle/first.jpg 0\n", encoding="utf-8"
    )
    (web / "info" / "val_filelist.txt").write_text("valid.jpg 49\n", encoding="utf-8")
    web_train = WebVision50(web, "train")
    assert web_train.targets == [49, 0]
    assert web_train[0][2] == 0
    assert WebVision50(web, "val").targets == [49]

    imagenet = tmp_path / "ILSVRC12"
    _image(imagenet / "ILSVRC2012_img_val" / "A.jpg")
    (imagenet / "ILSVRC2012_val_label.txt").write_text(
        "A.jpg 49\nB.jpg 500\n", encoding="utf-8"
    )
    ilsvrc = ILSVRC12_50(imagenet)
    assert ilsvrc.targets == [49]
    assert ilsvrc[0][2] == 0
    (imagenet / "ILSVRC2012_val_label.txt").write_text(
        "A.jpg 49 n01697457\nB.jpg 500 n04263257\n", encoding="utf-8"
    )
    assert ILSVRC12_50(imagenet).targets == [49]
    (imagenet / "ILSVRC2012_val_label.txt").write_text(
        "A.jpg 49 n01440764\n", encoding="utf-8"
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "must map"):
        ILSVRC12_50(imagenet)


def _check_real_cache_manifest_guard(tmp_path):
    root = tmp_path / "Animal10N"
    _image(root / "training" / "0_a.jpg")
    base = Animal10N(root, "train")
    manifest = real_manifest_sha256(base)
    path = tmp_path / "train_l2.pt"
    torch.save({
        "features": torch.ones(1, 8), "labels": torch.tensor([0]),
        "indices": torch.arange(1), "normalized": True,
        "backbone": "vit_b16_imagenet", "dataset": "animal10n",
        "split": "train", "num_classes": 10, "sample_count": 1,
        "manifest_sha256": manifest,
    }, path)
    assert load_real_features(path, "animal10n", "train", [0], "vit_b16_imagenet", manifest).shape == (1, 8)
    with unittest.TestCase().assertRaisesRegex(ValueError, "misaligned"):
        load_real_features(path, "animal10n", "train", [0], "vit_b16_imagenet", "wrong")


def _check_missing_root_and_evaluator(tmp_path):
    with unittest.TestCase().assertRaisesRegex(FileNotFoundError, "training"):
        Animal10N(tmp_path, "train")
    with unittest.TestCase().assertRaisesRegex(FileNotFoundError, "train_filelist_google"):
        WebVision50(tmp_path, "train")
    with unittest.TestCase().assertRaisesRegex(FileNotFoundError, "ILSVRC2012_img_val"):
        ILSVRC12_50(tmp_path)
    logits = torch.eye(5)
    metrics = evaluate_classification(logits, torch.arange(5))
    assert metrics == {"top1": 1.0, "top5": 1.0, "macro_f1": 1.0}


def _check_validation_checkpoint_uses_best_epoch(tmp_path):
    features = torch.eye(5)
    labels = torch.arange(5)
    head = torch.nn.Linear(5, 5)
    checkpoint = BestValidationCheckpoint(features, labels, torch.device("cpu"), tmp_path / "best.pt")
    with torch.no_grad():
        head.weight.copy_(torch.eye(5))
        head.bias.zero_()
    checkpoint(0, [head])
    with torch.no_grad():
        head.weight.zero_()
    checkpoint(1, [head])
    restored, saved = load_best_heads(tmp_path / "best.pt", 5, 5, torch.device("cpu"))
    assert saved["epoch"] == 1
    assert evaluate_classification(restored[0](features), labels)["top1"] == 1.0


def _check_chunked_knn_excludes_self():
    partitioner = GlobalLocalGMMPartitioner(10, k=2)
    vectors = torch.eye(8)
    neighbors = partitioner._scalable_neighbors(vectors)
    assert neighbors.shape == (8, 2)
    assert all(index not in neighbors[index] for index in range(8))


def _check_real_trainers_do_not_calibrate(tmp_path):
    features = torch.eye(10).repeat(2, 1)
    labels = torch.arange(10).repeat(2)
    partition = torch.zeros(20, dtype=torch.long)
    hook = BestValidationCheckpoint(features, labels, torch.device("cpu"), tmp_path / "ours.pt")
    ours = train_robust_linear_probe(
        features, labels, partition, features, labels, 10, 1, torch.device("cpu"),
        config=LinearProbeConfig(epochs=1, batch_size=20), epoch_hook=hook, calibrate=False,
    )
    assert hook.best_epoch == 1
    assert {"accuracy", "top5", "macro_f1"}.issubset(ours)
    assert "ece_raw" not in ours and "temperature" not in ours
    config = {"ce": {
        "epochs": 1, "batch_size": 20, "optimizer": "adamw",
        "learning_rate": 0.001, "weight_decay": 0.0001,
    }}
    baseline = train_stage2_baseline(
        "ce", features, labels, features, labels, 10, "animal10n", "real", None,
        1, torch.device("cpu"), config, compute_ece=False,
    )
    assert {"accuracy", "top5", "macro_f1"}.issubset(baseline.metrics)
    assert "ece_raw" not in baseline.metrics


def _check_classwise_coverage_and_paths(tmp_path):
    partitioner = ClasswiseGlobalLocalGMMPartitioner(2, k=2)
    assert partitioner.local_posterior_threshold == 0.8
    assert partitioner._global_state(
        np.array([0.72, 0.65, 0.04]), np.array([0.28, 0.35, 0.96])
    ).tolist() == [0, 1, 2]
    state = np.ones(200, dtype=np.int64)
    labels = np.repeat([0, 1], 100)
    margin = np.full(200, 0.1)
    local = np.full(200, 0.8)
    clean_prob = np.full(200, 0.6)
    promoted, mask = partitioner._ensure_class_coverage(
        state, margin, local, labels, clean_prob
    )
    assert promoted == [5, 5] and mask.sum() == 10
    assert all(np.bincount(labels[mask], minlength=2) == [5, 5])
    assert (state[mask] == 0).all()
    state[:] = 1
    local[:] = 0.2
    promoted, mask = partitioner._ensure_class_coverage(
        state, margin, local, labels, clean_prob
    )
    assert promoted == [0, 0] and not mask.any()

    from scripts.run_real_noise import _partition_dir
    cfg = {"output_dir": str(tmp_path), "global_posterior_threshold": 0.8,
           "local_posterior_threshold": 0.5, "knn_k": 20}
    old = _partition_dir(cfg, "animal10n", "vit_l16_imagenet", "abcdef012345", 1, "global")
    new = _partition_dir(cfg, "animal10n", "vit_l16_imagenet", "abcdef012345", 1)
    assert old != new and old.parent.name.startswith("real_g0.8")
    assert new.parent.name.startswith("real_classwise_c0.7_n0.95_l0.8")
    previous_classwise = _partition_dir(
        cfg, "animal10n", "vit_l16_imagenet", "abcdef012345", 1,
        classwise_thresholds=(0.8, 0.8, 0.5),
    )
    assert previous_classwise != new
    assert previous_classwise.parent.name.startswith("real_classwise_g0.8_l0.5")

    from scripts.run_real_noise import _stage1
    labels = torch.arange(2).repeat_interleave(100)
    features = torch.eye(2).repeat_interleave(100, dim=0)
    old.mkdir(parents=True)
    margins = np.tile(np.r_[np.linspace(-0.1, 0.0, 50), np.linspace(0.1, 0.2, 50)], 2)
    pd.DataFrame({
        "index": np.arange(200), "noisy_label": labels.numpy(),
        "partition": np.zeros(200), "global_margin": margins,
        "local_consistency": np.tile(np.r_[np.full(50, 0.2), np.full(50, 0.9)], 2),
    }).to_csv(old / "partition.csv", index=False)
    (old / "metrics.json").write_text(json.dumps({
        "protocol": "real_noise_stage1_v1", "feature_sha256": "abcdef012345",
        "num_samples": 200,
    }), encoding="utf-8")
    args = SimpleNamespace(dataset="animal10n", seed=1, partition_mode="classwise",
                           refit_statistics=False, force=False, clean_threshold=0.7,
                           global_noisy_threshold=0.95, local_noisy_threshold=0.8)
    _stage1(cfg, args, features, labels, "vit_l16_imagenet", "abcdef012345", 2)
    metadata = json.loads((new / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["partition_mode"] == "classwise"
    assert (metadata["clean_threshold"], metadata["global_noisy_threshold"],
            metadata["local_noisy_threshold"]) == (0.7, 0.95, 0.8)
    assert metadata["source_statistics"] is not None
    assert sum(metadata[key] for key in ("n_clean", "n_hard", "n_noisy")) == 200


class RealNoiseSmokeTests(unittest.TestCase):
    def test_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_real_manifests_and_mapping(Path(directory))

    def test_cache_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_real_cache_manifest_guard(Path(directory))

    def test_missing_root_and_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_missing_root_and_evaluator(Path(directory))

    def test_validation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_validation_checkpoint_uses_best_epoch(Path(directory))

    def test_chunked_knn(self):
        _check_chunked_knn_excludes_self()

    def test_trainers_without_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_real_trainers_do_not_calibrate(Path(directory))

    def test_classwise_coverage_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_classwise_coverage_and_paths(Path(directory))


if __name__ == "__main__":
    unittest.main()
