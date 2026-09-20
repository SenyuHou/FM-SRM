import tempfile
import unittest
from pathlib import Path

import torch

from lnl_foundation.features import feature_cache_path, load_features


class FeatureCacheTest(unittest.TestCase):
    def test_imagenet_alias_reuses_legacy_cache(self):
        with tempfile.TemporaryDirectory() as root:
            legacy = Path(root) / "cifar10" / "vit_b16" / "train_l2.pt"
            legacy.parent.mkdir(parents=True)
            legacy.touch()
            self.assertEqual(
                feature_cache_path(root, "cifar10", "vit_b16_imagenet"),
                legacy,
            )

    def test_backbones_have_isolated_cache_directories(self):
        with tempfile.TemporaryDirectory() as root:
            clip = feature_cache_path(root, "cifar10", "clip_vit_b16")
            imagenet = feature_cache_path(root, "cifar10", "vit_b16_imagenet")
            self.assertNotEqual(clip.parent, imagenet.parent)

    def test_legacy_payload_without_metadata_loads(self):
        labels = torch.tensor([0, 1])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "vit_b16" / "train_l2.pt"
            path.parent.mkdir()
            torch.save(
                {
                    "features": torch.nn.functional.normalize(torch.randn(2, 7), dim=1),
                    "clean_labels": labels,
                    "indices": torch.arange(2),
                    "normalized": True,
                },
                path,
            )
            features = load_features(path, labels, expected_backbone="vit_b16_imagenet")
            self.assertEqual(features.shape, (2, 7))

    def test_metadata_free_cache_is_not_accepted_for_new_backbone(self):
        labels = torch.tensor([0, 1])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "clip_vit_b16" / "train_l2.pt"
            path.parent.mkdir()
            torch.save(
                {
                    "features": torch.nn.functional.normalize(torch.randn(2, 7), dim=1),
                    "clean_labels": labels,
                    "indices": torch.arange(2),
                    "normalized": True,
                },
                path,
            )
            with self.assertRaises(ValueError):
                load_features(path, labels, expected_backbone="clip_vit_b16")


if __name__ == "__main__":
    unittest.main()
