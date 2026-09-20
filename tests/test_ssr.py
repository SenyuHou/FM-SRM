import unittest

import torch

from lnl_foundation.training.baseline_linear_probe import (
    _ssr_select,
    train_stage2_baseline,
)


class SSRTest(unittest.TestCase):
    def test_selection_uses_neighbor_agreement(self):
        neighbors = torch.tensor([
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1],
            [3, 4, 5],
            [4, 3, 5],
            [5, 3, 4],
        ])
        labels = torch.tensor([0, 0, 1, 1, 1, 1])

        selected, scores = _ssr_select(neighbors, labels, 2, 1.0)

        self.assertEqual(selected.tolist(), [0, 1, 3, 4, 5])
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_frozen_feature_training_smoke(self):
        generator = torch.Generator().manual_seed(7)
        class_zero = torch.randn(12, 4, generator=generator) * 0.1 - 1.0
        class_one = torch.randn(12, 4, generator=generator) * 0.1 + 1.0
        train_features = torch.cat((class_zero, class_one))
        labels = torch.tensor([0] * 12 + [1] * 12)
        config = {
            "epochs": 2,
            "batch_size": 8,
            "optimizer": "sgd",
            "learning_rate": 0.02,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "theta_selection": 0.0,
            "theta_relabel": 0.9,
            "knn_k": 3,
            "knn_block_size": 8,
            "mixup_alpha": 4.0,
            "feature_dropout": 0.0,
            "minimum_learning_rate_ratio": 0.02,
        }

        result = train_stage2_baseline(
            "ssr",
            train_features,
            labels,
            train_features,
            labels,
            num_classes=2,
            dataset="cifar10",
            noise_type="symmetric",
            noise_rate=0.6,
            seed=1,
            device=torch.device("cpu"),
            cfg={"ssr": config},
        )

        self.assertEqual(result.metrics["n_train"], 24)
        self.assertEqual(result.metrics["final_selected"], 24)
        self.assertIn("accuracy", result.metrics)
        self.assertIn("macro_f1", result.metrics)
        self.assertIn("ece_raw", result.metrics)


if __name__ == "__main__":
    unittest.main()
