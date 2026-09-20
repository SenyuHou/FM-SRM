import ast
import contextlib
import io
import random
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from lnl_foundation.partition.simifeat import nearest, neighbor_distribution, empirical_moments, model_moments, solve_hoc, rank_detection


class SimiFeatParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        source = Path(__file__).resolve().parents[2] / "SimiFeat-main/SimiFeat-main"
        cls.original = dict(torch=torch, np=np, F=F, random=random, time=time,
            smp=torch.nn.Softmax(dim=0), smt=torch.nn.Softmax(dim=1),
            global_var=SimpleNamespace(set_value=lambda *args: None))
        needed = {"count_real", "distCosine", "cosDistance", "count_knn_distribution", "count_y", "get_score", "func", "calc_func", "get_knn_acc_all_class"}
        for name in ("utils.py", "hoc.py", "main_fast.py"):
            module = ast.parse((source / name).read_text(encoding="utf-8"))
            functions = ast.Module(body=[node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in needed], type_ignores=[])
            exec(compile(functions, str(source / name), "exec"), cls.original)

    def test_neighbors_scores_and_empirical_moments(self):
        torch.manual_seed(19)
        features = torch.randn(90, 16)
        labels = torch.arange(90) % 4
        args = SimpleNamespace(num_classes=4, min_similarity=0)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = self.original["count_knn_distribution"](args, features, labels, 90, 10)
        idx, distance = nearest(features, 10, block_size=17)
        actual = neighbor_distribution(idx, distance, labels, 4)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        observed = empirical_moments(labels.numpy(), idx.numpy(), 4)
        original = self.original["count_y"](4, features, labels, 90)
        for a, b in zip(observed, original):
            torch.testing.assert_close(a, b / 90, rtol=0, atol=0)

    def test_hoc_moments_gradients_and_solver(self):
        torch.manual_seed(11)
        t = torch.randn(4, 4, requires_grad=True)
        p = torch.randn(4, 1, requires_grad=True)
        actual = model_moments(t.softmax(1), p.softmax(0))
        expected = self.original["count_real"](4, t.softmax(1), p.softmax(0), -1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.flatten(), b.flatten(), rtol=1e-5, atol=1e-7)
        weights = [torch.randn_like(a) for a in actual]
        la = sum((a * w).sum() for a, w in zip(actual, weights))
        lb = sum((b.reshape_as(w) * w).sum() for b, w in zip(expected, weights))
        ga = torch.autograd.grad(la, (t, p), retain_graph=True)
        gb = torch.autograd.grad(lb, (t, p))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        observed = [v.detach() for v in actual]
        torch.manual_seed(5)
        with contextlib.redirect_stdout(io.StringIO()):
            _, old_t, old_p, _ = self.original["calc_func"](4, observed, False, "cpu", max_step=25)
        torch.manual_seed(5)
        new_t, new_p, _, _ = solve_hoc(observed, 4, "cpu", steps=25)
        np.testing.assert_allclose(new_t, old_t.numpy(), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(new_p, old_p.numpy(), rtol=1e-4, atol=1e-6)

    def test_vote_rank_upstream_decisions(self):
        torch.manual_seed(4)
        features = torch.randn(100, 16)
        labels = torch.arange(100) % 4
        distribution = neighbor_distribution(*nearest(features, 10), labels, 4)
        score = -torch.log(distribution[torch.arange(100), labels] + 1e-8).numpy()
        t = np.full((4, 4), 0.1) + np.eye(4) * 0.6
        p = np.ones((4, 1)) / 4
        jitter = np.array([.01, -.03, .02, -.04])
        posterior = t * p / .25
        posterior[np.arange(4), np.arange(4)] += jitter
        args = SimpleNamespace(num_classes=4, min_similarity=0, method="rank1", Tii_offset=1)
        data = {"feature": features, "noisy_label": labels, "noise_or_not": np.zeros(100, dtype=bool), "index": np.arange(100)}
        with contextlib.redirect_stdout(io.StringIO()):
            rank = self.original["get_knn_acc_all_class"](args, data, k=10, sel_noisy=[], thre_noise_rate=posterior)
            args.method = "mv"
            vote = self.original["get_knn_acc_all_class"](args, data, k=10, sel_noisy=[])
        expected_rank = np.zeros(100, dtype=bool)
        expected_rank[rank] = True
        actual_rank, _ = rank_detection(score, labels.numpy(), t, p, jitter)
        np.testing.assert_array_equal(actual_rank, expected_rank)
        np.testing.assert_array_equal(np.sort(vote), np.flatnonzero(distribution.argmax(1).numpy() != labels.numpy()))


if __name__ == "__main__":
    unittest.main()
