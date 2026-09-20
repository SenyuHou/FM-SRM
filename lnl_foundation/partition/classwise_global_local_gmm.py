"""Class-conditional GMMs for real-noise Global-Local partitioning."""

import math

import numpy as np
import torch

from .global_local_gmm import (
    AMBIGUOUS, CLEAN, HARD, NOISY, RELIABLE_CLEAN, RELIABLE_NOISY,
    GlobalLocalGMMPartitioner,
)

DEFAULT_CLEAN_THRESHOLD = 0.7
DEFAULT_GLOBAL_NOISY_THRESHOLD = 0.95
DEFAULT_LOCAL_NOISY_THRESHOLD = 0.8


class ClasswiseGlobalLocalGMMPartitioner(GlobalLocalGMMPartitioner):
    """Fit each GMM within a noisy-label class; keep global geometry and KNN."""

    def __init__(self, *args, min_clean_fraction=0.05,
                 clean_threshold=DEFAULT_CLEAN_THRESHOLD,
                 global_noisy_threshold=DEFAULT_GLOBAL_NOISY_THRESHOLD, **kwargs):
        kwargs.setdefault("local_posterior_threshold", DEFAULT_LOCAL_NOISY_THRESHOLD)
        super().__init__(*args, **kwargs)
        if not 0 <= min_clean_fraction < 1:
            raise ValueError("min_clean_fraction must be in [0, 1).")
        if not (0.5 <= clean_threshold < 1 and 0.5 <= global_noisy_threshold < 1
                and clean_threshold + global_noisy_threshold > 1):
            raise ValueError("Classwise global thresholds must be in [0.5, 1) and not overlap.")
        self.min_clean_fraction = min_clean_fraction
        self.clean_threshold = clean_threshold
        self.global_noisy_threshold = global_noisy_threshold

    def _global_state(self, clean_prob, noisy_prob):
        state = np.full(len(clean_prob), AMBIGUOUS, dtype=np.int64)
        state[clean_prob >= self.clean_threshold] = RELIABLE_CLEAN
        state[noisy_prob >= self.global_noisy_threshold] = RELIABLE_NOISY
        return state

    def fit_predict(self, features: torch.Tensor, noisy_labels: torch.Tensor) -> dict:
        features = features.float().cpu()
        noisy_labels = noisy_labels.long().cpu()
        if (
            features.ndim != 2 or noisy_labels.ndim != 1
            or len(features) != len(noisy_labels) or len(features) < 2
            or not torch.isfinite(features).all()
            or (noisy_labels < 0).any() or (noisy_labels >= self.num_classes).any()
            or not 1 <= self.k < len(features)
        ):
            raise ValueError("Invalid features, labels, or k (require 1 <= k < N).")
        global_values = self._compute_global_margin(features, noisy_labels)
        local_consistency = self._compute_local_consistency(features, noisy_labels)
        result = self.fit_from_statistics(
            global_values["global_margin"], local_consistency, noisy_labels
        )
        result["positive_similarity"] = torch.as_tensor(
            global_values["positive_similarity"], dtype=torch.float32
        )
        result["negative_similarity"] = torch.as_tensor(
            global_values["negative_similarity"], dtype=torch.float32
        )
        return result

    def fit_from_statistics(self, global_margin, local_consistency, noisy_labels) -> dict:
        """Repartition saved statistics without recomputing frozen features or KNN."""
        if torch.is_tensor(global_margin):
            global_margin = global_margin.detach().cpu().numpy()
        if torch.is_tensor(local_consistency):
            local_consistency = local_consistency.detach().cpu().numpy()
        if torch.is_tensor(noisy_labels):
            noisy_labels = noisy_labels.detach().cpu().numpy()
        margin = np.array(global_margin, dtype=np.float64, copy=True)
        local = np.array(local_consistency, dtype=np.float64, copy=True)
        labels = np.array(noisy_labels, dtype=np.int64, copy=True)
        n = len(labels)
        if (
            margin.shape != (n,) or local.shape != (n,) or n < 2
            or not np.isfinite(margin).all() or not np.isfinite(local).all()
            or (local < 0).any() or (local > 1).any()
            or (labels < 0).any() or (labels >= self.num_classes).any()
        ):
            raise ValueError("Invalid classwise margins, local consistency, or labels.")

        clean_prob = np.full(n, 0.5, dtype=np.float64)
        noisy_prob = np.full(n, 0.5, dtype=np.float64)
        global_fallback = []
        for cls in range(self.num_classes):
            indices = np.flatnonzero(labels == cls)
            if not len(indices):
                continue
            values = margin[indices]
            if len(indices) < 2 or len(np.unique(values)) < 2:
                global_fallback.append(cls)
                continue
            gmm = self._make_gmm().fit(values.reshape(-1, 1))
            components = gmm.predict_proba(values.reshape(-1, 1))
            clean_component = int(np.argmax(gmm.means_.reshape(-1)))
            clean_prob[indices] = components[:, clean_component]
            noisy_prob[indices] = components[:, 1 - clean_component]

        global_state = self._global_state(clean_prob, noisy_prob)
        coverage_promoted, promoted_mask = self._ensure_class_coverage(
            global_state, margin, local, labels, clean_prob
        )
        partition = np.full(n, NOISY, dtype=np.int64)
        partition[global_state == RELIABLE_CLEAN] = CLEAN
        local_hard_prob = np.full(n, np.nan, dtype=np.float32)
        local_noisy_prob = np.full(n, np.nan, dtype=np.float32)
        local_fallback = []
        for cls in range(self.num_classes):
            indices = np.flatnonzero((labels == cls) & (global_state == AMBIGUOUS))
            if not len(indices):
                continue
            values = local[indices]
            if len(indices) < max(50, 2 * self.k) or len(np.unique(values)) < 2:
                local_fallback.append(cls)
                threshold = float(np.median(values))
                hard = values >= threshold
                local_hard_prob[indices] = hard.astype(np.float32)
                local_noisy_prob[indices] = (~hard).astype(np.float32)
            else:
                gmm = self._make_gmm().fit(values.reshape(-1, 1))
                components = gmm.predict_proba(values.reshape(-1, 1))
                noisy_component = int(np.argmin(gmm.means_.reshape(-1)))
                local_noisy_prob[indices] = components[:, noisy_component]
                local_hard_prob[indices] = components[:, 1 - noisy_component]
                hard = local_noisy_prob[indices] < self.local_posterior_threshold
            partition[indices[hard]] = HARD

        self.local_gmm_fallback = bool(local_fallback)
        return {
            "partition": torch.as_tensor(partition, dtype=torch.long),
            "global_margin": torch.as_tensor(margin, dtype=torch.float32),
            "global_clean_prob": torch.as_tensor(clean_prob, dtype=torch.float32),
            "global_noisy_prob": torch.as_tensor(noisy_prob, dtype=torch.float32),
            "global_state": torch.as_tensor(global_state, dtype=torch.long),
            "local_consistency": torch.as_tensor(local, dtype=torch.float32),
            "local_hard_prob": torch.as_tensor(local_hard_prob, dtype=torch.float32),
            "local_noisy_prob": torch.as_tensor(local_noisy_prob, dtype=torch.float32),
            "reliability_score": torch.as_tensor(clean_prob - noisy_prob, dtype=torch.float32),
            "local_gmm_fallback": self.local_gmm_fallback,
            "global_gmm_fallback_classes": global_fallback,
            "local_gmm_fallback_classes": local_fallback,
            "coverage_promoted_per_class": coverage_promoted,
            "coverage_promoted": torch.as_tensor(promoted_mask, dtype=torch.bool),
            "min_clean_fraction": self.min_clean_fraction,
            "clean_threshold": self.clean_threshold,
            "global_noisy_threshold": self.global_noisy_threshold,
            "local_noisy_threshold": self.local_posterior_threshold,
        }

    def _ensure_class_coverage(self, state, margin, local, labels, clean_prob):
        promoted = [0] * self.num_classes
        promoted_mask = np.zeros(len(labels), dtype=bool)
        for cls in range(self.num_classes):
            indices = np.flatnonzero(labels == cls)
            if not len(indices):
                continue
            clean_count = int((state[indices] == RELIABLE_CLEAN).sum())
            need = max(0, math.ceil(self.min_clean_fraction * len(indices)) - clean_count)
            if not need:
                continue
            candidates = indices[
                (state[indices] == AMBIGUOUS)
                & (margin[indices] > 0)
                & (clean_prob[indices] >= 0.5)
                & (local[indices] >= max(0.5, float(np.median(local[indices]))))
            ]
            order = np.lexsort((-local[candidates], -clean_prob[candidates]))
            chosen = candidates[order[:need]]
            state[chosen] = RELIABLE_CLEAN
            promoted_mask[chosen] = True
            promoted[cls] = len(chosen)
        return promoted, promoted_mask
