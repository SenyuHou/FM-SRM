"""Classification-only evaluation and validation-selected checkpoints."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate_classification(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    if logits.ndim != 2 or len(logits) != len(labels) or logits.shape[1] < 5:
        raise ValueError("Expected [N, C>=5] logits and N labels.")
    if not torch.isfinite(logits).all():
        raise ValueError("Evaluation logits contain NaN or Inf.")
    prediction = logits.argmax(dim=1)
    top5 = logits.topk(5, dim=1).indices.eq(labels[:, None]).any(dim=1)
    return {
        "top1": float(prediction.eq(labels).float().mean()),
        "top5": float(top5.float().mean()),
        "macro_f1": float(f1_score(labels.numpy(), prediction.numpy(),
                                   labels=np.arange(logits.shape[1]), average="macro", zero_division=0)),
    }


@torch.no_grad()
def linear_logits(models, features, device, reduction="mean", batch_size=4096):
    features = torch.as_tensor(features, dtype=torch.float32).cpu()
    outputs = []
    for model in models:
        previous = model.training
        model.eval()
        chunks = [model(features[start:start + batch_size].to(device)).float().cpu()
                  for start in range(0, len(features), batch_size)]
        model.train(previous)
        outputs.append(torch.cat(chunks))
    stacked = torch.stack(outputs)
    return stacked.sum(0) if reduction == "sum" else stacked.mean(0)


class BestValidationCheckpoint:
    def __init__(self, features, labels, device, path):
        self.features = features
        self.labels = labels
        self.device = device
        self.path = Path(path)
        self.best_top1 = -1.0
        self.best_epoch = None
        self.best_metrics = None

    def __call__(self, epoch, models, reduction="mean"):
        metrics = evaluate_classification(
            linear_logits(models, self.features, self.device, reduction), self.labels
        )
        if metrics["top1"] > self.best_top1:
            self.best_top1 = metrics["top1"]
            self.best_epoch = epoch + 1
            self.best_metrics = metrics
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": self.best_epoch,
                "validation_metrics": metrics,
                "reduction": reduction,
                "state_dicts": [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                                for model in models],
            }, self.path)
        return metrics


def load_best_heads(path, feature_dim, num_classes, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for state in checkpoint["state_dicts"]:
        model = torch.nn.Linear(feature_dim, num_classes).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, checkpoint
