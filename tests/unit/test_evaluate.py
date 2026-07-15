"""Unit tests for evaluate.run_inference. Synthetic dataset, no checkpoints."""

import torch
from torch.utils.data import DataLoader

from training.evaluate import run_inference


class TinyModel(torch.nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _fake_dataset(n=8):
    data = []
    for i in range(n):
        lbl = torch.zeros(14)
        lbl[i % 14] = 1
        data.append({"image": torch.randn(1, 8, 8), "labels": lbl, "filename": f"x{i}.png"})
    return data


def _collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def test_run_inference_shapes():
    loader = DataLoader(_fake_dataset(8), batch_size=2, collate_fn=_collate)
    model = TinyModel()
    targets, preds, latency = run_inference(model, loader, torch.device("cpu"))

    assert targets.shape == (8, 14)
    assert preds.shape == (8, 14)
    assert latency > 0.0


def test_predictions_in_probability_range():
    loader = DataLoader(_fake_dataset(4), batch_size=2, collate_fn=_collate)
    model = TinyModel()
    _, preds, _ = run_inference(model, loader, torch.device("cpu"))

    assert preds.min() >= 0.0 and preds.max() <= 1.0
