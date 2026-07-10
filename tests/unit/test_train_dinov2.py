"""Unit tests for train_dinov2 helpers. Synthetic fixtures, no GPU/network."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.train_dinov2 import evaluate, patient_level_split


@pytest.fixture
def fake_csv(tmp_path):
    df = pd.DataFrame({
        "Image Index": [f"img_{i}.png" for i in range(20)],
        "Finding Labels": ["No Finding"] * 20,
        "Patient ID": [i // 2 for i in range(20)],  # 10 patients, 2 images each
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path


def test_patient_split_no_overlap(fake_csv):
    train_idx, val_idx = patient_level_split(fake_csv, val_fraction=0.3, seed=42)

    df = pd.read_csv(fake_csv)
    train_patients = set(df.iloc[train_idx]["Patient ID"])
    val_patients = set(df.iloc[val_idx]["Patient ID"])

    assert train_patients.isdisjoint(val_patients), "patient leaked across splits"


def test_patient_split_covers_all(fake_csv):
    train_idx, val_idx = patient_level_split(fake_csv, val_fraction=0.3, seed=42)
    assert len(train_idx) + len(val_idx) == 20


def test_patient_split_reproducible(fake_csv):
    a1, a2 = patient_level_split(fake_csv, val_fraction=0.3, seed=42)
    b1, b2 = patient_level_split(fake_csv, val_fraction=0.3, seed=42)
    assert a1 == b1 and a2 == b2


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.linear = torch.nn.Linear(1 * 8 * 8, num_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def test_evaluate_returns_expected_keys():
    images = torch.randn(6, 1, 8, 8)
    labels = torch.zeros(6, 14)
    labels[0, 0] = 1
    labels[1, 3] = 1

    dataset = [{"image": img, "labels": lbl, "filename": f"x{i}.png"}
               for i, (img, lbl) in enumerate(zip(images, labels))]

    def collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    model = DummyModel()
    metrics = evaluate(model, loader, torch.device("cpu"))

    assert "mean_auc" in metrics
