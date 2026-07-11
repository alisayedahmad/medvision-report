"""Unit tests for train_unet helpers. Uses tiny non-pretrained model, no network 
needed"""

import numpy as np
import torch
from torch.utils.data import DataLoader


from models.unet_convnext import UNetClassifier

from training.train_unet import evaluate



def _fake_batch(n=4):
    # Create a fake dataset of n samples with random images and labels
    dataset = [
        {"image": torch.randn(1, 64, 64), "labels": torch.zeros(14), "filename": f"x{i}.png"}
        for i in range(n)
    ]
    dataset[0]["labels"][0] = 1
    dataset[1]["labels"][3] = 1
    return dataset


def _collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def test_evaluate_returns_expected_keys():
    model = UNetClassifier(num_classes=14, pretrained=False)
    loader = DataLoader(_fake_batch(4), batch_size=2, collate_fn=_collate)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert "mean_auc" in metrics


def test_evaluate_produces_valid_auc_range():



    model = UNetClassifier(num_classes=14, pretrained=False) 
    loader = DataLoader(_fake_batch(4), batch_size=2, collate_fn=_collate)
    metrics = evaluate(model, loader, torch.device("cpu"))
    

    for name, val in metrics.items():
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0
