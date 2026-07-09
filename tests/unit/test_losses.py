"""Unit tests for loss functions."""

import torch
import pytest

from models.losses import DiceLoss, FocalLoss, MultiLabelBCE


@pytest.fixture
def dummy_batch():
    logits = torch.randn(4, 14)
    targets = torch.zeros(4, 14)
    targets[0, [0, 3]] = 1.0
    targets[1, [7]] = 1.0
    return logits, targets


def test_focal_loss_runs(dummy_batch):
    loss = FocalLoss()(dummy_batch[0], dummy_batch[1])
    assert loss.shape == ()
    assert loss.item() > 0


def test_focal_loss_lower_for_correct_predictions():
    targets = torch.ones(4, 14)
    easy = FocalLoss()(torch.full((4, 14), 5.0), targets)  # confident correct
    hard = FocalLoss()(torch.full((4, 14), -5.0), targets)  # confident wrong
    assert easy < hard


def test_multi_label_bce_runs(dummy_batch):
    loss = MultiLabelBCE()(dummy_batch[0], dummy_batch[1])
    assert loss.shape == ()
    assert loss.item() > 0


def test_multi_label_bce_with_pos_weights(dummy_batch):
    weights = torch.ones(14) * 2.0
    loss = MultiLabelBCE(pos_weights=weights)(dummy_batch[0], dummy_batch[1])
    assert loss.item() > 0


def test_dice_loss_perfect_overlap():
    logits = torch.full((1, 1, 8, 8), 10.0)  # high confidence
    targets = torch.ones(1, 1, 8, 8)
    loss = DiceLoss()(logits, targets)
    assert loss.item() < 0.05  # near-zero


def test_dice_loss_no_overlap():
    logits = torch.full((1, 1, 8, 8), -10.0)  # confident wrong
    targets = torch.ones(1, 1, 8, 8)
    loss = DiceLoss()(logits, targets)
    assert loss.item() > 0.9  # near 1.0
