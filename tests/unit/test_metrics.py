"""Unit tests for evaluation metrics."""

import numpy as np

from models.metrics import compute_auc, dice_score, hausdorff_distance, sensitivity_specificity


def test_auc_perfect_predictions():
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    preds = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    result = compute_auc(targets, preds)
    assert result["mean_auc"] == 1.0


def test_auc_skips_empty_class():
    targets = np.array([[1, 0], [1, 0]])  # class 1 has no positives
    preds = np.array([[0.9, 0.1], [0.8, 0.2]])
    result = compute_auc(targets, preds)
    assert np.isnan(result.get("Effusion", float("nan")))


def test_dice_perfect_overlap():
    mask = np.ones((8, 8), dtype=np.uint8)
    assert dice_score(mask, mask) > 0.99


def test_dice_no_overlap():
    pred = np.zeros((8, 8), dtype=np.uint8)
    gt = np.ones((8, 8), dtype=np.uint8)
    assert dice_score(pred, gt) < 0.01


def test_hausdorff_identical_masks():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    assert hausdorff_distance(mask, mask) == 0.0


def test_hausdorff_empty_mask_returns_inf():
    pred = np.zeros((8, 8), dtype=np.uint8)
    gt = np.ones((8, 8), dtype=np.uint8)
    assert hausdorff_distance(pred, gt) == float("inf")


def test_sensitivity_specificity_perfect():
    targets = np.array([1, 1, 0, 0])
    preds = np.array([0.9, 0.8, 0.1, 0.2])
    result = sensitivity_specificity(targets, preds)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
