"""Evaluation metrics for classification and segmentation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Per-class and mean AUC. Skips classes with no positive samples."""
    n_classes = targets.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    results = {}
    valid_aucs = []

    for i, name in enumerate(class_names):
        if targets[:, i].sum() == 0:
            results[name] = float("nan")
            continue
        auc = roc_auc_score(targets[:, i], predictions[:, i])
        results[name] = auc
        valid_aucs.append(auc)

    results["mean_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return results


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice coefficient between two binary masks."""


    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()

    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Hausdorff distance between mask boundaries. Returns inf if either is empty."""

    from scipy.ndimage import distance_transform_edt


    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if not pred.any() or not gt.any():
        return float("inf")

    dt_pred = distance_transform_edt(~pred)
    dt_gt = distance_transform_edt(~gt)

    return float(max(dt_pred[gt].max(), dt_gt[pred].max()))


def sensitivity_specificity(targets: np.ndarray, predictions: np.ndarray, threshold: float = 0.5,) -> dict[str, float]:
    """Binary sensitivity and specificity at a given threshold."""


    preds_binary = (predictions >= threshold).astype(int)
    targets_binary = targets.astype(int)

    tp = ((preds_binary == 1) & (targets_binary == 1)).sum()
    tn = ((preds_binary == 0) & (targets_binary == 0)).sum()
    fp = ((preds_binary == 1) & (targets_binary == 0)).sum()
    fn = ((preds_binary == 0) & (targets_binary == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {"sensitivity": float(sensitivity), "specificity": float(specificity)}
