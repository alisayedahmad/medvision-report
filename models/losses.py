"""Loss functions for multi-label chest X-ray classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-label classification.

    Down-weights easy negatives so the model focuses on hard positives —
    critical when 'No Finding' dominates the dataset



    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        return (focal_weight * bce).mean()


class MultiLabelBCE(nn.Module):
    """Weighted BCE — allows per-pathology class weights  """

    def __init__(self, pos_weights: torch.Tensor | None = None):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights,
        )


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation masks """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
