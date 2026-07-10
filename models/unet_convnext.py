"""U-Net with ConvNeXt backbone for pixel-level pathology segmentation."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

NUM_PATHOLOGIES = 14


def build_unet(
    num_classes: int = NUM_PATHOLOGIES,
    encoder_name: str = "tu-convnext_tiny",
    pretrained: bool = True,
    in_channels: int = 1,
) -> nn.Module:
    """U-Net segmentation model. Single-channel input, one mask per pathology."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )


class UNetClassifier(nn.Module):
    """Wraps the U-Net to also output image-level classification logits.

    Segmentation gives per-pixel masks; global pooling over each mask
    gives a classification score per pathology. Same head, two outputs.
    """

    def __init__(self, num_classes: int = NUM_PATHOLOGIES, pretrained: bool = True):
        super().__init__()
        self.unet = build_unet(num_classes=num_classes, pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seg_logits = self.unet(x)                      # (B, C, H, W)
        cls_logits = self.pool(seg_logits).flatten(1)  # (B, C)
        return {"segmentation": seg_logits, "classification": cls_logits}
