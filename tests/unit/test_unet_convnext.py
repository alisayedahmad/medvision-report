"""Unit tests for U-Net ConvNeXt. Uses pretrained=False, no network needed."""

import torch

from models.unet_convnext import UNetClassifier, build_unet


def test_build_unet_output_shape():
    model = build_unet(num_classes=14, pretrained=False)
    out = model(torch.randn(1, 1, 224, 224))
    assert out.shape == (1, 14, 224, 224)


def test_unet_accepts_single_channel():
    model = build_unet(num_classes=14, pretrained=False, in_channels=1)
    out = model(torch.randn(2, 1, 128, 128))
    assert out.shape == (2, 14, 128, 128)


def test_classifier_returns_both_heads():
    model = UNetClassifier(num_classes=14, pretrained=False)
    out = model(torch.randn(1, 1, 224, 224))
    assert set(out.keys()) == {"segmentation", "classification"}


def test_classifier_output_shapes():
    model = UNetClassifier(num_classes=14, pretrained=False)
    out = model(torch.randn(2, 1, 224, 224))
    assert out["segmentation"].shape == (2, 14, 224, 224)
    assert out["classification"].shape == (2, 14)


def test_gradients_flow():
    model = UNetClassifier(num_classes=14, pretrained=False)
    out = model(torch.randn(1, 1, 128, 128))
    loss = out["classification"].sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
