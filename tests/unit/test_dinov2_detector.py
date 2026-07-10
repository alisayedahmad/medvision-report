"""Unit tests for DINOv2Classifier. Uses a stub backbone, no network needed  
"""

import torch
import torch.nn as nn

from models.dinov2_detector import DINOV2_HIDDEN_SIZE, DINOv2Classifier


class StubBackbone(nn.Module):
    """Fake DINOv2 returning (B, num_tokens, hidden) — mimics HF output shape """

    def __init__(self, hidden=DINOV2_HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(3, hidden, kernel_size=14, stride=14)

    def forward(self, x):
        feat = self.proj(x)                     # (B, hidden, H', W')
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H'*W', hidden)
        cls = tokens.mean(1, keepdim=True)        # fake CLS token
        return torch.cat([cls, tokens], dim=1)    # (B, 1+H'W', hidden)


def _model():
    return DINOv2Classifier(num_classes=14, backbone=StubBackbone())


def test_forward_output_shape():
    model = _model()
    x = torch.randn(2, 1, 224, 224)  # single-channel batch
    out = model(x)
    assert out.shape == (2, 14)


def test_accepts_single_channel_input():
    model = _model()
    x = torch.randn(1, 1, 224, 224)
    out = model(x)  # should not raise — channel repeat handles it
    assert out.shape == (1, 14)


def test_backbone_frozen_by_default():
    model = _model()
    
    backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
    assert not any(backbone_grads)


def test_head_is_trainable():

    model = _model()
    head_grads = [p.requires_grad for p in model.head.parameters()]

    assert all(head_grads)


def test_unfrozen_backbone_trains():


    model = DINOv2Classifier(num_classes=14, backbone=StubBackbone(), freeze_backbone=False)
    assert any(p.requires_grad for p in model.backbone.parameters())
