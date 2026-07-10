"""DINOv2 ViT-S backbone with a multi-label classification head and LoRA."""

from __future__ import annotations

import torch
import torch.nn as nn

DINOV2_HIDDEN_SIZE = 384  # ViT-S/14
NUM_PATHOLOGIES = 14


class DINOv2Classifier(nn.Module):
    """Frozen DINOv2 backbone + lightweight classification head.

    The backbone is loaded once and frozen; only the head (and optionally
    LoRA adapters) train. This keeps the trainable parameter count tiny —
    the whole point of the DINOv2-vs-U-Net comparison.
    """

    def __init__(
        self,
        num_classes: int = NUM_PATHOLOGIES,
        backbone: nn.Module | None = None,
        hidden_size: int = DINOV2_HIDDEN_SIZE,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else _load_dinov2()
        self.hidden_size = hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        return self.head(features)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled CLS-token features, shape (B, hidden_size).

        DINOv2 expects 3-channel input. X-rays are single channel, so we
        repeat the channel 3x before feeding the backbone.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        outputs = self.backbone(x)
        # HF DINOv2 returns last_hidden_state (B, num_tokens, hidden).
        # Token 0 is the CLS token.
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        return outputs[:, 0]


def _load_dinov2() -> nn.Module:
    """Load pretrained DINOv2-small from HuggingFace (runtime, needs network)."""
    from transformers import AutoModel
    return AutoModel.from_pretrained("facebook/dinov2-small")


def inject_lora(model: DINOv2Classifier, rank: int = 8, alpha: int = 16) -> DINOv2Classifier:
    """Attach LoRA adapters to the backbone attention layers."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model.backbone = get_peft_model(model.backbone, config)
    return model
