"""Confidence thresholding and uncertainty flagging:

The vision model's confidence is a noisy signal.
When it's low, the report needs to say so instead of committing to findings that aren't really there. This module decides when to raise the uncertainty flag
and what to tell the LLM about it

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UncertaintyAssessment:
    """Per-study uncertainty verdict for the report."""

    flag: bool                    # true if the report should warn the reader
    reason: str                   # short human-readable explanation
    max_confidence: float          # highest confidence across all pathologies
    mean_confidence: float          # mean over predictions above noise floor
    num_uncertain: int             # count of pathologies in the gray zone

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "reason": self.reason,
            "max_confidence": round(self.max_confidence, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "num_uncertain": self.num_uncertain,
        }


def assess_uncertainty(
    probabilities: np.ndarray,
    confidence_threshold: float = 0.5,
    uncertainty_threshold: float = 0.3,
    low_max_confidence: float = 0.6,
    high_uncertain_count: int = 3,
) -> UncertaintyAssessment:
    """Decide whether to flag this study as uncertain.

    Three failure modes to catch:
    
    1. Nothing is confident — max prob well below detection threshold
    2. Many predictions sit in the gray zone — model can't commit
    3. All findings are borderline — no strong signal above noise
    """
    max_conf = float(probabilities.max())

    # gray zone count
    in_gray = (probabilities >= uncertainty_threshold) & (probabilities < confidence_threshold)
    num_uncertain = int(in_gray.sum())

    # mean over non-noise predictions
    above_noise = probabilities[probabilities >= uncertainty_threshold]
    mean_conf = float(above_noise.mean()) if above_noise.size > 0 else 0.0

    # decide
    if max_conf < low_max_confidence and num_uncertain > 0:
        return UncertaintyAssessment(
            flag=True,
            reason=f"low peak confidence ({max_conf:.2f}) with {num_uncertain} uncertain finding(s)",
            max_confidence=max_conf,
            mean_confidence=mean_conf,
            num_uncertain=num_uncertain,
        )

    if num_uncertain >= high_uncertain_count:
        return UncertaintyAssessment(
            flag=True,
            reason=f"{num_uncertain} findings in the gray zone — model unable to commit",
            max_confidence=max_conf,
            mean_confidence=mean_conf,
            num_uncertain=num_uncertain,
        )

    return UncertaintyAssessment(
        flag=False,
        reason="confidence within expected range",
        max_confidence=max_conf,
        mean_confidence=mean_conf,
        num_uncertain=num_uncertain,
    )


def build_uncertainty_note(assessment: UncertaintyAssessment) -> str:
    """Human-readable disclaimer to append to the report when flagged."""
    if not assessment.flag:
        return ""

    return (
        "UNCERTAINTY NOTICE: this report was generated from predictions with "
        f"reduced confidence ({assessment.reason}). Clinical correlation and "
        "review by a qualified radiologist are strongly recommended."
    )
