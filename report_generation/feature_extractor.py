"""Extract structured clinical findings from model outputs.

Sits between vision model and LLM. Takes raw probabilities (and optionally
segmentation masks from U-Net) and produces a structured representation
that constrains what the LLM can say. No PyTorch dependency — numpy only


"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]


# 3x2 grid: rows split the image into upper/middle/lower thirds,
# columns split left/right at the midline. Maps directly to how
# radiologists describe location in reports


ANATOMICAL_ZONES = {
    (0, 0): "right upper",
    (0, 1): "left upper",
    (1, 0): "right middle",
    (1, 1): "left middle",
    (2, 0): "right lower",
    (2, 1): "left lower",
}

# Severity bins. Simple, documented, recalibrable
# These are heuristic — not validated against radiologist grading

SEVERITY_THRESHOLDS = {"severe": 0.85, "moderate": 0.65}


@dataclass
class Finding:
    """One detected pathology with everything the LLM needs to describe it """

    pathology: str
    confidence: float
    severity: str
    location: str  # anatomical zone or "diffuse"
    area_fraction: float | None = None  # fraction of image covered, U-Net only

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pathology": self.pathology,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "location": self.location,
        }
        if self.area_fraction is not None:
            d["area_fraction"] = round(self.area_fraction, 4)
        return d


@dataclass
class StudyFindings:
    """Complete structured output for one study image.

    The LLM prompt consumes this. `negatives` is just as important as
    `findings` — explicitly listing what's absent prevents the LLM from
    inventing findings the vision model never detected

    """

    findings: list[Finding] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    model_name: str = ""
    has_spatial_info: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "negatives": self.negatives,
            "uncertain": self.uncertain,
            "model_name": self.model_name,
            "has_spatial_info": self.has_spatial_info,
        }


def classify_severity(confidence: float, area_fraction: float | None = None) -> str:
    """Map confidence (and optionally lesion area) to severity label.

    When segmentation is available, large area boosts severity by one level.
    A 0.70-confidence finding covering 15% of the lung is more concerning
    than a 0.70 pinpoint detection

    """
    area_boost = area_fraction is not None and area_fraction > 0.10

    if confidence >= SEVERITY_THRESHOLDS["severe"]:
        return "severe"
    if confidence >= SEVERITY_THRESHOLDS["moderate"]:
        return "severe" if area_boost else "moderate"
    return "moderate" if area_boost else "mild"


def localize_from_mask(mask: np.ndarray) -> tuple[str, float]:
    """Find anatomical zone from a binary segmentation mask.

    Returns (zone_name, area_fraction). Zone is determined by the
    centroid of the positive region — not perfect, but robust enough
    for report-level location descriptions
    
    """
    binary = mask > 0.5
    total_pixels = binary.size
    positive_pixels = binary.sum()

    if positive_pixels == 0:
        return "diffuse", 0.0

    area_fraction = float(positive_pixels / total_pixels)

    # centroid of the positive region
    ys, xs = np.where(binary)
    cy = float(ys.mean())
    cx = float(xs.mean())

    h, w = mask.shape
    row = min(int(cy / h * 3), 2)  # 0=upper, 1=middle, 2=lower
    col = min(int(cx / w * 2), 1)  # 0=right, 1=left (radiological convention)

    zone = ANATOMICAL_ZONES[(row, col)]
    return zone, area_fraction


def extract_findings(
    probabilities: np.ndarray,
    segmentation_masks: np.ndarray | None = None,
    model_name: str = "unknown",
    confidence_threshold: float = 0.5,
    uncertainty_threshold: float = 0.3,
    pathology_names: list[str] | None = None,
) -> StudyFindings:
    """Main entry point. Probabilities in, structured findings out.

    Args:
        probabilities: shape (num_classes,) — sigmoid outputs per pathology
        segmentation_masks: shape (num_classes, H, W) or None
        model_name: for traceability in the report
        confidence_threshold: above this = detected finding
        uncertainty_threshold: between this and confidence = uncertain
        pathology_names: override default PATHOLOGIES list
    """
    names = pathology_names or PATHOLOGIES
    assert len(probabilities) == len(names), (
        f"got {len(probabilities)} probabilities for {len(names)} pathologies"
    )
    if segmentation_masks is not None:
        assert segmentation_masks.shape[0] == len(names), (
            f"got {segmentation_masks.shape[0]} masks for {len(names)} pathologies"
        )

    has_spatial = segmentation_masks is not None
    findings = []
    negatives = []
    uncertain = []

    for i, name in enumerate(names):
        prob = float(probabilities[i])

        if prob >= confidence_threshold:
            # detected — build a full finding
            if has_spatial:
                location, area_frac = localize_from_mask(segmentation_masks[i])
            else:
                location = "diffuse"
                area_frac = None

            severity = classify_severity(prob, area_frac)
            findings.append(Finding(
                pathology=name,
                confidence=prob,
                severity=severity,
                location=location,
                area_fraction=area_frac,
            ))

        elif prob < uncertainty_threshold:
            negatives.append(name)
        else:
            uncertain.append(name)

    # highest confidence first — the LLM should lead with the main finding
    findings.sort(key=lambda f: f.confidence, reverse=True)

    return StudyFindings(
        findings=findings,
        negatives=negatives,
        uncertain=uncertain,
        model_name=model_name,
        has_spatial_info=has_spatial,
    )
