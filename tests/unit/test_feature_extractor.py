"""Tests for structured feature extraction. No model, no data — pure numpy."""

import numpy as np
import pytest

from report_generation.feature_extractor import (
    PATHOLOGIES,
    Finding,
    StudyFindings,
    classify_severity,
    extract_findings,
    localize_from_mask,
)

NUM_CLASSES = len(PATHOLOGIES)


# --- classify_severity ---

def test_severity_high_confidence():
    assert classify_severity(0.90) == "severe"


def test_severity_moderate_confidence():
    assert classify_severity(0.70) == "moderate"


def test_severity_low_confidence():
    assert classify_severity(0.55) == "mild"


def test_severity_area_boost_promotes_moderate_to_severe():
    # 0.70 alone = moderate, but large area pushes it to severe
    assert classify_severity(0.70, area_fraction=0.15) == "severe"


def test_severity_area_boost_promotes_mild_to_moderate():
    assert classify_severity(0.55, area_fraction=0.12) == "moderate"


def test_severity_area_boost_ignored_when_small():
    # area below 10% threshold — no boost
    assert classify_severity(0.55, area_fraction=0.05) == "mild"


# --- localize_from_mask ---

def test_localize_empty_mask():
    mask = np.zeros((224, 224))
    zone, area = localize_from_mask(mask)
    assert zone == "diffuse"
    assert area == 0.0


def test_localize_top_left():
    # top-left quadrant in image = right upper in radiology convention
    mask = np.zeros((224, 224))
    mask[10:30, 10:50] = 1.0
    zone, area = localize_from_mask(mask)
    assert zone == "right upper"
    assert area > 0


def test_localize_bottom_right():
    mask = np.zeros((224, 224))
    mask[180:220, 140:220] = 1.0
    zone, area = localize_from_mask(mask)
    assert zone == "left lower"


def test_localize_center():
    mask = np.zeros((224, 224))
    mask[90:140, 50:110] = 1.0
    zone, area = localize_from_mask(mask)
    assert zone == "right middle"


def test_localize_area_fraction_is_correct():
    mask = np.zeros((100, 100))
    mask[0:10, 0:10] = 1.0  # 100 / 10000 = 0.01
    _, area = localize_from_mask(mask)
    assert abs(area - 0.01) < 1e-6


# --- extract_findings ---

def test_extract_clear_positive():
    probs = np.zeros(NUM_CLASSES)
    probs[2] = 0.87  # Consolidation
    result = extract_findings(probs, model_name="test")

    assert len(result.findings) == 1
    assert result.findings[0].pathology == "Consolidation"
    assert result.findings[0].confidence == pytest.approx(0.87, abs=1e-4)
    assert "Consolidation" not in result.negatives
    assert "Consolidation" not in result.uncertain


def test_extract_clear_negative():
    probs = np.full(NUM_CLASSES, 0.1)  # all well below uncertainty threshold
    result = extract_findings(probs)

    assert len(result.findings) == 0
    assert len(result.negatives) == NUM_CLASSES
    assert len(result.uncertain) == 0


def test_extract_uncertain_zone():
    probs = np.zeros(NUM_CLASSES)
    probs[0] = 0.40  # between 0.3 and 0.5 — uncertain
    result = extract_findings(probs)

    assert len(result.findings) == 0
    assert "Atelectasis" in result.uncertain
    assert "Atelectasis" not in result.negatives


def test_extract_mixed_findings():
    probs = np.zeros(NUM_CLASSES)
    probs[0] = 0.80   # Atelectasis — detected
    probs[4] = 0.60   # Effusion — detected
    probs[6] = 0.35   # Fibrosis — uncertain
    probs[8] = 0.10   # Infiltration — negative
    result = extract_findings(probs)

    assert len(result.findings) == 2
    assert len(result.uncertain) == 1
    assert "Fibrosis" in result.uncertain
    assert "Infiltration" in result.negatives


def test_findings_sorted_by_confidence():
    probs = np.zeros(NUM_CLASSES)
    probs[0] = 0.60
    probs[4] = 0.90
    probs[2] = 0.75
    result = extract_findings(probs)

    confidences = [f.confidence for f in result.findings]
    assert confidences == sorted(confidences, reverse=True)


def test_extract_with_segmentation_masks():
    probs = np.zeros(NUM_CLASSES)
    probs[2] = 0.80  # Consolidation

    masks = np.zeros((NUM_CLASSES, 224, 224))
    masks[2, 160:200, 130:200] = 1.0  # bottom-right = left lower

    result = extract_findings(probs, segmentation_masks=masks, model_name="unet")
    assert result.has_spatial_info
    assert result.findings[0].location == "left lower"
    assert result.findings[0].area_fraction > 0


def test_extract_without_masks_gives_diffuse():
    probs = np.zeros(NUM_CLASSES)
    probs[0] = 0.75
    result = extract_findings(probs, segmentation_masks=None, model_name="dinov2")

    assert not result.has_spatial_info
    assert result.findings[0].location == "diffuse"
    assert result.findings[0].area_fraction is None


def test_custom_thresholds():
    probs = np.zeros(NUM_CLASSES)
    probs[0] = 0.45  # would be uncertain at defaults, but detected at 0.4

    result = extract_findings(probs, confidence_threshold=0.4, uncertainty_threshold=0.2)
    assert len(result.findings) == 1

    result2 = extract_findings(probs, confidence_threshold=0.5, uncertainty_threshold=0.3)
    assert len(result2.findings) == 0
    assert "Atelectasis" in result2.uncertain


def test_shape_mismatch_raises():
    probs = np.zeros(10)  # wrong size
    with pytest.raises(AssertionError):
        extract_findings(probs)


def test_mask_shape_mismatch_raises():
    probs = np.zeros(NUM_CLASSES)
    masks = np.zeros((10, 224, 224))  # wrong number of channels
    with pytest.raises(AssertionError):
        extract_findings(probs, segmentation_masks=masks)


# --- serialization ---

def test_finding_to_dict():
    f = Finding("Pneumonia", 0.82, "moderate", "right lower", area_fraction=0.034)
    d = f.to_dict()
    assert d["pathology"] == "Pneumonia"
    assert d["area_fraction"] == 0.034
    assert isinstance(d["confidence"], float)


def test_finding_to_dict_no_area():
    f = Finding("Pneumonia", 0.82, "moderate", "diffuse")
    d = f.to_dict()
    assert "area_fraction" not in d


def test_study_findings_to_dict_roundtrip():
    probs = np.zeros(NUM_CLASSES)
    probs[2] = 0.88
    probs[12] = 0.55
    result = extract_findings(probs, model_name="dinov2")
    d = result.to_dict()

    assert isinstance(d, dict)
    assert len(d["findings"]) == 2
    assert isinstance(d["negatives"], list)
    assert d["model_name"] == "dinov2"


def test_no_findings_produces_valid_output():
    probs = np.zeros(NUM_CLASSES)
    result = extract_findings(probs)
    d = result.to_dict()

    assert d["findings"] == []
    assert len(d["negatives"]) == NUM_CLASSES
