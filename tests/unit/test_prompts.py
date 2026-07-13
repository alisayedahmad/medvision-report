"""Tests for versioned prompt templates and the build_prompt() function """

import pytest

from report_generation.feature_extractor import PATHOLOGIES, extract_findings
from report_generation.prompts import (
    CURRENT_VERSION,
    build_prompt,
    get_template,
    list_versions,
)

import numpy as np


# --- helpers ---

def _findings_with_detections() -> dict:
    """Simulates a study with two findings, some negatives, one uncertain """
    probs = np.zeros(len(PATHOLOGIES))
    probs[2] = 0.88   # Consolidation — detected
    probs[4] = 0.62   # Effusion — detected
    probs[6] = 0.35   # Fibrosis — uncertain
    # rest are < 0.3 — negative
    return extract_findings(probs, model_name="dinov2").to_dict()


def _findings_empty() -> dict:
    """Normal study — nothing detected."""
    probs = np.zeros(len(PATHOLOGIES))
    return extract_findings(probs, model_name="dinov2").to_dict()


def _findings_with_masks() -> dict:
    """U-Net output with segmentation masks."""
    probs = np.zeros(len(PATHOLOGIES))
    probs[2] = 0.80
    masks = np.zeros((len(PATHOLOGIES), 224, 224))
    masks[2, 160:200, 130:200] = 1.0
    return extract_findings(probs, segmentation_masks=masks, model_name="unet").to_dict()


# --- registry ---

def test_all_versions_registered():
    versions = list_versions()
    assert "v1" in versions
    assert "v2" in versions
    assert "v3" in versions


def test_current_version_exists():
    template = get_template()
    assert template.version == CURRENT_VERSION


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="unknown prompt version"):
        get_template("v999")


# --- prompt structure ---

@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_build_returns_system_user_version(version):
    result = build_prompt(_findings_with_detections(), version=version)
    assert "system" in result
    assert "user" in result
    assert result["version"] == version


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_build_with_empty_findings(version):
    result = build_prompt(_findings_empty(), version=version)
    assert "system" in result
    assert "user" in result
    # should mention no findings detected
    assert "no" in result["user"].lower() or "none" in result["user"].lower()


# --- v1 baseline ---

def test_v1_contains_detected_pathology():
    result = build_prompt(_findings_with_detections(), version="v1")
    assert "Consolidation" in result["user"]
    assert "Effusion" in result["user"]


# --- v2 adds negatives ---

def test_v2_contains_negatives():
    result = build_prompt(_findings_with_detections(), version="v2")
    assert "NOT detected" in result["user"] or "not detected" in result["user"].lower()
    # Atelectasis is a negative (prob=0), should appear
    assert "Atelectasis" in result["user"]


def test_v2_system_forbids_hallucination():
    result = build_prompt(_findings_with_detections(), version="v2")
    assert "ONLY" in result["system"] or "only" in result["system"].lower()


# --- v3 full constraints ---

def test_v3_contains_negatives():
    result = build_prompt(_findings_with_detections(), version="v3")
    assert "Atelectasis" in result["user"]


def test_v3_contains_uncertain():
    result = build_prompt(_findings_with_detections(), version="v3")
    assert "Fibrosis" in result["user"]
    assert "UNCERTAIN" in result["user"] or "uncertain" in result["user"].lower()


def test_v3_grounding_rule_in_system():
    result = build_prompt(_findings_with_detections(), version="v3")
    assert "trace" in result["system"].lower() or "ground" in result["system"].lower()


def test_v3_spatial_note_without_masks():
    result = build_prompt(_findings_with_detections(), version="v3")
    # dinov2, no masks — should warn about diffuse
    assert "diffuse" in result["user"].lower()


def test_v3_spatial_note_with_masks():
    result = build_prompt(_findings_with_masks(), version="v3")
    assert "segmentation" in result["user"].lower()


def test_v3_mentions_model_name():
    result = build_prompt(_findings_with_detections(), version="v3")
    assert "dinov2" in result["user"]


def test_v3_report_sections_requested():
    result = build_prompt(_findings_with_detections(), version="v3")
    user = result["user"].upper()
    assert "FINDINGS" in user
    assert "IMPRESSION" in user
    assert "RECOMMENDATION" in user


# --- convenience function ---

def test_build_prompt_defaults_to_current():
    result = build_prompt(_findings_with_detections())
    assert result["version"] == CURRENT_VERSION
