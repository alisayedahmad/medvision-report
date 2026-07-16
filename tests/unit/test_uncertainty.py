"""Tests for uncertainty assessment."""
import numpy as np
from report_generation.uncertainty import (
    assess_uncertainty,
    build_uncertainty_note,
)
def _probs(*values, size: int = 14) -> np.ndarray:

    """Build a probability vector of given size, first N slots set to values."""
    arr = np.zeros(size)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


# --- assess_uncertainty ---

def test_confident_finding_not_flagged():
    probs = _probs(0.85, 0.10, 0.05)
    result = assess_uncertainty(probs)
    assert result.flag is False
    assert result.max_confidence == 0.85
    assert result.num_uncertain == 0


def test_normal_study_not_flagged():
    probs = _probs()  # all zeros
    result = assess_uncertainty(probs)
    assert result.flag is False
    assert result.max_confidence == 0.0
    assert result.num_uncertain == 0


def test_low_peak_with_uncertain_is_flagged():
    # nothing confident (max 0.55 < 0.6), and one in gray zone
    probs = _probs(0.55, 0.35, 0.10)
    result = assess_uncertainty(probs)
    assert result.flag is True
    assert "low peak" in result.reason.lower()


def test_low_peak_without_uncertain_not_flagged():
    # everything just below detection but not in gray zone either
    probs = _probs(0.55, 0.10, 0.05)
    result = assess_uncertainty(probs)
    assert result.flag is False


def test_many_uncertain_findings_flagged():
    # 3 predictions in gray zone [0.3, 0.5), max above low_max_confidence
    # so we hit the "many uncertain" rule instead of "low peak"
    probs = _probs(0.75, 0.35, 0.40, 0.45)
    result = assess_uncertainty(probs)
    assert result.flag is True
    assert "gray zone" in result.reason.lower() or "unable to commit" in result.reason.lower()
    assert result.num_uncertain == 3


def test_two_uncertain_not_flagged_when_peak_confident():

    # 2 in gray zone but one very confident finding present
    probs = _probs(0.90, 0.35, 0.40, 0.10)
    result = assess_uncertainty(probs)
    assert result.flag is False


def test_gray_zone_boundaries():
    # exactly 0.3 = in gray zone (inclusive lower)
    # exactly 0.5 = NOT in gray zone (exclusive upper)
    probs = _probs(0.30, 0.50)
    result = assess_uncertainty(probs)
    assert result.num_uncertain == 1


def test_mean_confidence_ignores_noise():
    # only 0.85 and 0.40 are above 0.3 noise floor
    probs = _probs(0.85, 0.40, 0.05, 0.02)
    result = assess_uncertainty(probs)
    expected = (0.85 + 0.40) / 2
    assert abs(result.mean_confidence - expected) < 1e-6


def test_mean_confidence_zero_when_all_below_noise():
    probs = _probs(0.10, 0.05, 0.02)
    result = assess_uncertainty(probs)
    assert result.mean_confidence == 0.0


def test_custom_thresholds():

    probs = _probs(0.55, 0.25)
    # with defaults, max=0.55 < 0.6 low_max, 0 uncertain → not flagged
    r1 = assess_uncertainty(probs)
    assert r1.flag is False

    # lower low_max_confidence trigger → still no uncertain → not flagged
    r2 = assess_uncertainty(probs, low_max_confidence=0.5)
    assert r2.flag is False


def test_high_uncertain_count_configurable():

    # confident max avoids rule 1, only 2 uncertain — under default cutoff of 3
    probs = _probs(0.80, 0.40, 0.35, 0.10)
    r_default = assess_uncertainty(probs)
    assert r_default.flag is False

    r_strict = assess_uncertainty(probs, high_uncertain_count=2)
    assert r_strict.flag is True


# --- build_uncertainty_note ---

def test_note_empty_when_not_flagged():
    
    probs = _probs(0.85)
    result = assess_uncertainty(probs)
    assert build_uncertainty_note(result) == ""


def test_note_non_empty_when_flagged():
    probs = _probs(0.35, 0.40, 0.45)
    result = assess_uncertainty(probs)
    note = build_uncertainty_note(result)
    assert "UNCERTAINTY" in note
    assert "radiologist" in note.lower()


# --- serialization ---

def test_to_dict_shape():
    probs = _probs(0.85)
    result = assess_uncertainty(probs)
    d = result.to_dict()
    assert set(d.keys()) == {
        "flag", "reason", "max_confidence",
        "mean_confidence", "num_uncertain",
    }
    assert isinstance(d["flag"], bool)
    assert isinstance(d["num_uncertain"], int)
