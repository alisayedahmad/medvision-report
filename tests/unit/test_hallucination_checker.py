"""Tests for hallucination checker. All deterministic — no LLM needed."""

import numpy as np

from report_generation.feature_extractor import PATHOLOGIES, extract_findings
from report_generation.hallucination_checker import (
    check_report,
    find_mentioned_pathologies,
)


# --- helpers ---

def _findings(*detected: str, uncertain: list[str] | None = None) -> dict:
    """Build a StudyFindings dict with specified pathologies detected."""
    probs = np.zeros(len(PATHOLOGIES))
    for name in detected:
        probs[PATHOLOGIES.index(name)] = 0.85
    for name in uncertain or []:
        probs[PATHOLOGIES.index(name)] = 0.40
    return extract_findings(probs, model_name="test").to_dict()


# --- find_mentioned_pathologies ---

def test_finds_direct_mention():
    text = "Consolidation is present in the right lower lobe."
    assert "Consolidation" in find_mentioned_pathologies(text)


def test_finds_synonym():
    text = "Enlarged heart noted."
    assert "Cardiomegaly" in find_mentioned_pathologies(text)


def test_case_insensitive():
    text = "PNEUMONIA in the left lung."
    assert "Pneumonia" in find_mentioned_pathologies(text)


def test_multiple_mentions_deduplicated():
    text = "Consolidation seen. Consolidation persists."
    result = find_mentioned_pathologies(text)
    assert result == {"Consolidation"}


def test_word_boundary_prevents_substring_match():
    # "mass" is a synonym for "Mass" — but "mastectomy" should not trigger it
    text = "Post-mastectomy changes noted."
    result = find_mentioned_pathologies(text)
    assert "Mass" not in result


def test_finds_multiple_distinct_pathologies():
    text = "Effusion in the right base with associated consolidation."
    result = find_mentioned_pathologies(text)
    assert "Effusion" in result
    assert "Consolidation" in result


def test_multi_word_synonym_matches():
    text = "Pleural thickening bilaterally."
    assert "Pleural_Thickening" in find_mentioned_pathologies(text)


# --- negation handling ---

def test_negation_no():
    text = "No consolidation identified."
    assert "Consolidation" not in find_mentioned_pathologies(text)


def test_negation_without():
    text = "The lungs are without effusion."
    assert "Effusion" not in find_mentioned_pathologies(text)


def test_negation_absent():
    text = "Pneumothorax is absent."
    # "absent" is after the mention, so this actually IS a hallucination
    # of sorts — we only check backward. Documented limitation.
    # For real reports the pattern is "no X" / "X is not present"
    # This test documents current behavior.
    result = find_mentioned_pathologies(text)
    assert "Pneumothorax" in result  # backward-only negation, by design


def test_negation_negative_for():
    text = "Chest is negative for pneumonia."
    assert "Pneumonia" not in find_mentioned_pathologies(text)


def test_negation_does_not_cross_sentence():
    text = "No effusion. Consolidation present."
    result = find_mentioned_pathologies(text)
    assert "Effusion" not in result
    assert "Consolidation" in result


def test_negation_far_away_ignored():
    # negation cue too far to apply
    text = "No prior history of trauma or recent injuries or other conditions. Consolidation present."
    result = find_mentioned_pathologies(text)
    assert "Consolidation" in result


# --- check_report end-to-end ---

def test_clean_report_no_hallucination():
    findings = _findings("Consolidation")
    report = "FINDINGS: Consolidation in the right lower lobe. IMPRESSION: Pneumonic process."
    # Pneumonia is mentioned but not detected — this IS a hallucination
    result = check_report(report, findings)
    assert not result.is_clean
    assert "Pneumonia" in result.hallucinated


def test_report_with_only_grounded_mentions():
    findings = _findings("Consolidation", "Effusion")
    report = "Consolidation in the right base with small effusion. No other findings."
    result = check_report(report, findings)
    assert result.is_clean
    assert set(result.grounded) == {"Consolidation", "Effusion"}
    assert result.hallucinated == []


def test_report_with_hallucination():
    findings = _findings("Consolidation")
    report = "Consolidation noted. Also concerning for pneumothorax."
    result = check_report(report, findings)
    assert not result.is_clean
    assert "Pneumothorax" in result.hallucinated
    assert "Consolidation" in result.grounded


def test_report_misses_detected_finding():
    findings = _findings("Consolidation", "Cardiomegaly")
    report = "Consolidation in the right lower lobe."
    result = check_report(report, findings)
    assert result.is_clean  # no hallucination
    assert "Cardiomegaly" in result.missed


def test_uncertain_findings_allowed_to_mention():
    findings = _findings("Consolidation", uncertain=["Effusion"])
    report = "Consolidation seen. Small effusion cannot be excluded."
    result = check_report(report, findings)
    assert result.is_clean  # Effusion is uncertain, OK to mention


def test_negated_pathologies_not_flagged():
    findings = _findings("Consolidation")
    report = "Consolidation in the right lower lobe. No effusion, no pneumothorax."
    result = check_report(report, findings)
    assert result.is_clean


def test_hallucination_rate_computation():
    findings = _findings("Consolidation")
    # 3 pathologies mentioned, 1 grounded, 2 hallucinated
    report = "Consolidation, pneumonia, and pneumothorax all seen."
    result = check_report(report, findings)
    assert result.hallucination_rate == 2 / 3


def test_empty_report():
    findings = _findings("Consolidation")
    result = check_report("", findings)
    assert result.is_clean
    assert result.hallucination_rate == 0.0
    assert "Consolidation" in result.missed


def test_empty_findings_and_mention_is_hallucination():
    findings = _findings()  # nothing detected
    report = "Consolidation noted."
    result = check_report(report, findings)
    assert not result.is_clean
    assert "Consolidation" in result.hallucinated


def test_empty_findings_normal_report_is_clean():
    findings = _findings()
    report = "No acute cardiopulmonary findings."
    result = check_report(report, findings)
    assert result.is_clean
    assert result.hallucination_rate == 0.0


# --- serialization ---

def test_to_dict_roundtrip():
    findings = _findings("Consolidation")
    report = "Consolidation and pneumothorax noted."
    result = check_report(report, findings)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert "hallucinated" in d
    assert "grounded" in d
    assert "is_clean" in d
    assert isinstance(d["hallucination_rate"], float)


def test_lists_sorted_deterministically():
    findings = _findings("Consolidation")
    report = "Pneumothorax and pneumonia and consolidation noted."
    result = check_report(report, findings)
    # deterministic order for reproducibility
    assert result.hallucinated == sorted(result.hallucinated)
