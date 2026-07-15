"""Grounded generation verification.

Every clinical claim in the generated report must trace back to a
detected finding. If the LLM mentions a pathology the vision model
never flagged, that's a hallucination — and in a medical context,
a critical safety failure.

This module scans the report text for pathology mentions and
cross-references them against the StudyFindings the report was
generated from. Simple, deterministic, no LLM calls needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Common radiology synonyms and lay terms mapped to canonical pathology names.
# Not exhaustive — covers the most frequent variants seen in real reports.
# When we hit false positives in evaluation, we grow this map.
_SYNONYMS: dict[str, str] = {
    # Atelectasis
    "atelectasis": "Atelectasis",
    "atelectatic": "Atelectasis",
    "collapse": "Atelectasis",
    "collapsed": "Atelectasis",
    # Cardiomegaly
    "cardiomegaly": "Cardiomegaly",
    "enlarged heart": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",
    # Consolidation
    "consolidation": "Consolidation",
    "consolidated": "Consolidation",
    "airspace disease": "Consolidation",
    # Edema
    "edema": "Edema",
    "oedema": "Edema",
    "pulmonary edema": "Edema",
    # Effusion
    "effusion": "Effusion",
    "pleural fluid": "Effusion",
    # Emphysema
    "emphysema": "Emphysema",
    "emphysematous": "Emphysema",
    "hyperinflation": "Emphysema",
    # Fibrosis
    "fibrosis": "Fibrosis",
    "fibrotic": "Fibrosis",
    "scarring": "Fibrosis",
    # Hernia
    "hernia": "Hernia",
    "hiatal hernia": "Hernia",
    # Infiltration
    "infiltrate": "Infiltration",
    "infiltration": "Infiltration",
    "infiltrates": "Infiltration",
    # Mass
    "mass": "Mass",
    "masses": "Mass",
    "tumor": "Mass",
    "tumour": "Mass",
    # Nodule
    "nodule": "Nodule",
    "nodules": "Nodule",
    "nodular": "Nodule",
    # Pleural_Thickening
    "pleural thickening": "Pleural_Thickening",
    "thickened pleura": "Pleural_Thickening",
    # Pneumonia
    "pneumonia": "Pneumonia",
    "pneumonic": "Pneumonia",
    # Pneumothorax
    "pneumothorax": "Pneumothorax",
    "collapsed lung": "Pneumothorax",
}

# Negation cues — if a pathology mention is preceded by one of these
# within a small window, it's a negation ("no consolidation") and
# doesn't count as a claim of presence.
_NEGATION_CUES = {
    "no", "not", "without", "absent", "denies", "denied",
    "negative", "free of", "clear of", "ruled out", "unremarkable",
    "excludes", "excluded", "resolution of", "resolved",
}

# How many words before the pathology to scan for negation
_NEGATION_WINDOW = 4


@dataclass
class HallucinationReport:
    """Result of checking one generated report against its findings."""

    hallucinated: list[str] = field(default_factory=list)  # mentioned but not detected
    grounded: list[str] = field(default_factory=list)      # mentioned and detected
    missed: list[str] = field(default_factory=list)         # detected but not mentioned
    is_clean: bool = True
    hallucination_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hallucinated": self.hallucinated,
            "grounded": self.grounded,
            "missed": self.missed,
            "is_clean": self.is_clean,
            "hallucination_rate": round(self.hallucination_rate, 4),
        }


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for consistent matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_negated(text: str, match_start: int) -> bool:
    """Check if a pathology mention is preceded by a negation cue.

    Looks back up to _NEGATION_WINDOW words. A period between the cue
    and the mention breaks the negation — different sentence.
    """
    before = text[:match_start]

    # Cut at the last sentence boundary — negation doesn't cross sentences
    last_period = max(before.rfind("."), before.rfind(";"), before.rfind(":"))
    if last_period >= 0:
        before = before[last_period + 1:]

    # Take the last N words
    words = before.split()
    window = words[-_NEGATION_WINDOW:] if len(words) > _NEGATION_WINDOW else words
    window_text = " ".join(window)

    return any(cue in window_text for cue in _NEGATION_CUES)


def find_mentioned_pathologies(report_text: str) -> set[str]:
    """Scan report text for pathology mentions, ignoring negated ones.

    Returns the set of canonical pathology names asserted as present.
    """
    normalized = _normalize(report_text)
    mentioned: set[str] = set()

    # Sort synonyms longest first so "pleural thickening" matches before "pleural"
    sorted_synonyms = sorted(_SYNONYMS.keys(), key=len, reverse=True)

    for synonym in sorted_synonyms:
        # word boundaries to avoid substring false matches
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        for match in pattern.finditer(normalized):
            if not _is_negated(normalized, match.start()):
                mentioned.add(_SYNONYMS[synonym])
                break  # one confirmed non-negated mention is enough

    return mentioned


def check_report(
    report_text: str,
    findings_dict: dict[str, Any],
) -> HallucinationReport:
    """Verify generated report against structured findings.

    Args:
        report_text: the free-text report the LLM produced
        findings_dict: StudyFindings.to_dict() the report was generated from

    Returns:
        HallucinationReport with hallucinated / grounded / missed pathologies.
    """
    mentioned = find_mentioned_pathologies(report_text)
    detected = {f["pathology"] for f in findings_dict.get("findings", [])}

    # Uncertain pathologies are OK to mention (as equivocal) —
    # the v3 prompt explicitly asks the LLM to acknowledge them.
    uncertain = set(findings_dict.get("uncertain", []))
    allowed_to_mention = detected | uncertain

    hallucinated = sorted(mentioned - allowed_to_mention)
    grounded = sorted(mentioned & detected)
    missed = sorted(detected - mentioned)

    # Rate = fraction of mentions that were unfounded
    rate = len(hallucinated) / len(mentioned) if mentioned else 0.0

    return HallucinationReport(
        hallucinated=hallucinated,
        grounded=grounded,
        missed=missed,
        is_clean=len(hallucinated) == 0,
        hallucination_rate=rate,
    )
