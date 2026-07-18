"""CheXpert-style clinical label extraction and F1 evaluation.

BLEU and ROUGE measure text similarity. This measures clinical correctness
It extracts what the report *says* about each pathology — positive, negative,
uncertain, or not mentioned — and compares against a reference


Reuses the synonym map and negation logic from hallucination_checker
Adds uncertainty detection on top. No Java, no external binary, runs in CI


"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from report_generation.feature_extractor import PATHOLOGIES
from report_generation.hallucination_checker import (
    _NEGATION_CUES,
    _NEGATION_WINDOW,
    _SYNONYMS,
    _normalize,
)


class Label(Enum):
    """Four-way label, same as CheXpert convention."""
    POSITIVE = 1
    NEGATIVE = -1
    UNCERTAIN = 0
    BLANK = None  # not mentioned at all


# words that make a finding uncertain rather than positive
_UNCERTAINTY_CUES = {
    "possible", "possibly", "probable", "probably", "likely",
    "suspicious", "suspected", "suggest", "suggests", "suggesting",
    "cannot exclude", "cannot rule out", "may represent",
    "questionable", "equivocal", "borderline", "potential",
    "could represent", "consider", "differential",
}

_UNCERTAINTY_WINDOW = 5


def _is_negated(text: str, match_start: int) -> bool:
    """Same logic as hallucination_checker — negation cue within N words."""
    before = text[:match_start]

    last_boundary = max(before.rfind("."), before.rfind(";"), before.rfind(":"))
    if last_boundary >= 0:
        before = before[last_boundary + 1:]

    words = before.split()
    window = words[-_NEGATION_WINDOW:] if len(words) > _NEGATION_WINDOW else words
    window_text = " ".join(window)

    return any(cue in window_text for cue in _NEGATION_CUES)


def _is_uncertain(text: str, match_start: int) -> bool:
    """Check if a pathology mention is hedged with uncertainty language."""
    before = text[:match_start]

    last_boundary = max(before.rfind("."), before.rfind(";"), before.rfind(":"))
    if last_boundary >= 0:
        before = before[last_boundary + 1:]

    words = before.split()
    window = words[-_UNCERTAINTY_WINDOW:] if len(words) > _UNCERTAINTY_WINDOW else words
    window_text = " ".join(window)

    return any(cue in window_text for cue in _UNCERTAINTY_CUES)


def extract_labels(report_text: str) -> dict[str, Label]:

    """Extract a label for each of the 14 pathologies from free text

    Priority: negation beats uncertainty beats positive.
    If a pathology isn't mentioned at all, it's BLANK.
    """
    normalized = _normalize(report_text)
    labels = {p: Label.BLANK for p in PATHOLOGIES}

    sorted_synonyms = sorted(_SYNONYMS.keys(), key=len, reverse=True)

    for synonym in sorted_synonyms:
        pattern = re.compile(r"\b" + re.escape(synonym) + r"\b")
        canonical = _SYNONYMS[synonym]

        # already labeled by a longer synonym match — skip
        if labels[canonical] != Label.BLANK:
            continue

        for match in pattern.finditer(normalized):
            if _is_negated(normalized, match.start()):
                labels[canonical] = Label.NEGATIVE
            elif _is_uncertain(normalized, match.start()):
                labels[canonical] = Label.UNCERTAIN
            else:
                labels[canonical] = Label.POSITIVE
            break  # first match decides

    return labels


def labels_to_vector(labels: dict[str, Label]) -> list[int | None]:
    """Convert label dict to ordered vector matching PATHOLOGIES order."""
    return [labels[p].value for p in PATHOLOGIES]


# -------------------------------------------------------------------
# F1 computation
# -------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """Precision / recall / F1 for one pathology class."""
    pathology: str
    precision: float
    recall: float
    f1: float
    support: int  # how many references had this label positive

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathology": self.pathology,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
        }


@dataclass
class CheXpertScores:
    """Full evaluation result across all pathologies."""
    per_class: list[ClassMetrics]
    macro_f1: float
    macro_precision: float
    macro_recall: float
    accuracy: float  # exact label match rate across all (report, pathology) pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_class": [c.to_dict() for c in self.per_class],
            "macro_f1": round(self.macro_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "accuracy": round(self.accuracy, 4),
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def compute_chexpert_f1(
    generated_reports: list[str],
    reference_reports: list[str],
    positive_label: Label = Label.POSITIVE,
) -> CheXpertScores:
    """Compare extracted labels between generated and reference reports.

    Treats positive_label as the "positive class" for precision/recall.
    Default is POSITIVE — did we correctly identify present pathologies?
    """
    assert len(generated_reports) == len(reference_reports), (
        f"got {len(generated_reports)} generated vs {len(reference_reports)} references"
    )

    n = len(generated_reports)
    if n == 0:
        return CheXpertScores(
            per_class=[], macro_f1=0.0, macro_precision=0.0,
            macro_recall=0.0, accuracy=0.0,
        )

    # extract labels for every report
    gen_labels = [extract_labels(r) for r in generated_reports]
    ref_labels = [extract_labels(r) for r in reference_reports]

    # per-class binary F1 (positive_label vs everything else)
    per_class = []
    total_correct = 0
    total_cells = 0

    for pathology in PATHOLOGIES:
        tp = fp = fn = support = 0

        for i in range(n):
            g = gen_labels[i][pathology]
            r = ref_labels[i][pathology]

            if g == r:
                total_correct += 1
            total_cells += 1

            is_gen_pos = g == positive_label
            is_ref_pos = r == positive_label

            if is_ref_pos:
                support += 1

            if is_gen_pos and is_ref_pos:
                tp += 1
            elif is_gen_pos and not is_ref_pos:
                fp += 1
            elif not is_gen_pos and is_ref_pos:
                fn += 1

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class.append(ClassMetrics(
            pathology=pathology,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        ))

    # macro averages — unweighted, each class counts equally
    macro_p = _safe_div(sum(c.precision for c in per_class), len(per_class))
    macro_r = _safe_div(sum(c.recall for c in per_class), len(per_class))
    macro_f1 = _safe_div(sum(c.f1 for c in per_class), len(per_class))
    accuracy = _safe_div(total_correct, total_cells)

    return CheXpertScores(
        per_class=per_class,
        macro_f1=macro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        accuracy=accuracy,
    )
