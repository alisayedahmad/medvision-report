"""Batch hallucination analysis across a full dataset.

Runs hallucination_checker.check_report on every (report, findings) pair
and aggregates the results. Answers questions like: what fraction of
reports contain at least one hallucination? Which pathologies get
invented most often? How does hallucination rate vary with confidence?


"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from report_generation.hallucination_checker import HallucinationReport, check_report


@dataclass
class BatchHallucinationResult:
    """Aggregated hallucination stats across N reports
    """

    n_reports: int
    n_clean: int  # reports with zero hallucinations
    n_dirty: int
    clean_rate: float
    mean_hallucination_rate: float
    median_hallucination_rate: float
    # which pathologies get hallucinated and how often
    hallucinated_counts: dict[str, int]
    missed_counts: dict[str, int]
    # indices of the worst offenders for debugging
    dirty_indices: list[int]
    per_report: list[HallucinationReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reports": self.n_reports,
            "n_clean": self.n_clean,
            "n_dirty": self.n_dirty,
            "clean_rate": round(self.clean_rate, 4),
            "mean_hallucination_rate": round(self.mean_hallucination_rate, 4),
            "median_hallucination_rate": round(self.median_hallucination_rate, 4),
            "hallucinated_counts": dict(self.hallucinated_counts),
            "missed_counts": dict(self.missed_counts),
            "dirty_indices": self.dirty_indices,
        }


def run_hallucination_analysis(
    reports: list[str],
    findings: list[dict[str, Any]],
) -> BatchHallucinationResult:
    """Check every report against its findings and aggregate.

    Args:
        reports: generated report texts
        findings: list of StudyFindings.to_dict() dicts, one per report
    """
    assert len(reports) == len(findings), (
        f"got {len(reports)} reports vs {len(findings)} findings"
    )

    n = len(reports)
    if n == 0:
        return BatchHallucinationResult(
            n_reports=0, n_clean=0, n_dirty=0, clean_rate=0.0,
            mean_hallucination_rate=0.0, median_hallucination_rate=0.0,
            hallucinated_counts={}, missed_counts={},
            dirty_indices=[], per_report=[],
        )

    per_report: list[HallucinationReport] = []
    hallucinated_counter: Counter[str] = Counter()
    missed_counter: Counter[str] = Counter()
    dirty_indices: list[int] = []

    for i in range(n):
        result = check_report(reports[i], findings[i])
        per_report.append(result)

        hallucinated_counter.update(result.hallucinated)
        missed_counter.update(result.missed)

        if not result.is_clean:
            dirty_indices.append(i)

    rates = [r.hallucination_rate for r in per_report]
    n_clean = sum(1 for r in per_report if r.is_clean)

    return BatchHallucinationResult(
        n_reports=n,
        n_clean=n_clean,
        n_dirty=n - n_clean,
        clean_rate=n_clean / n,
        mean_hallucination_rate=statistics.mean(rates),
        median_hallucination_rate=statistics.median(rates),
        hallucinated_counts=dict(hallucinated_counter.most_common()),
        missed_counts=dict(missed_counter.most_common()),
        dirty_indices=dirty_indices,
        per_report=per_report,
    )
