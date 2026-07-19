"""Tests for evaluation.hallucination_rate.

All lightweight, no heavy deps.
Tests cover clean reports, dirty reports,per-pathology counting, and edge cases

"""

from evaluation.hallucination_rate import run_hallucination_analysis


def _findings(detected, uncertain=None):
    """Helper — build a findings dict like StudyFindings.to_dict()."""
    return {
        "findings": [{"pathology": p} for p in detected],
        "uncertain": uncertain or [],
    }


class TestBatchHallucination:

    def test_all_clean(self):
        reports = [
            "Consolidation in the right lung.",
            "No effusion identified.",
        ]
        findings = [
            _findings(["Consolidation"]),
            _findings([]),  # no findings, and report says "no effusion" — clean
        ]
        result = run_hallucination_analysis(reports, findings)
        assert result.n_clean == 2
        assert result.n_dirty == 0
        assert result.clean_rate == 1.0

    def test_one_dirty(self):
        reports = [
            "Consolidation and pneumonia present.",  # pneumonia not detected
            "No effusion.",
        ]
        findings = [
            _findings(["Consolidation"]),  # pneumonia is hallucinated
            _findings([]),
        ]
        result = run_hallucination_analysis(reports, findings)
        assert result.n_dirty == 1
        assert result.dirty_indices == [0]

    def test_hallucinated_counts(self):
        reports = [
            "Pneumonia present.",
            "Pneumonia and effusion noted.",
        ]
        findings = [
            _findings([]),  # pneumonia hallucinated
            _findings([]),  # both hallucinated
        ]
        result = run_hallucination_analysis(reports, findings)
        assert result.hallucinated_counts["Pneumonia"] == 2
        assert result.hallucinated_counts["Effusion"] == 1

    def test_missed_counts(self):
        reports = [
            "Normal chest radiograph.",  # doesn't mention the detected finding
        ]
        findings = [
            _findings(["Consolidation"]),
        ]
        result = run_hallucination_analysis(reports, findings)
        assert result.missed_counts["Consolidation"] == 1

    def test_empty_input(self):
        result = run_hallucination_analysis([], [])
        assert result.n_reports == 0
        assert result.clean_rate == 0.0

    def test_mismatched_lengths_raises(self):
        try:
            run_hallucination_analysis(["a"], [_findings([]), _findings([])])
            assert False, "should have raised"
        except AssertionError:
            pass

    def test_mean_and_median_rates(self):
        reports = [
            "Consolidation and pneumonia.",  # 1 hallucinated out of 2 mentions
            "No effusion.",                  # clean
        ]
        findings = [
            _findings(["Consolidation"]),
            _findings([]),
        ]
        result = run_hallucination_analysis(reports, findings)
        assert result.mean_hallucination_rate > 0.0
        assert result.median_hallucination_rate >= 0.0

    def test_to_dict_keys(self):
        result = run_hallucination_analysis(
            ["Consolidation."], [_findings(["Consolidation"])]
        )
        d = result.to_dict()
        expected = {
            "n_reports", "n_clean", "n_dirty", "clean_rate",
            "mean_hallucination_rate", "median_hallucination_rate",
            "hallucinated_counts", "missed_counts", "dirty_indices",
        }
        assert set(d.keys()) == expected

    def test_uncertain_not_counted_as_hallucination(self):
        reports = ["Possible pneumonia noted."]
        findings = [_findings([], uncertain=["Pneumonia"])]
        result = run_hallucination_analysis(reports, findings)
        # hallucination_checker allows uncertain mentions
        assert result.n_clean == 1
