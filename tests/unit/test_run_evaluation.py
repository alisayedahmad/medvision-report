"""Tests for evaluation.run_evaluation.

Tests the orchestrator — loading pairs, running all metrics, saving results.
Uses the sample_pairs.json fixture. BERTScore disabled for speed.
"""

import json
from pathlib import Path

import pytest

from evaluation.run_evaluation import load_pairs, run_full_evaluation, main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_PAIRS = FIXTURES / "sample_pairs.json"


class TestLoadPairs:

    def test_loads_valid_json(self):
        pairs = load_pairs(SAMPLE_PAIRS)
        assert len(pairs) == 3
        assert "generated" in pairs[0]
        assert "reference" in pairs[0]

    def test_rejects_missing_keys(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('[{"generated": "hello"}]')
        with pytest.raises(ValueError, match="missing"):
            load_pairs(bad)

    def test_rejects_non_list(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"generated": "hello", "reference": "hi"}')
        with pytest.raises(ValueError, match="list"):
            load_pairs(bad)


class TestRunFullEvaluation:

    def test_returns_all_sections(self):
        pairs = load_pairs(SAMPLE_PAIRS)
        results = run_full_evaluation(pairs, use_bertscore=False)
        assert "metadata" in results
        assert "nlp_metrics" in results
        assert "chexpert" in results
        assert "hallucination" in results

    def test_metadata_counts(self):
        pairs = load_pairs(SAMPLE_PAIRS)
        results = run_full_evaluation(pairs, use_bertscore=False)
        assert results["metadata"]["n_pairs"] == 3
        assert results["metadata"]["bertscore_enabled"] is False

    def test_nlp_metrics_present(self):
        pairs = load_pairs(SAMPLE_PAIRS)
        results = run_full_evaluation(pairs, use_bertscore=False)
        assert "corpus" in results["nlp_metrics"]
        assert len(results["nlp_metrics"]["per_pair"]) == 3

    def test_chexpert_has_14_classes(self):
        pairs = load_pairs(SAMPLE_PAIRS)
        results = run_full_evaluation(pairs, use_bertscore=False)
        assert len(results["chexpert"]["per_class"]) == 14

    def test_hallucination_included_when_findings_present(self):
        pairs = load_pairs(SAMPLE_PAIRS)
        results = run_full_evaluation(pairs, use_bertscore=False)
        assert results["hallucination"] is not None
        assert "n_reports" in results["hallucination"]

    def test_hallucination_skipped_without_findings(self):
        pairs = [
            {"generated": "Consolidation.", "reference": "Consolidation."},
        ]
        results = run_full_evaluation(pairs, use_bertscore=False)
        assert results["hallucination"] is None


class TestCLI:

    def test_main_writes_output(self, tmp_path):
        output = tmp_path / "results.json"
        main(["--input", str(SAMPLE_PAIRS), "--output", str(output), "--no-bertscore"])
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["metadata"]["n_pairs"] == 3
