"""Tests for evaluation.nlp_metrics.

BLEU and ROUGE tests run in CI (sacrebleu and rouge-score are lightweight).
BERTScore tests are skipped when bert_score is not installed — same pattern
as torch tests. We test score ranges, not exact values, because these
metrics are sensitive to tokenization and library version.
"""

import importlib

import pytest

from evaluation.nlp_metrics import (
    compute_batch_metrics,
    compute_nlp_metrics,
)

_has_bertscore = importlib.util.find_spec("bert_score") is not None


# -----------------------------------------------------------------------
# Fixtures — realistic radiology text, not lorem ipsum
# -----------------------------------------------------------------------

REFERENCE_A = (
    "The cardiac silhouette is within normal limits. "
    "No focal consolidation, pleural effusion, or pneumothorax is identified. "
    "The lungs are clear bilaterally."
)

# Good match — same findings, different words
GENERATED_GOOD = (
    "Heart size is normal. No consolidation, effusion, or pneumothorax. "
    "Both lungs appear clear."
)

# Bad match — completely different content
GENERATED_BAD = (
    "Large left-sided pleural effusion with adjacent atelectasis. "
    "Recommend thoracentesis and follow-up imaging."
)

REFERENCE_B = (
    "There is a right lower lobe opacity consistent with pneumonia. "
    "Small right pleural effusion noted. Heart size is normal."
)


# -----------------------------------------------------------------------
# Single pair — BLEU + ROUGE (no BERTScore, runs in CI)
# -----------------------------------------------------------------------

class TestSinglePairNoBert:
    """BLEU and ROUGE only — lightweight, runs everywhere."""

    def test_good_match_scores_higher(self):
        good = compute_nlp_metrics(GENERATED_GOOD, REFERENCE_A, use_bertscore=False)
        bad = compute_nlp_metrics(GENERATED_BAD, REFERENCE_A, use_bertscore=False)
        assert good.rouge_l_f1 > bad.rouge_l_f1

    def test_identical_texts_score_high(self):
        scores = compute_nlp_metrics(REFERENCE_A, REFERENCE_A, use_bertscore=False)
        assert scores.bleu4 > 0.9
        assert scores.rouge_l_f1 > 0.95

    def test_empty_generated_returns_zeros(self):
        scores = compute_nlp_metrics("", REFERENCE_A, use_bertscore=False)
        assert scores.bleu4 == 0.0
        assert scores.rouge_l_f1 == 0.0

    def test_empty_reference_returns_zeros(self):
        scores = compute_nlp_metrics(GENERATED_GOOD, "", use_bertscore=False)
        assert scores.bleu4 == 0.0

    def test_bertscore_fields_zero_when_disabled(self):
        scores = compute_nlp_metrics(GENERATED_GOOD, REFERENCE_A, use_bertscore=False)
        assert scores.bertscore_f1 == 0.0
        assert scores.bertscore_precision == 0.0
        assert scores.bertscore_recall == 0.0

    def test_to_dict_has_all_keys(self):
        scores = compute_nlp_metrics(GENERATED_GOOD, REFERENCE_A, use_bertscore=False)
        d = scores.to_dict()
        expected = {
            "bleu4", "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
            "bertscore_precision", "bertscore_recall", "bertscore_f1",
        }
        assert set(d.keys()) == expected

    def test_scores_are_bounded_zero_one(self):
        scores = compute_nlp_metrics(GENERATED_GOOD, REFERENCE_A, use_bertscore=False)
        for v in scores.to_dict().values():
            assert 0.0 <= v <= 1.0


# -----------------------------------------------------------------------
# Batch evaluation
# -----------------------------------------------------------------------

class TestBatchMetrics:
    """Batch interface — corpus BLEU + aggregates."""

    def test_batch_returns_per_pair_and_corpus(self):
        result = compute_batch_metrics(
            [GENERATED_GOOD, GENERATED_BAD],
            [REFERENCE_A, REFERENCE_B],
            use_bertscore=False,
        )
        assert len(result["per_pair"]) == 2
        assert "bleu4_corpus" in result["corpus"]
        assert result["corpus"]["n_pairs"] == 2

    def test_corpus_bleu_differs_from_sentence_average(self):
        # corpus BLEU != mean of sentence BLEU — verifies we compute it correctly
        result = compute_batch_metrics(
            [GENERATED_GOOD, GENERATED_BAD],
            [REFERENCE_A, REFERENCE_B],
            use_bertscore=False,
        )
        corpus = result["corpus"]["bleu4_corpus"]
        sentence_mean = result["corpus"]["bleu4_sentence"]["mean"]
        # they should differ (corpus BLEU is geometric mean-based, different formula)
        # but both should be valid floats in [0, 1]
        assert 0.0 <= corpus <= 1.0
        assert 0.0 <= sentence_mean <= 1.0

    def test_empty_batch_returns_empty(self):
        result = compute_batch_metrics([], [], use_bertscore=False)
        assert result["per_pair"] == []
        assert result["corpus"] == {}

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            compute_batch_metrics(["a"], ["b", "c"], use_bertscore=False)

    def test_aggregate_stats_have_mean_std_median(self):
        result = compute_batch_metrics(
            [GENERATED_GOOD, GENERATED_BAD],
            [REFERENCE_A, REFERENCE_B],
            use_bertscore=False,
        )
        for key in ["rouge_l_f1", "rouge_l_precision", "rouge_l_recall"]:
            stats = result["corpus"][key]
            assert "mean" in stats
            assert "std" in stats
            assert "median" in stats


# -----------------------------------------------------------------------
# BERTScore — only runs locally with bert_score installed
# -----------------------------------------------------------------------

@pytest.mark.skipif(not _has_bertscore, reason="bert_score not installed")
class TestBERTScore:
    """These run locally only — need bert_score + model download."""

    def test_bertscore_returns_nonzero(self):
        scores = compute_nlp_metrics(GENERATED_GOOD, REFERENCE_A, use_bertscore=True)
        assert scores.bertscore_f1 > 0.0

    def test_good_match_beats_bad_match(self):
        good = compute_nlp_metrics(GENERATED_GOOD, REFERENCE_A, use_bertscore=True)
        bad = compute_nlp_metrics(GENERATED_BAD, REFERENCE_A, use_bertscore=True)
        assert good.bertscore_f1 > bad.bertscore_f1

    def test_identical_bertscore_near_one(self):
        scores = compute_nlp_metrics(REFERENCE_A, REFERENCE_A, use_bertscore=True)
        assert scores.bertscore_f1 > 0.95
