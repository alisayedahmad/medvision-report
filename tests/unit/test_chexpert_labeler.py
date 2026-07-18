"""Tests for evaluation.chexpert_labeler.

All CPU, no heavy deps — runs in CI.
Tests cover label extraction (positive/negative/uncertain/blank),
F1 computation, and edge cases.
"""

from evaluation.chexpert_labeler import (
    Label,
    compute_chexpert_f1,
    extract_labels,
    labels_to_vector,
)


# -------------------------------------------------------------------
# Label extraction from free text
# -------------------------------------------------------------------

class TestExtractLabels:

    def test_positive_mention(self):
        labels = extract_labels("There is consolidation in the right lower lobe.")
        assert labels["Consolidation"] == Label.POSITIVE

    def test_negated_mention(self):
        labels = extract_labels("No consolidation identified.")
        assert labels["Consolidation"] == Label.NEGATIVE

    def test_uncertain_mention(self):
        labels = extract_labels("Possible consolidation in the right lower lobe.")
        assert labels["Consolidation"] == Label.UNCERTAIN

    def test_not_mentioned_is_blank(self):
        labels = extract_labels("Heart size is normal.")
        assert labels["Consolidation"] == Label.BLANK

    def test_multiple_pathologies(self):
        text = (
            "There is consolidation in the right lung. "
            "No pleural effusion. "
            "Possible pneumothorax."
        )
        labels = extract_labels(text)
        assert labels["Consolidation"] == Label.POSITIVE
        assert labels["Effusion"] == Label.NEGATIVE
        assert labels["Pneumothorax"] == Label.UNCERTAIN

    def test_synonym_maps_to_canonical(self):
        labels = extract_labels("Enlarged heart noted on imaging.")
        assert labels["Cardiomegaly"] == Label.POSITIVE

    def test_negation_does_not_cross_sentence(self):
        text = "No effusion. Consolidation is present."
        labels = extract_labels(text)
        assert labels["Effusion"] == Label.NEGATIVE
        assert labels["Consolidation"] == Label.POSITIVE

    def test_empty_report_all_blank(self):
        labels = extract_labels("")
        for p in labels:
            assert labels[p] == Label.BLANK

    def test_negation_beats_uncertainty(self):
        # "no possible" — negation takes priority
        labels = extract_labels("No possible consolidation seen.")
        assert labels["Consolidation"] == Label.NEGATIVE

    def test_all_14_pathologies_present(self):
        labels = extract_labels("Normal chest radiograph.")
        assert len(labels) == 14


class TestLabelsToVector:

    def test_vector_length(self):
        labels = extract_labels("Normal study.")
        vec = labels_to_vector(labels)
        assert len(vec) == 14

    def test_positive_is_1(self):
        labels = extract_labels("Consolidation present.")
        vec = labels_to_vector(labels)
        # Consolidation is index 2 in PATHOLOGIES
        assert vec[2] == 1

    def test_blank_is_none(self):
        labels = extract_labels("Normal study.")
        vec = labels_to_vector(labels)
        assert all(v is None for v in vec)


# -------------------------------------------------------------------
# F1 computation
# -------------------------------------------------------------------

class TestCheXpertF1:

    def test_perfect_match(self):
        reports = ["Consolidation in the right lung. No effusion."]
        result = compute_chexpert_f1(reports, reports)
        # accuracy is 1.0 — every label matches
        assert result.accuracy == 1.0
        # consolidation specifically: both positive -> F1=1.0
        con = next(c for c in result.per_class if c.pathology == "Consolidation")
        assert con.f1 == 1.0

    def test_complete_mismatch(self):
        gen = ["Consolidation present."]
        ref = ["No consolidation. Effusion noted."]
        result = compute_chexpert_f1(gen, ref)
        # consolidation: gen=positive, ref=negative -> FP
        # effusion: gen=blank, ref=positive -> FN
        con = next(c for c in result.per_class if c.pathology == "Consolidation")
        assert con.f1 == 0.0
        eff = next(c for c in result.per_class if c.pathology == "Effusion")
        assert eff.f1 == 0.0

    def test_partial_match(self):
        gen = ["Consolidation present. No effusion."]
        ref = ["Consolidation in the right lung. No effusion."]
        result = compute_chexpert_f1(gen, ref)
        con = next(c for c in result.per_class if c.pathology == "Consolidation")
        assert con.f1 == 1.0  # both positive

    def test_support_counts_reference_positives(self):
        gen = ["Consolidation. Effusion."]
        ref = ["Consolidation present."]
        result = compute_chexpert_f1(gen, ref)
        con = next(c for c in result.per_class if c.pathology == "Consolidation")
        assert con.support == 1
        eff = next(c for c in result.per_class if c.pathology == "Effusion")
        assert eff.support == 0

    def test_multiple_reports(self):
        gen = [
            "Consolidation present.",
            "No consolidation.",
        ]
        ref = [
            "Consolidation present.",
            "Consolidation present.",
        ]
        result = compute_chexpert_f1(gen, ref)
        con = next(c for c in result.per_class if c.pathology == "Consolidation")
        # report 1: TP, report 2: FN -> recall=0.5, precision=1.0
        assert con.recall == 0.5
        assert con.precision == 1.0

    def test_empty_lists(self):
        result = compute_chexpert_f1([], [])
        assert result.macro_f1 == 0.0
        assert result.per_class == []

    def test_mismatched_lengths_raises(self):
        try:
            compute_chexpert_f1(["a"], ["b", "c"])
            assert False, "should have raised"
        except AssertionError:
            pass

    def test_to_dict_shape(self):
        result = compute_chexpert_f1(
            ["Consolidation present."],
            ["Consolidation present."],
        )
        d = result.to_dict()
        assert "per_class" in d
        assert "macro_f1" in d
        assert "accuracy" in d
        assert len(d["per_class"]) == 14
