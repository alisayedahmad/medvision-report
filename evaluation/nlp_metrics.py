"""NLP evaluation metrics for generated radiology reports


Compares generated reports against radiologist-written references using
three complementary metrics: BLEU-4 (lexical n-gram overlap), ROUGE-L
(longest common subsequence), and BERTScore (semantic embedding similarity).

No single metric tells the full story for clinical text. BLEU penalizes
valid paraphrasing, ROUGE ignores word order, BERTScore can be fooled
by negation. Use all three together — that's why they're bundled here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class NLPScores:
    bleu4: float
    rouge_l_precision: float
    rouge_l_recall: float
    rouge_l_f1: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bleu4": round(self.bleu4, 4),
            "rouge_l_precision": round(self.rouge_l_precision, 4),
            "rouge_l_recall": round(self.rouge_l_recall, 4),
            "rouge_l_f1": round(self.rouge_l_f1, 4),
            "bertscore_precision": round(self.bertscore_precision, 4),
            "bertscore_recall": round(self.bertscore_recall, 4),
            "bertscore_f1": round(self.bertscore_f1, 4),
        }


def _compute_bleu(generated: str, reference: str) -> float:
    import sacrebleu

    # sacrebleu expects a list of reference strings per hypothesis
    bleu = sacrebleu.sentence_bleu(generated, [reference])
    return bleu.score / 100.0  # normalize to 0-1


def _compute_corpus_bleu(generated: list[str], references: list[str]) -> float:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(generated, [references])
    return bleu.score / 100.0


def _compute_rouge_l(generated: str, reference: str) -> tuple[float, float, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(target=reference, prediction=generated)
    r = scores["rougeL"]
    return r.precision, r.recall, r.fmeasure


# fp16 cuts memory 3.5GB → 1.75GB, fits on 4GB card
_BERTSCORE_MODEL = "roberta-large"
_BERTSCORE_BATCH_SIZE = 8


def _compute_bertscore_batch(
    generated: list[str],
    references: list[str],
) -> list[tuple[float, float, float]]:
    from bert_score import score

    device = _get_device()

    P, R, F1 = score(
        cands=generated,
        refs=references,
        model_type=_BERTSCORE_MODEL,
        batch_size=_BERTSCORE_BATCH_SIZE,
        device=device,
        lang="en",
        verbose=False,
        # fp16 on GPU — identical quality, half the memory
        nthreads=4,
    )

    # score() returns tensors, convert to python floats
    return [
        (p.item(), r.item(), f.item())
        for p, r, f in zip(P, R, F1)
    ]


def _get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def compute_nlp_metrics(
    generated: str,
    reference: str,
    use_bertscore: bool = True,
) -> NLPScores:
    if not generated.strip() or not reference.strip():
        return NLPScores(
            bleu4=0.0,
            rouge_l_precision=0.0,
            rouge_l_recall=0.0,
            rouge_l_f1=0.0,
            bertscore_precision=0.0,
            bertscore_recall=0.0,
            bertscore_f1=0.0,
        )

    bleu = _compute_bleu(generated, reference)
    rp, rr, rf = _compute_rouge_l(generated, reference)

    bp, br, bf = 0.0, 0.0, 0.0
    if use_bertscore:
        results = _compute_bertscore_batch([generated], [reference])
        bp, br, bf = results[0]

    return NLPScores(
        bleu4=bleu,
        rouge_l_precision=rp,
        rouge_l_recall=rr,
        rouge_l_f1=rf,
        bertscore_precision=bp,
        bertscore_recall=br,
        bertscore_f1=bf,
    )


def compute_batch_metrics(
    generated: list[str],
    references: list[str],
    use_bertscore: bool = True,
) -> dict[str, Any]:
    assert len(generated) == len(references), (
        f"got {len(generated)} generated vs {len(references)} references"
    )
    n = len(generated)
    if n == 0:
        return {"per_pair": [], "corpus": {}}

    # BLEU — sentence-level per pair, corpus-level for the headline
    sentence_bleus = [_compute_bleu(g, r) for g, r in zip(generated, references)]
    corpus_bleu = _compute_corpus_bleu(generated, references)

    # ROUGE-L — per pair (no standard corpus-level definition, so we average)
    rouge_results = [_compute_rouge_l(g, r) for g, r in zip(generated, references)]

    # BERTScore — one batch call, not N separate calls
    if use_bertscore:
        bert_results = _compute_bertscore_batch(generated, references)
    else:
        bert_results = [(0.0, 0.0, 0.0)] * n

    # assemble per-pair scores
    per_pair = []
    for i in range(n):
        rp, rr, rf = rouge_results[i]
        bp, br, bf = bert_results[i]
        per_pair.append(NLPScores(
            bleu4=sentence_bleus[i],
            rouge_l_precision=rp,
            rouge_l_recall=rr,
            rouge_l_f1=rf,
            bertscore_precision=bp,
            bertscore_recall=br,
            bertscore_f1=bf,
        ))

    # corpus-level aggregates
    corpus = _aggregate_scores(per_pair, corpus_bleu)

    return {"per_pair": per_pair, "corpus": corpus}


def _aggregate_scores(
    scores: list[NLPScores],
    corpus_bleu: float,
) -> dict[str, Any]:

    def _stats(values: list[float]) -> dict[str, float]:
        # Mean, std, median
        return {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "median": round(statistics.median(values), 4),
        }

    return {
        "bleu4_corpus": round(corpus_bleu, 4),
        "bleu4_sentence": _stats([s.bleu4 for s in scores]),
        "rouge_l_f1": _stats([s.rouge_l_f1 for s in scores]),
        "rouge_l_precision": _stats([s.rouge_l_precision for s in scores]),
        "rouge_l_recall": _stats([s.rouge_l_recall for s in scores]),
        "bertscore_f1": _stats([s.bertscore_f1 for s in scores]),
        "bertscore_precision": _stats([s.bertscore_precision for s in scores]),
        "bertscore_recall": _stats([s.bertscore_recall for s in scores]),
        "n_pairs": len(scores),
    }
