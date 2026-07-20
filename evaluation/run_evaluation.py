"""Full evaluation benchmark runner.

Ties nlp_metrics, chexpert_labeler, and hallucination_rate into one
command. Takes a JSON file of (generated, reference, findings) triples,
runs all metrics, saves results.

Two modes:
  1. From pre-generated pairs — pass a JSON file, skip the pipeline
  2. End-to-end — load images, run vision + LLM, then evaluate

Mode 1 is what you use 95% of the time. Generate reports once, iterate
on evaluation without re-running the expensive pipeline.

Usage:
    python -m evaluation.run_evaluation --input pairs.json --output results/nlp_evaluation.json
    python -m evaluation.run_evaluation --input pairs.json --no-bertscore  # fast, CPU-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from evaluation.chexpert_labeler import compute_chexpert_f1
from evaluation.hallucination_rate import run_hallucination_analysis
from evaluation.nlp_metrics import compute_batch_metrics


def load_pairs(path: Path) -> list[dict[str, Any]]:
    """Load evaluation pairs from JSON.

    Expected format — list of objects, each with:
        "generated": str   — the LLM-generated report
        "reference": str   — the radiologist-written report
        "findings": dict   — StudyFindings.to_dict() (optional, needed for hallucination check)
    """
    with open(path) as f:
        pairs = json.load(f)

    if not isinstance(pairs, list):
        raise ValueError(f"expected a JSON list, got {type(pairs).__name__}")

    for i, p in enumerate(pairs):
        if "generated" not in p or "reference" not in p:
            raise ValueError(f"pair {i} missing 'generated' or 'reference' key")

    return pairs


def run_full_evaluation(
    pairs: list[dict[str, Any]],
    use_bertscore: bool = True,
) -> dict[str, Any]:
    """Run all evaluation metrics on a list of (generated, reference) pairs.

    Returns a single dict with everything — nlp scores, chexpert F1,
    hallucination stats, and metadata.
    """
    generated = [p["generated"] for p in pairs]
    references = [p["reference"] for p in pairs]
    has_findings = all("findings" in p for p in pairs)

    start = time.time()

    # NLP metrics — BLEU, ROUGE, BERTScore
    print(f"running NLP metrics on {len(pairs)} pairs (bertscore={use_bertscore})...")
    nlp_results = compute_batch_metrics(generated, references, use_bertscore=use_bertscore)

    # CheXpert-style clinical F1
    print("running CheXpert label extraction and F1...")
    chexpert_results = compute_chexpert_f1(generated, references)

    # hallucination analysis — only if we have findings
    hallucination_results = None
    if has_findings:
        print("running hallucination analysis...")
        findings = [p["findings"] for p in pairs]
        hall_result = run_hallucination_analysis(generated, findings)
        hallucination_results = hall_result.to_dict()
    else:
        print("skipping hallucination analysis — no findings in input pairs")

    elapsed = time.time() - start

    # per-pair NLP scores as dicts for JSON serialization
    per_pair_dicts = [s.to_dict() for s in nlp_results["per_pair"]]

    return {
        "metadata": {
            "n_pairs": len(pairs),
            "bertscore_enabled": use_bertscore,
            "elapsed_seconds": round(elapsed, 2),
        },
        "nlp_metrics": {
            "corpus": nlp_results["corpus"],
            "per_pair": per_pair_dicts,
        },
        "chexpert": chexpert_results.to_dict(),
        "hallucination": hallucination_results,
    }


def save_results(results: dict[str, Any], path: Path) -> None:
    """Write results to JSON. Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full NLP evaluation benchmark on generated reports."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="JSON file with evaluation pairs (generated + reference + optional findings)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/nlp_evaluation.json"),
        help="where to save results (default: results/nlp_evaluation.json)",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="skip BERTScore for faster evaluation (CPU-only metrics only)",
    )
    args = parser.parse_args(argv)

    pairs = load_pairs(args.input)
    results = run_full_evaluation(pairs, use_bertscore=not args.no_bertscore)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
