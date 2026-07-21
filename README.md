# MedVision-Report

A pipeline that reads chest X-rays, detects pathologies with a fine-tuned vision model, and generates a structured clinical report with an LLM — grounded in what the vision model actually found, not free-form guessing.

> Research and educational project. Not a medical device, not for clinical use. See [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).

## Motivation

Most "AI reads X-rays" demos either hallucinate findings or need a huge model and an API call per image. I wanted to see how far a small fine-tuned vision model (DINOv2 ViT-S or U-Net) gets if I force the LLM to only describe what the vision model detected — never anything else. The vision model outputs structured findings (pathology, location, size, confidence), and those findings are the only thing the LLM is allowed to talk about.

The goal is a system that is cheap to run and auditable — every sentence in the generated report traces back to a specific detection.

## Current status

Phases 0 through 5 are done. The pipeline works end-to-end: image in, report out, evaluated against radiologist-written reports. Phases 6 (API) and 7 (deployment) are next.

### Results so far

**Vision model comparison on ChestX-ray14:**

| Model | Mean AUC | Trainable params | Inference |
|-------|----------|-----------------|-----------|
| DINOv2 ViT-S + LoRA | 0.7469 | 153k | 6.04 ms/img |
| U-Net ConvNeXt | 0.6772 | 32M | 6.32 ms/img |

DINOv2 with LoRA wins with 208x fewer trainable parameters and higher AUC. Self-supervised features transfer to medical imaging better than expected.

**Report evaluation** uses four metrics: BLEU-4 (lexical overlap), ROUGE-L (longest common subsequence), BERTScore with roberta-large (semantic similarity), and CheXpert-style clinical label F1 (did the report get the diagnosis right, regardless of wording). Hallucination rate is tracked per pathology.

## How it works

X-ray image → preprocessing (normalization, lung masking) → DINOv2 + LoRA → probability vector (14 pathologies) → structured feature extraction (findings, negatives, uncertainties) → LLM prompt with grounding constraints → generated report + hallucination check + uncertainty flag.

The structured feature extraction layer between the vision model and the LLM is the key design choice. The LLM never sees the image. It only sees a structured list of what was detected and what wasn't. It cannot hallucinate a finding that the vision model didn't produce.

## What's built

**Phase 1 — Preprocessing:** DICOM/PNG reading, Hounsfield normalization, histogram equalization, lung segmentation masking, PyTorch Dataset for ChestX-ray14.

**Phase 2 — Vision models:** DINOv2 ViT-S/14 with LoRA injection, U-Net with ConvNeXt backbone. Dice loss, Focal loss, multi-label BCE. AUC, Dice score, Hausdorff distance metrics.

**Phase 3 — Training and evaluation:** Full training scripts with the comparison above. LoRA config tuning, evaluation on test split.

**Phase 4 — Report generation:** Feature extraction from probabilities to structured findings. Versioned prompt templates (v1/v2/v3) with anti-hallucination constraints. Async LLM client with retry and caching (Anthropic + OpenAI). Hallucination checker with synonym maps and negation detection. Uncertainty flagging with three failure-mode heuristics. Full orchestrator returning a Report dataclass.

**Phase 5 — NLP evaluation:** BLEU-4 via sacrebleu, ROUGE-L via rouge-score, BERTScore via roberta-large in fp16. CheXpert-style label extraction (positive/negative/uncertain/blank) with per-class F1. Batch hallucination analysis. Single CLI command to run everything: `python -m evaluation.run_evaluation --input pairs.json`.

## Setup

```bash
git clone https://github.com/alisayedahmad/medvision-report.git
cd medvision-report
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For training and inference (local only, not CI):
```bash
pip install torch torchvision monai transformers peft pandas pillow scikit-learn
```

For NLP evaluation:
```bash
pip install -r requirements-eval.txt
```

## Running evaluation

```bash
python -m evaluation.run_evaluation --input pairs.json --output results/nlp_evaluation.json
```

Add `--no-bertscore` to skip the heavy model and run BLEU + ROUGE only.

## Tests

```bash
python -m pytest -v
```

CI runs on every push — lightweight tests only (no torch, no GPU). Full test suite runs locally with all deps installed.

## Layout

```
medvision-report/
├── data/                  # preprocessing + dataset loading
├── models/                # DINOv2 + LoRA, U-Net ConvNeXt
├── training/              # training + evaluation scripts
├── report_generation/     # feature extraction, prompts, LLM client, hallucination checks
├── evaluation/            # NLP metrics, CheXpert labeler, hallucination rate, benchmark runner
├── api/                   # FastAPI app (Phase 6 — next)
├── deployment/            # Docker + Kubernetes manifests (Phase 7)
├── monitoring/            # custom Prometheus metrics, drift detection
├── tests/                 # unit + integration tests
├── results/               # evaluation outputs
├── docs/                  # architecture notes, disclaimer, prompt changelog
└── configs/               # model/training/deployment configs
```

## Datasets

**NIH ChestX-ray14** — 112,120 frontal chest X-rays, 14 pathology labels. Used for training and vision model evaluation.

**OpenI** — 7,470 X-rays paired with radiologist-written reports. Used for NLP evaluation — comparing generated reports against real ones.

## Hardware

Developed and tested on RTX A2000 Laptop (4GB VRAM). BERTScore runs in fp16 on GPU, falls back to CPU automatically.

## License

MIT. Research use only. See the medical disclaimer before using this work in any other context.