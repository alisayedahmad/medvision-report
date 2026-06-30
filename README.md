# MedVision-Report

I'm building a pipeline that reads chest X-rays, detects pathologies with a fine-tuned vision model, and generates a structured clinical report with an LLM — grounded in what the vision model actually found, not free-form guessing.

Right now there's no code yet, just the project skeleton. I wanted the folder structure decided before writing anything, so I'm not reorganizing every other week.

> Research/educational project. Not a medical device, not for clinical use. Details in [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).

## The idea

Most "AI reads X-rays" demos either hallucinate findings or need a huge model and an API call per image. I want to see how far a small fine-tuned vision model (DINOv2 ViT-S or U-Net) gets if I force the LLM to only describe what the vision model detected — never anything else. The vision model outputs structured findings (pathology, location, size, confidence), and those findings are the only thing the LLM is allowed to talk about.

If it works, I'll have something that's cheap to run and auditable — you can always trace a sentence in the report back to a specific detection.

Backbones: DINOv2 ViT-S/14 + LoRA, vs U-Net + ConvNeXt, trained on NIH ChestX-ray14. Generated reports get scored against real radiologist reports from OpenI (BLEU-4, ROUGE-L, BERTScore, CheXpert labeler F1).

## How I'm building this

Step by step, not all at once:

1. Data + preprocessing — DICOM/PNG loading, normalization, lung masking
2. Vision model — train + evaluate DINOv2 and U-Net on ChestX-ray14
3. Structured feature extraction — detections → the format the LLM consumes
4. LLM report generation — prompting, grounding checks, uncertainty flagging
5. NLP evaluation — score generated reports against OpenI
6. API — FastAPI wrapper around the pipeline
7. Deployment — Docker, then Kubernetes, then monitoring

I'm only moving to the next step once the current one actually runs and has a test or two. No skipping ahead.

## Layout

```
medvision-report/
├── data/                # preprocessing + dataset loading
├── models/               # vision model architectures
├── training/             # training + evaluation scripts
├── report_generation/    # feature extraction, prompts, LLM client, hallucination checks
├── evaluation/            # NLP metrics, CheXpert labeler, benchmark runner
├── api/                  # FastAPI app
├── deployment/            # Docker + Kubernetes manifests
├── monitoring/            # custom Prometheus metrics, drift detection
├── notebooks/             # exploration and analysis
├── tests/                 # unit + integration tests
├── results/               # evaluation outputs (metrics, not weights)
├── docs/                  # architecture notes, disclaimer, prompt changelog
└── configs/                # model/training/deployment configs
```

Most folders are just `.gitkeep` placeholders for now — wanted the shape of the thing visible from the first commit, fills in as I go.

## Setup

```bash
git clone <your-repo-url>
cd medvision-report
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nothing to run yet — just tooling installed (pytest, ruff, black).

## Datasets

- **NIH ChestX-ray14** — 112,120 frontal chest X-rays, 14 pathology labels. Download instructions go in `data/README.md` once I start on preprocessing.
- **OpenI** — 7,470 X-rays paired with real radiologist reports, used for evaluation later.

## License

MIT, research use only — see the disclaimer before using this for anything beyond that.
