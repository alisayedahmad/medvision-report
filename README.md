# MedVision-Report

I'm building a pipeline that reads chest X-rays, detects pathologies with a fine-tuned vision model, and generates a structured clinical report with an LLM — grounded in what the vision model actually found, not free-form guessing.

> Research/educational project. Not a medical device, not for clinical use. Details in [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).

## The idea

Most "AI reads X-rays" demos either hallucinate findings or need a huge model and an API call per image. I want to see how far a small fine-tuned vision model gets if I force the LLM to only describe what the vision model detected — never anything else. The vision model outputs structured findings (pathology, location, confidence, severity), and those findings are the only thing the LLM is allowed to talk about.

If it works, I'll have something that's cheap to run and auditable — you can always trace a sentence in the report back to a specific detection.

Backbones: DINOv2 ViT-S/14 + LoRA, vs U-Net + ConvNeXt, trained on NIH ChestX-ray14. Generated reports get scored against real radiologist reports from OpenI (BLEU-4, ROUGE-L, BERTScore, CheXpert labeler F1).

## Progress

Step by step, not all at once:

- ✅ **Phase 0** — repo structure, CI, licence
- ✅ **Phase 1** — preprocessing (DICOM/PNG loading, normalization, lung masking)
- ✅ **Phase 2** — vision models (DINOv2 + LoRA, U-Net + ConvNeXt, losses, metrics)
- ✅ **Phase 3** — training + evaluation on ChestX-ray14 sample
- 🚧 **Phase 4** — LLM report generation (feature extraction + prompts + LLM client + hallucination checker done, uncertainty + orchestrator remaining)
- ⏳ **Phase 5** — NLP evaluation (BLEU, ROUGE, BERTScore, CheXpert labeler)
- ⏳ **Phase 6** — FastAPI service
- ⏳ **Phase 7** — Docker → Kubernetes → monitoring

I only move to the next step once the current one runs and has tests. No skipping ahead.

## First results (Phase 3)

Trained on the ChestX-ray14 sample (5,606 images, patient-level split):

| Model | Mean AUC | Trainable params | Inference (RTX A2000) |
|-------|----------|------------------|-----------------------|
| DINOv2 + LoRA | **0.7469** | 153k | 6.04 ms/img |
| U-Net + ConvNeXt | 0.6772 | 32M | 6.32 ms/img |

LoRA on frozen DINOv2 wins with 208× fewer trainable parameters. This confirms the first hypothesis from the README — self-supervised features transfer to medical imaging well enough that most of the work is done by the backbone, and the head just needs to specialize.

Full NIH ChestX-ray14 training run is on the roadmap once Phase 5 is done and I have proper NLP evaluation to justify the compute.

## Layout

```
medvision-report/
├── data/                 # preprocessing + dataset loading
├── models/               # vision model architectures + losses + metrics
├── training/             # training + evaluation scripts
├── report_generation/    # feature extraction, prompts, LLM client, hallucination checks
├── evaluation/           # NLP metrics, CheXpert labeler, benchmark runner
├── api/                  # FastAPI app
├── deployment/           # Docker + Kubernetes manifests
├── monitoring/           # custom Prometheus metrics, drift detection
├── notebooks/            # exploration and analysis
├── tests/                # unit + integration tests
├── results/              # evaluation outputs (metrics, not weights)
├── docs/                 # architecture notes, disclaimer, prompt changelog
└── configs/              # model/training/deployment configs
```

## Setup

```bash
git clone https://github.com/alisayedahmad/medvision-report.git
cd medvision-report
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# heavy ML deps (torch, monai, transformers, peft, PIL, pandas)
# installed separately — see docs/replication.md
```

Run tests:

```bash
pytest -v
```

## Datasets

- **NIH ChestX-ray14** — 112,120 frontal chest X-rays, 14 pathology labels. Currently working from the sample subset (5,606 images) — see `data/README.md`.
- **OpenI** — 7,470 X-rays paired with real radiologist reports. Used for Phase 5 evaluation.

## License

MIT, research use only — see the disclaimer before using this for anything beyond that.
