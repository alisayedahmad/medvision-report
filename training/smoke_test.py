"""Smoke test: 1 epoch on a small subset to verify GPU memory fits.

Run before committing to a real training job. Reports peak VRAM usage

and catches OOM early. Meant to be quick ....

"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import ChestXray14Dataset
from models.dinov2_detector import DINOv2Classifier, inject_lora
from models.losses import FocalLoss
from models.unet_convnext import UNetClassifier
from training.train_dinov2 import Config, patient_level_split


def run_smoke(model_name: str, cfg: Config, num_batches: int = 20) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA — this test is meant to check GPU memory")

    print(f"\n=== SMOKE TEST: {model_name} ===")
    print(f"device: {device}")
    print(f"batch_size: {cfg.training['batch_size']} | image_size: {cfg.data['image_size']}")

    dataset = ChestXray14Dataset(
        data_dir=cfg.data["data_dir"],
        labels_csv=cfg.data["labels_csv"],
        image_size=cfg.data["image_size"],
        use_clahe=cfg.data["use_clahe"],
        use_lung_mask=cfg.data["use_lung_mask"],
    )

    train_idx, _ = patient_level_split(
        cfg.data["labels_csv"], cfg.split["val_fraction"], cfg.split["seed"],
    )
    train_idx = train_idx[:num_batches * cfg.training["batch_size"]]

    loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        num_workers=0,  # keep deterministic for the smoke test
    )

    if model_name == "dinov2":
        model = DINOv2Classifier(num_classes=cfg.model["num_classes"])
        model = inject_lora(model, rank=cfg.model["lora_rank"], alpha=cfg.model["lora_alpha"])
    elif model_name == "unet":
        model = UNetClassifier(
            num_classes=cfg.model["num_classes"],
            pretrained=cfg.model.get("pretrained", True),
        )
    else:
        raise ValueError(f"unknown model: {model_name}")

    model = model.to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    start = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(images)
            logits = output["classification"] if isinstance(output, dict) else output
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            print(f"first batch OK - loss={loss.item():.4f}")

    elapsed = time.time() - start
    print(f"processed {len(loader)} batches in {elapsed:.1f}s ({elapsed / len(loader):.2f}s/batch)")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"peak VRAM: {peak_mb:.0f} MB / {total_mb:.0f} MB ({100 * peak_mb / total_mb:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dinov2", "unet", "both"], default="both")
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()

    if args.model in ("dinov2", "both"):
        cfg = Config.from_yaml("configs/training/lora.yaml")
        try:
            run_smoke("dinov2", cfg, args.num_batches)
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM on dinov2: {e}\n-> reduce batch_size in configs/training/lora.yaml")

    if args.model in ("unet", "both"):
        cfg = Config.from_yaml("configs/training/unet.yaml")
        try:
            run_smoke("unet", cfg, args.num_batches)
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM on unet: {e}\n-> reduce batch_size in configs/training/unet.yaml")

    print("\nsmoke test done. if VRAM stayed under 90%, you're safe to run the full training.")


if __name__ == "__main__":
    main()
