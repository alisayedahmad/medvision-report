"""Fine-tune DINOv2 + LoRA on ChestX-ray14 for multi-label classification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import PATHOLOGIES, ChestXray14Dataset
from models.dinov2_detector import DINOv2Classifier, inject_lora
from models.losses import FocalLoss
from models.metrics import compute_auc


@dataclass
class Config:
    data: dict
    split: dict
    model: dict
    training: dict
    logging: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def patient_level_split(
    labels_csv: str | Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices by patient ID so no patient appears in both sets.

    Random per-image splits leak patients across train/val, inflating
    metrics. NIH ChestX-ray14 has multiple images per patient — must
    split at the patient level.
    """
    df = pd.read_csv(labels_csv)
    patients = df["Patient ID"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_val_patients = int(len(patients) * val_fraction)
    val_patients = set(patients[:n_val_patients])

    train_idx = df.index[~df["Patient ID"].isin(val_patients)].tolist()
    val_idx = df.index[df["Patient ID"].isin(val_patients)].tolist()

    return train_idx, val_idx


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    log_every_n_steps: int,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

        if step % log_every_n_steps == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss={loss.item() * grad_accum_steps:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_targets, all_preds = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(batch["labels"].numpy())

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    return compute_auc(targets, preds, class_names=PATHOLOGIES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/lora.yaml")
    parser.add_argument("--use-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = ChestXray14Dataset(
        data_dir=cfg.data["data_dir"],
        labels_csv=cfg.data["labels_csv"],
        image_size=cfg.data["image_size"],
        use_clahe=cfg.data["use_clahe"],
        use_lung_mask=cfg.data["use_lung_mask"],
    )

    train_idx, val_idx = patient_level_split(
        cfg.data["labels_csv"], cfg.split["val_fraction"], cfg.split["seed"],
    )
    print(f"train: {len(train_idx)} images | val: {len(val_idx)} images")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        num_workers=cfg.training["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.training["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = DINOv2Classifier(num_classes=cfg.model["num_classes"])
    model = inject_lora(model, rank=cfg.model["lora_rank"], alpha=cfg.model["lora_alpha"])
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training["learning_rate"],
        weight_decay=cfg.training["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.training["mixed_precision"] and device.type == "cuda" else None

    if args.use_mlflow:
        import mlflow
        mlflow.set_experiment("dinov2_lora_chestxray14")
        mlflow.start_run()
        mlflow.log_params({
            "lora_rank": cfg.model["lora_rank"],
            "batch_size": cfg.training["batch_size"],
            "lr": cfg.training["learning_rate"],
            "epochs": cfg.training["num_epochs"],
        })

    ckpt_dir = Path(cfg.logging["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(cfg.training["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg.training["grad_accum_steps"], cfg.logging["log_every_n_steps"], epoch,
        )
        metrics = evaluate(model, val_loader, device)
        mean_auc = metrics["mean_auc"]

        print(f"epoch {epoch} | train_loss={train_loss:.4f} | val_mean_auc={mean_auc:.4f}")

        if args.use_mlflow:
            import mlflow
            mlflow.log_metrics({"train_loss": train_loss, "val_mean_auc": mean_auc}, step=epoch)

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  -> saved best model (auc={best_auc:.4f})")

    if args.use_mlflow:
        import mlflow
        mlflow.end_run()

    print(f"training done. best val mean AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
