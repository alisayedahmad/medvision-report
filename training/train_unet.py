"""Fine-tune U-Net ConvNeXt on ChestX-ray14 for multi-label classification.

The sample dataset has no pixel-level segmentation masks, so we train the
classification head only (global pooling over segmentation logits) with the
same patient-level split and metrics as the DINOv2 run for a fair comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import PATHOLOGIES, ChestXray14Dataset
from models.losses import FocalLoss
from models.metrics import compute_auc
from models.unet_convnext import UNetClassifier
from training.train_dinov2 import Config, patient_level_split


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
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
            outputs = model(images)
            loss = criterion(outputs["classification"], targets) / grad_accum_steps

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
        outputs = model(images)
        preds = torch.sigmoid(outputs["classification"]).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(batch["labels"].numpy())

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    return compute_auc(targets, preds, class_names=PATHOLOGIES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/unet.yaml")
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

    model = UNetClassifier(
        num_classes=cfg.model["num_classes"],
        pretrained=cfg.model["pretrained"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training["learning_rate"],
        weight_decay=cfg.training["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.training["mixed_precision"] and device.type == "cuda" else None

    if args.use_mlflow:
        import mlflow
        mlflow.set_experiment("unet_convnext_chestxray14")
        mlflow.start_run()
        mlflow.log_params({
            "encoder": cfg.model["encoder_name"],
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
