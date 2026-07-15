"""Evaluate both trained models on the validation set and dump comparison JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import PATHOLOGIES, ChestXray14Dataset
from models.dinov2_detector import DINOv2Classifier, inject_lora
from models.metrics import compute_auc
from models.unet_convnext import UNetClassifier
from training.train_dinov2 import Config, patient_level_split


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (targets, predictions, mean_latency_ms_per_image)."""
    model.eval()
    all_targets, all_preds = [], []
    total_time = 0.0
    total_images = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["labels"].numpy()

        start = time.perf_counter()
        output = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        logits = output["classification"] if isinstance(output, dict) else output
        preds = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets)
        total_time += elapsed
        total_images += images.shape[0]

    mean_latency_ms = (total_time / total_images) * 1000
    return np.concatenate(all_targets), np.concatenate(all_preds), mean_latency_ms


def build_dinov2(cfg: Config, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = DINOv2Classifier(num_classes=cfg.model["num_classes"])
    model = inject_lora(model, rank=cfg.model["lora_rank"], alpha=cfg.model["lora_alpha"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


def build_unet(cfg: Config, ckpt_path: Path, device: torch.device) -> torch.nn.Module:

    model = UNetClassifier(

        num_classes=cfg.model["num_classes"],
        pretrained=False,  # weights come from checkpoint
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


def evaluate_model(
    name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())


    targets, preds, latency_ms = run_inference(model, loader, device)
    aucs = compute_auc(targets, preds, class_names=PATHOLOGIES)


    return {
        "model": name,
        "trainable_params": trainable,
        "total_params": total,
        "mean_latency_ms_per_image": round(latency_ms, 2),
        "per_pathology_auc": {k: round(v, 4) for k, v in aucs.items() if k != "mean_auc"},
        "mean_auc": round(aucs["mean_auc"], 4),

    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2-config", default="configs/training/lora.yaml")
    parser.add_argument("--unet-config", default="configs/training/unet.yaml")
    parser.add_argument("--dinov2-ckpt", default="checkpoints/dinov2_lora/best.pt")
    parser.add_argument("--unet-ckpt", default="checkpoints/unet_convnext/best.pt")
    parser.add_argument("--output", default="results/model_comparison.json")
    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg_dinov2 = Config.from_yaml(args.dinov2_config)



    dataset = ChestXray14Dataset(
        data_dir=cfg_dinov2.data["data_dir"],
        labels_csv=cfg_dinov2.data["labels_csv"],
        image_size=cfg_dinov2.data["image_size"],
        use_clahe=cfg_dinov2.data["use_clahe"],
        use_lung_mask=cfg_dinov2.data["use_lung_mask"],
    )

    _, val_idx = patient_level_split(
        cfg_dinov2.data["labels_csv"],
        cfg_dinov2.split["val_fraction"],
        cfg_dinov2.split["seed"],
    )
    print(f"eval set: {len(val_idx)} images")

    val_loader = DataLoader(

        Subset(dataset, val_idx),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


    results = []

    print("\n--- DINOv2 + LoRA ---")
    model = build_dinov2(cfg_dinov2, Path(args.dinov2_ckpt), device)
    r = evaluate_model("dinov2_lora", model, val_loader, device)
    print(f"  mean AUC: {r['mean_auc']} | latency: {r['mean_latency_ms_per_image']} ms/img")
    results.append(r)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n--- U-Net ConvNeXt ---")
    cfg_unet = Config.from_yaml(args.unet_config)
    model = build_unet(cfg_unet, Path(args.unet_ckpt), device)
    r = evaluate_model("unet_convnext", model, val_loader, device)
    print(f"  mean AUC: {r['mean_auc']} | latency: {r['mean_latency_ms_per_image']} ms/img")
    results.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
