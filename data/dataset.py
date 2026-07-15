
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.preprocessing.dicom_reader import read_medical_image
from data.preprocessing.lung_segmentation import apply_lung_mask, get_lung_mask
from data.preprocessing.normalization import preprocess_image

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

class ChestXray14Dataset(Dataset):
    """Loads NIH ChestX-ray14 images with multi-label pathology targets.

    Expected layout:
        data_dir/
            images/           <- PNG files (00000001_000.png, ...)
            Data_Entry_2017.csv
    """

    def __init__(
        self,
        data_dir: str | Path,
        labels_csv: str | Path | None = None,
        split_csv: str | Path | None = None,
        use_clahe: bool = True,
        use_lung_mask: bool = True,
        image_size: int = 224,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.use_clahe = use_clahe
        self.use_lung_mask = use_lung_mask
        self.image_size = image_size
        self.mean = mean
        self.std = std

        labels_path = Path(labels_csv) if labels_csv else self.data_dir / "Data_Entry_2017.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels CSV not found at {labels_path}")

        df = pd.read_csv(labels_path)

        # Filter to split if provided (train/val/test list of filenames)
        if split_csv is not None:
            split_files = set(pd.read_csv(split_csv, header=None)[0])
            df = df[df["Image Index"].isin(split_files)]

        self.filenames = df["Image Index"].tolist()
        self.labels = self._parse_labels(df["Finding Labels"])

    def _parse_labels(self, label_series: pd.Series) -> np.ndarray:
        """Convert pipe-separated label strings to multi-hot vectors."""
        pathology_to_idx = {p: i for i, p in enumerate(PATHOLOGIES)}
        out = np.zeros((len(label_series), len(PATHOLOGIES)), dtype=np.float32)

        for i, label_str in enumerate(label_series):
            if label_str == "No Finding":
                continue
            for pathology in label_str.split("|"):
                pathology = pathology.strip()
                if pathology in pathology_to_idx:
                    out[i, pathology_to_idx[pathology]] = 1.0

        return out

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_dir / self.filenames[idx]
        pixels, meta = read_medical_image(img_path)

        if self.use_lung_mask:
            mask = get_lung_mask(pixels)
            pixels = apply_lung_mask(pixels, mask)

        pixels = preprocess_image(
            pixels, use_clahe=self.use_clahe, mean=self.mean, std=self.std,
        )

        pixels = self._resize(pixels)

        # (H, W) -> (1, H, W) single channel tensor
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        labels = torch.from_numpy(self.labels[idx])

        return {"image": tensor, "labels": labels, "filename": self.filenames[idx]}

    def _resize(self, pixels: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(
            pixels, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR,
        )