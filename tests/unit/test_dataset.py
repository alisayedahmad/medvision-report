"""Unit tests for ChestXray14Dataset. Synthetic CSV + PNGs, no real data needed."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data.dataset import PATHOLOGIES, ChestXray14Dataset


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake NIH ChestX-ray14 directory structure."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    filenames = ["img_001.png", "img_002.png", "img_003.png"]
    labels = ["Atelectasis|Effusion", "No Finding", "Pneumonia"]

    for fname in filenames:
        arr = np.random.randint(30, 220, (128, 128), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname)

    df = pd.DataFrame({"Image Index": filenames, "Finding Labels": labels})
    df.to_csv(tmp_path / "Data_Entry_2017.csv", index=False)

    return tmp_path, filenames, labels


def test_dataset_length(fake_dataset):
    data_dir, filenames, _ = fake_dataset
    ds = ChestXray14Dataset(data_dir, use_lung_mask=False)
    assert len(ds) == len(filenames)


def test_getitem_returns_expected_keys(fake_dataset):
    data_dir, _, _ = fake_dataset
    ds = ChestXray14Dataset(data_dir, use_lung_mask=False)
    sample = ds[0]
    assert set(sample.keys()) == {"image", "labels", "filename"}


def test_image_shape_and_dtype(fake_dataset):
    data_dir, _, _ = fake_dataset
    ds = ChestXray14Dataset(data_dir, image_size=224, use_lung_mask=False)
    sample = ds[0]

    assert sample["image"].shape == (1, 224, 224)
    assert sample["image"].dtype == torch.float32


def test_multi_label_encoding(fake_dataset):
    data_dir, _, _ = fake_dataset
    ds = ChestXray14Dataset(data_dir, use_lung_mask=False)

    # img_001: "Atelectasis|Effusion" -> two 1s
    sample_0 = ds[0]
    assert sample_0["labels"][PATHOLOGIES.index("Atelectasis")] == 1.0
    assert sample_0["labels"][PATHOLOGIES.index("Effusion")] == 1.0
    assert sample_0["labels"].sum() == 2.0

    # img_002: "No Finding" -> all zeros
    sample_1 = ds[1]
    assert sample_1["labels"].sum() == 0.0


def test_split_csv_filters(fake_dataset):
    data_dir, _, _ = fake_dataset

    split_path = data_dir / "split.csv"
    pd.DataFrame(["img_001.png", "img_003.png"]).to_csv(split_path, index=False, header=False)

    ds = ChestXray14Dataset(data_dir, split_csv=split_path, use_lung_mask=False)
    assert len(ds) == 2


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChestXray14Dataset(tmp_path)


import torch  # noqa: E402 — keep imports at top in real code, here it's fine for test clarity
