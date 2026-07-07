"""Unit tests for lung_segmentation. Synthetic images only."""

import numpy as np

from data.preprocessing.lung_segmentation import apply_lung_mask, get_lung_mask


def _make_fake_xray(size=128):
    """Two bright rectangles on a black background — mimics left/right lungs."""
    img = np.zeros((size, size), dtype=np.float32)
    img[20:100, 10:50] = 200   # left lung
    img[20:100, 70:115] = 200  # right lung
    return img


def test_mask_is_binary_uint8(tmp_path):
    img = _make_fake_xray()
    mask = get_lung_mask(img)

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_mask_covers_bright_lung_regions():
    img = _make_fake_xray()
    mask = get_lung_mask(img)

    # the bright lung rectangles should be mostly inside the mask
    left_lung_coverage = mask[20:100, 10:50].mean()
    right_lung_coverage = mask[20:100, 70:115].mean()
    assert left_lung_coverage > 0.7, f"Left lung coverage too low: {left_lung_coverage:.2f}"
    assert right_lung_coverage > 0.7, f"Right lung coverage too low: {right_lung_coverage:.2f}"


def test_mask_suppresses_background():
    img = _make_fake_xray()
    mask = get_lung_mask(img)

    # top-left corner is pure black background — should be masked out
    background_coverage = mask[0:10, 0:10].mean()
    assert background_coverage < 0.1


def test_apply_lung_mask_zeros_outside_region():
    img = np.ones((64, 64), dtype=np.float32) * 150
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1

    result = apply_lung_mask(img, mask)

    assert result[0, 0] == 0.0           # outside mask → zeroed
    assert result[30, 30] == 150.0       # inside mask → unchanged


def test_apply_lung_mask_output_dtype():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8).astype(np.float32)
    mask = np.ones((64, 64), dtype=np.uint8)
    result = apply_lung_mask(img, mask)

    assert result.dtype == np.float32
