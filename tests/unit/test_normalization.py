"""Unit tests for normalization. Synthetic arrays only, no real dataset needed """

import numpy as np

from data.preprocessing.normalization import apply_clahe, normalize_pixel_values, preprocess_image


def test_clahe_output_shape_and_range():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8).astype(np.float32)
    result = apply_clahe(img)

    assert result.shape == img.shape
    assert result.dtype == np.float32
    assert result.min() >= 0 and result.max() <= 255


def test_clahe_increases_local_contrast_on_low_contrast_region():
    # flat mid-gray image with a barely-different patch - low contrast on purpose
    img = np.full((64, 64), 128, dtype=np.float32)
    img[20:40, 20:40] = 135
    result = apply_clahe(img)

    original_range = img[20:40, 20:40].max() - img.min()
    enhanced_range = result[20:40, 20:40].max() - result.min()
    assert enhanced_range >= original_range


def test_normalize_pixel_values_default_range():
    img = np.array([[0.0, 255.0], [128.0, 64.0]], dtype=np.float32)
    result = normalize_pixel_values(img)

    # 0 -> (0/255 - 0.5)/0.5 = -1.0 ; 255 -> (1.0 - 0.5)/0.5 = 1.0
    assert np.isclose(result[0, 0], -1.0)
    assert np.isclose(result[0, 1], 1.0)


def test_normalize_pixel_values_custom_stats():
    img = np.array([[255.0]], dtype=np.float32)
    result = normalize_pixel_values(img, mean=1.0, std=1.0)
    assert np.isclose(result[0, 0], 0.0)


def test_preprocess_image_chains_clahe_and_normalize():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8).astype(np.float32)
    result = preprocess_image(img)

    assert result.shape == img.shape
    assert result.dtype == np.float32
    # normalized with mean=0.5, std=0.5 -> values must land in [-1, 1]
    assert result.min() >= -1.0 - 1e-3 and result.max() <= 1.0 + 1e-3



def test_preprocess_image_without_clahe():
    
    img = np.array([[0.0, 255.0]], dtype=np.float32)
    result = preprocess_image(img, use_clahe=False)


    assert np.isclose(result[0, 0], -1.0)
    assert np.isclose(result[0, 1], 1.0)
