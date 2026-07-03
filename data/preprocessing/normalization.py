
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(pixels: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization

    Input: float32 array, any range
    Output: float32 array in [0, 255]
    """
    img_uint8 = np.clip(pixels, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(img_uint8)
    return result.astype(np.float32)


def normalize_pixel_values(pixels: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Scale to [0,1] then standardize with domain-specific mean/std.

    ImageNet stats (mean=0.485, std=0.229 etc.) are wrong here — replace with stats computed on the actual
    training set once we have one
    """

    
    scaled = pixels / 255.0
    return (scaled - mean) / std


def preprocess_image(pixels: np.ndarray, use_clahe: bool = True, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Full preprocessing chain: CLAHE (optional) -> normalization. Model-ready output."""
    if use_clahe:
        pixels = apply_clahe(pixels)
    return normalize_pixel_values(pixels, mean=mean, std=std)
