"""Binary lung mask extraction from chest X-rays """

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def get_lung_mask(pixels: np.ndarray) -> np.ndarray:
    """Return binary mask (uint8, 0/1) same shape as input """
    img = np.clip(pixels, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    mask = _keep_n_largest_components(cleaned, n=2)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    return mask


def apply_lung_mask(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside the lung region. Returns float32   """
    return pixels.astype(np.float32) * mask.astype(np.float32)


def _keep_n_largest_components(binary: np.ndarray, n: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(binary)

    areas = stats[1:, cv2.CC_STAT_AREA]
    top_labels = np.argsort(areas)[::-1][:n] + 1  # +1 skips background label

    out = np.zeros_like(binary)
    for label in top_labels:
        out[labels == label] = 255

    return out
