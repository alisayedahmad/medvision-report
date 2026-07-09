"""Read chest X-rays from DICOM or PNG into a normalized numpy array."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None  # DICOM optional, PNG still works


@dataclass
class ImageMetadata:
    source_format: str  # "dicom" or "png"
    original_dtype: str
    photometric_interpretation: Optional[str] = None
    pixel_spacing_mm: Optional[tuple[float, float]] = None
    was_inverted: bool = False


def read_medical_image(path: str | Path) -> tuple[np.ndarray, ImageMetadata]:
    """Return (float32 array in [0,255], shape (H,W)), metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No image at {path}")

    suffix = path.suffix.lower()
    if suffix == ".dcm":
        return _read_dicom(path)
    elif suffix in (".png", ".jpg", ".jpeg"):
        return _read_standard_image(path)
    else:
        raise ValueError(f"Unsupported extension '{suffix}' for {path}")


def _read_dicom(path: Path) -> tuple[np.ndarray, ImageMetadata]:
    if pydicom is None:
        raise ImportError("pydicom required for .dcm files: pip install pydicom")

    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array.astype(np.float32)

    # MONOCHROME1 = inverted grayscale. Uncorrected, image reads as a negative.
    photometric = getattr(dataset, "PhotometricInterpretation", None)
    was_inverted = False
    if photometric == "MONOCHROME1":
        pixels = pixels.max() - pixels
        was_inverted = True

    # Needed later for lesion size in cm, captured here to avoid re-parsing.
    spacing = getattr(dataset, "PixelSpacing", None)
    spacing_mm = tuple(float(v) for v in spacing) if spacing is not None else None

    pixels = _rescale_to_uint8_range(pixels)

    metadata = ImageMetadata(
        source_format="dicom",
        original_dtype=str(dataset.pixel_array.dtype),
        photometric_interpretation=photometric,
        pixel_spacing_mm=spacing_mm,
        was_inverted=was_inverted,
    )
    return pixels, metadata


def _read_standard_image(path: Path) -> tuple[np.ndarray, ImageMetadata]:
    with Image.open(path) as img:
        img = img.convert("L")  # collapse fake RGB duplicates to single channel
        pixels = np.array(img, dtype=np.float32)

    metadata = ImageMetadata(source_format="png", original_dtype="uint8")
    return pixels, metadata


def _rescale_to_uint8_range(pixels: np.ndarray) -> np.ndarray:
    p_min, p_max = pixels.min(), pixels.max()
    if p_max - p_min < 1e-6:
        return np.zeros_like(pixels)  # flat image guard, avoids div by zero
    return (pixels - p_min) / (p_max - p_min) * 255.0