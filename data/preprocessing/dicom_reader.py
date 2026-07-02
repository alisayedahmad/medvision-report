from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import pydicom
else:
    try:
        import pydicom
    except ImportError:
        pydicom = None  


@dataclass
class ImageMetadata:
    """Metadata describing image source and preprocessing applied."""

    source_format: str
    original_dtype: str
    photometric_interpretation: Optional[str] = None
    pixel_spacing_mm: Optional[tuple[float, float]] = None
    was_inverted: bool = False


def read_medical_image(path: str | Path) -> tuple[np.ndarray, ImageMetadata]:
        """
        Load a chest x-ray from disk (DICOM, PNG, JPG) and return:
            - a float32 numpy array in [0, 255], shape (H, W)
                (single channel — no fake RGB)
            - ImageMetadata describing source and preprocessing

        Raises:
                FileNotFoundError: if path does not exist
                ValueError: if extension unsupported or pixels unreadable
        """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Aucune image trouvée à {path}")

    suffix = path.suffix.lower()

    if suffix == ".dcm":
        return _read_dicom(path)
    elif suffix in (".png", ".jpg", ".jpeg"):
        return _read_standard_image(path)
    else:
        raise ValueError(
            f"Extension '{suffix}' non supportée pour {path}. "
            "Attendu : .dcm, .png, .jpg ou .jpeg."
        )


def _read_dicom(path: Path) -> tuple[np.ndarray, ImageMetadata]:
    if pydicom is None:
        raise ImportError(
            "pydicom is not installed but is required to read .dcm files. "
            "Install it with: pip install pydicom"
        )

    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array.astype(np.float32)

    # Some DICOMs use inverted grayscale (MONOCHROME1). Fix here.
    photometric = getattr(dataset, "PhotometricInterpretation", None)
    was_inverted = False
    if photometric == "MONOCHROME1":
        pixels = pixels.max() - pixels
        was_inverted = True

    # PixelSpacing gives real-world pixel size in mm. Save for later.
    spacing = getattr(dataset, "PixelSpacing", None)
    spacing_mm = tuple(float(v) for v in spacing) if spacing is not None else None

    # Rescale pixel values to 0-255 regardless of original bit depth.
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
        # Force grayscale: some PNGs are RGB duplicates. Keep one channel.
        img = img.convert("L")
        pixels = np.array(img, dtype=np.float32)

    metadata = ImageMetadata(
        source_format="png",
        original_dtype="uint8",
    )
    return pixels, metadata


def _rescale_to_uint8_range(pixels: np.ndarray) -> np.ndarray:
    """Rescale image min-max to [0, 255]. Handles flat images safely."""
    p_min, p_max = pixels.min(), pixels.max()
    if p_max - p_min < 1e-6:
        return np.zeros_like(pixels)
    return (pixels - p_min) / (p_max - p_min) * 255.0
