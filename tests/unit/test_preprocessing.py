"""Unit tests for dicom_reader. Uses synthetic fixtures, no real dataset needed."""

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from data.preprocessing.dicom_reader import read_medical_image


def _make_fake_png(tmp_path, values=None):
    path = tmp_path / "fake_xray.png"
    arr = values if values is not None else np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _make_fake_dicom(tmp_path, photometric="MONOCHROME2"):
    path = tmp_path / "fake_xray.dcm"

    # Raw value for a visually bright region is inverted between the two
    # photometric interpretations - simulates real DICOM semantics.
    arr = np.zeros((64, 64), dtype=np.uint16)
    if photometric == "MONOCHROME1":
        arr[:, :] = 4000
        arr[10:20, 10:20] = 0
    else:
        arr[10:20, 10:20] = 4000

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.Rows, ds.Columns = arr.shape
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.PixelSpacing = ["0.14", "0.14"]
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(path, write_like_original=False)
    return path


def test_reads_png_as_grayscale_float32(tmp_path):
    path = _make_fake_png(tmp_path)
    pixels, meta = read_medical_image(path)

    assert pixels.dtype == np.float32
    assert pixels.shape == (64, 64)
    assert meta.source_format == "png"
    assert pixels.min() >= 0 and pixels.max() <= 255


def test_reads_normal_dicom(tmp_path):
    path = _make_fake_dicom(tmp_path, photometric="MONOCHROME2")
    pixels, meta = read_medical_image(path)

    assert meta.source_format == "dicom"
    assert meta.was_inverted is False
    assert meta.pixel_spacing_mm == (0.14, 0.14)
    assert pixels[10:20, 10:20].mean() > pixels[0:5, 0:5].mean()


def test_monochrome1_dicom_gets_inverted(tmp_path):
    path = _make_fake_dicom(tmp_path, photometric="MONOCHROME1")
    pixels, meta = read_medical_image(path)

    assert meta.was_inverted is True
    assert pixels[10:20, 10:20].mean() > pixels[0:5, 0:5].mean()


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_medical_image("/nonexistent/path.png")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "fake.bmp"
    path.write_bytes(b"not a real image")
    with pytest.raises(ValueError):
        read_medical_image(path)