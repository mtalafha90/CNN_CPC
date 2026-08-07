"""Tests for reading awkward DICOM series layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rsna_knee.dicom import read_dicom_series

pytest.importorskip("torch", reason="rsna_knee.dicom imports torch")


def _dataset(rows: int = 8, cols: int = 8, frames: int | None = None, value: int = 1):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    if frames is None:
        ds.PixelData = np.full((rows, cols), value, dtype=np.uint16).tobytes()
    else:
        ds.NumberOfFrames = frames
        stack = np.stack(
            [np.full((rows, cols), i + 1, dtype=np.uint16) for i in range(frames)]
        )
        ds.PixelData = stack.tobytes()
    return ds


def _write(ds, path: Path):
    import pydicom

    path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def test_reads_files_without_a_dcm_suffix(tmp_path: Path):
    for index in range(3):
        ds = _dataset(value=index + 1)
        ds.ImagePositionPatient = [0.0, float(index * 4), 0.0]
        ds.InstanceNumber = index + 1
        _write(ds, tmp_path / f"IM_{index:04d}")
    assert read_dicom_series(tmp_path).shape == (3, 8, 8)


def test_reads_mixed_supported_suffixes_in_one_series(tmp_path: Path):
    suffixes = [".dcm", "", ".ima"]
    for index, suffix in enumerate(suffixes):
        ds = _dataset(value=index + 1)
        ds.ImagePositionPatient = [0.0, float(index * 4), 0.0]
        ds.InstanceNumber = index + 1
        _write(ds, tmp_path / f"IM_{index:04d}{suffix}")
    assert read_dicom_series(tmp_path).shape == (3, 8, 8)


def test_reads_an_enhanced_multiframe_instance(tmp_path: Path):
    _write(_dataset(frames=5), tmp_path / "multiframe.dcm")
    volume = read_dicom_series(tmp_path)
    assert volume.shape == (5, 8, 8)
    assert volume[0].mean() < volume[-1].mean()


def test_mixed_slice_sizes_are_normalised(tmp_path: Path):
    big = _dataset(rows=16, cols=16, value=2)
    big.ImagePositionPatient = [0.0, 0.0, 0.0]
    _write(big, tmp_path / "a.dcm")
    small = _dataset(rows=8, cols=8, value=3)
    small.ImagePositionPatient = [0.0, 4.0, 0.0]
    _write(small, tmp_path / "b.dcm")
    assert read_dicom_series(tmp_path).shape == (2, 16, 16)


def test_monochrome1_is_inverted(tmp_path: Path):
    ds = _dataset(value=5)
    ds.PhotometricInterpretation = "MONOCHROME1"
    pixels = np.zeros((8, 8), dtype=np.uint16)
    pixels[0, 0] = 100
    ds.PixelData = pixels.tobytes()
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    _write(ds, tmp_path / "a.dcm")
    volume = read_dicom_series(tmp_path)
    assert volume[0, 0, 0] == 0
    assert volume[0, 1, 1] == 100


def test_empty_directory_raises(tmp_path: Path):
    with pytest.raises(RuntimeError):
        read_dicom_series(tmp_path)
