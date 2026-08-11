"""Tests for the label-free slice-coverage audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.slice_audit import (
    audit_slice_coverage,
    count_series_slices,
    format_summary,
    summarise_coverage,
)


def _write_series(directory: Path, n: int, thickness: float = 3.0, spacing: float = 4.0):
    """Write `n` header-only DICOM instances with real geometry."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    for index in range(n):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.SliceThickness = thickness
        ds.ImagePositionPatient = [0.0, 0.0, float(index) * spacing]
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
        ds.InstanceNumber = index + 1
        ds.Rows = ds.Columns = 4
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((4, 4), dtype=np.uint16).tobytes()
        pydicom.dcmwrite(directory / f"{index:03d}.dcm", ds, enforce_file_format=True)


def test_counts_slices_in_a_series(tmp_path: Path):
    _write_series(tmp_path / "s", n=37)
    info = count_series_slices(tmp_path / "s")
    assert info["n_slices"] == 37


def test_derives_spacing_from_positions_when_tag_absent(tmp_path: Path):
    """SpacingBetweenSlices is often missing; positions still give the answer."""
    _write_series(tmp_path / "s", n=10, spacing=4.5)
    info = count_series_slices(tmp_path / "s")
    assert info["spacing"] == pytest.approx(4.5, abs=1e-6)


def test_missing_directory_is_not_an_error():
    assert count_series_slices("/no/such/path")["n_slices"] == 0


def test_multiframe_instance_counts_all_frames(tmp_path: Path):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory = tmp_path / "mf"
    directory.mkdir()
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.NumberOfFrames = 28
    ds.Rows = ds.Columns = 4
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((28, 4, 4), dtype=np.uint16).tobytes()
    pydicom.dcmwrite(directory / "a.dcm", ds, enforce_file_format=True)

    assert count_series_slices(directory)["n_slices"] == 28


# --- Summary arithmetic ----------------------------------------------------


def _frame(counts, spacing=4.0, plane="Sagittal"):
    return pd.DataFrame(
        {
            "StudyInstanceUID": [f"s{i}" for i in range(len(counts))],
            "SeriesInstanceUID": [f"x{i}" for i in range(len(counts))],
            "plane": [plane] * len(counts),
            "found": [True] * len(counts),
            "n_slices": counts,
            "slice_thickness": [3.0] * len(counts),
            "spacing": [spacing] * len(counts),
        }
    )


def test_full_coverage_when_series_are_shorter_than_the_budget():
    summary = summarise_coverage(_frame([8, 10, 12]), n_sampled=16)
    assert summary["series_undersampled_fraction"] == 0.0
    assert summary["fraction_of_slices_seen"]["median"] == 1.0
    assert summary["sampling_stride_slices"]["median"] == 1.0
    assert "NOT limiting" in format_summary(summary)


def test_undersampling_is_detected_and_quantified():
    """40-slice series sampled at 16 positions skip roughly 3 of every 5 slices."""
    summary = summarise_coverage(_frame([40] * 10), n_sampled=16)

    assert summary["series_undersampled_fraction"] == 1.0
    assert summary["fraction_of_slices_seen"]["median"] == pytest.approx(0.4)
    assert summary["sampling_stride_slices"]["median"] == pytest.approx(2.5)
    assert summary["gap_between_sampled_positions_mm"]["median"] == pytest.approx(10.0)


def test_severe_undersampling_flags_the_focal_lesion_risk():
    summary = summarise_coverage(_frame([48] * 5), n_sampled=16)
    text = format_summary(summary)
    assert "focal targets" in text
    assert "ACL" in text


def test_mild_undersampling_is_reported_as_mild():
    summary = summarise_coverage(_frame([24] * 5), n_sampled=16)
    assert "Mild skipping" in format_summary(summary)


def test_summary_reports_per_plane_breakdown():
    frame = pd.concat([_frame([40] * 3, plane="Sagittal"), _frame([20] * 2, plane="Axial")])
    summary = summarise_coverage(frame, n_sampled=16)
    assert summary["by_plane"]["Sagittal"]["median_slices"] == 40
    assert summary["by_plane"]["Axial"]["n_series"] == 2


def test_unreadable_series_are_excluded_not_counted_as_zero_slices():
    """A missing series must not drag the median slice count towards zero."""
    frame = _frame([40, 40, 40])
    frame.loc[0, "n_slices"] = 0
    frame.loc[0, "found"] = False

    summary = summarise_coverage(frame, n_sampled=16)

    assert summary["n_series_audited"] == 3
    assert summary["n_series_readable"] == 2
    assert summary["slices_per_series"]["median"] == 40


def test_empty_audit_is_reported_not_crashed():
    summary = summarise_coverage(_frame([]).astype({"n_slices": float}), n_sampled=16)
    assert summary["n_series_readable"] == 0
    assert "no readable series" in format_summary(summary)


def test_end_to_end_audit_over_a_small_tree(tmp_path: Path):
    for study, n in (("study0", 40), ("study1", 12)):
        _write_series(tmp_path / "train_series" / study / "ser", n=n)
    series = pd.DataFrame(
        {
            "StudyInstanceUID": ["study0", "study1"],
            "SeriesInstanceUID": ["ser", "ser"],
            "Anatomical_Plane": ["Sagittal", "Coronal"],
        }
    )

    frame, summary = audit_slice_coverage(series, tmp_path, split="train", n_sampled=16, workers=1)

    assert set(frame["n_slices"]) == {40, 12}
    assert summary["n_series_readable"] == 2
    assert summary["series_undersampled_fraction"] == pytest.approx(0.5)
