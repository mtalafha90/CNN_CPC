"""Tests for the exact B13 slice-exposure audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.slice_audit import (
    audit_slice_coverage,
    count_series_slices,
    format_summary,
    sampling_exposure,
    summarise_coverage,
)


def _write_series(
    directory: Path,
    n: int,
    *,
    thickness: float = 3.0,
    spacing: float = 4.0,
    orientation=(0, 1, 0, 0, 0, -1),
    position_axis: int = 0,
):
    """Write header-valid single-frame MR DICOM instances with real geometry."""
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
        position = [0.0, 0.0, 0.0]
        position[position_axis] = float(index) * spacing
        ds.ImagePositionPatient = position
        ds.ImageOrientationPatient = list(orientation)
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


def test_spacing_uses_orientation_projected_position_not_ipp_z(tmp_path: Path):
    """Sagittal-like geometry varies in X while Z can remain constant."""
    _write_series(
        tmp_path / "s",
        n=10,
        spacing=4.5,
        orientation=(0, 1, 0, 0, 0, -1),  # normal points along X
        position_axis=0,
    )
    info = count_series_slices(tmp_path / "s")
    assert info["spacing"] == pytest.approx(4.5, abs=1e-6)
    assert info["spacing_source"] == "orientation_projected_IPP"


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
    ds.SpacingBetweenSlices = 3.5
    ds.Rows = ds.Columns = 4
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((28, 4, 4), dtype=np.uint16).tobytes()
    pydicom.dcmwrite(directory / "a.dcm", ds, enforce_file_format=True)

    info = count_series_slices(directory)
    assert info["n_slices"] == 28
    assert info["spacing"] == pytest.approx(3.5)


# --- Exact B13 sampling ----------------------------------------------------


def test_40_slice_series_is_fully_exposed_by_eval_tta_union():
    """The old 16/40=40% proxy badly understated the real 2.5D TTA exposure."""
    exposure = sampling_exposure(40)
    assert exposure["center_positions_per_view"] == 16
    assert exposure["triplet_references_per_view"] == 48
    assert exposure["eval_unique_slices"] == 40
    assert exposure["eval_fraction_seen"] == pytest.approx(1.0)
    assert exposure["eval_max_unsampled_run_slices"] == 0


def test_long_series_can_still_have_real_multi_slice_gaps():
    exposure = sampling_exposure(120)
    assert exposure["eval_fraction_seen"] < 0.70
    assert exposure["eval_max_unsampled_run_slices"] >= 3


def test_training_expected_exposure_accounts_for_triplets_gap_and_jitter():
    exposure = sampling_exposure(40)
    # One random training view sees substantially more than the 16 center slices,
    # but unlike the evaluation TTA union it does not necessarily see everything.
    assert exposure["train_expected_fraction_per_view"] > 0.70
    assert exposure["train_expected_fraction_per_view"] < 1.0
    assert exposure["train_possible_fraction"] == pytest.approx(1.0)


def _frame(n_frames):
    rows = []
    for i, n in enumerate(n_frames):
        exposure = sampling_exposure(int(n))
        rows.append(
            {
                "StudyInstanceUID": f"s{i}",
                "SeriesInstanceUID": f"x{i}",
                "plane": "Sagittal",
                "found": True,
                "n_slices": int(n),
                "slice_thickness": 3.0,
                "spacing": 4.0,
                **exposure,
            }
        )
    return pd.DataFrame(rows)


def test_summary_reports_actual_eval_exposure_not_center_fraction():
    summary = summarise_coverage(_frame([40] * 5))
    assert summary["eval_fraction_seen"]["median"] == pytest.approx(1.0)
    assert summary["series_with_complete_eval_exposure_fraction"] == pytest.approx(1.0)
    assert "not supported" in format_summary(summary)


def test_summary_flags_material_long_series_gaps_without_target_tuning():
    summary = summarise_coverage(_frame([120] * 5))
    text = format_summary(summary)
    assert summary["series_with_eval_unsampled_run_ge_3_fraction"] == pytest.approx(1.0)
    assert "plausible global bottleneck" in text
    assert "target-wise" in text


def test_unreadable_series_are_excluded_from_summary():
    frame = _frame([40, 40])
    missing = frame.iloc[[0]].copy()
    missing["found"] = False
    missing["n_slices"] = 0
    combined = pd.concat([missing, frame.iloc[[1]]], ignore_index=True)
    summary = summarise_coverage(combined)
    assert summary["n_series_audited"] == 2
    assert summary["n_series_readable"] == 1


def test_end_to_end_audit_over_small_tree(tmp_path: Path):
    for study, n in (("study0", 40), ("study1", 120)):
        _write_series(tmp_path / "train_series" / study / "ser", n=n)
    series = pd.DataFrame(
        {
            "StudyInstanceUID": ["study0", "study1"],
            "SeriesInstanceUID": ["ser", "ser"],
            "Anatomical_Plane": ["Sagittal", "Coronal"],
        }
    )

    frame, summary = audit_slice_coverage(
        series,
        tmp_path,
        split="train",
        workers=1,
    )
    assert set(frame["n_slices"]) == {40, 120}
    assert summary["n_series_readable"] == 2
    assert summary["eval_fraction_seen"]["min"] < 1.0
    assert summary["eval_fraction_seen"]["max"] == pytest.approx(1.0)
