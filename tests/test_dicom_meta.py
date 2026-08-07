"""Tests for DICOM-derived series metadata and the CSV fallbacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data import (
    backfill_series_metadata,
    coerce_bool,
    load_series_csv,
    normalise_plane,
    select_series,
)
from rsna_knee.dicom_meta import (
    is_fluid_sensitive,
    plane_from_orientation,
    read_series_metadata,
    weighting_from_parameters,
)

# Direction cosines as real scanners write them for each acquisition plane.
SAGITTAL = [0, 1, 0, 0, 0, -1]
CORONAL = [1, 0, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    "orientation,expected",
    [(SAGITTAL, "Sagittal"), (CORONAL, "Coronal"), (AXIAL, "Axial")],
)
def test_plane_from_orientation(orientation, expected):
    assert plane_from_orientation(orientation) == expected


def test_plane_detection_tolerates_oblique_acquisition():
    """Knee series are often angled slightly; a tilted sagittal is still sagittal."""
    assert plane_from_orientation([0.05, 0.99, 0.0, 0.0, 0.0, -1.0]) == "Sagittal"


@pytest.mark.parametrize("bad", [None, [1, 0, 0], [0, 0, 0, 0, 0, 0]])
def test_plane_detection_returns_none_when_undecidable(bad):
    """None, not a guess: the caller must be able to keep the CSV value."""
    assert plane_from_orientation(bad) is None


@pytest.mark.parametrize(
    "te,tr,expected",
    [(10.0, 600.0, "t1"), (30.0, 3000.0, "pd"), (80.0, 4000.0, "t2")],
)
def test_weighting_from_timings(te, tr, expected):
    assert weighting_from_parameters(te, tr)[0] == expected


def test_short_inversion_time_is_stir():
    weighting, fat = weighting_from_parameters(30.0, 4000.0, inversion_time=150.0)
    assert weighting == "stir"
    assert fat


def test_fat_suppression_read_from_scan_options():
    assert weighting_from_parameters(30.0, 3000.0, scan_options="FS")[1]
    assert not weighting_from_parameters(30.0, 3000.0, scan_options="")[1]


def test_missing_timings_give_unknown():
    assert weighting_from_parameters(None, None)[0] == "unknown"


def test_fluid_sensitive_mapping():
    assert is_fluid_sensitive("t2") and is_fluid_sensitive("pd") and is_fluid_sensitive("stir")
    assert not is_fluid_sensitive("t1")


# --- The boolean coercion bug this replaces --------------------------------


def test_coerce_bool_handles_missing_values():
    """NaN must be False. `astype(bool)` turns it into True."""
    values = pd.Series([True, False, np.nan])
    assert coerce_bool(values).tolist() == [True, False, False]


def test_coerce_bool_handles_string_flags():
    """The string "False" is truthy in Python; it must not become True here."""
    values = pd.Series(["True", "False", "yes", "no", ""])
    assert coerce_bool(values).tolist() == [True, False, True, False, False]


def test_coerce_bool_handles_numeric_flags():
    assert coerce_bool(pd.Series([1, 0, 1.0, np.nan])).tolist() == [True, False, True, False]


def test_normalise_plane_maps_abbreviations_and_blanks():
    values = pd.Series(["SAG", "coronal", "Ax", None, "nonsense"])
    assert normalise_plane(values).tolist() == ["Sagittal", "Coronal", "Axial", "", ""]


def test_load_series_csv_does_not_mark_missing_flags_as_true(tmp_path: Path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s1"],
            "SeriesInstanceUID": ["a", "b"],
            "Fluid_Sensitive": ["False", np.nan],
            "Fat_Suppression": ["True", np.nan],
            "Anatomical_Plane": ["SAG", ""],
        }
    ).to_csv(path, index=False)

    df = load_series_csv(path)

    assert df["Fluid_Sensitive"].tolist() == [False, False]
    assert df["Fat_Suppression"].tolist() == [True, False]
    assert df["Anatomical_Plane"].tolist() == ["Sagittal", ""]


def test_select_series_routes_fluid_and_structural_correctly():
    """A structural series must not win the fluid slot once flags parse properly."""
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s1"],
            "SeriesInstanceUID": ["fluid", "structural"],
            "Fluid_Sensitive": [True, False],
            "Fat_Suppression": [True, False],
            "Anatomical_Plane": ["Sagittal", "Sagittal"],
        }
    )

    routed = select_series(df, "s1", mode="dual")

    assert routed["sagittal_fluid"] == "fluid"
    assert routed["sagittal_structural"] == "structural"


# --- Backfill from DICOM headers -------------------------------------------


def _write_series(directory: Path, orientation, te, tr, options="", n=2):
    """Write a tiny but valid DICOM series."""
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
        ds.ImageOrientationPatient = orientation
        ds.ImagePositionPatient = [0.0, 0.0, float(index * 4)]
        ds.InstanceNumber = index + 1
        ds.EchoTime = te
        ds.RepetitionTime = tr
        ds.ScanOptions = options
        ds.Rows = ds.Columns = 8
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.full((8, 8), index + 1, dtype=np.uint16).tobytes()
        pydicom.dcmwrite(directory / f"{index:03d}.dcm", ds, enforce_file_format=True)


def test_read_series_metadata_from_headers(tmp_path: Path):
    _write_series(tmp_path / "series", CORONAL, te=80.0, tr=4000.0, options="FS")

    metadata = read_series_metadata(tmp_path / "series")

    assert metadata["Anatomical_Plane"] == "Coronal"
    assert metadata["weighting"] == "t2"
    assert metadata["Fluid_Sensitive"] is True
    assert metadata["Fat_Suppression"] is True


def test_read_series_metadata_on_missing_directory():
    assert read_series_metadata("/nonexistent/path")["Anatomical_Plane"] is None


def test_backfill_repairs_blank_plane(tmp_path: Path):
    """A blank plane hides the series from routing; backfill should recover it."""
    study = "study1"
    series = "series1"
    _write_series(tmp_path / "train_series" / study / series, SAGITTAL, te=30.0, tr=3000.0)

    df = pd.DataFrame(
        {
            "StudyInstanceUID": [study],
            "SeriesInstanceUID": [series],
            "Fluid_Sensitive": [False],
            "Fat_Suppression": [False],
            "Anatomical_Plane": [""],
        }
    )
    assert select_series(df, study)["sagittal"] is None  # invisible before repair

    repaired, stats = backfill_series_metadata(df, tmp_path, split="train")

    assert stats == {"missing": 1, "inspected": 1, "repaired": 1}
    assert repaired.loc[0, "Anatomical_Plane"] == "Sagittal"
    assert bool(repaired.loc[0, "Fluid_Sensitive"]) is True  # PD is fluid sensitive
    assert select_series(repaired, study)["sagittal"] == series


def test_backfill_leaves_populated_rows_untouched(tmp_path: Path):
    """The CSV is authoritative wherever it is filled in."""
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1"],
            "SeriesInstanceUID": ["a"],
            "Fluid_Sensitive": [True],
            "Fat_Suppression": [True],
            "Anatomical_Plane": ["Axial"],
        }
    )

    repaired, stats = backfill_series_metadata(df, tmp_path, split="train")

    assert stats["missing"] == 0
    assert repaired.equals(df)
