from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data import backfill_series_metadata, coerce_bool, load_series_csv, normalise_plane, select_series
from rsna_knee.dicom_meta import is_fluid_sensitive, plane_from_orientation, read_series_metadata, weighting_from_parameters

SAGITTAL = [0, 1, 0, 0, 0, -1]
CORONAL = [1, 0, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize("orientation,expected", [(SAGITTAL,"Sagittal"),(CORONAL,"Coronal"),(AXIAL,"Axial")])
def test_plane_from_orientation(orientation, expected):
    assert plane_from_orientation(orientation) == expected


def test_plane_detection_tolerates_oblique_acquisition():
    assert plane_from_orientation([0.05, 0.99, 0.0, 0.0, 0.0, -1.0]) == "Sagittal"


@pytest.mark.parametrize("bad", [None, [1,0,0], [0,0,0,0,0,0]])
def test_plane_detection_returns_none_when_undecidable(bad):
    assert plane_from_orientation(bad) is None


@pytest.mark.parametrize("te,tr,expected", [(10.0,600.0,"t1"),(30.0,3000.0,"pd"),(80.0,4000.0,"t2")])
def test_weighting_from_timings(te, tr, expected):
    assert weighting_from_parameters(te, tr)[0] == expected


def test_short_inversion_time_is_stir():
    weighting, fat = weighting_from_parameters(30.0, 4000.0, inversion_time=150.0)
    assert weighting == "stir" and fat


def test_fluid_sensitive_mapping():
    assert all(is_fluid_sensitive(x) for x in ("t2","pd","stir"))
    assert not is_fluid_sensitive("t1")


def test_coerce_bool_conservative_public_default():
    values = pd.Series([True, False, np.nan, "yes", "no"])
    assert coerce_bool(values).tolist() == [True, False, False, True, False]


def test_coerce_bool_can_preserve_unknown():
    values = pd.Series(["True", "False", np.nan, "nonsense"])
    result = coerce_bool(values, preserve_unknown=True)
    assert bool(result.iloc[0]) is True and bool(result.iloc[1]) is False
    assert pd.isna(result.iloc[2]) and pd.isna(result.iloc[3])


def test_normalise_plane_maps_abbreviations_and_blanks():
    values = pd.Series(["SAG", "coronal", "Ax", None, "nonsense"])
    assert normalise_plane(values).tolist() == ["Sagittal", "Coronal", "Axial", "", ""]


def test_load_series_csv_preserves_missing_flags_for_backfill(tmp_path: Path):
    path = tmp_path / "series.csv"
    pd.DataFrame({
        "StudyInstanceUID": ["s1", "s1"],
        "SeriesInstanceUID": ["a", "b"],
        "Fluid_Sensitive": ["False", np.nan],
        "Fat_Suppression": ["True", np.nan],
        "Anatomical_Plane": ["SAG", ""],
    }).to_csv(path, index=False)
    df = load_series_csv(path)
    assert bool(df.loc[0,"Fluid_Sensitive"]) is False
    assert bool(df.loc[0,"Fat_Suppression"]) is True
    assert pd.isna(df.loc[1,"Fluid_Sensitive"]) and pd.isna(df.loc[1,"Fat_Suppression"])
    assert df["Anatomical_Plane"].tolist() == ["Sagittal", ""]


def test_select_series_routes_fluid_and_structural_correctly():
    df = pd.DataFrame({
        "StudyInstanceUID": ["s1","s1"],
        "SeriesInstanceUID": ["fluid","structural"],
        "Fluid_Sensitive": [True,False],
        "Fat_Suppression": [True,False],
        "Anatomical_Plane": ["Sagittal","Sagittal"],
    })
    routed = select_series(df, "s1", mode="dual")
    assert routed["sagittal_fluid"] == "fluid"
    assert routed["sagittal_structural"] == "structural"


def _write_series(directory: Path, orientation, te, tr, options="", n=2):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    for index in range(n):
        ds = Dataset(); ds.file_meta = FileMetaDataset(); ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian; ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"; ds.file_meta.MediaStorageSOPInstanceUID = generate_uid(); ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID; ds.SeriesInstanceUID = series_uid; ds.Modality = "MR"; ds.ImageOrientationPatient = orientation; ds.ImagePositionPatient = [0.0,0.0,float(index*4)]; ds.InstanceNumber = index+1; ds.EchoTime = te; ds.RepetitionTime = tr; ds.ScanOptions = options; ds.Rows = ds.Columns = 8; ds.BitsAllocated = ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"; ds.PixelData = np.full((8,8),index+1,dtype=np.uint16).tobytes()
        pydicom.dcmwrite(directory / f"{index:03d}.dcm", ds, enforce_file_format=True)


def test_read_series_metadata_from_headers(tmp_path: Path):
    _write_series(tmp_path/"series", CORONAL, te=80.0, tr=4000.0, options="FS")
    metadata = read_series_metadata(tmp_path/"series")
    assert metadata["Anatomical_Plane"] == "Coronal"
    assert metadata["weighting"] == "t2"
    assert metadata["Fluid_Sensitive"] is True and metadata["Fat_Suppression"] is True


def test_backfill_repairs_each_missing_field_independently(tmp_path: Path):
    study, series = "study1", "series1"
    _write_series(tmp_path/"train_series"/study/series, SAGITTAL, te=30.0, tr=3000.0, options="FS")
    df = pd.DataFrame({
        "StudyInstanceUID": [study],
        "SeriesInstanceUID": [series],
        "Fluid_Sensitive": pd.Series([pd.NA], dtype="boolean"),
        "Fat_Suppression": pd.Series([pd.NA], dtype="boolean"),
        "Anatomical_Plane": [""],
    })
    repaired, stats = backfill_series_metadata(df, tmp_path, split="train")
    assert repaired.loc[0,"Anatomical_Plane"] == "Sagittal"
    assert bool(repaired.loc[0,"Fluid_Sensitive"]) is True
    assert bool(repaired.loc[0,"Fat_Suppression"]) is True
    assert stats["repaired_plane"] == stats["repaired_fluid"] == stats["repaired_fat_suppression"] == 1


def test_backfill_repairs_missing_flag_even_when_plane_is_present(tmp_path: Path):
    study, series = "study1", "series1"
    _write_series(tmp_path/"train_series"/study/series, AXIAL, te=80.0, tr=4000.0, options="FS")
    df = pd.DataFrame({
        "StudyInstanceUID": [study],
        "SeriesInstanceUID": [series],
        "Fluid_Sensitive": pd.Series([pd.NA], dtype="boolean"),
        "Fat_Suppression": pd.Series([True], dtype="boolean"),
        "Anatomical_Plane": ["Axial"],
    })
    repaired, stats = backfill_series_metadata(df, tmp_path, split="train")
    assert bool(repaired.loc[0,"Fluid_Sensitive"]) is True
    assert stats["repaired_fluid"] == 1
    assert stats["repaired_plane"] == 0
