"""Slice thickness against series length, and what a 2.5D triplet really spans.

Two things are easy to get wrong here and are pinned deliberately:

* the frames a pass actually touches. Each centre pulls three frames, so
  counting one per centre understates the reading on any long series;
* which spacing to believe. Measured positions beat the tags, but only when
  filename order follows geometric order — otherwise the estimate is nonsense
  and the tags must win.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.slice_geometry_scan import (
    DEFAULT_CENTRES,
    frames_read,
    geometry_from_headers,
    scan,
    series_table,
    summarise,
)


def _write_series(
    directory: Path,
    n: int,
    *,
    thickness: float | None = 3.0,
    between: float | None = None,
    pitch: float = 4.0,
    acquisition: str | None = "2D",
    orientation=(0, 1, 0, 0, 0, -1),
    positions: list[float] | None = None,
) -> None:
    """Write header-valid single-frame MR instances with real geometry."""
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
        if thickness is not None:
            ds.SliceThickness = thickness
        if between is not None:
            ds.SpacingBetweenSlices = between
        if acquisition is not None:
            ds.MRAcquisitionType = acquisition
        offset = positions[index] if positions is not None else float(index) * pitch
        # The orientation used here has a slice normal along x, so the position
        # is moved along x for the projection to recover the pitch.
        ds.ImagePositionPatient = [float(offset), 0.0, 0.0]
        ds.ImageOrientationPatient = list(orientation)
        ds.InstanceNumber = index + 1
        ds.PixelSpacing = [0.3, 0.3]
        ds.Rows = ds.Columns = 4
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((4, 4), dtype=np.uint16).tobytes()
        pydicom.dcmwrite(directory / f"{index:03d}.dcm", ds, enforce_file_format=True)


def _dataset(root: Path, spec: dict[str, list[tuple[str, str, dict]]]) -> Path:
    """spec: {study: [(series, plane, kwargs for _write_series), ...]}"""
    rows = []
    for study, entries in spec.items():
        for name, plane, kwargs in entries:
            rows.append(
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": name,
                    "Anatomical_Plane": plane,
                    "Fluid_Sensitive": "Yes",
                    "Fat_Suppression": "No",
                }
            )
            _write_series(root / "train_series" / study / name, **kwargs)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "train_series.csv", index=False)
    return root


# --- what one pass actually reads ---------------------------------------------


def test_a_short_series_is_read_completely():
    assert frames_read(10, centres=32, gap=1) == 10


def test_a_long_series_reads_three_frames_per_centre():
    """The triplets are disjoint once the centres are far enough apart."""
    assert frames_read(320, centres=32, gap=1) == 96


def test_counting_one_frame_per_centre_understates_the_reading():
    """This is the correction to the earlier inventory, stated as a test."""
    assert frames_read(320, centres=32, gap=1) > min(320, 32)


def test_the_reading_is_capped_at_three_frames_per_centre():
    """However long the series, one pass touches at most three per centre."""
    assert frames_read(1000, centres=32, gap=1) == 96
    assert frames_read(1000, centres=32, gap=2) == 96


def test_an_empty_series_reads_nothing():
    assert frames_read(0) == 0


# --- which spacing to believe -------------------------------------------------


def test_the_pitch_is_measured_from_the_image_positions(tmp_path):
    folder = tmp_path / "s"
    _write_series(folder, 12, thickness=3.0, pitch=4.5)
    result = geometry_from_headers(folder)

    assert result["spacing_source"] == "image_positions"
    assert result["slice_spacing_mm"] == pytest.approx(4.5)


def test_only_the_sampled_headers_are_opened(tmp_path):
    folder = tmp_path / "s"
    _write_series(folder, 40)
    result = geometry_from_headers(folder, samples=3)

    assert result["frames"] == 40
    assert result["headers_read"] == 3


def test_unordered_positions_fall_back_to_the_tag(tmp_path):
    """If filename order is not geometric order, the span is not the pitch."""
    folder = tmp_path / "s"
    # Samples land on files 0, 2 and 4, so the disorder is put on file 2.
    _write_series(folder, 5, thickness=3.0, between=5.0, positions=[0, 4, 40, 8, 12])
    result = geometry_from_headers(folder)

    assert result["spacing_source"] == "SpacingBetweenSlices"
    assert result["slice_spacing_mm"] == pytest.approx(5.0)


def test_thickness_is_the_last_resort(tmp_path):
    folder = tmp_path / "s"
    _write_series(
        folder, 5, thickness=3.5, between=None, positions=[0.0] * 5
    )
    result = geometry_from_headers(folder)

    assert result["spacing_source"] == "SliceThickness"
    assert result["slice_spacing_mm"] == pytest.approx(3.5)


def test_a_series_with_no_geometry_at_all_is_marked_unavailable(tmp_path):
    folder = tmp_path / "s"
    _write_series(folder, 4, thickness=None, between=None, positions=[0.0] * 4)
    result = geometry_from_headers(folder)

    assert result["spacing_source"] == "unavailable"
    assert np.isnan(result["slice_spacing_mm"])


def test_an_implausible_pitch_is_refused(tmp_path):
    """A 400 mm step is a broken header, not a knee."""
    folder = tmp_path / "s"
    _write_series(folder, 4, thickness=3.0, pitch=400.0)
    result = geometry_from_headers(folder)

    assert result["spacing_source"] == "SliceThickness"


def test_the_acquisition_type_is_recorded(tmp_path):
    folder = tmp_path / "s"
    _write_series(folder, 8, acquisition="3D", pitch=0.6)
    assert geometry_from_headers(folder)["acquisition_type"] == "3D"


def test_a_missing_folder_is_reported_not_raised(tmp_path):
    result = geometry_from_headers(tmp_path / "nowhere")
    assert result["series_dir_found"] is False
    assert result["frames"] == 0


# --- the table ----------------------------------------------------------------


def test_the_triplet_span_is_twice_the_gap_times_the_spacing(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {"a": [("s1", "Sagittal", {"n": 10, "pitch": 4.0})]},
    )
    frame = series_table(data_root=root, workers=1)

    assert frame.loc[0, "slice_spacing_mm"] == pytest.approx(4.0)
    assert frame.loc[0, "triplet_span_mm"] == pytest.approx(8.0)


def test_a_wider_gap_widens_the_span(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {"a": [("s1", "Sagittal", {"n": 10, "pitch": 4.0})]},
    )
    frame = series_table(data_root=root, gap=2, workers=1)
    assert frame.loc[0, "triplet_span_mm"] == pytest.approx(16.0)


def test_a_series_with_no_recoverable_plane_is_left_out(tmp_path):
    """A blank plane is repaired from the headers first; only a series whose
    orientation is degenerate as well drops out."""
    root = _dataset(
        tmp_path / "d",
        {
            "a": [
                ("s1", "Sagittal", {"n": 6}),
                ("s2", "Oblique", {"n": 6, "orientation": (0, 0, 0, 0, 0, 0)}),
            ]
        },
    )
    assert list(series_table(data_root=root, workers=1)["SeriesInstanceUID"]) == ["s1"]


def test_the_threaded_and_serial_paths_agree(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {
            "a": [("s1", "Sagittal", {"n": 8, "pitch": 4.0})],
            "b": [("s2", "Coronal", {"n": 12, "pitch": 0.6, "acquisition": "3D"})],
        },
    )
    serial = series_table(data_root=root, workers=1)
    threaded = series_table(data_root=root, workers=4)
    pd.testing.assert_frame_equal(serial, threaded)


# --- the summary --------------------------------------------------------------


def _two_kinds(tmp_path) -> Path:
    """One thick 2D stack that fits, one thin 3D volume that does not."""
    return _dataset(
        tmp_path / "d",
        {
            "a": [("thick", "Sagittal", {"n": 30, "pitch": 4.0, "acquisition": "2D"})],
            "b": [
                (
                    "thin",
                    "Sagittal",
                    {"n": 200, "pitch": 0.6, "acquisition": "3D", "thickness": 0.6},
                )
            ],
        },
    )


def test_the_loss_is_attributed_to_the_acquisition_that_caused_it(tmp_path):
    root = _two_kinds(tmp_path)
    frame = series_table(data_root=root, workers=1)
    result = summarise(frame, centres=DEFAULT_CENTRES, gap=1)

    by_type = {row["band"]: row for row in result["acquisition_type"]}
    assert by_type["2D"]["frames_never_read"] == 0
    assert by_type["3D"]["frames_never_read"] == 200 - frames_read(200)
    assert by_type["3D"]["share_of_frames_never_read"] == pytest.approx(1.0)


def test_both_reading_figures_are_reported(tmp_path):
    root = _two_kinds(tmp_path)
    result = summarise(series_table(data_root=root, workers=1), centres=32, gap=1)
    loss = result["reading_loss"]

    assert loss["frames_total"] == 230
    assert loss["frames_read_centres_only"] == 30 + 32
    assert loss["frames_read_triplets"] == 30 + frames_read(200)
    assert loss["frames_never_read"] < loss["frames_never_read_centres_only"]


def test_the_bands_place_each_series(tmp_path):
    root = _two_kinds(tmp_path)
    result = summarise(series_table(data_root=root, workers=1), centres=32, gap=1)

    spans = {row["band"]: row["series"] for row in result["triplet_span_bands"]}
    assert spans["<2 mm"] == 1  # 2 x 0.6 mm
    assert spans["8-12 mm"] == 1  # 2 x 4.0 mm

    lengths = {row["band"]: row["series"] for row in result["length_bands"]}
    assert lengths["<=32"] == 1
    assert lengths[">160"] == 1


def test_a_series_with_no_spacing_lands_in_the_missing_band(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {
            "a": [
                (
                    "s1",
                    "Sagittal",
                    {
                        "n": 4,
                        "thickness": None,
                        "between": None,
                        "positions": [0.0] * 4,
                    },
                )
            ]
        },
    )
    result = summarise(series_table(data_root=root, workers=1), centres=32, gap=1)

    bands = {row["band"]: row["series"] for row in result["triplet_span_bands"]}
    assert bands["<missing>"] == 1
    assert result["series_with_a_known_spacing"] == 0


def test_long_and_thin_shows_up_as_a_negative_correlation(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {
            "a": [
                ("s1", "Sagittal", {"n": 20, "pitch": 5.0}),
                ("s2", "Sagittal", {"n": 40, "pitch": 3.0}),
                ("s3", "Sagittal", {"n": 80, "pitch": 1.5}),
                ("s4", "Sagittal", {"n": 160, "pitch": 0.6}),
            ]
        },
    )
    result = summarise(series_table(data_root=root, workers=1), centres=32, gap=1)
    assert result["spearman_frames_vs_spacing"] == pytest.approx(-1.0)


# --- output -------------------------------------------------------------------


def test_the_result_and_the_table_are_written(tmp_path):
    root = _two_kinds(tmp_path)
    out_csv = tmp_path / "out" / "series.csv"
    out_json = tmp_path / "out" / "summary.json"

    result = scan(data_root=root, workers=1, out_csv=out_csv, out_json=out_json)

    assert json.loads(out_json.read_text()) == result
    assert len(pd.read_csv(out_csv)) == 2
    assert result["data_root"] == str(root.resolve())


def test_max_series_limits_the_work(tmp_path):
    root = _two_kinds(tmp_path)
    assert scan(data_root=root, workers=1, max_series=1)["series"] == 1
