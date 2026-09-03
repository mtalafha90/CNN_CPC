"""How much of the dataset the loader never reads.

The number that matters is frames never read, and it is easy to get wrong in the
flattering direction: counting short series as losses, or counting every frame of
a long one as read. Both are pinned here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.data_inventory import inventory


def _dataset(root, series_spec):
    """series_spec: {study: [(series, plane, n_frames), ...]}"""
    rows, series_rows = [], []
    for study, entries in series_spec.items():
        row = {"StudyInstanceUID": study, "Report": "text"}
        for target in TARGETS:
            row[target] = None
        rows.append(row)
        for name, plane, frames in entries:
            series_rows.append(
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": name,
                    "Anatomical_Plane": plane,
                    "Fluid_Sensitive": "Yes",
                    "Fat_Suppression": "No",
                }
            )
            folder = root / "train_series" / study / name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(frames):
                (folder / f"{index:04d}.dcm").write_bytes(b"x")

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    return root


# --- plane eligibility --------------------------------------------------------


def test_an_unrecognised_plane_is_excluded_and_counted(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {"a": [("s1", "Sagittal", 10), ("s2", "Oblique", 10)]},
    )
    result = inventory(data_root=root)

    assert result["plane"]["recognised"] == 1
    assert result["plane"]["excluded_unknown_plane"] == 1


def test_a_study_whose_series_are_all_excluded_is_counted(tmp_path):
    root = _dataset(tmp_path / "d", {"a": [("s1", "Oblique", 10)]})
    assert inventory(data_root=root)["studies_with_zero_eligible_series"] == 1


def test_a_study_with_one_good_series_is_not(tmp_path):
    root = _dataset(
        tmp_path / "d", {"a": [("s1", "Oblique", 10), ("s2", "Axial", 10)]}
    )
    assert inventory(data_root=root)["studies_with_zero_eligible_series"] == 0


def test_the_plane_distribution_is_reported(tmp_path):
    root = _dataset(
        tmp_path / "d",
        {"a": [("s1", "Sagittal", 5), ("s2", "Sagittal", 5), ("s3", "Coronal", 5)]},
    )
    assert inventory(data_root=root)["plane"]["distribution"] == {
        "Sagittal": 2,
        "Coronal": 1,
    }


# --- what the loader reads ----------------------------------------------------


def test_a_short_series_is_read_completely(tmp_path):
    """Centres repeat on a short series; nothing is lost."""
    root = _dataset(tmp_path / "d", {"a": [("s1", "Sagittal", 10)]})
    slices = inventory(data_root=root, centres=32)["slices"]

    assert slices["frames_total"] == 10
    assert slices["frames_read"] == 10
    assert slices["frames_never_read"] == 0
    assert slices["series_shorter_than_centres"] == 1


def test_a_long_series_loses_its_excess(tmp_path):
    root = _dataset(tmp_path / "d", {"a": [("s1", "Sagittal", 100)]})
    slices = inventory(data_root=root, centres=32)["slices"]

    assert slices["frames_total"] == 100
    assert slices["frames_read"] == 32
    assert slices["frames_never_read"] == 68
    assert slices["series_longer_than_centres"] == 1


def test_a_series_of_exactly_the_centre_count_loses_nothing(tmp_path):
    root = _dataset(tmp_path / "d", {"a": [("s1", "Sagittal", 32)]})
    slices = inventory(data_root=root, centres=32)["slices"]

    assert slices["frames_never_read"] == 0
    assert slices["series_longer_than_centres"] == 0
    assert slices["series_shorter_than_centres"] == 0


def test_the_fraction_is_over_frames_not_over_series(tmp_path):
    """One long series can dominate many short ones, and should."""
    root = _dataset(
        tmp_path / "d",
        {"a": [("s1", "Sagittal", 320)] + [(f"s{i}", "Sagittal", 4) for i in range(2, 12)]},
    )
    slices = inventory(data_root=root, centres=32)["slices"]

    assert slices["frames_total"] == 360
    assert slices["frames_read"] == 32 + 40
    assert slices["fraction_never_read"] == pytest.approx(288 / 360)


def test_an_excluded_series_is_not_counted_in_the_slice_totals(tmp_path):
    """It never reaches the loader, so its frames are not a loss."""
    root = _dataset(
        tmp_path / "d", {"a": [("s1", "Sagittal", 10), ("s2", "Oblique", 500)]}
    )
    assert inventory(data_root=root)["slices"]["frames_total"] == 10


def test_the_centre_count_is_adjustable(tmp_path):
    root = _dataset(tmp_path / "d", {"a": [("s1", "Sagittal", 100)]})
    assert inventory(data_root=root, centres=16)["slices"]["frames_never_read"] == 84


# --- output -------------------------------------------------------------------


def test_a_missing_series_folder_is_reported_not_raised(tmp_path):
    root = _dataset(tmp_path / "d", {"a": [("s1", "Sagittal", 5)]})
    import shutil

    shutil.rmtree(root / "train_series" / "a" / "s1")
    result = inventory(data_root=root)

    assert result["series_folders_not_found_count"] == 1
    assert result["slices"]["frames_total"] == 0


def test_the_result_is_written_as_json(tmp_path):
    root = _dataset(tmp_path / "d", {"a": [("s1", "Sagittal", 5)]})
    out = tmp_path / "inventory.json"

    result = inventory(data_root=root, out_json=out)
    assert json.loads(out.read_text()) == result
