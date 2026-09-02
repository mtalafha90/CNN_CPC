"""Collecting the 58 expert studies into one folder.

These are the only hard labels the project has. An export that quietly drops a
study, or that half-overwrites an earlier one, is worse than no export, so the
counts and the refusals are pinned here rather than trusted.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.export_gold_studies import (
    COPY,
    HARDLINK,
    SYMLINK,
    EXPECTED_GOLD_STUDIES,
    export,
    gold_series,
    gold_studies,
    plan,
)


def _dataset(root, *, gold=EXPECTED_GOLD_STUDIES, report_only=2, series_per_study=2, slices=3):
    """A miniature competition data root: tables plus train_series/<study>/<series>."""
    rows, series_rows = [], []
    for index in range(gold + report_only):
        uid = f"study{index:04d}"
        is_gold = index < gold
        row = {"StudyInstanceUID": uid, "Report": f"report {index}"}
        for position, target in enumerate(TARGETS):
            row[target] = float((index + position) % 2) if is_gold else None
        rows.append(row)
        for which in range(series_per_study):
            name = f"{uid}_series{which}"
            series_rows.append(
                {
                    "StudyInstanceUID": uid,
                    "SeriesInstanceUID": name,
                    "Fluid_Sensitive": "Yes",
                    "Fat_Suppression": "No",
                    "Anatomical_Plane": "Sagittal",
                }
            )
            folder = root / "train_series" / uid / name
            folder.mkdir(parents=True, exist_ok=True)
            for slice_index in range(slices):
                (folder / f"{slice_index:03d}.dcm").write_bytes(
                    bytes([index % 251, which, slice_index])
                )

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    return root


@pytest.fixture
def data_root(tmp_path):
    return _dataset(tmp_path / "data")


# --- selecting ---------------------------------------------------------------


def test_exactly_the_expert_studies_are_selected(data_root):
    selected = gold_studies(data_root)

    assert len(selected) == EXPECTED_GOLD_STUDIES
    assert selected["StudyInstanceUID"].tolist()[0] == "study0000"
    assert "study0058" not in selected["StudyInstanceUID"].tolist()


def test_every_original_column_survives(data_root):
    selected = gold_studies(data_root)

    assert "Report" in selected.columns
    for target in TARGETS:
        assert target in selected.columns


def test_a_train_csv_with_the_wrong_count_is_refused(tmp_path):
    root = _dataset(tmp_path / "small", gold=3)
    with pytest.raises(ValueError, match="not 58"):
        gold_studies(root)


def test_series_rows_are_taken_unparsed(data_root):
    """load_series_csv rewrites plane and fluid; a copy must stay the source."""
    uids = gold_studies(data_root)["StudyInstanceUID"].astype(str).tolist()
    selected = gold_series(data_root, uids)

    assert selected["Fluid_Sensitive"].iloc[0] == "Yes"
    assert selected["Anatomical_Plane"].iloc[0] == "Sagittal"
    assert len(selected) == EXPECTED_GOLD_STUDIES * 2


def test_a_study_with_no_series_row_is_refused(data_root):
    frame = pd.read_csv(data_root / "train_series.csv")
    frame = frame.loc[frame["StudyInstanceUID"] != "study0000"]
    frame.to_csv(data_root / "train_series.csv", index=False)

    uids = gold_studies(data_root)["StudyInstanceUID"].astype(str).tolist()
    with pytest.raises(ValueError, match="no series row"):
        gold_series(data_root, uids)


# --- planning ----------------------------------------------------------------


def test_the_plan_totals_every_file_and_its_size(data_root):
    prepared = plan(data_root=data_root)

    assert prepared["file_count"] == EXPECTED_GOLD_STUDIES * 2 * 3
    assert prepared["bytes"] == prepared["file_count"] * 3
    assert prepared["series_missing"] == []


def test_a_series_folder_that_is_absent_is_reported_not_raised(data_root):
    """One unreadable series must not forfeit the other 115."""
    import shutil

    shutil.rmtree(data_root / "train_series" / "study0000" / "study0000_series0")
    prepared = plan(data_root=data_root)

    assert prepared["series_missing"] == ["study0000/study0000_series0"]
    assert len(prepared["items"]) == EXPECTED_GOLD_STUDIES * 2 - 1


# --- exporting ---------------------------------------------------------------


def test_a_dry_run_writes_nothing(data_root, tmp_path):
    out = tmp_path / "gold"
    summary = export(data_root=data_root, out_root=out, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["files"] == EXPECTED_GOLD_STUDIES * 2 * 3
    assert not out.exists()


def test_the_export_mirrors_the_competition_layout(data_root, tmp_path):
    out = tmp_path / "gold"
    export(data_root=data_root, out_root=out)

    assert (out / "train.csv").is_file()
    assert (out / "train_series.csv").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "train_series" / "study0000" / "study0000_series0" / "000.dcm").is_file()


def test_the_exported_tables_hold_only_the_expert_studies(data_root, tmp_path):
    out = tmp_path / "gold"
    export(data_root=data_root, out_root=out)

    written = pd.read_csv(out / "train.csv")
    series = pd.read_csv(out / "train_series.csv")

    assert len(written) == EXPECTED_GOLD_STUDIES
    assert len(series) == EXPECTED_GOLD_STUDIES * 2
    assert "study0058" not in written["StudyInstanceUID"].astype(str).tolist()


def test_the_copied_pixels_are_identical(data_root, tmp_path):
    out = tmp_path / "gold"
    export(data_root=data_root, out_root=out)

    source = data_root / "train_series" / "study0007" / "study0007_series1" / "002.dcm"
    target = out / "train_series" / "study0007" / "study0007_series1" / "002.dcm"
    assert target.read_bytes() == source.read_bytes()


def test_the_manifest_records_what_happened(data_root, tmp_path):
    out = tmp_path / "gold"
    summary = export(data_root=data_root, out_root=out)
    written = json.loads((out / "manifest.json").read_text())

    assert written == summary
    assert written["studies"] == EXPECTED_GOLD_STUDIES
    assert written["files_written"] == written["files"]
    assert written["mode"] == COPY


@pytest.mark.parametrize("mode", [COPY, HARDLINK, SYMLINK])
def test_every_mode_produces_readable_pixels(data_root, tmp_path, mode):
    out = tmp_path / f"gold_{mode}"
    export(data_root=data_root, out_root=out, mode=mode)

    target = out / "train_series" / "study0003" / "study0003_series0" / "001.dcm"
    source = data_root / "train_series" / "study0003" / "study0003_series0" / "001.dcm"
    assert target.read_bytes() == source.read_bytes()


def test_a_copy_survives_the_source_being_deleted(data_root, tmp_path):
    """Which is the whole reason copy is the default."""
    import shutil

    out = tmp_path / "gold"
    export(data_root=data_root, out_root=out, mode=COPY)
    shutil.rmtree(data_root / "train_series")

    target = out / "train_series" / "study0001" / "study0001_series0" / "000.dcm"
    assert len(target.read_bytes()) == 3


def test_an_unknown_mode_is_refused(data_root, tmp_path):
    with pytest.raises(ValueError, match="mode must be"):
        export(data_root=data_root, out_root=tmp_path / "gold", mode="move")


# --- refusals ----------------------------------------------------------------


def test_a_directory_that_already_holds_files_is_refused(data_root, tmp_path):
    out = tmp_path / "gold"
    out.mkdir()
    (out / "something.txt").write_text("do not clobber me")

    with pytest.raises(ValueError, match="already holds files"):
        export(data_root=data_root, out_root=out)


def test_an_empty_directory_is_accepted(data_root, tmp_path):
    out = tmp_path / "gold"
    out.mkdir()
    assert export(data_root=data_root, out_root=out)["files_written"] > 0


def test_writing_over_the_data_root_is_refused(data_root):
    with pytest.raises(ValueError, match="over the data root"):
        export(data_root=data_root, out_root=data_root)


def test_writing_into_a_source_series_folder_is_refused(data_root):
    inside = data_root / "train_series" / "study0000" / "study0000_series0"
    with pytest.raises(ValueError, match="source series folder"):
        export(data_root=data_root, out_root=inside)


def test_writing_beside_the_source_inside_the_data_root_is_allowed(data_root):
    """<root>/gold_58 is exactly what was asked for."""
    summary = export(data_root=data_root, out_root=data_root / "gold_58")
    assert summary["files_written"] == EXPECTED_GOLD_STUDIES * 2 * 3


def test_the_source_is_never_modified(data_root, tmp_path):
    before = sorted(
        (str(p.relative_to(data_root)), p.stat().st_size)
        for p in data_root.rglob("*")
        if p.is_file()
    )
    export(data_root=data_root, out_root=tmp_path / "gold")
    after = sorted(
        (str(p.relative_to(data_root)), p.stat().st_size)
        for p in data_root.rglob("*")
        if p.is_file()
    )
    assert before == after
