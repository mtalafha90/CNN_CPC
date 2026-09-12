"""Finding what a machine-to-machine copy left behind.

The property that matters, and the one an existence check cannot have: a series
directory that exists is not a series directory that is complete. An
interrupted transfer leaves 40 of 160 slices, every existence check passes, and
the model trains on a third of a knee without anything raising.

So these tests are mostly about truncation, not absence. Absence is the easy
half and `dicom_coverage` already finds it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.copy_audit import (
    COPY_AUDIT_VERSION,
    check_against_table,
    compare_manifests,
    format_check,
    format_comparison,
    rsync_list,
    scan_manifest,
    series_files,
)


def _tree(root, layout: dict, *, split: str = "train") -> None:
    """Build a data root: {study: {series: number_of_slices}}, plus the table."""
    rows = []
    for study, series_map in layout.items():
        for series, slices in series_map.items():
            rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series})
            if slices is None:
                continue  # named in the table, absent on disk
            directory = root / f"{split}_images" / study / series
            directory.mkdir(parents=True)
            for index in range(slices):
                (directory / f"{index:04d}.dcm").write_bytes(b"x" * 100)
    pd.DataFrame(rows).to_csv(root / f"{split}_series.csv", index=False)


# --- scanning ------------------------------------------------------------------


def test_the_scan_counts_files_and_bytes_per_series(tmp_path):
    _tree(tmp_path, {"studyA": {"s1": 3, "s2": 2}})
    manifest = scan_manifest(tmp_path)

    assert manifest["series"]["studyA/s1"] == {"files": 3, "bytes": 300, "present": True}
    assert manifest["totals"] == {
        "series_in_table": 2,
        "series_present": 2,
        "series_absent": 0,
        "files": 5,
        "bytes": 500,
    }


def test_a_series_named_by_the_table_but_absent_is_recorded_not_skipped(tmp_path):
    """Skipping it would let a short copy report a complete manifest."""
    _tree(tmp_path, {"studyA": {"s1": 3, "s2": None}})
    manifest = scan_manifest(tmp_path)

    assert manifest["series"]["studyA/s2"]["present"] is False
    assert manifest["totals"]["series_absent"] == 1


def test_the_table_drives_the_scan_not_the_directory_tree(tmp_path):
    """A stray directory must not pad the count into looking complete."""
    _tree(tmp_path, {"studyA": {"s1": 2}})
    stray = tmp_path / "train_images" / "studyA" / "not_in_table"
    stray.mkdir(parents=True)
    (stray / "0000.dcm").write_bytes(b"x")

    manifest = scan_manifest(tmp_path)
    assert set(manifest["series"]) == {"studyA/s1"}


def test_only_dicom_suffixes_are_counted(tmp_path):
    _tree(tmp_path, {"studyA": {"s1": 2}})
    (tmp_path / "train_images" / "studyA" / "s1" / "notes.txt").write_text("hello")

    assert scan_manifest(tmp_path)["series"]["studyA/s1"]["files"] == 2


def test_series_files_are_returned_in_a_stable_order(tmp_path):
    directory = tmp_path / "s"
    directory.mkdir()
    for name in ("0002.dcm", "0000.dcm", "0001.dcm"):
        (directory / name).write_bytes(b"x")

    assert [path.name for path in series_files(directory)] == [
        "0000.dcm", "0001.dcm", "0002.dcm"
    ]


# --- the comparison, which is the whole point ----------------------------------


def test_a_truncated_series_is_found_by_file_count(tmp_path):
    """The failure an existence check cannot see."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 160}})
    _tree(destination, {"studyA": {"s1": 40}})

    result = compare_manifests(scan_manifest(source), scan_manifest(destination))

    assert result["absent"] == []
    assert len(result["short"]) == 1
    assert result["short"][0]["files"] == [160, 40]


def test_a_transfer_cut_inside_one_file_is_found_by_bytes(tmp_path):
    """Right number of files, wrong number of bytes. A count-only check passes."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 3}})
    _tree(destination, {"studyA": {"s1": 3}})
    # Truncate the last file, as an interrupted copy would.
    (destination / "train_images" / "studyA" / "s1" / "0002.dcm").write_bytes(b"x" * 10)

    result = compare_manifests(scan_manifest(source), scan_manifest(destination))

    assert result["short"], "equal file counts hid a truncated file"
    assert result["short"][0]["bytes"] == [300, 210]


def test_a_whole_copy_reports_nothing_to_do(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 4, "s2": 2}})
    _tree(destination, {"studyA": {"s1": 4, "s2": 2}})

    result = compare_manifests(scan_manifest(source), scan_manifest(destination))

    assert result["absent"] == [] and result["short"] == []
    assert result["complete"] == 2
    assert result["bytes_missing"] == 0
    assert "arrived whole" in format_comparison(result)


def test_a_series_missing_at_the_source_is_not_blamed_on_the_copy(tmp_path):
    """Re-copying cannot fix what the source never had."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 2, "s2": None}})
    _tree(destination, {"studyA": {"s1": 2, "s2": None}})

    result = compare_manifests(scan_manifest(source), scan_manifest(destination))

    assert result["absent"] == []
    assert result["expected_series"] == 1, "only series the source holds are expected"


def test_the_bytes_missing_figure_adds_absent_and_short(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 4, "s2": 2}})
    _tree(destination, {"studyA": {"s1": 1, "s2": None}})

    result = compare_manifests(scan_manifest(source), scan_manifest(destination))

    # s1 is short by 300 bytes; s2 is absent and worth 200.
    assert result["bytes_missing"] == 500


def test_a_destination_series_that_is_larger_is_not_reported(tmp_path):
    """Only shortfalls matter; an extra file is not a failed copy."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 2}})
    _tree(destination, {"studyA": {"s1": 3}})

    result = compare_manifests(scan_manifest(source), scan_manifest(destination))
    assert result["short"] == [] and result["absent"] == []


# --- the list you actually feed to rsync ---------------------------------------


def test_the_rsync_list_names_directories_not_files():
    """A file path would copy one slice; the trailing slash copies the series."""
    result = {"absent": ["studyA/s2"], "short": [{"series": "studyA/s1"}]}
    paths = rsync_list(result, split="train")

    assert paths == ["train_images/studyA/s1/", "train_images/studyA/s2/"]
    assert all(path.endswith("/") for path in paths)


def test_the_rsync_list_covers_short_series_as_well_as_absent_ones():
    """Re-copying only the absent ones leaves the truncated ones truncated."""
    result = {"absent": ["a/1"], "short": [{"series": "b/2"}]}
    assert len(rsync_list(result)) == 2


def test_the_rsync_list_follows_the_layout_that_is_actually_used():
    result = {"absent": ["a/1"], "short": []}
    assert rsync_list(result, layout="{split}_series") == ["train_series/a/1/"]


def test_the_rsync_list_is_sorted_and_free_of_duplicates():
    result = {"absent": ["b/2", "a/1"], "short": [{"series": "a/1"}]}
    assert rsync_list(result) == ["train_images/a/1/", "train_images/b/2/"]


# --- the one-machine fallback, and its honesty ---------------------------------


def test_the_check_finds_series_the_table_names_and_the_disk_lacks(tmp_path):
    _tree(tmp_path, {"studyA": {"s1": 5, "s2": None}})
    result = check_against_table(tmp_path)

    assert result["absent"] == ["studyA/s2"]


def test_the_check_says_it_cannot_see_truncation(tmp_path):
    """A clean result here must not read as proof the copy finished."""
    _tree(tmp_path, {"studyA": {"s1": 5}})
    printed = format_check(check_against_table(tmp_path))

    assert "cannot detect a truncated series" in printed
    assert "not proof the copy finished" in printed


def test_the_check_flags_a_very_thin_series_separately_from_a_missing_one(tmp_path):
    """Reported, not failed: a localiser really can be two images."""
    _tree(tmp_path, {"studyA": {"s1": 1, "s2": 20}})
    result = check_against_table(tmp_path)

    assert result["thin"] == ["studyA/s1"]
    assert result["absent"] == []


# --- the command line ----------------------------------------------------------


def test_the_command_line_renders(capsys, monkeypatch):
    import sys

    from rsna_knee import copy_audit

    monkeypatch.setattr(sys, "argv", ["copy-audit", "--help"])
    with pytest.raises(SystemExit) as exit_code:
        copy_audit.main()

    assert exit_code.value.code == 0
    printed = capsys.readouterr().out
    for flag in ("scan", "compare", "check", "--source-manifest", "--rsync-list"):
        assert flag in printed


def test_scan_writes_a_manifest_a_later_compare_can_read(tmp_path, monkeypatch, capsys):
    """The round trip, run rather than assumed: scan on one machine, compare on
    another, with JSON as the only thing that crosses."""
    import sys

    from rsna_knee import copy_audit

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 4}})
    _tree(destination, {"studyA": {"s1": 1}})
    manifest = tmp_path / "source.json"

    monkeypatch.setattr(
        sys, "argv",
        ["copy-audit", "scan", "--data-root", str(source), "--out", str(manifest)],
    )
    copy_audit.main()

    listing = tmp_path / "recopy.txt"
    monkeypatch.setattr(
        sys, "argv",
        ["copy-audit", "compare", "--data-root", str(destination),
         "--source-manifest", str(manifest), "--rsync-list", str(listing)],
    )
    with pytest.raises(SystemExit) as exit_code:
        copy_audit.main()

    assert exit_code.value.code == 1, "an incomplete copy must exit non-zero"
    assert listing.read_text().strip() == "train_images/studyA/s1/"
    assert json.loads(manifest.read_text())["version"] == COPY_AUDIT_VERSION


def test_a_complete_copy_exits_zero(tmp_path, monkeypatch):
    import sys

    from rsna_knee import copy_audit

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _tree(source, {"studyA": {"s1": 4}})
    _tree(destination, {"studyA": {"s1": 4}})
    manifest = tmp_path / "source.json"

    monkeypatch.setattr(
        sys, "argv",
        ["copy-audit", "scan", "--data-root", str(source), "--out", str(manifest)],
    )
    copy_audit.main()
    monkeypatch.setattr(
        sys, "argv",
        ["copy-audit", "compare", "--data-root", str(destination),
         "--source-manifest", str(manifest)],
    )
    with pytest.raises(SystemExit) as exit_code:
        copy_audit.main()

    assert exit_code.value.code == 0
