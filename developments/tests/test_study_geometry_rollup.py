"""Rolling the series geometry up to the studies the model predicts.

The thing worth pinning is the within-study spread. A corpus histogram can look
tidy while every individual study mixes a 1 mm triplet with a 13 mm one, and
that is the fault that would matter, so the spread and the "mixed" flag are
tested directly rather than through the summary.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.study_geometry_rollup import (
    load_series_geometry,
    rollup,
    study_table,
)


def _series_csv(path, rows):
    """rows: (study, series, triplet_span_mm, frames, frames_never_read)"""
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": series,
                "Anatomical_Plane": "Sagittal",
                "triplet_span_mm": span,
                "slice_spacing_mm": None if span is None else span / 2.0,
                "frames": frames,
                "frames_never_read": lost,
            }
            for study, series, span, frames, lost in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# --- reading the scan's output ------------------------------------------------


def test_a_table_that_is_not_the_scans_output_is_refused(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="slice_geometry_scan"):
        load_series_geometry(path)


def test_the_scans_own_output_is_accepted(tmp_path):
    path = _series_csv(tmp_path / "s.csv", [("a", "s1", 6.0, 30, 0)])
    assert len(load_series_geometry(path)) == 1


# --- the per-study table ------------------------------------------------------


def test_one_study_reports_the_range_of_depths_it_hands_over(tmp_path):
    path = _series_csv(
        tmp_path / "s.csv",
        [
            ("a", "s1", 1.2, 200, 104),
            ("a", "s2", 6.6, 30, 0),
            ("a", "s3", 13.0, 16, 0),
        ],
    )
    table = study_table(load_series_geometry(path))

    assert len(table) == 1
    row = table.iloc[0]
    assert row["series"] == 3
    assert row["thinnest_triplet_mm"] == pytest.approx(1.2)
    assert row["median_triplet_mm"] == pytest.approx(6.6)
    assert row["thickest_triplet_mm"] == pytest.approx(13.0)
    assert row["triplet_spread_mm"] == pytest.approx(11.8)
    assert row["triplet_spread_ratio"] == pytest.approx(13.0 / 1.2)


def test_a_study_that_mixes_a_thin_volume_with_a_thick_stack_is_flagged(tmp_path):
    path = _series_csv(
        tmp_path / "s.csv",
        [("a", "s1", 1.2, 200, 104), ("a", "s2", 9.0, 20, 0)],
    )
    assert bool(study_table(load_series_geometry(path)).iloc[0]["mixes_thin_and_thick"])


def test_a_consistent_study_is_not_flagged(tmp_path):
    path = _series_csv(
        tmp_path / "s.csv",
        [("a", "s1", 6.0, 30, 0), ("a", "s2", 7.0, 32, 0)],
    )
    assert not bool(
        study_table(load_series_geometry(path)).iloc[0]["mixes_thin_and_thick"]
    )


def test_the_loss_is_expressed_as_a_share_of_the_studys_own_frames(tmp_path):
    path = _series_csv(
        tmp_path / "s.csv",
        [("a", "s1", 1.2, 200, 100), ("a", "s2", 6.0, 100, 0)],
    )
    row = study_table(load_series_geometry(path)).iloc[0]

    assert row["frames"] == 300
    assert row["frames_never_read"] == 100
    assert row["fraction_never_read"] == pytest.approx(100 / 300)


def test_a_study_with_no_measurable_spacing_does_not_break_the_rollup(tmp_path):
    path = _series_csv(tmp_path / "s.csv", [("a", "s1", None, 20, 0)])
    row = study_table(load_series_geometry(path)).iloc[0]

    assert row["series"] == 1
    assert row["series_with_a_span"] == 0
    assert pd.isna(row["thinnest_triplet_mm"])


# --- the summary --------------------------------------------------------------


def _mixed_corpus(tmp_path):
    """Two studies that mix depths, one that does not."""
    return _series_csv(
        tmp_path / "s.csv",
        [
            ("wide1", "s1", 1.2, 200, 104),
            ("wide1", "s2", 9.0, 20, 0),
            ("wide2", "s3", 1.6, 180, 84),
            ("wide2", "s4", 8.2, 24, 0),
            ("tight", "s5", 6.0, 30, 0),
            ("tight", "s6", 6.6, 32, 0),
        ],
    )


def test_the_mixed_studies_are_counted(tmp_path):
    result = rollup(series_csv=_mixed_corpus(tmp_path))
    block = result["all_studies"]

    assert block["studies"] == 3
    assert block["studies_mixing_thin_and_thick"]["studies"] == 2
    assert block["studies_mixing_thin_and_thick"]["fraction"] == pytest.approx(2 / 3)


def test_the_wide_spread_count_uses_the_ratio_not_the_millimetres(tmp_path):
    result = rollup(series_csv=_mixed_corpus(tmp_path))
    wide = result["all_studies"]["studies_spread_wider_than_ratio"]

    # 9.0/1.2 and 8.2/1.6 clear 2x; 6.6/6.0 does not.
    assert wide["studies"] == 2


def test_only_the_studies_that_lose_frames_are_counted(tmp_path):
    result = rollup(series_csv=_mixed_corpus(tmp_path))
    loss = result["all_studies"]["loss"]

    assert loss["studies_losing_any_frame"] == 2
    assert loss["frames_never_read"] == 188
    assert loss["share_of_own_frames_lost"]["n"] == 2


def test_a_corpus_where_nothing_is_lost_reports_no_share(tmp_path):
    path = _series_csv(tmp_path / "s.csv", [("a", "s1", 6.0, 30, 0)])
    loss = rollup(series_csv=path)["all_studies"]["loss"]

    assert loss["studies_losing_any_frame"] == 0
    assert loss["share_of_own_frames_lost"] == {"n": 0}


# --- the expert split and the legacy comparison -------------------------------


def _data_root(tmp_path, studies_with_gold):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    rows, series_rows = [], []
    for study, series_names in {
        "wide1": ["s1", "s2"],
        "wide2": ["s3", "s4"],
        "tight": ["s5", "s6"],
    }.items():
        row = {"StudyInstanceUID": study, "Report": "text"}
        for target in TARGETS:
            row[target] = 1 if study in studies_with_gold else None
        rows.append(row)
        for index, name in enumerate(series_names):
            series_rows.append(
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": name,
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": "Yes" if index == 0 else "No",
                    "Fat_Suppression": "Yes" if index == 0 else "No",
                }
            )
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    return root


def test_the_expert_studies_are_reported_separately(tmp_path):
    result = rollup(
        series_csv=_mixed_corpus(tmp_path),
        data_root=_data_root(tmp_path, {"tight"}),
    )

    assert result["expert_gold"]["studies"] == 1
    assert result["report_only"]["studies"] == 2
    assert result["expert_gold"]["studies_mixing_thin_and_thick"]["studies"] == 0


def test_the_legacy_dual_subset_is_reported_as_a_comparison(tmp_path):
    """Both series of each study survive dual selection here, one per stream."""
    result = rollup(
        series_csv=_mixed_corpus(tmp_path),
        data_root=_data_root(tmp_path, set()),
    )
    legacy = result["legacy_dual_subset"]

    assert legacy["series"] == 6
    assert legacy["studies"] == 3
    assert "not the current input" in legacy["note"]


def test_the_policy_is_stated_in_the_result(tmp_path):
    result = rollup(series_csv=_mixed_corpus(tmp_path))
    assert "all_repaired_anatomical_series_v1" in result["policy"]


# --- output -------------------------------------------------------------------


def test_the_study_table_and_the_summary_are_written(tmp_path):
    result = rollup(
        series_csv=_mixed_corpus(tmp_path),
        out_csv=tmp_path / "out" / "studies.csv",
        out_json=tmp_path / "out" / "summary.json",
    )

    assert json.loads((tmp_path / "out" / "summary.json").read_text()) == result
    assert len(pd.read_csv(tmp_path / "out" / "studies.csv")) == 3
