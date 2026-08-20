"""A subset must be self-contained, or it trains on blanks and says nothing.

Two failures matter here and neither announces itself. A subset without the
expert-labelled studies trains perfectly well and cannot be scored at all. A
subset whose CSV lists studies it did not copy loads perfectly well too -- the
readers find no folder, return zeros, and the model trains on blank images with
the presence flag quietly off.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.make_subset import (
    _series_directory,
    choose_studies,
    copy_subset,
    measure,
    write_tables,
)

TARGETS = ("ACL", "MCL", "Effusion")


def _studies(n_expert: int, n_report: int) -> pd.DataFrame:
    rows = []
    for i in range(n_expert):
        rows.append({"StudyInstanceUID": f"expert-{i}", "ACL": 1.0, "MCL": 0.0, "Effusion": 1.0})
    for i in range(n_report):
        rows.append({"StudyInstanceUID": f"report-{i}", "ACL": None, "MCL": None, "Effusion": None})
    return pd.DataFrame(rows)


def _series(study_uids, per_study: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"StudyInstanceUID": uid, "SeriesInstanceUID": f"{uid}-s{j}"}
            for uid in study_uids
            for j in range(per_study)
        ]
    )


def _lay_out_images(root: Path, series: pd.DataFrame, *, files_each: int = 3) -> None:
    for _, row in series.iterrows():
        directory = root / "train_series" / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
        directory.mkdir(parents=True, exist_ok=True)
        for k in range(files_each):
            (directory / f"{k}.dcm").write_bytes(b"x" * 100)


def test_every_expert_study_is_kept_however_small_the_subset():
    """There are 58 in the whole release and they are the only scoring surface."""
    studies = _studies(n_expert=5, n_report=100)
    expert, sampled = choose_studies(studies, TARGETS, count=8, seed=1)
    assert len(expert) == 5
    assert len(sampled) == 3
    assert all(uid.startswith("expert-") for uid in expert)


def test_asking_for_fewer_studies_than_there_are_experts_still_keeps_them_all():
    studies = _studies(n_expert=5, n_report=100)
    expert, sampled = choose_studies(studies, TARGETS, count=2, seed=1)
    assert len(expert) == 5
    assert sampled == []


def test_the_same_seed_gives_the_same_subset():
    """A result is only reproducible if the data behind it is."""
    studies = _studies(n_expert=2, n_report=50)
    _, first = choose_studies(studies, TARGETS, count=12, seed=7)
    _, second = choose_studies(studies, TARGETS, count=12, seed=7)
    _, other = choose_studies(studies, TARGETS, count=12, seed=8)
    assert first == second
    assert first != other


def test_measuring_finds_the_series_and_adds_up_their_size(tmp_path):
    studies = _studies(n_expert=1, n_report=1)
    uids = studies["StudyInstanceUID"].astype(str).tolist()
    series = _series(uids, per_study=2)
    _lay_out_images(tmp_path, series, files_each=3)

    found, total = measure(tmp_path, "train", series, uids)
    assert len(found) == 4              # 2 studies x 2 series
    assert total == 4 * 3 * 100


def test_a_series_listed_but_absent_is_skipped_not_counted(tmp_path, capsys):
    studies = _studies(n_expert=1, n_report=0)
    uids = studies["StudyInstanceUID"].astype(str).tolist()
    series = _series(uids, per_study=2)
    _lay_out_images(tmp_path, series.iloc[:1], files_each=2)   # only the first exists

    found, total = measure(tmp_path, "train", series, uids)
    assert len(found) == 1
    assert total == 2 * 100
    assert "no folder on disk" in capsys.readouterr().out


def test_the_copy_uses_the_layout_the_loaders_try_first(tmp_path):
    source = tmp_path / "source"
    series = _series(["study-a"], per_study=1)
    _lay_out_images(source, series)

    found, _ = measure(source, "train", series, ["study-a"])
    out = tmp_path / "subset"
    copy_subset(out, "train", found)

    assert _series_directory(out, "train", "study-a", "study-a-s0") is not None
    assert (out / "train_series" / "study-a" / "study-a-s0" / "0.dcm").is_file()


def test_the_tables_list_only_what_was_copied(tmp_path):
    """A CSV row without images trains on blanks instead of raising."""
    studies = _studies(n_expert=1, n_report=3)
    uids = studies["StudyInstanceUID"].astype(str).tolist()
    series = _series(uids, per_study=1)

    kept_studies = uids[:2]
    kept_series = {(uid, f"{uid}-s0") for uid in kept_studies}

    out = tmp_path / "subset"
    out.mkdir()
    write_tables(out, "train", studies, series, kept_studies, kept_series, None)

    written_studies = pd.read_csv(out / "train.csv")
    written_series = pd.read_csv(out / "train_series.csv")
    assert len(written_studies) == 2
    assert len(written_series) == 2
    assert set(written_studies["StudyInstanceUID"]) == set(kept_studies)


def test_the_label_file_is_trimmed_to_match(tmp_path):
    studies = _studies(n_expert=0, n_report=4)
    uids = studies["StudyInstanceUID"].astype(str).tolist()
    series = _series(uids, per_study=1)

    labels = tmp_path / "labels.csv"
    pd.DataFrame({"StudyInstanceUID": uids, "ACL__state": ["positive"] * 4}).to_csv(
        labels, index=False
    )

    out = tmp_path / "subset"
    out.mkdir()
    write_tables(out, "train", studies, series, uids[:2],
                 {(u, f"{u}-s0") for u in uids[:2]}, labels)

    written = pd.read_csv(out / "training_targets.csv")
    assert len(written) == 2
    assert set(written["StudyInstanceUID"]) == set(uids[:2])


def test_copying_twice_does_not_duplicate_or_fail(tmp_path):
    """Re-running the same command must be safe -- subsets get rebuilt often."""
    source = tmp_path / "source"
    series = _series(["study-a"], per_study=1)
    _lay_out_images(source, series)
    found, _ = measure(source, "train", series, ["study-a"])

    out = tmp_path / "subset"
    copy_subset(out, "train", found)
    copy_subset(out, "train", found)

    directory = out / "train_series" / "study-a" / "study-a-s0"
    assert sorted(p.name for p in directory.iterdir()) == ["0.dcm", "1.dcm", "2.dcm"]


def test_an_unknown_layout_is_reported_as_missing(tmp_path):
    assert _series_directory(tmp_path, "train", "nope", "nope") is None
