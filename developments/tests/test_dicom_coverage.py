"""The check that turns a mid-epoch crash into a message you can act on.

`RuntimeError: B42 study has no readable MRI series` names no study, no path and
no cause, and arrives after the config, the base checkpoint, the label export
and the whole series index have loaded. On a Windows drive mounted under WSL
that is twenty minutes of waiting for four words.

Two causes produce that identical crash and need opposite fixes -- part of the
images missing, or every image present under a directory name the loader does
not look in. These tests pin that the diagnostic tells them apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.dicom_coverage import (  # noqa: E402
    describe_layout,
    format_coverage,
    require_dicom_coverage,
    study_coverage,
)


def _index(*studies: tuple) -> dict:
    """A series index shaped like audit_variable_series_surface's output."""
    return {
        uid: [{"series_uid": series} for series in series_uids]
        for uid, series_uids in studies
    }


def _make(root: Path, layout: str, studies: dict) -> None:
    """Create empty series directories in one of the accepted layouts."""
    for uid, series_uids in studies.items():
        for series in series_uids:
            (root / layout / uid / series).mkdir(parents=True, exist_ok=True)


# --- counting ---------------------------------------------------------------


def test_a_complete_dataset_reports_no_missing_studies(tmp_path):
    index = _index(("study-a", ("s1", "s2")), ("study-b", ("s3",)))
    _make(tmp_path, "train_series", {"study-a": ("s1", "s2"), "study-b": ("s3",)})

    report = study_coverage(tmp_path, index)
    assert report["studies_missing"] == 0
    assert report["studies_complete"] == 2
    assert report["series_found"] == report["series_total"] == 3


def test_a_partly_copied_dataset_is_counted_correctly(tmp_path):
    """This is the 5090 case: full tables, only some of the images."""
    index = _index(
        ("study-a", ("s1", "s2")), ("study-b", ("s3",)), ("study-c", ("s4",))
    )
    _make(tmp_path, "train_series", {"study-a": ("s1", "s2")})

    report = study_coverage(tmp_path, index)
    assert report["studies_complete"] == 1
    assert report["studies_missing"] == 2
    assert report["series_found"] == 2
    assert report["studies_missing_fraction"] == pytest.approx(2 / 3)
    assert report["missing_examples"] == ["study-b", "study-c"]


def test_a_study_with_some_series_is_partial_not_missing(tmp_path):
    """The loader masks a series it cannot read, so a partial study still runs."""
    index = _index(("study-a", ("s1", "s2", "s3")))
    _make(tmp_path, "train_series", {"study-a": ("s1",)})

    report = study_coverage(tmp_path, index)
    assert report["studies_partial"] == 1
    assert report["studies_missing"] == 0


def test_every_accepted_layout_is_found(tmp_path):
    """find_series_dir tries three; the diagnostic must agree with all three."""
    for layout in ("train_series", "train_images"):
        root = tmp_path / layout
        root.mkdir()
        _make(root, layout, {"study-a": ("s1",)})
        assert study_coverage(root, _index(("study-a", ("s1",))))["studies_missing"] == 0

    bare = tmp_path / "bare"
    (bare / "study-a" / "s1").mkdir(parents=True)
    assert study_coverage(bare, _index(("study-a", ("s1",))))["studies_missing"] == 0


def test_the_test_split_is_looked_up_under_its_own_directory(tmp_path):
    _make(tmp_path, "test_series", {"study-a": ("s1",)})
    index = _index(("study-a", ("s1",)))
    assert study_coverage(tmp_path, index, split="test")["studies_missing"] == 0
    assert study_coverage(tmp_path, index, split="train")["studies_missing"] == 1


# --- telling the two causes apart -------------------------------------------


def test_a_wrong_layout_is_reported_as_a_layout_problem(tmp_path):
    """Every folder present, under a name the loader does not look in.

    Identical crash to missing files, opposite fix, so the message must not
    say the images are missing.
    """
    _make(tmp_path, "train_dicoms", {"study-a": ("s1",), "study-b": ("s2",)})
    index = _index(("study-a", ("s1",)), ("study-b", ("s2",)))

    report = study_coverage(tmp_path, index)
    text = format_coverage(report, describe_layout(tmp_path))

    assert report["series_found"] == 0
    assert "layout rather" in text
    assert "train_dicoms" in text, "it must show what is actually there"
    assert "train_series/<study>/<series>" in text, "and what it expected"


def test_partly_missing_images_are_reported_as_missing_images(tmp_path):
    _make(tmp_path, "train_series", {"study-a": ("s1",)})
    index = _index(("study-a", ("s1",)), ("study-b", ("s2",)))

    text = format_coverage(study_coverage(tmp_path, index), describe_layout(tmp_path))
    assert "only part of the images" in text
    assert "layout rather" not in text


def test_the_message_explains_why_the_fingerprint_still_matched(tmp_path):
    """The confusing part: the run got far enough to resolve 3,801 studies."""
    _make(tmp_path, "train_series", {"study-a": ("s1",)})
    index = _index(("study-a", ("s1",)), ("study-b", ("s2",)))

    text = format_coverage(study_coverage(tmp_path, index), describe_layout(tmp_path))
    assert "fingerprint matched" in text


def test_describe_layout_survives_a_data_root_that_is_not_there(tmp_path):
    layout = describe_layout(tmp_path / "nope")
    assert layout["exists"] is False
    assert layout["entries"] == []


# --- refusing to start ------------------------------------------------------


def test_a_complete_surface_is_allowed_through(tmp_path):
    _make(tmp_path, "train_series", {"study-a": ("s1",)})
    report = require_dicom_coverage(tmp_path, _index(("study-a", ("s1",))))
    assert report["studies_missing"] == 0


def test_a_missing_study_stops_the_run_with_a_useful_message(tmp_path):
    _make(tmp_path, "train_series", {"study-a": ("s1",)})
    index = _index(("study-a", ("s1",)), ("study-b", ("s2",)))

    with pytest.raises(RuntimeError) as caught:
        require_dicom_coverage(tmp_path, index, label="training surface")

    message = str(caught.value)
    assert "training surface" in message
    assert "study-b" in message, "it must name a study you can go and look for"
    assert "1 of 2 studies" in message
    assert "B42 study has no readable MRI series" in message, (
        "it must connect itself to the error the user would otherwise see"
    )


def test_a_partial_study_does_not_stop_the_run(tmp_path):
    """Masking one unreadable series is normal; it is not a reason to refuse."""
    _make(tmp_path, "train_series", {"study-a": ("s1",)})
    require_dicom_coverage(tmp_path, _index(("study-a", ("s1", "s2"))))


# --- wired into B53 ---------------------------------------------------------


def test_b53_checks_coverage_before_it_builds_a_dataset():
    """Checking after the dataset is built would still cost the wait."""
    import inspect

    from rsna_knee import b53_augmented_training as b53

    source = inspect.getsource(b53.train_b53)
    check_at = source.index("require_dicom_coverage(")
    build_at = source.index("_build_train_dataset(")
    loop_at = source.index("for epoch in range(1")

    assert check_at < build_at < loop_at, (
        "the coverage check must run before the dataset and the epoch loop"
    )


def test_b53_checks_both_surfaces():
    """A complete training set and an absent validation set still fails."""
    import inspect

    from rsna_knee import b53_augmented_training as b53

    source = inspect.getsource(b53.train_b53)
    assert 'label="training surface"' in source
    assert 'label="validation surface"' in source


def test_b53_records_the_coverage_in_its_checkpoint():
    """So a finished run says how much of the data it actually saw."""
    import inspect

    from rsna_knee import b53_augmented_training as b53

    assert '"dicom_coverage": coverage,' in inspect.getsource(b53.train_b53)
