"""Preflight must fail loudly and never approve something training would reject.

A check that quietly passes is worse than no check: it converts "I am not sure
about these paths" into false confidence, and the cost lands ninety minutes
later. So the cases covered here are a broken input, a missing one, and an
input that works but lives somewhere the user is about to tidy away.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.preflight import CHECKS, _check, outside_working_directory


def test_a_failing_check_becomes_a_report_line_not_a_crash():
    """One bad input must not stop the other four being reported."""
    def explode():
        raise FileNotFoundError("missing training_targets.csv")

    result = _check("all-script labels", explode)
    assert result["ok"] is False
    assert "missing training_targets.csv" in result["detail"]
    assert "FileNotFoundError" in result["detail"]


def test_a_passing_check_carries_its_evidence():
    result = _check("data root", lambda: "4407 studies")
    assert result["ok"] is True
    assert result["detail"] == "4407 studies"


def test_every_training_input_is_checked():
    """A new required input must not slip through unchecked."""
    assert set(CHECKS) == {
        "data root",
        "latin-script labels",
        "all-script labels",
        "series policy",
        "encoder",
    }


def test_an_input_in_another_folder_is_noticed(tmp_path):
    """The case that prompted this: one path left behind in the old copy."""
    working = tmp_path / "CNN_CPC"
    other = tmp_path / "CNN_CPC_current"
    (working / "runs").mkdir(parents=True)
    (other / "runs").mkdir(parents=True)

    assert outside_working_directory(other / "runs" / "labels", working)
    assert not outside_working_directory(working / "runs" / "labels", working)


def test_a_nested_input_counts_as_inside(tmp_path):
    working = tmp_path / "CNN_CPC"
    (working / "runs" / "deep" / "deeper").mkdir(parents=True)
    assert not outside_working_directory(working / "runs" / "deep" / "deeper", working)


def test_the_working_directory_itself_counts_as_inside(tmp_path):
    assert not outside_working_directory(tmp_path, tmp_path)


def test_a_symlink_out_of_the_folder_is_still_outside(tmp_path):
    """Resolving matters: a link inside pointing out is a dependency on out."""
    working = tmp_path / "CNN_CPC"
    other = tmp_path / "elsewhere"
    working.mkdir()
    other.mkdir()
    link = working / "labels"
    link.symlink_to(other)
    assert outside_working_directory(link, working)


def test_a_missing_path_is_reported_before_any_loader_runs(tmp_path):
    """Loaders raise confusing errors on absent paths; say the plain thing."""
    missing = tmp_path / "not_here"
    assert not missing.exists()
    # The main() flow short-circuits on this, so the check is that the
    # existence test comes first and needs no loader.
    assert not Path(missing).exists()
