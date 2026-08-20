"""Every checkpoint path is checked before any model is built.

Scoring an ensemble names several paths, and one of them is usually typed from
memory. Checking them as they are reached means a typo in the last path is
found after the first model has been built -- and in the test stage, after the
scans have started being read. Checking all of them first turns that into an
instant, readable error.

The message matters as much as the timing: the usual cause is a run directory
called something slightly different from what was typed, so the error says what
is actually there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model._implementation import resolve_checkpoints


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_existing_checkpoints_come_back_resolved(tmp_path):
    first = _touch(tmp_path / "a" / "model.pt")
    second = _touch(tmp_path / "b" / "model.pt")
    assert resolve_checkpoints([first, second]) == [first.resolve(), second.resolve()]


def test_a_single_path_may_be_given_on_its_own(tmp_path):
    path = _touch(tmp_path / "model.pt")
    assert resolve_checkpoints(path) == [path.resolve()]
    assert resolve_checkpoints(str(path)) == [path.resolve()]


def test_a_later_bad_path_is_caught_before_the_first_is_loaded(tmp_path):
    """This is the whole point: no work happens before every path is known good."""
    good = _touch(tmp_path / "seed7" / "model.pt")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoints([good, tmp_path / "finetune" / "model.pt"])


def test_a_missing_file_names_the_checkpoints_beside_it(tmp_path):
    _touch(tmp_path / "run" / "model.pt")
    with pytest.raises(FileNotFoundError, match="model.pt"):
        resolve_checkpoints(tmp_path / "run" / "modle.pt")


def test_a_missing_run_directory_names_the_directories_that_exist(tmp_path):
    """The real case: the run was called finetune_1stage, not finetune."""
    _touch(tmp_path / "finetune_1stage" / "train" / "model.pt")
    with pytest.raises(FileNotFoundError, match="finetune_1stage"):
        resolve_checkpoints(tmp_path / "finetune" / "train" / "model.pt")


def test_an_empty_directory_says_so_rather_than_listing_nothing(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="no .pt files"):
        resolve_checkpoints(tmp_path / "run" / "model.pt")


def test_no_checkpoints_at_all_is_refused():
    with pytest.raises(ValueError, match="no checkpoint given"):
        resolve_checkpoints([])
