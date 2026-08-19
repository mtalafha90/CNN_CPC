"""The submission notebook must submit what was chosen, not what was attached.

The notebook once used every `.pt` file it could find. That reads as helpful
and is not: a model dataset gathers checkpoints across a competition, so
uploading one for later quietly turned a single-model submission into an
ensemble of whatever happened to be sitting there. The run then differed from
the run that was intended, and the manifest gave no hint of it.

These tests read the notebook source and hold it to naming its models.
"""

from __future__ import annotations

from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "submission_notebook.py"


@pytest.fixture(scope="module")
def source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def test_the_models_are_named_rather_than_swept_up(source):
    assert "SUBMIT = [" in source, "the notebook must name the checkpoints it submits"
    assert "MODEL_PATHS = sorted(MINE.rglob" not in source, (
        "using every attached .pt lets the dataset's contents decide the "
        "experiment"
    )


def test_a_named_model_that_is_missing_stops_the_run(source):
    """Silently dropping a name would be the same failure in a new costume."""
    assert "missing = [name for name in SUBMIT if name not in available]" in source
    assert "which is not attached" in source


def test_attached_but_unsubmitted_models_are_reported(source):
    """The reader must be able to see what was left out, not just what went in."""
    assert "not submitting:" in source


def test_the_checkpoints_are_described_before_the_expensive_cell(source):
    """A filename is a typed label; the payload is the training run's record."""
    assert "encoder_sha256_initial" in source
    assert "fine-tuned" in source
    prediction = source.index("predict_test_set")
    assert source.index("encoder_sha256_initial") < prediction


def test_dinov3_is_still_refused_offline(source):
    assert "uses DINOv3" in source
