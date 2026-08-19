"""The submission manifest must describe the run it actually came from.

A manifest exists to be believed without rerunning anything, so a field that
is written as a constant is worse than a missing one: it reads as evidence and
carries none.  `encoder_frozen` was such a field.  Every experiment before
encoder fine-tuning existed had a frozen encoder, so the training code stored
`True` outright -- and then fine-tuning arrived and the field kept saying
`True` about runs whose encoder had plainly moved.

The fix is to read the two encoder fingerprints instead.  They are taken
before and after training, so they cannot agree with each other unless the
weights really did stay put.
"""

from __future__ import annotations

from pathlib import Path

from testing.test import encoder_stayed_frozen

TRAINING_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "developments"
    / "src"
    / "rsna_knee"
    / "phase9_matched_supervision_training.py"
)


def test_a_frozen_run_is_reported_frozen():
    assert encoder_stayed_frozen(
        {"encoder_sha256_initial": "abc", "encoder_sha256_final": "abc"}
    )


def test_a_fine_tuned_run_is_not_reported_frozen():
    """This is the case the old constant got wrong."""
    assert not encoder_stayed_frozen(
        {
            "encoder_sha256_initial": "abc",
            "encoder_sha256_final": "def",
            "encoder_frozen": True,  # the stale claim, which must not win
        }
    )


def test_the_fingerprints_outrank_the_stored_flag():
    assert encoder_stayed_frozen(
        {
            "encoder_sha256_initial": "abc",
            "encoder_sha256_final": "abc",
            "encoder_frozen": False,
        }
    )


def test_a_checkpoint_without_fingerprints_falls_back_to_the_flag():
    assert encoder_stayed_frozen({"encoder_frozen": True})
    assert not encoder_stayed_frozen({})


def test_training_no_longer_writes_the_field_as_a_constant():
    """Guard the source, not just the reader.

    The reader can recover the truth for old checkpoints, but new ones should
    be written honestly in the first place.
    """
    source = TRAINING_SOURCE.read_text(encoding="utf-8")
    assert '"encoder_frozen": True,' not in source
    assert '"encoder_frozen": not encoder_moved,' in source
