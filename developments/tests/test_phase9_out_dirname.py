"""The matched-supervision trainer's output directory name is cosmetic only.

`out_dirname` was added so the clean interface can lay runs out by supervision
surface rather than by internal arm name. It must change where files land and
nothing else, so the existing default layout and every audited quantity stay
exactly as they were.
"""

from __future__ import annotations

import inspect

import pytest

from rsna_knee import phase9_matched_supervision_training as trainer


def test_out_dirname_defaults_to_the_arm_name():
    parameter = inspect.signature(trainer.train_phase9_arm).parameters["out_dirname"]
    assert parameter.default is None, "the historical layout must remain the default"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_arm_is_still_validated_against_a_closed_set():
    """The set is closed, so a typo cannot quietly become a new arm.

    It was a pair -- the two report-label surfaces the matched comparison ran
    on -- until `llm_fill` was added deliberately as a third. What this guards
    is that the list stays exhaustive, not that it never grows.
    """
    assert trainer.PHASE9_ARMS == ("control", "candidate", "llm_fill")


@pytest.mark.parametrize("directory", ["a/b", "a\\b", "..", "."])
def test_a_path_cannot_be_smuggled_in_as_a_directory_name(directory):
    """A separator would let a run escape its experiment directory."""
    with pytest.raises(ValueError, match="single directory name"):
        trainer.train_phase9_arm(
            {},
            arm="control",
            b6_root="b6",
            phase8_root="phase8",
            series_policy_path="policy.json",
            report_ssl_checkpoint="encoder.pt",
            out_root="runs",
            out_dirname=directory,
        )


def test_an_invalid_arm_is_still_refused_before_any_work():
    with pytest.raises(ValueError, match="Phase 9 arm must be one of"):
        trainer.train_phase9_arm(
            {},
            arm="all-script",
            b6_root="b6",
            phase8_root="phase8",
            series_policy_path="policy.json",
            report_ssl_checkpoint="encoder.pt",
            out_root="runs",
        )
