"""The encoder source is validated before any work begins.

Swapping the encoder is the one change the frozen working model permits, so the
selection has to be checked up front rather than part-way into a training
session that costs an hour and a half.
"""

from __future__ import annotations

import inspect

import pytest

from rsna_knee import phase9_matched_supervision_training as trainer


def test_report_aligned_remains_the_default():
    parameters = inspect.signature(trainer.train_phase9_arm).parameters
    assert parameters["encoder_source"].default == "report-aligned"
    assert parameters["dinov3_variant"].default == "tiny"


def test_only_the_two_encoder_sources_are_offered():
    assert trainer.PHASE9_ENCODER_SOURCES == ("report-aligned", "dinov3")


def test_an_unknown_encoder_source_is_refused_before_loading_anything():
    with pytest.raises(ValueError, match="encoder_source must be one of"):
        trainer.train_phase9_arm(
            {},
            arm="control",
            b6_root="b6",
            phase8_root="phase8",
            series_policy_path="policy.json",
            report_ssl_checkpoint="encoder.pt",
            out_root="runs",
            encoder_source="dinov2",
        )


def test_the_arm_is_still_checked_first():
    """Arm validation must not be displaced by the new encoder check."""
    with pytest.raises(ValueError, match="Phase 9 arm must be one of"):
        trainer.train_phase9_arm(
            {},
            arm="nonsense",
            b6_root="b6",
            phase8_root="phase8",
            series_policy_path="policy.json",
            report_ssl_checkpoint="encoder.pt",
            out_root="runs",
            encoder_source="dinov3",
        )
