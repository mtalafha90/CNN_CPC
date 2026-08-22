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


def test_dinov3_checkpoint_is_rebuilt_before_its_weights_load(monkeypatch):
    """The Phase-9 loader must reconstruct the swapped encoder without a download."""

    final_sha = "f" * 64
    payload = {
        "experiment": trainer.PHASE9_EXPERIMENT,
        "phase9_version": trainer.PHASE9_VERSION,
        "arm": "candidate",
        "fixed_endpoint": True,
        "completed_epochs": trainer.PHASE9_FIXED_EPOCHS,
        "validation_used_for_checkpoint_selection": False,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "report_only_studies_exposed": trainer.REPORT_ONLY_STUDIES,
        "training_series": trainer.PHASE9_EXPECTED_REPORT_ONLY_SERIES,
        "stochastic_path_matched_after_model_construction": True,
        "encoder_sha256_initial": final_sha,
        "encoder_sha256_final": final_sha,
        "encoder_trainable_stages": 0,
        "encoder": {"variant": "tiny"},
        "model_spec": {"encoder_source": "dinov3", "dinov3_variant": "tiny"},
        "model_state": {"state": "from-checkpoint"},
    }
    calls = []

    class FakeModel:
        def load_state_dict(self, state, *, strict):
            calls.append(("load", state, strict))

        def to(self, device):
            calls.append(("to", str(device)))
            return self

    model = FakeModel()
    monkeypatch.setattr(trainer.torch, "load", lambda *args, **kwargs: payload)
    def build(spec, *, pretrained_weights):
        calls.append(("build", spec, pretrained_weights))
        return model

    monkeypatch.setattr(trainer, "build_b34_model", build)

    def attach(rebuilt, *, variant, pretrained_weights):
        rebuilt.encoder = object()
        calls.append(("attach", rebuilt, variant, pretrained_weights))

    monkeypatch.setattr(trainer, "attach_dinov3_encoder", attach)
    monkeypatch.setattr(trainer, "freeze_encoder", lambda rebuilt: calls.append(("freeze", rebuilt)))
    monkeypatch.setattr(trainer, "encoder_state_sha256", lambda encoder: final_sha)

    rebuilt, restored_payload = trainer.load_phase9_checkpoint("candidate.pt", device="cpu")

    assert rebuilt is model
    assert restored_payload is payload
    assert calls.index(("attach", model, "tiny", False)) < calls.index(
        ("load", {"state": "from-checkpoint"}, True)
    )
