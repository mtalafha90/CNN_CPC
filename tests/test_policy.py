from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsna_knee.budget import RuntimeBudget
from rsna_knee.inference import _load_checkpoint_payload
from rsna_knee.policy import validate_competition_config, validate_submission_path


def _safe_training_config() -> dict:
    return {
        "competition_mode": True,
        "runtime_budget_hours": 8.5,
        "runtime_reserve_minutes": 10,
        "requested_gpus": 1,
        "pretrained": False,
        "allow_external_pretrained": False,
        "tta_center_offsets": [-1, 0, 1],
        "validation_tta_offsets": [-1, 0, 1],
    }


def test_competition_budget_must_be_strictly_below_nine_hours():
    with pytest.raises(ValueError, match="strictly below 9"):
        validate_competition_config(
            {"competition_mode": True, "runtime_budget_hours": 9.0, "requested_gpus": 1},
            purpose="train",
        )


def test_competition_policy_rejects_multiple_gpus():
    with pytest.raises(ValueError, match="single-GPU only"):
        validate_competition_config(
            {"competition_mode": True, "runtime_budget_hours": 8.5, "requested_gpus": 2},
            purpose="train",
        )


def test_validation_tta_must_match_submission_tta():
    config = {**_safe_training_config(), "validation_tta_offsets": [0]}
    with pytest.raises(ValueError, match="validation_tta_offsets"):
        validate_competition_config(config, purpose="train")


def test_stage2_cannot_set_single_root_and_candidate_list_together():
    config = {
        **_safe_training_config(),
        "cotrain_stage1_root": "runs/random",
        "cotrain_stage1_candidates": ["runs/random", "runs/ssl"],
    }
    with pytest.raises(ValueError, match="either cotrain_stage1_root"):
        validate_competition_config(config, purpose="train")


def test_external_pretrained_is_off_by_default():
    with pytest.raises(ValueError, match="forbids external pretrained"):
        validate_competition_config({**_safe_training_config(), "pretrained": True}, purpose="train")


def test_attached_ssl_requires_explicit_competition_data_source():
    base = {**_safe_training_config(), "ssl_encoder_checkpoint": "/kaggle/input/ssl/ssl_encoder.pt"}
    with pytest.raises(ValueError, match="ssl_checkpoint_source"):
        validate_competition_config(base, purpose="train")
    safe = {**base, "ssl_checkpoint_source": "competition_training_data"}
    validate_competition_config(safe, purpose="train")


def test_mounted_ssl_payload_must_have_competition_data_provenance(tmp_path):
    bad_path = tmp_path / "bad_ssl.pt"
    torch.save({"encoder": {}, "source": "unknown", "config": _safe_training_config()}, bad_path)
    config = {
        **_safe_training_config(),
        "ssl_encoder_checkpoint": str(bad_path),
        "ssl_checkpoint_source": "competition_training_data",
    }
    with pytest.raises(ValueError, match="provenance"):
        validate_competition_config(config, purpose="train")

    good_path = tmp_path / "good_ssl.pt"
    torch.save(
        {"encoder": {}, "source": "competition_training_data", "config": _safe_training_config()},
        good_path,
    )
    validate_competition_config({**config, "ssl_encoder_checkpoint": str(good_path)}, purpose="train")


def test_inference_rejects_checkpoint_from_unsafe_training_policy(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save(
        {
            "model": {},
            "model_spec": {},
            "stream_names": [],
            "fold": 0,
            "stage": "stage1",
            "validation_tta_offsets": [-1, 0, 1],
            "config": {
                "competition_mode": True,
                "runtime_budget_hours": 8.5,
                "runtime_reserve_minutes": 10,
                "requested_gpus": 2,
                "pretrained": False,
                "tta_center_offsets": [-1, 0, 1],
                "validation_tta_offsets": [-1, 0, 1],
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="single-GPU only"):
        _load_checkpoint_payload(path)


def test_submission_filename_is_enforced():
    config = {**_safe_training_config(), "submission_filename": "submission.csv"}
    validate_competition_config(config, purpose="infer")
    validate_submission_path("submission.csv", config)
    with pytest.raises(ValueError, match="submission.csv"):
        validate_submission_path("predictions.csv", config)


def test_runtime_budget_rejects_nine_hours():
    with pytest.raises(ValueError, match="strictly <9"):
        RuntimeBudget(max_hours=9.0)


def test_runtime_budget_exposes_work_deadline_before_hard_deadline():
    budget = RuntimeBudget(max_hours=0.1, reserve_minutes=1.0)
    assert budget.work_deadline_monotonic < budget.hard_deadline_monotonic
    assert budget.remaining_work_seconds < budget.remaining_seconds


def test_runtime_budget_can_start_short_work():
    budget = RuntimeBudget(max_hours=0.1, reserve_minutes=0.0)
    assert budget.can_start(1.0)
