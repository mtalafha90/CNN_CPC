from __future__ import annotations

import pytest

from rsna_knee.budget import RuntimeBudget
from rsna_knee.policy import validate_competition_config, validate_submission_path


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


def test_external_pretrained_is_off_by_default():
    with pytest.raises(ValueError, match="forbids external pretrained"):
        validate_competition_config(
            {
                "competition_mode": True,
                "runtime_budget_hours": 8.5,
                "requested_gpus": 1,
                "pretrained": True,
                "allow_external_pretrained": False,
            },
            purpose="train",
        )


def test_submission_filename_is_enforced():
    config = {
        "competition_mode": True,
        "runtime_budget_hours": 8.5,
        "requested_gpus": 1,
        "submission_filename": "submission.csv",
    }
    validate_competition_config(config, purpose="infer")
    validate_submission_path("submission.csv", config)
    with pytest.raises(ValueError, match="submission.csv"):
        validate_submission_path("predictions.csv", config)


def test_runtime_budget_rejects_nine_hours():
    with pytest.raises(ValueError, match="strictly <9"):
        RuntimeBudget(max_hours=9.0)


def test_runtime_budget_can_start_short_work():
    budget = RuntimeBudget(max_hours=0.1, reserve_minutes=0.0)
    assert budget.can_start(1.0)
