from __future__ import annotations

import pytest

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b37_highres_sparse_submission import (
    _largest_submission_index,
    projected_remaining_seconds,
    require_b37_submission_contract,
)
from rsna_knee.b39_b37_five_offset_tta import (
    B39_NUMBERED_CONTAINER,
    B39_RUN_ROOT,
    B39_RUNTIME_BUDGET_HOURS,
    B39_RUNTIME_RESERVE_MINUTES,
    B39_TTA_OFFSETS,
    require_b39_five_offset_contract,
)


def test_b39_uses_a_permanent_numbered_inference_run_root() -> None:
    assert B39_NUMBERED_CONTAINER == "runs/074_Experiment_B39_b37_five_offset_tta"
    assert B39_RUN_ROOT == (
        "runs/074_Experiment_B39_b37_five_offset_tta/"
        "b39_b37_five_offset_tta"
    )


def test_b39_is_exactly_the_symmetric_five_offset_recipe() -> None:
    config = dict(_read_config("config/b39_b37_five_offset_tta.yaml"))
    assert tuple(config["b7_eval_tta_offsets"]) == B39_TTA_OFFSETS
    assert config["runtime_budget_hours"] == B39_RUNTIME_BUDGET_HOURS
    assert config["runtime_reserve_minutes"] == B39_RUNTIME_RESERVE_MINUTES
    policy = require_b39_five_offset_contract(config)
    assert policy["crop_fraction"] == 0.90


def test_b39_rejects_a_different_tta_or_runtime_recipe() -> None:
    config = dict(_read_config("config/b39_b37_five_offset_tta.yaml"))

    wrong_offsets = dict(config)
    wrong_offsets["b7_eval_tta_offsets"] = [-1, 0, 1]
    with pytest.raises(ValueError, match="b7_eval_tta_offsets"):
        require_b39_five_offset_contract(wrong_offsets)

    wrong_reserve = dict(config)
    wrong_reserve["runtime_reserve_minutes"] = 30
    with pytest.raises(ValueError, match="runtime_reserve_minutes"):
        require_b39_five_offset_contract(wrong_reserve)


def test_original_b37_submission_remains_the_three_offset_endpoint() -> None:
    config = dict(_read_config("config/b37_highres_sparse_448.yaml"))
    policy = require_b37_submission_contract(config)
    assert tuple(config["b7_eval_tta_offsets"]) == (-1, 0, 1)
    assert policy["crop_fraction"] == 0.90


def test_remaining_budget_uses_full_study_wall_times() -> None:
    projected = projected_remaining_seconds(
        [10.0, 20.0],
        remaining_studies=3,
    )
    assert projected == pytest.approx(60.75)

    assert projected_remaining_seconds([], remaining_studies=1) == 180.0
    with pytest.raises(ValueError, match="safety_factor"):
        projected_remaining_seconds([1.0], remaining_studies=1, safety_factor=0.9)


def test_submission_preflight_chooses_the_largest_series_study() -> None:
    class Dataset:
        study_uids = ["three", "twelve", "seven"]
        series_records = {
            "three": [None] * 3,
            "twelve": [None] * 12,
            "seven": [None] * 7,
        }

        def __len__(self):
            return len(self.study_uids)

    assert _largest_submission_index(Dataset()) == 1
