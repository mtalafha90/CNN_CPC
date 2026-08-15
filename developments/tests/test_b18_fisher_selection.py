from __future__ import annotations

import pytest

from rsna_knee.b18_fisher_selection import (
    B18_CANDIDATE_EPOCHS,
    B18_SELECTION_METRIC,
    B18_TIE_BREAK,
    require_b18_contract,
    select_best_epoch,
)


def _config() -> dict:
    return {
        "seed": 2026,
        "competition_mode": True,
        "requested_gpus": 1,
        "allow_external_pretrained": True,
        "pretrained": True,
        "b7_epochs": 5,
        "b7_max_batches_per_epoch": 1560,
        "b7_encoder_lr": 0.0,
        "b7_head_lr": 1e-4,
        "b17_encoder_frozen": True,
        "b17_label_smoothing": 0.0,
        "b17_robust_loss": "none",
        "b18_expert_selection": True,
        "b18_selection_metric": B18_SELECTION_METRIC,
        "b18_selection_tie_break": B18_TIE_BREAK,
        "b18_candidate_epochs": B18_CANDIDATE_EPOCHS,
        "b18_save_candidate_checkpoints": True,
        "b7_eval_tta_offsets": [-1, 0, 1],
        "b7_eval_batch_size": 2,
    }


def test_b18_contract_accepts_only_selection_change():
    require_b18_contract(_config())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("b7_epochs", 4),
        ("b7_encoder_lr", 1e-5),
        ("b17_encoder_frozen", False),
        ("b17_label_smoothing", 0.1),
        ("b17_robust_loss", "sce"),
        ("b18_expert_selection", False),
        ("b18_selection_metric", "per_target_auc"),
        ("b18_selection_tie_break", "latest_epoch"),
        ("b18_candidate_epochs", 4),
        ("b18_save_candidate_checkpoints", False),
        ("b7_eval_tta_offsets", [0]),
        ("b7_eval_batch_size", 1),
    ],
)
def test_b18_contract_rejects_recipe_drift(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        require_b18_contract(config)


def _history(scores):
    return [
        {"epoch": i + 1, "expert_selection_macro_auc": float(score)}
        for i, score in enumerate(scores)
    ]


def test_b18_selects_highest_global_macro_auc():
    best = select_best_epoch(_history([0.60, 0.63, 0.62, 0.61, 0.59]))
    assert best["epoch"] == 2
    assert best["expert_selection_macro_auc"] == pytest.approx(0.63)


def test_b18_numerical_tie_prefers_earliest_epoch():
    best = select_best_epoch(_history([0.60, 0.64, 0.64 + 5e-13, 0.62, 0.61]))
    assert best["epoch"] == 2


def test_b18_selection_requires_all_five_epochs():
    with pytest.raises(ValueError):
        select_best_epoch(_history([0.60, 0.61, 0.62, 0.63]))


def test_b18_selection_rejects_nonfinite_score():
    values = [0.60, 0.61, float("nan"), 0.63, 0.62]
    with pytest.raises(ValueError):
        select_best_epoch(_history(values))
