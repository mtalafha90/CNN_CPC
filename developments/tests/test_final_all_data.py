from __future__ import annotations

import pytest

from rsna_knee.final_all_data_training import (
    FINAL_ACTIVE_CELLS,
    FINAL_BATCHES,
    FINAL_B6_CELLS,
    FINAL_B6_SERIES,
    FINAL_B6_STUDIES,
    FINAL_ENCODER_LR,
    FINAL_EPOCHS,
    FINAL_GOLD_CELLS,
    FINAL_GOLD_SERIES,
    FINAL_GOLD_STUDIES,
    FINAL_GOLD_WEIGHT,
    FINAL_HEAD_LR,
    FINAL_TRAINING_SERIES,
    FINAL_TRAINING_STUDIES,
    require_final_contract,
)


def _config() -> dict:
    return {
        "seed": 2026,
        "competition_mode": True,
        "requested_gpus": 1,
        "allow_external_pretrained": True,
        "pretrained": True,
        "b7_epochs": FINAL_EPOCHS,
        "b7_max_batches_per_epoch": FINAL_BATCHES,
        "b7_encoder_lr": FINAL_ENCODER_LR,
        "b7_head_lr": FINAL_HEAD_LR,
        "final_include_gold": True,
        "final_gold_weight": FINAL_GOLD_WEIGHT,
        "final_encoder_frozen": True,
        "final_additional_label_smoothing": 0.0,
        "final_robust_loss": "none",
    }


def test_final_contract_accepts_predeclared_production_recipe():
    require_final_contract(_config())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("b7_epochs", 4),
        ("b7_max_batches_per_epoch", 1560),
        ("b7_encoder_lr", 1e-5),
        ("b7_head_lr", 2e-4),
        ("final_include_gold", False),
        ("final_gold_weight", 2.0),
        ("final_encoder_frozen", False),
        ("final_additional_label_smoothing", 0.1),
        ("final_robust_loss", "sce"),
    ],
)
def test_final_contract_rejects_posthoc_variants(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        require_final_contract(config)


def test_final_surface_accounting_is_exact():
    assert FINAL_TRAINING_STUDIES == FINAL_B6_STUDIES + FINAL_GOLD_STUDIES == 3178
    assert FINAL_ACTIVE_CELLS == FINAL_B6_CELLS + FINAL_GOLD_CELLS == 14819
    assert FINAL_TRAINING_SERIES == FINAL_B6_SERIES + FINAL_GOLD_SERIES == 17811
    assert FINAL_BATCHES == 1589
