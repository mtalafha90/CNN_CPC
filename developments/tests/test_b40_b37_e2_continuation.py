from __future__ import annotations

from pathlib import Path

import pytest

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b40_b37_e2_continuation import (
    B40_ADDITIONAL_EPOCHS,
    B40_COMPLETED_EPOCHS,
    B40_PARENT_EPOCHS,
    require_b40_continuation_contract,
)


def _config() -> dict:
    root = Path(__file__).resolve().parents[2]
    return dict(_read_config(root / "config" / "b40_b37_e2_continuation.yaml"))


def test_b40_declares_one_optimizer_reset_epoch_from_b37_e2():
    config = _config()
    policy = require_b40_continuation_contract(config)

    assert config["b40_parent_completed_epochs"] == B40_PARENT_EPOCHS == 2
    assert config["b40_additional_epochs"] == B40_ADDITIONAL_EPOCHS == 1
    assert B40_COMPLETED_EPOCHS == 3
    assert config["b40_optimizer_reset"] is True
    assert policy["crop_fraction"] == pytest.approx(0.90)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("b40_additional_epochs", 2),
        ("b40_optimizer_reset", False),
        ("b40_head_lr", 2e-4),
        ("b40_encoder_lr_scale", 0.10),
        ("b40_weight_decay", 0.0),
        ("b40_grad_clip", 2.0),
    ],
)
def test_b40_rejects_unregistered_changes(key: str, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        require_b40_continuation_contract(config)
