from __future__ import annotations

import pytest
import torch
from torch import nn

from rsna_knee.b17_training import (
    B17_EPOCHS,
    B17_ENCODER_LR,
    B17_HEAD_LR,
    encoder_state_sha256,
    freeze_encoder,
    require_b17_contract,
)


def _config() -> dict:
    return {
        "seed": 2026,
        "competition_mode": True,
        "requested_gpus": 1,
        "allow_external_pretrained": True,
        "pretrained": True,
        "b7_epochs": B17_EPOCHS,
        "b7_max_batches_per_epoch": 1560,
        "b7_encoder_lr": B17_ENCODER_LR,
        "b7_head_lr": B17_HEAD_LR,
        "b17_encoder_frozen": True,
        "b17_label_smoothing": 0.0,
        "b17_robust_loss": "none",
    }


def test_b17_contract_accepts_frozen_recipe():
    require_b17_contract(_config())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("b7_epochs", 4),
        ("b7_encoder_lr", 1e-5),
        ("b7_max_batches_per_epoch", 1559),
        ("b17_encoder_frozen", False),
        ("b17_label_smoothing", 0.1),
        ("b17_robust_loss", "sce"),
    ],
)
def test_b17_contract_rejects_unfrozen_or_extra_interventions(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        require_b17_contract(config)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
        self.head = nn.Linear(3, 1)


def test_freeze_encoder_disables_gradients_and_train_mode():
    model = _ToyModel()
    model.train()
    assert model.encoder.training
    freeze_encoder(model)
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_encoder_state_sha256_is_stable_and_sensitive():
    torch.manual_seed(17)
    model = _ToyModel()
    first = encoder_state_sha256(model.encoder)
    second = encoder_state_sha256(model.encoder)
    assert first == second
    with torch.no_grad():
        next(model.encoder.parameters()).add_(0.01)
    third = encoder_state_sha256(model.encoder)
    assert third != first
