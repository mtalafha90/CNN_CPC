from __future__ import annotations

import pytest
import torch
from torch import nn

from rsna_knee.discriminative_training import _discriminative_optimizer_bundle, _policy_payload


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.GELU())
        self.head = nn.Linear(4, 2)


class DummyRuntime:
    device = torch.device("cpu")
    use_scaler = False


def test_optimizer_uses_smaller_encoder_lr_and_disjoint_groups():
    model = TinyModel()
    config = {
        "lr": 1e-4,
        "encoder_lr": 1e-5,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
    }
    optimizer, scheduler, _ = _discriminative_optimizer_bundle(model, config, 8, DummyRuntime())

    assert [group["name"] for group in optimizer.param_groups] == ["heads", "encoder"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)

    head_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    encoder_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert head_ids.isdisjoint(encoder_ids)
    assert encoder_ids == {id(p) for p in model.encoder.parameters()}
    assert head_ids | encoder_ids == {id(p) for p in model.parameters()}

    scheduler.step()
    assert optimizer.param_groups[1]["lr"] < optimizer.param_groups[0]["lr"]


def test_optimizer_rejects_encoder_lr_above_head_lr():
    model = TinyModel()
    with pytest.raises(ValueError, match="must not exceed"):
        _discriminative_optimizer_bundle(
            model,
            {"lr": 1e-4, "encoder_lr": 2e-4, "min_lr": 1e-6},
            8,
            DummyRuntime(),
        )


def test_policy_records_single_intervention():
    payload = _policy_payload(
        {
            "lr": 1e-4,
            "encoder_lr": 1e-5,
            "min_lr": 1e-6,
            "ssl_encoder_checkpoint": "/tmp/ssl.pt",
            "ssl_checkpoint_source": "competition_training_data",
        }
    )
    assert payload["candidate"] == "B2_discriminative_ssl_finetune"
    assert payload["encoder_to_head_lr_ratio"] == pytest.approx(0.1)
    assert payload["encoder_frozen_epochs"] == 0
    assert payload["only_optimizer_differs_from_standard_train_fold"] is True
