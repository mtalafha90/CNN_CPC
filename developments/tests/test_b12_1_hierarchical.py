from pathlib import Path

import pytest
import torch
import yaml

from rsna_knee.b12_1_hierarchical import (
    B12_1_AGGREGATION,
    B12_1_ARCHITECTURE,
    LearnedSeriesPool,
    b12_1_model_spec,
)
from rsna_knee.b12_1_training import (
    B12_1_EXPERIMENT,
    _require_b12_1_contract,
)


def test_learned_series_pool_returns_one_token_per_series():
    pool = LearnedSeriesPool(dim=32, heads=8, dropout=0.0).eval()
    slices = torch.randn(5, 16, 32)
    with torch.no_grad():
        token = pool(slices)
    assert token.shape == (5, 32)
    assert torch.isfinite(token).all()


def test_learned_series_pool_is_token_order_invariant():
    torch.manual_seed(7)
    pool = LearnedSeriesPool(dim=32, heads=8, dropout=0.0).eval()
    slices = torch.randn(3, 6, 32)
    permutation = torch.tensor([5, 1, 4, 0, 3, 2])
    with torch.no_grad():
        a = pool(slices)
        b = pool(slices[:, permutation])
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_b12_1_model_spec_freezes_hierarchical_aggregation():
    spec = b12_1_model_spec(
        {
            "b7_n_slices": 16,
            "b7_transformer_heads": 8,
            "b12_1_series_pool_heads": 8,
        },
        normalize_input=False,
    )
    assert spec["architecture"] == B12_1_ARCHITECTURE
    assert spec["aggregation"] == B12_1_AGGREGATION
    assert spec["series_pool_heads"] == 8
    assert "no series-position embedding" in spec["metadata_embeddings"]


def test_b12_1_contract_accepts_frozen_recipe():
    _require_b12_1_contract(
        {
            "b12_1_experiment_name": B12_1_EXPERIMENT,
            "b7_epochs": 4,
            "b7_batch_size": 2,
            "b12_1_series_pool_heads": 8,
            "b12_use_physical_scale": False,
        }
    )


def test_b12_1_contract_rejects_pool_head_tuning():
    with pytest.raises(ValueError, match="series_pool_heads=8"):
        _require_b12_1_contract(
            {
                "b12_1_experiment_name": B12_1_EXPERIMENT,
                "b7_epochs": 4,
                "b7_batch_size": 2,
                "b12_1_series_pool_heads": 4,
                "b12_use_physical_scale": False,
            }
        )


def test_b12_1_contract_rejects_extra_epochs():
    with pytest.raises(ValueError, match="exactly four epochs"):
        _require_b12_1_contract(
            {
                "b12_1_experiment_name": B12_1_EXPERIMENT,
                "b7_epochs": 5,
                "b7_batch_size": 2,
                "b12_1_series_pool_heads": 8,
                "b12_use_physical_scale": False,
            }
        )


def test_b12_1_checked_in_config_matches_frozen_b12_controls():
    path = Path(__file__).resolve().parents[1] / "configs" / "b12_1_hierarchical.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "b12_1_experiment_name": B12_1_EXPERIMENT,
        "b12_1_series_pool_heads": 8,
        "b12_use_physical_scale": False,
        "b7_n_slices": 16,
        "b7_image_size": 224,
        "b7_triplet_gap": 1,
        "b7_batch_size": 2,
        "b7_encoder_batch_size": 24,
        "b7_gradient_checkpointing": True,
        "b7_dropout": 0.25,
        "b7_transformer_layers": 2,
        "b7_transformer_heads": 8,
        "b7_transformer_ff_mult": 2.0,
        "b7_pathology_layers": 1,
        "b7_epochs": 4,
        "b7_max_batches_per_epoch": 1560,
        "b7_encoder_lr": 0.00001,
        "b7_head_lr": 0.0001,
        "b7_min_lr": 0.000001,
        "b7_weight_decay": 0.0001,
        "b7_grad_clip": 1.0,
        "b7_noise_std": 0.02,
        "b7_slice_dropout": 0.08,
        "b7_train_gap_choices": [1, 2],
        "b7_center_jitter": 2,
        "b7_rotation_deg": 5.0,
        "b7_translate_frac": 0.03,
        "b7_scale_jitter": 0.05,
        "b7_gamma_jitter": 0.12,
        "b7_bias_field_strength": 0.08,
        "b7_eval_tta_offsets": [-1, 0, 1],
        "b7_eval_batch_size": 2,
        "b7_n_bootstrap": 5000,
    }
    for key, value in expected.items():
        assert config[key] == value, f"frozen B12.1 control drifted: {key}"
