import pytest
import torch

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
