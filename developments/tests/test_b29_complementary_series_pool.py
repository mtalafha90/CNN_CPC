from __future__ import annotations

import torch

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b29_complementary_series_pool import (
    B29_EXPECTED_GATE_PARAMETERS,
    B29_EXPECTED_NEW_PARAMETERS,
    B29_EXPECTED_QUERY_PARAMETERS,
    ComplementarySeriesPoolKneeMILNet,
)


def _model(cls, *, dropout: float = 0.0):
    return cls(
        2,
        pretrained_weights=False,
        normalize_input=False,
        dropout=dropout,
        encoder_batch_size=4,
        gradient_checkpointing=False,
        transformer_layers=1,
        transformer_heads=8,
        transformer_ff_mult=1.0,
        pathology_layers=1,
        series_pool_heads=8,
    )


def _batch():
    volumes = torch.randn(2, 3, 2, 3, 64, 64)
    present = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    meta = torch.tensor(
        [
            [[1, 1, 1], [2, 2, 2], [0, 0, 0]],
            [[3, 2, 2], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    return volumes, present, meta


def test_b29_has_exact_new_parameter_contract_and_zero_gate():
    model = _model(ComplementarySeriesPoolKneeMILNet)
    assert model.complementary_query.numel() == B29_EXPECTED_QUERY_PARAMETERS == 768
    assert model.complementary_gate.numel() == B29_EXPECTED_GATE_PARAMETERS == 768
    assert (
        model.complementary_query.numel() + model.complementary_gate.numel()
        == B29_EXPECTED_NEW_PARAMETERS
        == 1536
    )
    assert torch.count_nonzero(model.complementary_gate).item() == 0
    assert torch.count_nonzero(model.effective_complementary_gate()).item() == 0


def test_b29_zero_gate_is_functionally_b20_equivalent():
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    candidate = _model(ComplementarySeriesPoolKneeMILNet).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"complementary_query", "complementary_gate"}
    assert unexpected == []

    volumes, present, meta = _batch()
    with torch.no_grad():
        y_base = base(volumes, present, meta)
        y_candidate = candidate(volumes, present, meta)
    assert torch.allclose(y_base, y_candidate, atol=1e-5, rtol=1e-5)


def test_b29_zero_gate_preserves_b20_training_rng_path():
    # The complementary branch must not consume dropout RNG while its gate is
    # zero. With identical shared weights and an identical RNG reset, B20 and
    # B29 should therefore match even in training mode with dropout enabled.
    base = _model(HierarchicalSeriesKneeMILNet, dropout=0.25).train()
    candidate = _model(ComplementarySeriesPoolKneeMILNet, dropout=0.25).train()
    candidate.load_state_dict(base.state_dict(), strict=False)
    volumes, present, meta = _batch()

    torch.manual_seed(9917)
    y_base = base(volumes, present, meta)
    torch.manual_seed(9917)
    y_candidate = candidate(volumes, present, meta)
    assert torch.allclose(y_base, y_candidate, atol=1e-5, rtol=1e-5)


def test_b29_complementary_weights_are_finite_probabilities():
    model = _model(ComplementarySeriesPoolKneeMILNet).eval()
    active = torch.randn(4, 7, model.encoder.out_dim)
    weights = model._complementary_weights(active)
    assert weights.shape == (4, 7)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6)


def test_b29_gate_is_bounded_by_tanh():
    model = _model(ComplementarySeriesPoolKneeMILNet)
    with torch.no_grad():
        model.complementary_gate.fill_(100.0)
    gate = model.effective_complementary_gate()
    assert torch.isfinite(gate).all()
    assert float(gate.max()) <= 1.0
    assert float(gate.min()) >= -1.0


def test_b29_gradient_stages_gate_then_query():
    model = _model(ComplementarySeriesPoolKneeMILNet)
    logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.sum().backward()

    gate_grad = model.complementary_gate.grad
    query_grad = model.complementary_query.grad
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert torch.count_nonzero(gate_grad).item() > 0
    # Exact zero gate mathematically blocks the complementary-query gradient on
    # the first backward pass; this is the intended safe staged activation.
    assert torch.count_nonzero(query_grad).item() == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.complementary_gate.fill_(0.05)
    model(*_batch()).sum().backward()
    query_grad = model.complementary_query.grad
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert torch.count_nonzero(query_grad).item() > 0


def test_b29_bf16_autocast_and_empty_study_stay_finite():
    model = _model(ComplementarySeriesPoolKneeMILNet)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert torch.isfinite(model.complementary_gate.grad).all()
    assert torch.isfinite(model.complementary_query.grad).all()

    volumes, _, meta = _batch()
    empty = torch.zeros(volumes.shape[0], volumes.shape[1], dtype=torch.bool)
    empty_logits = model(volumes, empty, torch.zeros_like(meta))
    assert torch.isfinite(empty_logits).all()
