from __future__ import annotations

import torch

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b29_complementary_series_pool import ComplementarySeriesPoolKneeMILNet
from rsna_knee.b31_local_context_complementary_pool import (
    B31_EXPECTED_CONTEXT_PARAMETERS,
    B31_EXPECTED_GATE_PARAMETERS,
    B31_EXPECTED_NEW_PARAMETERS,
    B31_EXPECTED_QUERY_PARAMETERS,
    LocalContextComplementarySeriesPoolKneeMILNet,
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


def test_b31_parameter_contract_and_exact_zero_context():
    model = _model(LocalContextComplementarySeriesPoolKneeMILNet)
    assert model.complementary_query.numel() == B31_EXPECTED_QUERY_PARAMETERS == 768
    assert model.complementary_gate.numel() == B31_EXPECTED_GATE_PARAMETERS == 768
    assert model.local_context.weight.numel() == B31_EXPECTED_CONTEXT_PARAMETERS == 2304
    assert B31_EXPECTED_NEW_PARAMETERS == 3840
    assert model.local_context.groups == 768
    assert model.local_context.kernel_size == (3,)
    assert model.local_context.bias is None
    assert torch.count_nonzero(model.local_context.weight).item() == 0
    assert torch.count_nonzero(model.complementary_gate).item() == 0


def test_b31_zero_gate_is_exact_b20_function():
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    candidate = _model(LocalContextComplementarySeriesPoolKneeMILNet).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {
        "complementary_query",
        "complementary_gate",
        "local_context.weight",
    }
    assert unexpected == []
    with torch.no_grad():
        y0 = base(*_batch())
        y1 = candidate(*_batch())
    # Different random batches above are intentional bad practice; use one fixed batch below.
    batch = _batch()
    with torch.no_grad():
        y0 = base(*batch)
        y1 = candidate(*batch)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_b31_zero_gate_preserves_training_rng_path():
    base = _model(HierarchicalSeriesKneeMILNet, dropout=0.25).train()
    candidate = _model(LocalContextComplementarySeriesPoolKneeMILNet, dropout=0.25).train()
    candidate.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    torch.manual_seed(9917)
    y0 = base(*batch)
    torch.manual_seed(9917)
    y1 = candidate(*batch)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_b31_zero_context_reproduces_b29_scoring_and_summary():
    b29 = _model(ComplementarySeriesPoolKneeMILNet).eval()
    b31 = _model(LocalContextComplementarySeriesPoolKneeMILNet).eval()
    missing, unexpected = b31.load_state_dict(b29.state_dict(), strict=False)
    assert set(missing) == {"local_context.weight"}
    assert unexpected == []
    active = torch.randn(5, 16, b29.encoder.out_dim)
    with torch.no_grad():
        raw_weights = b29._complementary_weights(active)
        ctx_weights, contextual = b31._contextual_weights(active)
        raw_summary = b29._complementary_summary(active)
        ctx_summary, _, _ = b31._contextual_complementary_summary(active)
    assert torch.equal(contextual, active)
    assert torch.allclose(raw_weights, ctx_weights, atol=1e-7, rtol=1e-7)
    assert torch.allclose(raw_summary, ctx_summary, atol=1e-7, rtol=1e-7)


def test_b31_first_backward_moves_gate_only_then_context_and_query_couple():
    model = _model(LocalContextComplementarySeriesPoolKneeMILNet)
    model(*_batch()).sum().backward()
    gate_grad = model.complementary_gate.grad
    query_grad = model.complementary_query.grad
    context_grad = model.local_context.weight.grad
    assert gate_grad is not None and torch.count_nonzero(gate_grad).item() > 0
    assert query_grad is not None and torch.count_nonzero(query_grad).item() == 0
    assert context_grad is not None and torch.count_nonzero(context_grad).item() == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.complementary_gate.fill_(0.05)
    model(*_batch()).sum().backward()
    assert torch.count_nonzero(model.complementary_query.grad).item() > 0
    assert torch.count_nonzero(model.local_context.weight.grad).item() > 0


def test_b31_context_changes_scores_not_values():
    model = _model(LocalContextComplementarySeriesPoolKneeMILNet).eval()
    active = torch.randn(4, 16, model.encoder.out_dim)
    with torch.no_grad():
        model.local_context.weight.fill_(0.02)
        summary, weights, contextual = model._contextual_complementary_summary(active)
        expected = torch.sum(weights[:, :, None] * active, dim=1)
        expected = torch.nn.functional.layer_norm(expected.float(), (expected.shape[-1],)).to(active.dtype)
    assert not torch.allclose(contextual, active)
    assert torch.allclose(summary, expected, atol=1e-6, rtol=1e-6)


def test_b31_attention_audit_is_finite_and_bounded():
    model = _model(LocalContextComplementarySeriesPoolKneeMILNet).eval()
    model.enable_attention_audit(True, reset=True)
    with torch.no_grad():
        _ = model(*_batch())
    state = model.attention_audit_state(reset=True)
    assert state["series_count"] == 3
    for key in (
        "raw_b29_attention_entropy_normalized_mean",
        "context_attention_entropy_normalized_mean",
        "raw_vs_context_js_divergence_normalized_mean",
        "raw_vs_context_top1_agreement",
        "raw_vs_context_top3_overlap_fraction_mean",
    ):
        value = state[key]
        assert value is not None
        assert 0.0 <= value <= 1.0
    for key in (
        "raw_attention_adjacent_absdiff_mean",
        "context_attention_adjacent_absdiff_mean",
        "context_delta_norm_ratio_mean",
        "context_delta_norm_ratio_max",
        "effective_residual_norm_ratio_mean",
        "effective_residual_norm_ratio_max",
    ):
        value = state[key]
        assert value is not None and value >= 0.0
        assert torch.isfinite(torch.tensor(value))
    assert model.attention_audit_state()["series_count"] == 0


def test_b31_bf16_and_empty_study_are_finite():
    model = _model(LocalContextComplementarySeriesPoolKneeMILNet)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert torch.isfinite(model.complementary_gate.grad).all()
    assert torch.isfinite(model.complementary_query.grad).all()
    assert torch.isfinite(model.local_context.weight.grad).all()

    volumes, _, meta = _batch()
    empty = torch.zeros(volumes.shape[0], volumes.shape[1], dtype=torch.bool)
    empty_logits = model(volumes, empty, torch.zeros_like(meta))
    assert torch.isfinite(empty_logits).all()
