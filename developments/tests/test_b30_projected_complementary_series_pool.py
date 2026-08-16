from __future__ import annotations

import torch

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b30_projected_complementary_series_pool import (
    B30_EXPECTED_GATE_PARAMETERS,
    B30_EXPECTED_NEW_PARAMETERS,
    B30_EXPECTED_QUERY_PARAMETERS,
    ProjectedComplementarySeriesPoolKneeMILNet,
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


def test_b30_has_exact_new_parameter_contract_and_zero_gate():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet)
    assert model.complementary_query.numel() == B30_EXPECTED_QUERY_PARAMETERS == 768
    assert model.complementary_gate.numel() == B30_EXPECTED_GATE_PARAMETERS == 768
    assert (
        model.complementary_query.numel() + model.complementary_gate.numel()
        == B30_EXPECTED_NEW_PARAMETERS
        == 1536
    )
    assert torch.count_nonzero(model.complementary_gate).item() == 0
    assert torch.count_nonzero(model.effective_complementary_gate()).item() == 0


def test_b30_zero_gate_is_functionally_b20_equivalent():
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    candidate = _model(ProjectedComplementarySeriesPoolKneeMILNet).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"complementary_query", "complementary_gate"}
    assert unexpected == []

    volumes, present, meta = _batch()
    with torch.no_grad():
        y_base = base(volumes, present, meta)
        y_candidate = candidate(volumes, present, meta)
    assert torch.allclose(y_base, y_candidate, atol=1e-5, rtol=1e-5)


def test_b30_zero_gate_preserves_b20_training_rng_path():
    base = _model(HierarchicalSeriesKneeMILNet, dropout=0.25).train()
    candidate = _model(ProjectedComplementarySeriesPoolKneeMILNet, dropout=0.25).train()
    candidate.load_state_dict(base.state_dict(), strict=False)
    volumes, present, meta = _batch()

    torch.manual_seed(9917)
    y_base = base(volumes, present, meta)
    torch.manual_seed(9917)
    y_candidate = candidate(volumes, present, meta)
    assert torch.allclose(y_base, y_candidate, atol=1e-5, rtol=1e-5)


def test_b30_complementary_path_detaches_shared_projection_parameters():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet)
    active = torch.randn(4, 7, model.encoder.out_dim, requires_grad=True)
    summary, weights = model._projected_complementary_summary(active)
    assert torch.isfinite(summary).all()
    assert torch.isfinite(weights).all()

    # A non-uniform deterministic probe avoids the zero-gradient symmetry of
    # summing all LayerNorm outputs equally.
    probe = torch.linspace(-1.0, 1.0, summary.shape[-1], dtype=summary.dtype)
    (summary * probe[None, :]).sum().backward()

    assert active.grad is not None and torch.count_nonzero(active.grad).item() > 0
    q_grad = model.complementary_query.grad
    assert q_grad is not None and torch.count_nonzero(q_grad).item() > 0
    assert model.series_pool.attention.in_proj_weight.grad is None
    assert model.series_pool.attention.in_proj_bias.grad is None
    assert model.series_pool.attention.out_proj.weight.grad is None
    assert model.series_pool.attention.out_proj.bias.grad is None
    assert model.series_pool.norm.weight.grad is None
    assert model.series_pool.norm.bias.grad is None


def test_b30_projected_attention_weights_are_finite_probabilities():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet).eval()
    active = torch.randn(4, 7, model.encoder.out_dim)
    _, weights = model._projected_complementary_summary(active)
    assert weights.shape == (4, 8, 7)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(
        weights.float().sum(dim=-1),
        torch.ones(4, 8),
        atol=1e-6,
        rtol=1e-6,
    )


def test_b30_gate_is_bounded_by_tanh():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet)
    with torch.no_grad():
        model.complementary_gate.fill_(100.0)
    gate = model.effective_complementary_gate()
    assert torch.isfinite(gate).all()
    assert float(gate.max()) <= 1.0
    assert float(gate.min()) >= -1.0


def test_b30_gradient_stages_gate_then_query():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet)
    logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.sum().backward()

    gate_grad = model.complementary_gate.grad
    query_grad = model.complementary_query.grad
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert torch.count_nonzero(gate_grad).item() > 0
    assert torch.count_nonzero(query_grad).item() == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.complementary_gate.fill_(0.05)
    model(*_batch()).sum().backward()
    query_grad = model.complementary_query.grad
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert torch.count_nonzero(query_grad).item() > 0


def test_b30_attention_audit_is_finite_bounded_and_resettable():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet).eval()
    model.enable_attention_audit(True, reset=True)
    with torch.no_grad():
        _ = model(*_batch())
    state = model.attention_audit_state(reset=True)
    assert state["series_count"] == 3
    for key in (
        "primary_attention_entropy_normalized_mean",
        "complementary_attention_entropy_normalized_mean",
        "js_divergence_normalized_mean",
        "top1_slice_agreement",
        "top3_slice_overlap_fraction_mean",
        "effective_residual_norm_ratio_mean",
        "effective_residual_norm_ratio_max",
    ):
        assert state[key] is not None
        assert torch.isfinite(torch.tensor(state[key]))
    assert 0.0 <= state["primary_attention_entropy_normalized_mean"] <= 1.0
    assert 0.0 <= state["complementary_attention_entropy_normalized_mean"] <= 1.0
    assert 0.0 <= state["js_divergence_normalized_mean"] <= 1.0
    assert 0.0 <= state["top1_slice_agreement"] <= 1.0
    assert 0.0 <= state["top3_slice_overlap_fraction_mean"] <= 1.0
    assert model.attention_audit_state()["series_count"] == 0


def test_b30_bf16_autocast_and_empty_study_stay_finite():
    model = _model(ProjectedComplementarySeriesPoolKneeMILNet)
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
