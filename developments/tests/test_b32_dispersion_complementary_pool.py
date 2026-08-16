from __future__ import annotations

import torch
import torch.nn.functional as F

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b29_complementary_series_pool import ComplementarySeriesPoolKneeMILNet
from rsna_knee.b32_dispersion_complementary_pool import (
    B32_EXPECTED_DISPERSION_GATE_PARAMETERS,
    B32_EXPECTED_MEAN_GATE_PARAMETERS,
    B32_EXPECTED_NEW_PARAMETERS,
    B32_EXPECTED_QUERY_PARAMETERS,
    B32_VARIANCE_EPS,
    DispersionComplementarySeriesPoolKneeMILNet,
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


def test_b32_parameter_contract_and_exact_zero_gates():
    model = _model(DispersionComplementarySeriesPoolKneeMILNet)
    assert model.complementary_query.numel() == B32_EXPECTED_QUERY_PARAMETERS == 768
    assert model.complementary_gate.numel() == B32_EXPECTED_MEAN_GATE_PARAMETERS == 768
    assert model.dispersion_gate.numel() == B32_EXPECTED_DISPERSION_GATE_PARAMETERS == 768
    assert B32_EXPECTED_NEW_PARAMETERS == 2304
    assert torch.count_nonzero(model.complementary_gate).item() == 0
    assert torch.count_nonzero(model.dispersion_gate).item() == 0
    assert not hasattr(model, "local_context")


def test_b32_zero_gates_are_exact_b20_function():
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    candidate = _model(DispersionComplementarySeriesPoolKneeMILNet).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {
        "complementary_query",
        "complementary_gate",
        "dispersion_gate",
    }
    assert unexpected == []
    batch = _batch()
    with torch.no_grad():
        y0 = base(*batch)
        y1 = candidate(*batch)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_b32_zero_gates_preserve_b20_training_rng_path():
    base = _model(HierarchicalSeriesKneeMILNet, dropout=0.25).train()
    candidate = _model(DispersionComplementarySeriesPoolKneeMILNet, dropout=0.25).train()
    candidate.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    torch.manual_seed(9917)
    y0 = base(*batch)
    torch.manual_seed(9917)
    y1 = candidate(*batch)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_b32_zero_dispersion_gate_reproduces_b29():
    b29 = _model(ComplementarySeriesPoolKneeMILNet).eval()
    b32 = _model(DispersionComplementarySeriesPoolKneeMILNet).eval()
    missing, unexpected = b32.load_state_dict(b29.state_dict(), strict=False)
    assert set(missing) == {"dispersion_gate"}
    assert unexpected == []

    with torch.no_grad():
        b29.complementary_gate.fill_(0.05)
        b32.complementary_gate.copy_(b29.complementary_gate)
        batch = _batch()
        y29 = b29(*batch)
        y32 = b32(*batch)
    assert torch.allclose(y29, y32, atol=1e-5, rtol=1e-5)


def test_b32_weighted_moments_match_manual_formula():
    model = _model(DispersionComplementarySeriesPoolKneeMILNet).eval()
    active = torch.randn(5, 16, model.encoder.out_dim)
    with torch.no_grad():
        weights, mean_summary, dispersion_summary, mu_raw, sigma_raw = model._weighted_moments(active)
        w = model._complementary_weights(active).float()
        x = active.float()
        mu_expected = torch.sum(w[:, :, None] * x, dim=1)
        var_expected = torch.sum(w[:, :, None] * (x - mu_expected[:, None, :]).square(), dim=1)
        sigma_expected = torch.sqrt(var_expected.clamp_min(0.0) + B32_VARIANCE_EPS)
        mean_expected = F.layer_norm(mu_expected, (mu_expected.shape[-1],)).to(active.dtype)
        dispersion_expected = F.layer_norm(
            sigma_expected, (sigma_expected.shape[-1],)
        ).to(active.dtype)
    assert torch.allclose(weights.float(), w, atol=1e-7, rtol=1e-7)
    assert torch.allclose(mu_raw, mu_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(sigma_raw, sigma_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(mean_summary, mean_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(dispersion_summary, dispersion_expected, atol=1e-6, rtol=1e-6)
    assert torch.all(sigma_raw > 0)


def test_b32_first_backward_moves_both_gates_then_query_couples():
    model = _model(DispersionComplementarySeriesPoolKneeMILNet)
    logits = model(*_batch())
    probe = torch.linspace(-1.0, 1.0, logits.numel(), dtype=logits.dtype).reshape_as(logits)
    (logits * probe).sum().backward()

    mean_gate_grad = model.complementary_gate.grad
    dispersion_gate_grad = model.dispersion_gate.grad
    query_grad = model.complementary_query.grad
    assert mean_gate_grad is not None and torch.count_nonzero(mean_gate_grad).item() > 0
    assert dispersion_gate_grad is not None and torch.count_nonzero(dispersion_gate_grad).item() > 0
    assert query_grad is not None and torch.count_nonzero(query_grad).item() == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.complementary_gate.fill_(0.05)
        model.dispersion_gate.fill_(0.05)
    logits = model(*_batch())
    (logits * probe).sum().backward()
    assert torch.count_nonzero(model.complementary_query.grad).item() > 0


def test_b32_gates_are_bounded_by_tanh():
    model = _model(DispersionComplementarySeriesPoolKneeMILNet)
    with torch.no_grad():
        model.complementary_gate.fill_(100.0)
        model.dispersion_gate.fill_(-100.0)
    mean_gate = model.effective_complementary_gate()
    dispersion_gate = model.effective_dispersion_gate()
    assert torch.isfinite(mean_gate).all() and torch.isfinite(dispersion_gate).all()
    assert float(mean_gate.max()) <= 1.0 and float(mean_gate.min()) >= -1.0
    assert float(dispersion_gate.max()) <= 1.0 and float(dispersion_gate.min()) >= -1.0


def test_b32_dispersion_audit_is_finite_and_bounded():
    model = _model(DispersionComplementarySeriesPoolKneeMILNet).eval()
    model.enable_dispersion_audit(True, reset=True)
    with torch.no_grad():
        _ = model(*_batch())
    state = model.dispersion_audit_state(reset=True)
    assert state["series_count"] == 3
    assert 0.0 <= state["attention_entropy_normalized_mean"] <= 1.0
    for key in (
        "weighted_mean_vs_uniform_mean_norm_ratio_mean",
        "weighted_dispersion_vs_uniform_dispersion_norm_ratio_mean",
        "raw_dispersion_to_raw_mean_norm_ratio_mean",
        "mean_residual_norm_ratio_mean",
        "dispersion_residual_norm_ratio_mean",
        "combined_residual_norm_ratio_mean",
        "combined_residual_norm_ratio_max",
    ):
        value = state[key]
        assert value is not None and value >= 0.0
        assert torch.isfinite(torch.tensor(value))
    cosine = state["mean_dispersion_residual_cosine_mean"]
    assert cosine is not None and -1.0 <= cosine <= 1.0
    assert torch.isfinite(torch.tensor(cosine))
    assert model.dispersion_audit_state()["series_count"] == 0


def test_b32_bf16_and_empty_study_are_finite():
    model = _model(DispersionComplementarySeriesPoolKneeMILNet)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert torch.isfinite(model.complementary_gate.grad).all()
    assert torch.isfinite(model.dispersion_gate.grad).all()
    assert torch.isfinite(model.complementary_query.grad).all()

    volumes, _, meta = _batch()
    empty = torch.zeros(volumes.shape[0], volumes.shape[1], dtype=torch.bool)
    empty_logits = model(volumes, empty, torch.zeros_like(meta))
    assert torch.isfinite(empty_logits).all()
