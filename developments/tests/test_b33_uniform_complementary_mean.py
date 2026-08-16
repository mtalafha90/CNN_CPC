from __future__ import annotations

import torch
import torch.nn.functional as F

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b33_uniform_complementary_mean import (
    B33_EXPECTED_GATE_PARAMETERS,
    B33_EXPECTED_NEW_PARAMETERS,
    UniformComplementaryMeanKneeMILNet,
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


def test_b33_parameter_contract_has_only_one_new_gate():
    model = _model(UniformComplementaryMeanKneeMILNet)
    assert model.uniform_complementary_gate.numel() == B33_EXPECTED_GATE_PARAMETERS == 768
    assert B33_EXPECTED_NEW_PARAMETERS == 768
    assert torch.count_nonzero(model.uniform_complementary_gate).item() == 0
    assert not hasattr(model, "complementary_query")
    assert not hasattr(model, "local_context")
    assert not hasattr(model, "dispersion_gate")


def test_b33_shared_initialization_matches_b20_exactly():
    torch.manual_seed(314159)
    base = _model(HierarchicalSeriesKneeMILNet)
    torch.manual_seed(314159)
    candidate = _model(UniformComplementaryMeanKneeMILNet)
    candidate_state = candidate.state_dict()
    for name, value in base.state_dict().items():
        assert name in candidate_state
        assert torch.equal(value, candidate_state[name])


def test_b33_zero_gate_is_exact_b20_function():
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    candidate = _model(UniformComplementaryMeanKneeMILNet).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"uniform_complementary_gate"}
    assert unexpected == []
    batch = _batch()
    with torch.no_grad():
        y0 = base(*batch)
        y1 = candidate(*batch)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_b33_zero_gate_preserves_training_rng_path():
    base = _model(HierarchicalSeriesKneeMILNet, dropout=0.25).train()
    candidate = _model(UniformComplementaryMeanKneeMILNet, dropout=0.25).train()
    candidate.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    torch.manual_seed(9917)
    y0 = base(*batch)
    torch.manual_seed(9917)
    y1 = candidate(*batch)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_b33_uniform_summary_is_exact_mean_plus_parameter_free_ln():
    model = _model(UniformComplementaryMeanKneeMILNet).eval()
    active = torch.randn(5, 16, model.encoder.out_dim)
    with torch.no_grad():
        got = model._uniform_complementary_summary(active)
        expected = active.float().mean(dim=1)
        expected = F.layer_norm(expected, (expected.shape[-1],)).to(active.dtype)
    assert torch.allclose(got, expected, atol=1e-7, rtol=1e-7)


def test_b33_gate_is_bounded_and_gets_gradient_immediately():
    model = _model(UniformComplementaryMeanKneeMILNet)
    logits = model(*_batch())
    logits.sum().backward()
    grad = model.uniform_complementary_gate.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0

    with torch.no_grad():
        model.uniform_complementary_gate.fill_(100.0)
    gate = model.effective_uniform_gate()
    assert torch.isfinite(gate).all()
    assert float(gate.max()) <= 1.0
    assert float(gate.min()) >= -1.0


def test_b33_nonzero_gate_changes_output():
    model = _model(UniformComplementaryMeanKneeMILNet).eval()
    batch = _batch()
    with torch.no_grad():
        y0 = model(*batch)
        model.uniform_complementary_gate.fill_(0.1)
        y1 = model(*batch)
    assert not torch.allclose(y0, y1)


def test_b33_uniform_audit_is_finite_and_bounded():
    model = _model(UniformComplementaryMeanKneeMILNet).eval()
    model.enable_uniform_audit(True, reset=True)
    with torch.no_grad():
        _ = model(*_batch())
    state = model.uniform_audit_state(reset=True)
    assert state["series_count"] == 3
    assert state["raw_uniform_mean_norm_to_primary_mean"] is not None
    assert state["raw_uniform_mean_norm_to_primary_mean"] >= 0.0
    cosine = state["uniform_summary_to_primary_cosine_mean"]
    assert cosine is not None and -1.0 <= cosine <= 1.0
    for key in (
        "uniform_minus_primary_norm_ratio_mean",
        "uniform_minus_primary_norm_ratio_max",
        "effective_residual_norm_ratio_mean",
        "effective_residual_norm_ratio_max",
    ):
        value = state[key]
        assert value is not None and value >= 0.0
        assert torch.isfinite(torch.tensor(value))
    assert model.uniform_audit_state()["series_count"] == 0


def test_b33_bf16_and_empty_study_are_finite():
    model = _model(UniformComplementaryMeanKneeMILNet)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert torch.isfinite(model.uniform_complementary_gate.grad).all()

    volumes, _, meta = _batch()
    empty = torch.zeros(volumes.shape[0], volumes.shape[1], dtype=torch.bool)
    empty_logits = model(volumes, empty, torch.zeros_like(meta))
    assert torch.isfinite(empty_logits).all()
