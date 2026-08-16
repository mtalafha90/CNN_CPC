from __future__ import annotations

import torch

from rsna_knee.b12_1_hierarchical import HierarchicalSeriesKneeMILNet
from rsna_knee.b28_max_evidence_residual import (
    B28_EXPECTED_GATE_PARAMETERS,
    MaxEvidenceResidualKneeMILNet,
)


def _model(cls):
    return cls(
        1,
        pretrained_weights=False,
        normalize_input=False,
        dropout=0.0,
        encoder_batch_size=4,
        gradient_checkpointing=False,
        transformer_layers=1,
        transformer_heads=8,
        transformer_ff_mult=1.0,
        pathology_layers=1,
        series_pool_heads=8,
    )


def _batch():
    volumes = torch.randn(2, 3, 1, 3, 64, 64)
    present = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    meta = torch.tensor(
        [
            [[1, 1, 1], [2, 2, 2], [0, 0, 0]],
            [[3, 2, 2], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    return volumes, present, meta


def test_b28_has_exact_zero_initialized_feature_gate():
    model = _model(MaxEvidenceResidualKneeMILNet)
    assert model.max_residual_gate.numel() == B28_EXPECTED_GATE_PARAMETERS == 768
    assert torch.count_nonzero(model.max_residual_gate).item() == 0
    assert torch.count_nonzero(model.effective_max_residual_gate()).item() == 0


def test_b28_zero_gate_is_functionally_b20_equivalent():
    base = _model(HierarchicalSeriesKneeMILNet).eval()
    candidate = _model(MaxEvidenceResidualKneeMILNet).eval()
    missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
    assert missing == ["max_residual_gate"]
    assert unexpected == []

    volumes, present, meta = _batch()
    with torch.no_grad():
        y_base = base(volumes, present, meta)
        y_candidate = candidate(volumes, present, meta)
    assert torch.allclose(y_base, y_candidate, atol=1e-5, rtol=1e-5)


def test_b28_content_recovery_removes_position_and_metadata_terms():
    model = _model(MaxEvidenceResidualKneeMILNet).eval()
    b, k, s, d = 1, 2, 1, model.encoder.out_dim
    present = torch.tensor([[1, 1]], dtype=torch.bool)
    meta = torch.tensor([[[1, 1, 1], [2, 2, 2]]], dtype=torch.long)
    content = torch.randn(b, k, s, d)
    plane = model.plane_embedding(meta[:, :, 0])
    fluid = model.fluid_embedding(meta[:, :, 1])
    fat = model.fat_embedding(meta[:, :, 2])
    metadata = plane + fluid + fat
    assembled = content + model.slice_position[None, None, :, :] + metadata[:, :, None, :]
    recovered = model._content_slice_features(assembled, present, meta)
    assert torch.allclose(recovered, content, atol=1e-6, rtol=1e-6)


def test_b28_gate_is_bounded_by_tanh():
    model = _model(MaxEvidenceResidualKneeMILNet)
    with torch.no_grad():
        model.max_residual_gate.fill_(100.0)
    gate = model.effective_max_residual_gate()
    assert torch.isfinite(gate).all()
    assert float(gate.max()) <= 1.0
    assert float(gate.min()) >= -1.0


def test_b28_backward_reaches_gate_from_zero_initialization():
    model = _model(MaxEvidenceResidualKneeMILNet)
    logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    grad = model.max_residual_gate.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0


def test_b28_bf16_autocast_and_empty_study_stay_finite():
    model = _model(MaxEvidenceResidualKneeMILNet)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*_batch())
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    assert torch.isfinite(model.max_residual_gate.grad).all()

    volumes, _, meta = _batch()
    empty = torch.zeros(volumes.shape[0], volumes.shape[1], dtype=torch.bool)
    empty_logits = model(volumes, empty, torch.zeros_like(meta))
    assert torch.isfinite(empty_logits).all()
