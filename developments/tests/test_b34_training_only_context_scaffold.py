from __future__ import annotations

import torch

from rsna_knee.b34_training_only_context_scaffold import (
    B34_EXPECTED_CONTEXT_PARAMETERS,
    B34_EXPECTED_NEW_PARAMETERS,
    TrainingOnlyContextScaffoldKneeMILNet,
)


def _model():
    return TrainingOnlyContextScaffoldKneeMILNet(
        2,
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


def test_b34_parameter_contract_matches_b31_capacity():
    model = _model()
    assert model.complementary_query.numel() == 768
    assert model.complementary_gate.numel() == 768
    assert model.local_context.weight.numel() == B34_EXPECTED_CONTEXT_PARAMETERS == 2304
    assert B34_EXPECTED_NEW_PARAMETERS == 3840
    assert torch.count_nonzero(model.local_context.weight).item() == 0
    assert torch.count_nonzero(model.complementary_gate).item() == 0


def test_b34_context_is_active_only_in_training_mode():
    model = _model()
    active = torch.randn(5, 16, model.encoder.out_dim)
    with torch.no_grad():
        model.local_context.weight.fill_(0.02)

    model.train()
    train_context = model._contextualized_slice_features(active)
    assert not torch.allclose(train_context, active)

    model.eval()
    eval_context = model._contextualized_slice_features(active)
    assert torch.equal(eval_context, active)
    state = model.b34_state()
    assert state["training_context_active"] is False
    assert state["eval_context_exact_bypass"] is True
    assert state["inference_context_parameters_used"] == 0


def test_b34_eval_summary_is_exact_raw_query_summary_even_with_nonzero_scaffold():
    model = _model().eval()
    active = torch.randn(4, 16, model.encoder.out_dim)
    with torch.no_grad():
        model.local_context.weight.fill_(0.03)
        raw = model._complementary_summary(active)
        contextual, _, features = model._contextual_complementary_summary(active)
    assert torch.equal(features, active)
    assert torch.allclose(contextual, raw, atol=1e-7, rtol=1e-7)


def test_b34_training_scaffold_receives_gradient_when_outer_gate_is_open():
    model = _model().train()
    active = torch.randn(3, 16, model.encoder.out_dim)
    with torch.no_grad():
        model.complementary_gate.fill_(0.05)
    summary, _, _ = model._contextual_complementary_summary(active)
    loss = summary.square().mean()
    loss.backward()
    grad = model.local_context.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0
