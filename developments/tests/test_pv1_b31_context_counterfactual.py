from __future__ import annotations

import torch

from rsna_knee.b31_local_context_complementary_pool import LocalContextComplementarySeriesPoolKneeMILNet
from rsna_knee.prospective_weak_v1_b31_context_counterfactual import (
    classify_primary_counterfactual,
    zero_b31_local_context_for_counterfactual,
)


def _model():
    return LocalContextComplementarySeriesPoolKneeMILNet(
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


def test_context_counterfactual_zeros_only_local_context():
    model = _model().eval()
    with torch.no_grad():
        model.local_context.weight.fill_(0.0125)
    before_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    before, after = zero_b31_local_context_for_counterfactual(model)

    assert before["weight_l2"] > 0.0
    assert after["weight_l2"] == 0.0
    assert torch.count_nonzero(model.local_context.weight).item() == 0

    for name, value in model.state_dict().items():
        if name == "local_context.weight":
            assert torch.count_nonzero(value).item() == 0
        else:
            assert torch.equal(value, before_state[name]), name


def test_context_counterfactual_rejects_already_zero_context():
    model = _model().eval()
    try:
        zero_b31_local_context_for_counterfactual(model)
    except ValueError as exc:
        assert "already zero" in str(exc)
    else:
        raise AssertionError("expected zero-context checkpoint to be rejected")


def test_context_counterfactual_interpretation_rule():
    assert classify_primary_counterfactual({"ci_lower": 0.001, "ci_upper": 0.02}) == (
        "trained_context_directly_improves_final_inference"
    )
    assert classify_primary_counterfactual({"ci_lower": -0.02, "ci_upper": -0.001}) == (
        "trained_context_is_harmful_at_inference_despite_training_path"
    )
    assert classify_primary_counterfactual({"ci_lower": -0.01, "ci_upper": 0.01}) == (
        "direct_inference_effect_unresolved_optimization_path_remains_plausible"
    )
