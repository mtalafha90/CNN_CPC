"""B34 training-only local-context scaffold on the frozen B31/B29 hierarchy.

B34 is defined before any PV2 result is observed.  It directly tests the global
mechanistic hypothesis produced by the B31 context-zero counterfactual: the B31
local-context branch may help optimization even though its learned weights are
not needed by the final inference function.

Training:
    H = X + DWConv1d_k3(LN0(X))
    w = softmax(<H,q>/sqrt(D))

Evaluation/inference:
    H = X exactly
    w = softmax(<X,q>/sqrt(D))

The value sum always uses the original B20 slice tokens X, exactly as in B31.
The depthwise context parameters are therefore a training scaffold only.  They
remain trainable and are checkpointed for audit, but model.eval() bypasses them
exactly without modifying checkpoint weights.

B34 adds no parameters relative to B31 and no target-specific mechanism.  The
complete B31/B29/B20 construction path is retained, including zero initialization
of both the outer complementary gate and the depthwise context convolution.
"""
from __future__ import annotations

import copy

import torch

from .b31_local_context_complementary_pool import (
    B31_CONTEXT_VERSION,
    B31_EXPECTED_CONTEXT_PARAMETERS,
    B31_EXPECTED_GATE_PARAMETERS,
    B31_EXPECTED_NEW_PARAMETERS,
    B31_EXPECTED_QUERY_PARAMETERS,
    LocalContextComplementarySeriesPoolKneeMILNet,
    b31_model_spec,
)

B34_ARCHITECTURE = "b31_training_only_local_context_scaffold_eval_bypass_v1"
B34_AGGREGATION = "b29_query_with_b31_context_scaffold_during_training_and_exact_b29_scoring_at_eval_v1"
B34_SCAFFOLD_VERSION = "train_only_zero_init_depthwise_context_eval_exact_bypass_v1"
B34_EXPECTED_QUERY_PARAMETERS = B31_EXPECTED_QUERY_PARAMETERS
B34_EXPECTED_GATE_PARAMETERS = B31_EXPECTED_GATE_PARAMETERS
B34_EXPECTED_CONTEXT_PARAMETERS = B31_EXPECTED_CONTEXT_PARAMETERS
B34_EXPECTED_NEW_PARAMETERS = B31_EXPECTED_NEW_PARAMETERS


class TrainingOnlyContextScaffoldKneeMILNet(LocalContextComplementarySeriesPoolKneeMILNet):
    """B31 training path with exact local-context bypass under ``eval()``."""

    def _contextualized_slice_features(self, active_slice_features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Exact inference contract: trained local-context weights are not read.
            return active_slice_features
        return super()._contextualized_slice_features(active_slice_features)

    def b34_state(self) -> dict:
        return {
            "scaffold_version": B34_SCAFFOLD_VERSION,
            "training_context_active": bool(self.training),
            "eval_context_exact_bypass": True,
            "inference_context_parameters_used": 0,
            "complementary_pool": self.complementary_state(),
            "local_context_scaffold": self.local_context_state(),
        }


def b34_model_spec(config: dict, *, normalize_input: bool) -> dict:
    spec = copy.deepcopy(b31_model_spec(config, normalize_input=normalize_input))
    spec["architecture"] = B34_ARCHITECTURE
    spec["aggregation"] = B34_AGGREGATION
    spec["b34_scaffold_version"] = B34_SCAFFOLD_VERSION
    spec["b34_new_parameter_count"] = B34_EXPECTED_NEW_PARAMETERS
    spec["b34_training_scaffold"] = (
        "B31 parameter-free LN + zero-init bias-free depthwise Conv1d(k=3) affects complementary scores only while model.training"
    )
    spec["b34_inference"] = (
        "model.eval() bypasses local_context exactly; complementary query scores original B20 slice tokens"
    )
    spec["b34_inference_context_parameters_used"] = 0
    spec["b34_hypothesis"] = (
        "local context improves the optimization trajectory rather than contributing materially to the final inference function"
    )
    return spec


def build_b34_model(
    spec: dict,
    *,
    encoder_state: dict | None = None,
    pretrained_weights: bool = False,
) -> TrainingOnlyContextScaffoldKneeMILNet:
    if spec.get("architecture") != B34_ARCHITECTURE:
        raise ValueError("not a B34 training-only context-scaffold model spec")
    if spec.get("aggregation") != B34_AGGREGATION:
        raise ValueError("B34 aggregation policy mismatch")
    if spec.get("b34_scaffold_version") != B34_SCAFFOLD_VERSION:
        raise ValueError("B34 scaffold version mismatch")
    if int(spec.get("b34_new_parameter_count", -1)) != B34_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B34 new-parameter count mismatch")
    if int(spec.get("b31_context_parameter_count", -1)) != B34_EXPECTED_CONTEXT_PARAMETERS:
        raise ValueError("B34 inherited context parameter count mismatch")
    if spec.get("b31_context_version") != B31_CONTEXT_VERSION:
        raise ValueError("B34 inherited B31 context version mismatch")
    if encoder_state is not None and pretrained_weights:
        raise ValueError("encoder_state and pretrained_weights are mutually exclusive")

    model = TrainingOnlyContextScaffoldKneeMILNet(
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=bool(pretrained_weights),
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
        transformer_ff_mult=float(spec["transformer_ff_mult"]),
        pathology_layers=int(spec["pathology_layers"]),
        series_pool_heads=int(spec["series_pool_heads"]),
    )
    if int(model.complementary_query.numel()) != B34_EXPECTED_QUERY_PARAMETERS:
        raise ValueError("B34 query dimension changed")
    if int(model.complementary_gate.numel()) != B34_EXPECTED_GATE_PARAMETERS:
        raise ValueError("B34 gate dimension changed")
    if int(model.local_context.weight.numel()) != B34_EXPECTED_CONTEXT_PARAMETERS:
        raise ValueError("B34 scaffold dimension changed")
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model
