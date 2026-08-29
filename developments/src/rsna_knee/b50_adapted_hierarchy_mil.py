"""B50 — let the study hierarchy adapt to the features it is now being fed.

Every B37-descended model freezes the whole B34 aggregation before training
starts (`b37_highres_sparse_mil.py:313-315`). What trains is the encoder's final
stage and a sparse head whose output reaches the score through a gate the
completed runs measured at `|tanh(g)|` of about 0.022. So roughly 98% of every
0.714 submission is produced by 18.96M parameters that have not received a
gradient since B34 -- trained at 224 pixels, before the resolution change and
before every experiment from B37 onward.

Three measurements on the 58 expert studies say why that matters. Comparing
saved predictions and counting the study pairs each pair of models orders
differently, which is the only thing an ROC AUC can respond to:

    224 base  ->  448 encoder + tail        0.119
    B41 geometry -> B42 geometry            0.036
    B37 geometry -> B42 geometry            0.0145
    B42 global -> B42 combined (local MIL)  0.013

The encoder change is the only one that ever moved the model, and it is the only
one that produced a real jump on hidden data (0.694 -> 0.714). Everything since
has been refining components worth at most a hundredth. The hierarchy sits
between the encoder and the answer, it transmitted that 0.119 while frozen, and
it has never been allowed to adapt to what the encoder now produces.

B50 changes exactly one thing: whether it can.

    frozen_hierarchy_control      requires_grad_(False), exactly as B37-B49
    adapted_hierarchy_candidate   the aggregation trains at a reduced rate

Two details that are easy to get wrong and are tested here.

**Gradients are not training mode.** The base is kept in `eval()` throughout,
exactly as B37 keeps it. That is not incidental: B34's local-context scaffold is
defined to be active only while `model.training`, and its inference contract is
that `model.eval()` bypasses it exactly. Setting the base to train mode would
silently switch the architecture on. `requires_grad` and `train()/eval()` are
independent, so the hierarchy can learn while the scaffold stays bypassed.

**`local_context` stays frozen.** It is bypassed under eval, so it would receive
no gradient anyway, and leaving it out keeps the B34 inference contract exactly
reconstructible rather than merely equivalent.

Nothing here modifies B36, B37, B42, B46, B48 or B49, all of which are completed
frozen experiments that import the module this subclasses.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectSparseMILResidual,
    require_b42_contract,
)

B50_VERSION = "b50_adapted_hierarchy_mil_v1"
B50_EXPERIMENT = "B50_ADAPTED_STUDY_HIERARCHY"
B50_RUN_ROOT = "runs/083_Experiment_B50_adapted_study_hierarchy"

# The encoder tail's own scale. It is the only reduced learning rate this
# project has already run successfully at 448, so it is the one value that does
# not need to be invented. Frozen before any result: there is no sweep.
B50_HIERARCHY_LR_SCALE = 0.05

# Bypassed under eval, so it would collect no gradient. Excluded explicitly so
# the B34 inference contract stays exactly reconstructible.
B50_ALWAYS_FROZEN_PREFIXES = ("encoder.", "local_context.")

# Measured from the deployed B34 specification, not asserted from memory:
# 46,775,148 total, less the 27,820,128-parameter encoder and the 2,304-parameter
# bypassed local-context scaffold. A test rebuilds the real model and re-derives
# this rather than trusting the subtraction.
B50_EXPECTED_HIERARCHY_PARAMETERS = 18_952_716

B50_ARMS = ("frozen_hierarchy_control", "adapted_hierarchy_candidate")


def hierarchy_parameter_names(base: nn.Module) -> list[str]:
    """Every base parameter B50 is allowed to unfreeze, in a stable order."""
    return [
        name
        for name, _ in base.named_parameters()
        if not name.startswith(B50_ALWAYS_FROZEN_PREFIXES)
    ]


class B50AdaptedHierarchySparseMILResidual(B42ConstantAreaAspectSparseMILResidual):
    """B42 exactly, except that the study aggregation may receive gradients."""

    def __init__(self, *args, adapt_hierarchy: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adapt_hierarchy = bool(adapt_hierarchy)
        self.hierarchy_names = hierarchy_parameter_names(self.base)
        if self.adapt_hierarchy:
            lookup = dict(self.base.named_parameters())
            for name in self.hierarchy_names:
                lookup[name].requires_grad_(True)
        # The base stays in eval mode either way. See the module docstring: this
        # keeps B34's training-only local-context scaffold bypassed, which is
        # part of the frozen inference contract and not a detail of B50's.
        self.base.eval()

    def train(self, mode: bool = True):
        """Never put the base in training mode, whatever B50 unfroze."""
        super().train(mode)
        self.base.eval()
        self.base.encoder.eval()
        self.head.train(mode)
        return self

    def hierarchy_parameters(self) -> list[nn.Parameter]:
        lookup = dict(self.base.named_parameters())
        return [lookup[name] for name in self.hierarchy_names]

    def trainable_parameter_summary(self) -> dict:
        """What is actually learning, so the control can be checked against B42."""
        encoder = sum(
            p.numel() for p in self.base.encoder.parameters() if p.requires_grad
        )
        hierarchy = sum(p.numel() for p in self.hierarchy_parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "adapt_hierarchy": self.adapt_hierarchy,
            "encoder_trainable_parameters": int(encoder),
            "hierarchy_trainable_parameters": int(hierarchy),
            "head_trainable_parameters": int(head),
            "hierarchy_parameters_available": int(
                sum(p.numel() for p in self.hierarchy_parameters())
            ),
        }

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": B50_VERSION,
                "experiment": B50_EXPERIMENT,
                "trainable": self.trainable_parameter_summary(),
                "base_module_mode": "eval throughout; B34 local-context scaffold bypassed",
            }
        )
        return state


def b50_parameter_groups(
    model: B50AdaptedHierarchySparseMILResidual,
    *,
    head_lr: float,
    encoder_lr_scale: float,
    hierarchy_lr_scale: float = B50_HIERARCHY_LR_SCALE,
) -> list[dict]:
    """Head at full rate, encoder tail and hierarchy at reduced rates.

    The hierarchy is pretrained and large relative to 4,349 report-labelled
    studies, so it moves slowly or it simply memorises the label noise faster.
    """
    encoder = [p for p in model.base.encoder.parameters() if p.requires_grad]
    hierarchy = [p for p in model.hierarchy_parameters() if p.requires_grad]
    head = [p for p in model.head.parameters() if p.requires_grad]

    groups = [{"params": head, "lr": float(head_lr), "name": "sparse_head"}]
    if encoder:
        groups.append(
            {
                "params": encoder,
                "lr": float(head_lr) * float(encoder_lr_scale),
                "name": "encoder_tail",
            }
        )
    if hierarchy:
        groups.append(
            {
                "params": hierarchy,
                "lr": float(head_lr) * float(hierarchy_lr_scale),
                "name": "study_hierarchy",
            }
        )
    return groups


def require_b50_contract(config: dict) -> dict:
    """Everything B42 froze, plus the one thing B50 is allowed to move."""
    crop_policy = require_b42_contract(config)

    arm = str(config.get("b50_arm", "adapted_hierarchy_candidate"))
    if arm not in B50_ARMS:
        raise ValueError(f"B50 arm must be one of {B50_ARMS}; got {arm!r}")

    scale = float(config.get("b50_hierarchy_lr_scale", B50_HIERARCHY_LR_SCALE))
    if not np.isclose(scale, B50_HIERARCHY_LR_SCALE, atol=1e-12, rtol=0):
        raise ValueError(
            f"B50 freezes b50_hierarchy_lr_scale={B50_HIERARCHY_LR_SCALE}; got {scale}. "
            "The value is frozen before any result; there is no sweep."
        )

    return {
        "crop_policy": crop_policy,
        "arm": arm,
        "adapt_hierarchy": arm == "adapted_hierarchy_candidate",
        "hierarchy_lr_scale": scale,
    }


def b50_state(arm: str = "adapted_hierarchy_candidate") -> dict:
    if arm not in B50_ARMS:
        raise ValueError(f"B50 arm must be one of {B50_ARMS}")
    return {
        "version": B50_VERSION,
        "experiment": B50_EXPERIMENT,
        "arm": arm,
        "adapt_hierarchy": arm == "adapted_hierarchy_candidate",
        "hierarchy_lr_scale": B50_HIERARCHY_LR_SCALE,
        "hierarchy_parameters": B50_EXPECTED_HIERARCHY_PARAMETERS,
        "always_frozen": list(B50_ALWAYS_FROZEN_PREFIXES),
        "base_module_mode": "eval throughout; gradients without training mode",
    }


__all__ = [
    "B50_ALWAYS_FROZEN_PREFIXES",
    "B50_ARMS",
    "B50_EXPECTED_HIERARCHY_PARAMETERS",
    "B50_EXPERIMENT",
    "B50_HIERARCHY_LR_SCALE",
    "B50_RUN_ROOT",
    "B50_VERSION",
    "B50AdaptedHierarchySparseMILResidual",
    "b50_parameter_groups",
    "b50_state",
    "hierarchy_parameter_names",
    "require_b50_contract",
]
