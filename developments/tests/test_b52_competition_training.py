"""B52 must actually change the three things it claims to change.

The whole argument for B52 is that the frozen contract left the model untrained:
one encoder stage of five, no augmentation, two epochs, no checkpoint selection.
If any of those silently stayed as it was, B52 would be another perturbation of
the same floor while reporting that it was not. Each is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from rsna_knee.b7_weak_supervision import make_b7_dataset_config
from rsna_knee.b52_competition_training import (
    B52_DEFAULT_ENCODER_STAGES,
    B52_DEFAULT_EPOCHS,
    B52_INHERITED_ENCODER_STAGES,
    B52_INHERITED_EPOCHS,
    b52_parameter_groups,
    macro_auc,
    masked_binary_targets,
)
from rsna_knee.constants import TARGETS
from rsna_knee.encoder_finetune import MAX_TRAINABLE_STAGES


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)


class _Model(nn.Module):
    """The shape b52_parameter_groups reads: base.encoder, hierarchy, head."""

    def __init__(self) -> None:
        super().__init__()
        self.base = _Base()
        self.head = nn.Linear(4, len(TARGETS))
        self._hierarchy = nn.Linear(4, 4)

    def hierarchy_parameters(self):
        return list(self._hierarchy.parameters())


# --- the three changes ------------------------------------------------------


def test_b52_trains_far_more_of_the_encoder_than_the_frozen_contract():
    assert B52_INHERITED_ENCODER_STAGES == 1, "the contract B52 departs from"
    assert B52_DEFAULT_ENCODER_STAGES == MAX_TRAINABLE_STAGES == 5
    assert B52_DEFAULT_ENCODER_STAGES > B52_INHERITED_ENCODER_STAGES


def test_b52_trains_for_more_than_two_epochs():
    assert B52_INHERITED_EPOCHS == 2
    assert B52_DEFAULT_EPOCHS > B52_INHERITED_EPOCHS


def test_the_augment_flag_actually_enables_all_nine_augmentations():
    """The central claim: `train=True` is the line that turns them back on.

    Both existing trainers call this with train=False, which zeroes every one.
    """
    config = {
        "b7_noise_std": 0.02,
        "b7_slice_dropout": 0.08,
        "b7_center_jitter": 2,
        "b7_rotation_deg": 5.0,
        "b7_translate_frac": 0.03,
        "b7_scale_jitter": 0.05,
        "b7_gamma_jitter": 0.12,
        "b7_bias_field_strength": 0.08,
    }
    off = make_b7_dataset_config(config, ".", train=False)
    on = make_b7_dataset_config(config, ".", train=True)

    fields = (
        "noise_std",
        "slice_dropout",
        "center_jitter",
        "rotation_deg",
        "translate_frac",
        "scale_jitter",
        "gamma_jitter",
        "bias_field_strength",
    )
    for field in fields:
        assert getattr(off, field) == 0, f"{field} should be disabled when train=False"
        assert getattr(on, field) > 0, f"{field} should be live when train=True"


# --- the optimiser ----------------------------------------------------------


def test_the_encoder_reaches_the_optimiser():
    """B42/B50 give the encoder lr 0.0; B52's whole point is that it trains."""
    model = _Model()
    groups = {g["name"]: g for g in b52_parameter_groups(
        model, head_lr=1e-4, encoder_lr_scale=0.10, hierarchy_lr_scale=0.05
    )}

    assert set(groups) == {"sparse_head", "encoder", "study_hierarchy"}
    assert groups["encoder"]["params"], "the encoder group must not be empty"
    assert groups["encoder"]["lr"] == pytest.approx(1e-5)
    assert groups["sparse_head"]["lr"] == pytest.approx(1e-4)
    assert groups["study_hierarchy"]["lr"] == pytest.approx(5e-6)


def test_a_frozen_encoder_is_refused():
    """Silently training nothing is the failure this whole experiment exists to end."""
    model = _Model()
    for parameter in model.base.encoder.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(RuntimeError, match="none of it requires gradients"):
        b52_parameter_groups(model, head_lr=1e-4, encoder_lr_scale=0.1, hierarchy_lr_scale=0.05)


def test_no_parameter_is_updated_twice():
    model = _Model()
    groups = b52_parameter_groups(
        model, head_lr=1e-4, encoder_lr_scale=0.1, hierarchy_lr_scale=0.05
    )
    seen = set()
    for group in groups:
        for parameter in group["params"]:
            assert id(parameter) not in seen
            seen.add(id(parameter))


def test_the_frozen_encoder_rate_is_not_inherited():
    """0.05 was chosen for one thawed stage; five stages is a different problem."""
    from rsna_knee.b52_competition_training import B52_DEFAULT_ENCODER_LR_SCALE
    from rsna_knee.encoder_finetune import DEFAULT_ENCODER_LR_SCALE

    assert DEFAULT_ENCODER_LR_SCALE == 0.05
    assert B52_DEFAULT_ENCODER_LR_SCALE > DEFAULT_ENCODER_LR_SCALE


# --- the selection metric ---------------------------------------------------


def test_the_state_boundary_is_the_one_the_split_was_built_with():
    """Soft targets are 0.85 and 0.05, so 0.5 is the boundary, not 1 and 0."""
    target = np.array([[0.85, 0.05, 0.50]])
    weight = np.array([[0.9, 0.9, 0.9]])
    masked = masked_binary_targets(target, weight)
    assert masked[0, 0] == 1.0
    assert masked[0, 1] == 0.0
    assert masked[0, 2] == 0.0, "0.50 is not above the boundary"


def test_an_unsupervised_cell_is_excluded_rather_than_called_negative():
    target = np.array([[0.85, 0.85]])
    weight = np.array([[0.9, 0.0]])
    masked = masked_binary_targets(target, weight)
    assert masked[0, 0] == 1.0
    assert np.isnan(masked[0, 1]), "a zero-weight cell must not become a negative"


def test_a_perfect_ranking_scores_one():
    rows = 8
    target = np.zeros((rows, len(TARGETS)))
    target[: rows // 2, :] = 0.85
    target[rows // 2 :, :] = 0.05
    weight = np.full((rows, len(TARGETS)), 0.9)
    prediction = np.tile(
        np.concatenate([np.ones(rows // 2), np.zeros(rows // 2)])[:, None],
        (1, len(TARGETS)),
    )

    scores = macro_auc(target, weight, prediction)
    assert scores["macro_auc"] == pytest.approx(1.0)
    assert scores["targets_defined"] == len(TARGETS)


def test_a_target_with_one_class_is_dropped_not_counted_as_half():
    rows = 6
    target = np.zeros((rows, len(TARGETS)))
    target[:, :] = 0.85
    target[rows // 2 :, 1:] = 0.05  # target 0 keeps only positives
    weight = np.full((rows, len(TARGETS)), 0.9)
    prediction = np.random.default_rng(0).random((rows, len(TARGETS)))

    scores = macro_auc(target, weight, prediction)
    assert not np.isfinite(scores["per_target_auc"][TARGETS[0]])
    assert scores["targets_defined"] == len(TARGETS) - 1
    assert np.isfinite(scores["macro_auc"]), "the remaining targets still score"


def test_reversing_the_ranking_scores_zero():
    rows = 8
    target = np.zeros((rows, len(TARGETS)))
    target[: rows // 2, :] = 0.85
    target[rows // 2 :, :] = 0.05
    weight = np.full((rows, len(TARGETS)), 0.9)
    prediction = np.tile(
        np.concatenate([np.zeros(rows // 2), np.ones(rows // 2)])[:, None],
        (1, len(TARGETS)),
    )
    assert macro_auc(target, weight, prediction)["macro_auc"] == pytest.approx(0.0)
