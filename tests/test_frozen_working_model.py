"""The frozen working model.

This is the reference the encoder experiment is measured against. Everything
pinned here is what "unchanged" means for the control arm: if a value below
moves, the comparison is no longer matched and any result from it is not
interpretable.

These assertions are deliberately literal. They are not describing intent --
they are the contract, and a failure here means either a mistake or a decision
that has to be taken explicitly rather than absorbed.
"""

from __future__ import annotations

import pytest

from model._implementation import network_spec, read_config
from model.architecture import TARGETS, describe
from model.preprocessing import CROP_POLICY

FROZEN_ARCHITECTURE = "b31_training_only_local_context_scaffold_eval_bypass_v1"
FROZEN_AGGREGATION = (
    "b29_query_with_b31_context_scaffold_during_training_"
    "and_exact_b29_scoring_at_eval_v1"
)
FROZEN_ENCODER_WIDTH = 768
FROZEN_SLICE_OFFSETS = (-1, 0, 1)
FROZEN_EPOCHS = 2


@pytest.fixture(scope="module")
def spec():
    return network_spec(read_config("config/current_model.yaml"), normalize_input=True)


def test_frozen_identity(spec):
    assert spec["architecture"] == FROZEN_ARCHITECTURE
    assert spec["aggregation"] == FROZEN_AGGREGATION


def test_frozen_input_geometry(spec):
    """The encoder swap must inherit this geometry exactly, or it is not matched."""
    assert spec["image_size"] == 224
    assert spec["n_slices"] == 16
    assert spec["in_channels"] == 3
    assert spec["triplet_gap"] == 1
    assert spec["normalize_input"] is True


def test_frozen_study_representation(spec):
    assert spec["transformer_layers"] == 2
    assert spec["transformer_heads"] == 8
    assert spec["transformer_ff_mult"] == 2.0
    assert spec["pathology_layers"] == 1
    assert spec["series_pool_heads"] == 8
    assert spec["dropout"] == 0.25


def test_frozen_zero_initialised_additions(spec):
    """Both additions start at exactly zero, so training begins from B20's function."""
    assert spec["b29_new_parameter_count"] == 1536
    assert spec["b31_context_parameter_count"] == 2304
    assert spec["b34_new_parameter_count"] == 3840
    assert spec["b34_inference_context_parameters_used"] == 0


def test_frozen_crop_policy():
    assert CROP_POLICY["version"] == "joint_focus_center_crop_only_v1"
    assert CROP_POLICY["crop_fraction"] == pytest.approx(0.90)


def test_frozen_inference_contract():
    description = describe()
    assert str(list(FROZEN_SLICE_OFFSETS)) in description["inference"]
    assert str(FROZEN_EPOCHS) in description["training_endpoint"] or "fixed" in (
        description["training_endpoint"]
    )


def test_frozen_target_set():
    assert TARGETS == (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    )


def test_expert_labels_never_train():
    assert describe()["expert_labels_in_gradients"] == 0


def test_encoder_width_is_the_swap_constraint():
    """Any replacement encoder must emit this width to leave the head unchanged.

    The whole study-level stack is built around it, so an encoder of a
    different width would silently change far more than the representation.
    """
    from model._implementation import ensure_developments_source

    ensure_developments_source()
    from rsna_knee.model import ConvNeXtSliceEncoder

    encoder = ConvNeXtSliceEncoder(pretrained_weights=False, normalize_input=True)
    assert encoder.out_dim == FROZEN_ENCODER_WIDTH
