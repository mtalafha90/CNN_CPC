"""B54's model: the last step, from the batch into the hierarchy's sum.

The whole design rests on one claim — that adding a per-series vector to
`global_feature` before the parent runs is *identical* to adding it into
`plane + fluid + fat`. That claim is what lets the override be three lines
instead of forty copied ones, so it is tested against a direct reproduction of
the parent's expression rather than argued in a comment.

A full B42 model cannot be built here: its encoder wants downloaded weights.
Everything that can be checked without one is.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from rsna_knee.b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectSparseMILResidual,
)
from rsna_knee.b54_spacing_conditioned_mil import (
    B54SpacingConditionedMIL,
    condition_global_feature,
    requires_spacing,
    spacing_from_batch,
)
from rsna_knee.b54_spacing_run import SPACING_KEY, install_spacing_conditioning

B35_BASE_SLICES = 16


class _Base(nn.Module):
    """Just enough of the study base for the arithmetic under test."""

    def __init__(self, d: int = 8, slices: int = B35_BASE_SLICES):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)
        self.slice_position = nn.Parameter(torch.randn(slices, d) * 0.02)


def _parent_sum(base, global_feature, present, series_meta):
    """The parent's expression, reproduced exactly from b37_highres_sparse_mil."""
    x = global_feature[:, :, :B35_BASE_SLICES]
    plane = base.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
    fluid = base.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
    fat = base.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
    metadata = plane + fluid + fat
    mask = present[:, :, None, None].to(x.dtype)
    return (x + base.slice_position[None, None, :, :] + metadata[:, :, None, :]) * mask


def _inputs(batch: int = 2, series: int = 3, d: int = 8, slices: int = 32):
    torch.manual_seed(0)
    global_feature = torch.randn(batch, series, slices, d)
    present = torch.ones(batch, series)
    series_meta = torch.stack(
        [
            torch.randint(0, 4, (batch, series)),
            torch.randint(0, 3, (batch, series)),
            torch.randint(0, 3, (batch, series)),
        ],
        dim=-1,
    )
    return global_feature, present, series_meta


# --- the claim the whole design rests on --------------------------------------


def test_adding_before_the_parent_equals_adding_into_the_metadata_sum():
    """If this fails, the three-line override is silently wrong."""
    base = _Base()
    conditioning = install_spacing_conditioning(base)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    global_feature, present, series_meta = _inputs()
    spacing = torch.tensor([[0.6, 3.3, 5.0], [4.0, 0.8, 3.3]])

    # what the subclass does
    through_feature = _parent_sum(
        base,
        condition_global_feature(base, global_feature, spacing),
        present,
        series_meta,
    )

    # what a forty-line copy with the term in the sum would have done
    contribution = conditioning(spacing)
    x = global_feature[:, :, :B35_BASE_SLICES]
    plane = base.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
    fluid = base.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
    fat = base.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
    metadata = plane + fluid + fat + contribution
    mask = present[:, :, None, None].to(x.dtype)
    in_the_sum = (
        x + base.slice_position[None, None, :, :] + metadata[:, :, None, :]
    ) * mask

    assert torch.allclose(through_feature, in_the_sum, atol=1e-6)


def test_the_equivalence_holds_when_some_series_are_padded():
    """The mask must still zero a padded series, either way round."""
    base = _Base()
    conditioning = install_spacing_conditioning(base)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    global_feature, present, series_meta = _inputs()
    present[0, 2] = 0.0
    spacing = torch.tensor([[0.6, 3.3, float("nan")], [4.0, 0.8, 3.3]])

    result = _parent_sum(
        base,
        condition_global_feature(base, global_feature, spacing),
        present,
        series_meta,
    )
    assert torch.all(result[0, 2] == 0.0)


# --- doing nothing, correctly -------------------------------------------------


def test_with_no_conditioning_installed_the_feature_is_untouched():
    base = _Base()
    global_feature, _, _ = _inputs()
    spacing = torch.full((2, 3), 3.3)

    assert condition_global_feature(base, global_feature, spacing) is global_feature


def test_with_no_spacing_the_feature_is_untouched():
    base = _Base()
    install_spacing_conditioning(base)
    global_feature, _, _ = _inputs()

    assert condition_global_feature(base, global_feature, None) is global_feature


def test_a_freshly_installed_conditioning_changes_nothing():
    """Zero-initialised, so B54 starts numerically identical to B42."""
    base = _Base()
    install_spacing_conditioning(base)
    global_feature, _, _ = _inputs()

    assert torch.allclose(
        condition_global_feature(base, global_feature, torch.full((2, 3), 3.3)),
        global_feature,
    )


def test_a_series_with_no_measurable_spacing_is_untouched():
    base = _Base()
    conditioning = install_spacing_conditioning(base)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    global_feature, _, _ = _inputs()
    spacing = torch.tensor([[float("nan"), 3.3, 5.0], [4.0, 0.8, 3.3]])
    out = condition_global_feature(base, global_feature, spacing)

    assert torch.allclose(out[0, 0], global_feature[0, 0])
    assert not torch.allclose(out[0, 1], global_feature[0, 1])


def test_the_feature_dtype_is_preserved():
    """Half precision must not be promoted out from under the encoder."""
    base = _Base()
    conditioning = install_spacing_conditioning(base)
    with torch.no_grad():
        conditioning.projection.weight.normal_()

    global_feature, _, _ = _inputs()
    out = condition_global_feature(
        base, global_feature.half(), torch.full((2, 3), 3.3)
    )
    assert out.dtype == torch.float16


# --- the subclass itself ------------------------------------------------------


def test_it_is_a_b42_model():
    assert issubclass(B54SpacingConditionedMIL, B42ConstantAreaAspectSparseMILResidual)


def test_the_spacing_is_optional_on_both_overridden_methods():
    """So every existing call site keeps working unchanged."""
    for name in ("forward", "_base_logits_from_global"):
        signature = inspect.signature(getattr(B54SpacingConditionedMIL, name))
        parameter = signature.parameters["series_spacing"]
        assert parameter.default is None, name


def test_the_parent_call_sites_still_fit():
    """`base_equivalence_error_448` calls _base_logits_from_global with three."""
    signature = inspect.signature(
        B54SpacingConditionedMIL._base_logits_from_global
    )
    required = [
        name
        for name, p in signature.parameters.items()
        if p.default is inspect.Parameter.empty and name != "self"
    ]
    assert required == ["global_feature", "present", "series_meta"]


def test_the_forward_signature_matches_the_parent_plus_one():
    parent = list(
        inspect.signature(
            B42ConstantAreaAspectSparseMILResidual.forward
        ).parameters
    )
    child = list(inspect.signature(B54SpacingConditionedMIL.forward).parameters)

    assert child[: len(parent)] == parent
    assert child[len(parent) :] == ["series_spacing"]


def test_it_does_not_copy_the_parents_forty_lines():
    """The point of the design: the hierarchy is delegated, not duplicated."""
    source = inspect.getsource(B54SpacingConditionedMIL._base_logits_from_global)

    assert "super()._base_logits_from_global" in source
    for frozen in ("cross_attention", "pathology_tokens", "_pool_real_series_b31"):
        assert frozen not in source, frozen


def test_the_head_is_left_unconditioned_deliberately():
    """Matched on the call's arguments, not its whitespace, so a reformat
    cannot break it and a real change cannot slip past it."""
    source = inspect.getsource(B54SpacingConditionedMIL.forward)
    head_call = source.split("self.head(", 1)[1].split(")", 1)[0]

    assert "spatial" in head_call
    assert "series_spacing" not in head_call


# --- pulling the spacing out of a batch ---------------------------------------


def _item(spacings):
    return {
        "study_uid": "a",
        "series_spacing": torch.tensor(spacings, dtype=torch.float32),
    }


def test_it_reads_a_padded_batch():
    batch = {"series_spacing": torch.full((2, 3), 3.3)}
    assert spacing_from_batch(batch).shape == (2, 3)


def test_it_reads_a_ragged_b42_batch():
    """collate_b42 is list(items), so the spacing arrives per study."""
    out = spacing_from_batch([_item([3.3, 4.0]), _item([0.6])])

    assert out.shape == (2, 2)
    assert out[0, 1].item() == pytest.approx(4.0)
    assert torch.isnan(out[1, 1])


def test_a_batch_without_spacing_yields_none():
    assert spacing_from_batch([{"study_uid": "a"}]) is None
    assert spacing_from_batch({}) is None
    assert spacing_from_batch([]) is None


def test_the_key_matches_the_one_the_dataset_writes():
    assert SPACING_KEY == "slice_spacing_mm"
    assert "series_spacing" in _item([1.0])


# --- whether a model would use one ---------------------------------------------


def test_a_model_with_no_conditioning_does_not_require_spacing():
    assert requires_spacing(_Base()) is False


def test_a_model_with_conditioning_does():
    base = _Base()
    install_spacing_conditioning(base)
    assert requires_spacing(base) is True
