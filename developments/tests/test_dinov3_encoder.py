"""The DINOv3 encoder must be interchangeable with the frozen one.

The value of this variant is that only the representation changes. These tests
hold that claim to account: same interface, same output width, same
normalisation, and a refusal rather than a silent reshape when a variant would
break it.

All of these run without network access -- `pretrained_weights=False` builds the
architecture only. Nothing here proves the published DINOv3 weights load; that
needs the real checkpoint and is verified on the machine that has it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from rsna_knee.dinov3_encoder import (  # noqa: E402
    DINOV3_CONVNEXT_MODELS,
    DROP_IN_VARIANTS,
    REQUIRED_OUTPUT_WIDTH,
    DinoV3SliceEncoder,
    attach_dinov3_encoder,
)
from rsna_knee.model import ConvNeXtSliceEncoder  # noqa: E402


def _frozen():
    return ConvNeXtSliceEncoder(pretrained_weights=False, normalize_input=True)


def _dinov3(**kwargs):
    kwargs.setdefault("pretrained_weights", False)
    return DinoV3SliceEncoder(**kwargs)


def test_output_width_matches_the_frozen_encoder():
    assert _dinov3().out_dim == _frozen().out_dim == REQUIRED_OUTPUT_WIDTH


def test_forward_produces_the_same_shape_as_the_frozen_encoder():
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        frozen = _frozen()(x)
        swapped = _dinov3()(x)
    assert frozen.shape == swapped.shape == (4, REQUIRED_OUTPUT_WIDTH)


def test_normalisation_statistics_are_identical():
    """DINOv3's ConvNeXt uses ImageNet statistics, so the input pipeline is unchanged."""
    frozen, swapped = _frozen(), _dinov3()
    assert torch.allclose(frozen.input_mean, swapped.input_mean)
    assert torch.allclose(frozen.input_std, swapped.input_std)


def test_normalisation_can_be_disabled_like_the_frozen_encoder():
    encoder = _dinov3(normalize_input=False)
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(encoder._normalize(x), x)


def test_single_channel_input_is_supported():
    encoder = _dinov3(in_channels=1)
    with torch.no_grad():
        out = encoder(torch.randn(2, 1, 224, 224))
    assert out.shape == (2, REQUIRED_OUTPUT_WIDTH)


def test_an_unknown_variant_is_refused_with_the_known_list():
    with pytest.raises(ValueError, match="unknown DINOv3 variant"):
        _dinov3(variant="enormous")


@pytest.mark.parametrize("variant", ["tiny", "small"])
def test_both_narrow_variants_drop_in_unchanged(variant):
    """Tiny and small share channel widths and differ only in depth."""
    assert _dinov3(variant=variant).out_dim == REQUIRED_OUTPUT_WIDTH


@pytest.mark.parametrize("variant", ["base", "large"])
def test_a_wider_variant_is_refused_rather_than_reshaped(variant):
    """Base and large emit 1024 and 1536, which would change the head too."""
    with pytest.raises(ValueError, match="drop-in variants"):
        _dinov3(variant=variant)


def test_every_declared_variant_names_a_dinov3_checkpoint():
    for name in DINOV3_CONVNEXT_MODELS.values():
        assert name.endswith(".dinov3_lvd1689m")


def test_the_drop_in_list_matches_what_actually_builds():
    """The advertised list must be the list that works, not an aspiration."""
    for variant in DINOV3_CONVNEXT_MODELS:
        if variant in DROP_IN_VARIANTS:
            assert _dinov3(variant=variant).out_dim == REQUIRED_OUTPUT_WIDTH
        else:
            with pytest.raises(ValueError):
                _dinov3(variant=variant)


def test_describe_records_provenance():
    described = _dinov3().describe()
    assert described["model_name"] == "convnext_tiny.dinov3_lvd1689m"
    assert "DINOv3" in described["pretraining"]
    assert described["out_dim"] == REQUIRED_OUTPUT_WIDTH


class _Head(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


def test_attaching_replaces_the_encoder_in_place():
    head = _Head(_frozen())
    original = head.encoder
    replacement = attach_dinov3_encoder(head, pretrained_weights=False)
    assert head.encoder is replacement
    assert head.encoder is not original
    assert head.encoder.out_dim == original.out_dim


def test_attaching_preserves_the_channel_and_normalisation_settings():
    head = _Head(ConvNeXtSliceEncoder(1, pretrained_weights=False, normalize_input=False))
    replacement = attach_dinov3_encoder(head, pretrained_weights=False)
    assert replacement.input_mean.shape[1] == 1
    assert replacement.normalize_input is False


def test_attaching_to_something_without_an_encoder_is_refused():
    with pytest.raises(ValueError, match="no encoder attribute"):
        attach_dinov3_encoder(torch.nn.Linear(2, 2), pretrained_weights=False)
