"""The wide-encoder variant must build at any published width and leave no trace.

Two things are being checked. That the head really is width-agnostic, so a
1024-d or 1536-d encoder needs no projection bolted on. And that the machinery
used to achieve it -- substituting an encoder class and rescaling frozen
parameter-count contracts -- is fully undone afterwards, so the supported path
is unaffected by having imported this variant.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from model._implementation import ensure_developments_source, read_config  # noqa: E402
from variants.dinov3_wide.encoder import (  # noqa: E402
    DINOV3_MODELS,
    WideDinoV3SliceEncoder,
)
from variants.dinov3_wide.model import (  # noqa: E402
    _contracts_scaled_to,
    build_wide_model,
    wide_model_spec,
)

ensure_developments_source()


def _small_spec(variant: str) -> dict:
    spec = dict(wide_model_spec(read_config("config/current_model.yaml"), variant))
    spec["n_slices"] = 2
    spec["encoder_batch_size"] = 4
    return spec


@pytest.mark.parametrize("variant,width", [(v, w) for v, (_, w) in DINOV3_MODELS.items()])
def test_encoder_emits_its_published_width(variant, width):
    encoder = WideDinoV3SliceEncoder(variant=variant, pretrained_weights=False)
    assert encoder.out_dim == width


@pytest.mark.parametrize("variant", ["tiny", "base", "large"])
def test_the_head_is_built_to_match_the_encoder(variant):
    spec = _small_spec(variant)
    model = build_wide_model(spec, pretrained_weights=False)
    width = int(spec["encoder_width"])
    assert model.encoder.out_dim == width
    # Depthwise k=3 context scales with the width; no projection is involved.
    assert model.local_context.weight.numel() == 3 * width
    assert model.local_context.groups == width
    assert model.local_context.bias is None


@pytest.mark.parametrize("variant", ["tiny", "base"])
def test_a_forward_pass_produces_twelve_finite_logits(variant):
    model = build_wide_model(_small_spec(variant), pretrained_weights=False)
    volumes = torch.randn(2, 3, 2, 3, 224, 224)
    present = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    meta = torch.tensor(
        [[[1, 1, 1], [2, 2, 2], [0, 0, 0]], [[3, 1, 2], [0, 0, 0], [0, 0, 0]]],
        dtype=torch.long,
    )
    with torch.no_grad():
        logits = model(volumes, present, meta)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_the_context_scaffold_still_starts_at_exactly_zero():
    """The zero-initialisation contract must survive the width change."""
    model = build_wide_model(_small_spec("base"), pretrained_weights=False)
    assert torch.count_nonzero(model.local_context.weight).item() == 0


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="unknown DINOv3 variant"):
        wide_model_spec(read_config("config/current_model.yaml"), "gigantic")


def test_the_frozen_encoder_class_is_restored_after_building():
    from rsna_knee import b12_1_hierarchical
    from rsna_knee.model import ConvNeXtSliceEncoder

    build_wide_model(_small_spec("base"), pretrained_weights=False)
    assert b12_1_hierarchical.ConvNeXtSliceEncoder is ConvNeXtSliceEncoder


def test_the_frozen_parameter_contracts_are_restored_after_building():
    from rsna_knee import b29_complementary_series_pool as b29
    from rsna_knee import b31_local_context_complementary_pool as b31

    build_wide_model(_small_spec("large"), pretrained_weights=False)
    assert b29.B29_EXPECTED_QUERY_PARAMETERS == 768
    assert b29.B29_EXPECTED_NEW_PARAMETERS == 1536
    assert b31.B31_EXPECTED_CONTEXT_PARAMETERS == 2304
    assert b31.B31_EXPECTED_NEW_PARAMETERS == 3840


def test_contract_scaling_is_restored_even_when_the_body_raises():
    from rsna_knee import b31_local_context_complementary_pool as b31

    with pytest.raises(RuntimeError, match="deliberate"):
        with _contracts_scaled_to(1024):
            assert b31.B31_EXPECTED_CONTEXT_PARAMETERS == 3072
            raise RuntimeError("deliberate")
    assert b31.B31_EXPECTED_CONTEXT_PARAMETERS == 2304


def test_the_supported_path_still_builds_the_frozen_width_afterwards():
    """Nothing here may change what the supported interface constructs."""
    from model._implementation import build_network, network_spec

    build_wide_model(_small_spec("base"), pretrained_weights=False)
    spec = dict(network_spec(read_config("config/current_model.yaml")))
    spec["n_slices"] = 2
    spec["encoder_batch_size"] = 4
    model = build_network(spec, pretrained_weights=False)
    assert model.encoder.out_dim == 768
    assert model.local_context.weight.numel() == 2304
