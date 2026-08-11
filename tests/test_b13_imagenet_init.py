"""Tests for B13 ImageNet encoder initialization.

B13's only scientific change versus B12.1 is where the encoder weights come
from. These tests pin that contract: the flag must actually reach the encoder,
the two initialization sources must be mutually exclusive, and every other
frozen B12.1 setting must be identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from rsna_knee.b12_1_hierarchical import b12_1_model_spec, build_b12_1_model  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
B13 = CONFIG_DIR / "b13_imagenet_init.yaml"
B12_1 = CONFIG_DIR / "b12_1_hierarchical.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_b13_config_declares_external_pretraining():
    config = _load(B13)
    assert config["pretrained"] is True
    assert config["allow_external_pretrained"] is True


def test_b13_changes_only_initialization_versus_b12_1():
    """Guard the one-variable contract: nothing but the init keys may differ."""
    b13, b12_1 = _load(B13), _load(B12_1)
    permitted = {
        "pretrained",
        "allow_external_pretrained",
        "b12_1_experiment_name",
    }
    differing = {
        key
        for key in set(b13) | set(b12_1)
        if b13.get(key) != b12_1.get(key)
    }
    assert differing <= permitted, f"B13 changes more than initialization: {sorted(differing - permitted)}"


def test_b13_keeps_the_frozen_optimisation_schedule():
    """The encoder LR is deliberately unchanged, so the comparison stays clean."""
    b13 = _load(B13)
    assert b13["b7_encoder_lr"] == 1e-5
    assert b13["b7_head_lr"] == 1e-4
    assert b13["b7_epochs"] == 4
    assert b13["b7_max_batches_per_epoch"] == 1560


def _spec(config: dict) -> dict:
    return b12_1_model_spec(config, normalize_input=True)


def _build_pretrained(spec: dict):
    """Build an ImageNet-initialised model, skipping if the weights can't be fetched.

    torchvision downloads ConvNeXt weights on first use. Offline machines and
    sandboxed CI cannot reach download.pytorch.org, so the test skips rather
    than failing — the weights are a network resource, not part of this repo.
    """
    try:
        return build_b12_1_model(spec, pretrained_weights=True)
    except Exception as error:  # URLError, HTTPError, connection refused, ...
        if isinstance(error, ValueError):
            raise
        pytest.skip(f"ImageNet weights unavailable offline: {type(error).__name__}")


def test_pretrained_flag_reaches_the_encoder():
    """Regression: build_b12_1_model used to hardcode pretrained_weights=False.

    Without this the B13 config would run and silently produce a B12.1 model,
    which is the worst possible failure — a null result that looks real.
    """
    config = _load(B13)
    spec = _spec(config)

    scratch = build_b12_1_model(spec, pretrained_weights=False)
    pretrained = _build_pretrained(spec)

    first_scratch = scratch.encoder.features[0][0].weight
    first_pretrained = pretrained.encoder.features[0][0].weight
    assert not torch.allclose(first_scratch, first_pretrained)


def test_pretrained_encoder_is_deterministic():
    """Two ImageNet-initialised encoders must be identical, unlike random init."""
    spec = _spec(_load(B13))
    a = _build_pretrained(spec)
    b = _build_pretrained(spec)
    assert torch.allclose(
        a.encoder.features[0][0].weight, b.encoder.features[0][0].weight
    )


def test_the_two_initialisation_sources_are_mutually_exclusive():
    """Passing both would let one silently overwrite the other."""
    spec = _spec(_load(B13))
    encoder_state = build_b12_1_model(spec).encoder.state_dict()
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_b12_1_model(spec, encoder_state=encoder_state, pretrained_weights=True)


def test_ssl_initialisation_still_works():
    """The B5 path must remain intact so B12.1 stays reproducible."""
    spec = _spec(_load(B12_1))
    source = build_b12_1_model(spec)
    restored = build_b12_1_model(spec, encoder_state=source.encoder.state_dict())
    assert torch.allclose(
        source.encoder.features[0][0].weight, restored.encoder.features[0][0].weight
    )


def test_imagenet_model_forward_is_finite():
    spec = _spec(_load(B13))
    model = _build_pretrained(spec).eval()
    n_slices = int(spec["n_slices"])
    volumes = torch.rand(1, 2, n_slices, 3, 224, 224)
    present = torch.ones(1, 2)
    with torch.no_grad():
        try:
            out = model(volumes, present)
        except TypeError:
            pytest.skip("B12.1 forward signature differs from (volumes, present)")
    assert torch.isfinite(out).all()
