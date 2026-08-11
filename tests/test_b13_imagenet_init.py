"""Regression tests for the clean standalone B13 ImageNet experiment."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from rsna_knee.b12_1_hierarchical import b12_1_model_spec, build_b12_1_model  # noqa: E402
from rsna_knee.b12_1_training import _require_b12_1_contract  # noqa: E402
from rsna_knee.b13_training import (  # noqa: E402
    B13_EXPERIMENT,
    B13_INITIALIZATION,
    B13_INPUT_NORMALIZATION,
    B13_SERIES_SIGNATURE,
    _require_b13_contract,
    train_b13,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
B13 = CONFIG_DIR / "b13_imagenet_init.yaml"
B12_1 = CONFIG_DIR / "b12_1_hierarchical.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_b13_config_is_directly_accepted_by_b13_contract():
    config = _load(B13)
    _require_b13_contract(config)
    assert config["b13_experiment_name"] == B13_EXPERIMENT
    assert config["allow_external_pretrained"] is True
    assert config["pretrained"] is True


def test_b13_config_cannot_masquerade_as_b12_1():
    with pytest.raises(ValueError, match="competition-only B5 encoder"):
        _require_b12_1_contract(_load(B13))


def test_b13_changes_only_encoder_protocol_and_administrative_identity_vs_b12_1():
    b13, b12_1 = _load(B13), _load(B12_1)
    permitted = {
        "pretrained",
        "allow_external_pretrained",
        "b13_experiment_name",
        "b12_1_experiment_name",
    }
    differing = {
        key
        for key in set(b13) | set(b12_1)
        if b13.get(key) != b12_1.get(key)
    }
    assert differing <= permitted, (
        "B13 config changes more than encoder protocol/admin identity: "
        f"{sorted(differing - permitted)}"
    )


def test_b13_trainer_has_no_b5_checkpoint_argument():
    assert "b5_checkpoint" not in inspect.signature(train_b13).parameters


def test_b13_freezes_encoder_protocol_metadata():
    assert B13_INITIALIZATION == "torchvision:convnext_tiny:IMAGENET1K_V1"
    assert B13_INPUT_NORMALIZATION == "imagenet_mean_std"
    assert B13_SERIES_SIGNATURE == "5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376"
    spec = b12_1_model_spec(_load(B13), normalize_input=True)
    assert spec["normalize_input"] is True


def test_b13_rejects_second_variable_encoder_lr_change():
    config = _load(B13)
    config["b7_encoder_lr"] = 2e-5
    with pytest.raises(ValueError, match="b7_encoder_lr"):
        _require_b13_contract(config)


def test_b13_rejects_second_variable_epoch_change():
    config = _load(B13)
    config["b7_epochs"] = 5
    with pytest.raises(ValueError, match="b7_epochs"):
        _require_b13_contract(config)


def test_b13_rejects_external_flag_removed():
    config = _load(B13)
    config["allow_external_pretrained"] = False
    with pytest.raises(ValueError, match="allow_external_pretrained"):
        _require_b13_contract(config)


def test_b13_rejects_pretrained_flag_removed():
    config = _load(B13)
    config["pretrained"] = False
    with pytest.raises(ValueError, match="pretrained=true"):
        _require_b13_contract(config)


def _spec(config: dict) -> dict:
    return b12_1_model_spec(config, normalize_input=True)


def _build_pretrained(spec: dict):
    try:
        return build_b12_1_model(spec, pretrained_weights=True)
    except Exception as error:  # network/cache errors in offline CI
        if isinstance(error, ValueError):
            raise
        pytest.skip(f"ImageNet weights unavailable offline: {type(error).__name__}")


def test_pretrained_flag_reaches_encoder_when_weights_available():
    spec = _spec(_load(B13))
    scratch = build_b12_1_model(spec, pretrained_weights=False)
    pretrained = _build_pretrained(spec)
    first_scratch = scratch.encoder.features[0][0].weight
    first_pretrained = pretrained.encoder.features[0][0].weight
    assert not torch.allclose(first_scratch, first_pretrained)


def test_two_initialization_sources_are_mutually_exclusive():
    spec = _spec(_load(B13))
    encoder_state = build_b12_1_model(spec, pretrained_weights=False).encoder.state_dict()
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_b12_1_model(spec, encoder_state=encoder_state, pretrained_weights=True)
