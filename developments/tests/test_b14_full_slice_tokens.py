"""Regression tests for B14 ImageNet full slice-token aggregation."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from rsna_knee.b12_1_hierarchical import b12_1_model_spec, build_b12_1_model  # noqa: E402
from rsna_knee.b12_variable_series import b12_model_spec, build_b12_model  # noqa: E402
from rsna_knee.b14_training import (  # noqa: E402
    B14_AGGREGATION,
    B14_EXPERIMENT,
    B14_INITIALIZATION,
    B14_INPUT_NORMALIZATION,
    B14_SERIES_SIGNATURE,
    _require_b14_contract,
    train_b14,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
B13 = CONFIG_DIR / "b13_imagenet_init.yaml"
B14 = CONFIG_DIR / "b14_imagenet_full_tokens.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_b14_config_is_directly_accepted_by_contract():
    config = _load(B14)
    _require_b14_contract(config)
    assert config["b14_experiment_name"] == B14_EXPERIMENT
    assert config["b14_aggregation"] == B14_AGGREGATION
    assert config["allow_external_pretrained"] is True
    assert config["pretrained"] is True


def test_b14_config_matches_b13_except_identity_and_aggregation():
    b13, b14 = _load(B13), _load(B14)
    permitted = {
        "b13_experiment_name",
        "b14_experiment_name",
        "b14_aggregation",
    }
    differing = {
        key
        for key in set(b13) | set(b14)
        if b13.get(key) != b14.get(key)
    }
    assert differing <= permitted, (
        "B14 changes more than aggregation/admin identity versus B13: "
        f"{sorted(differing - permitted)}"
    )


def test_b14_trainer_has_no_b5_or_b13_checkpoint_argument():
    params = inspect.signature(train_b14).parameters
    assert "b5_checkpoint" not in params
    assert "b13_checkpoint" not in params


def test_b14_freezes_imagenet_and_series_metadata():
    assert B14_INITIALIZATION == "torchvision:convnext_tiny:IMAGENET1K_V1"
    assert B14_INPUT_NORMALIZATION == "imagenet_mean_std"
    assert B14_AGGREGATION == "all_real_series_x_16_slice_tokens_v1"
    assert B14_SERIES_SIGNATURE == "5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376"
    spec = b12_model_spec(_load(B14), normalize_input=True)
    assert spec["normalize_input"] is True
    assert spec["architecture"] == "variable_series_pathology_queries_v1"


def test_b14_has_full_slice_memory_not_series_pool():
    config = _load(B14)
    torch.manual_seed(123)
    b14 = build_b12_model(b12_model_spec(config, normalize_input=True))
    assert not hasattr(b14, "series_pool")
    assert b14.n_slices == 16


def test_b14_and_b13_shared_random_initialization_matches():
    config14 = _load(B14)
    config13 = _load(B13)
    torch.manual_seed(2026)
    b14 = build_b12_model(b12_model_spec(config14, normalize_input=True))
    torch.manual_seed(2026)
    b13 = build_b12_1_model(
        b12_1_model_spec(config13, normalize_input=True),
        pretrained_weights=False,
    )
    # B13 creates the series pool only after all parameters shared with B14.
    shared_names = [
        "slice_position",
        "pathology_tokens",
        "target_weight",
        "context.layers.0.linear1.weight",
        "cross_attention.in_proj_weight",
    ]
    state14, state13 = b14.state_dict(), b13.state_dict()
    for name in shared_names:
        assert torch.equal(state14[name], state13[name]), name


def test_b14_rejects_aggregation_drift():
    config = _load(B14)
    config["b14_aggregation"] = "one_token_per_series"
    with pytest.raises(ValueError, match="aggregation"):
        _require_b14_contract(config)


def test_b14_rejects_encoder_lr_change():
    config = _load(B14)
    config["b7_encoder_lr"] = 2e-5
    with pytest.raises(ValueError, match="b7_encoder_lr"):
        _require_b14_contract(config)


def test_b14_rejects_epoch_change():
    config = _load(B14)
    config["b7_epochs"] = 5
    with pytest.raises(ValueError, match="b7_epochs"):
        _require_b14_contract(config)


def test_b14_rejects_imagenet_flag_removal():
    config = _load(B14)
    config["pretrained"] = False
    with pytest.raises(ValueError, match="pretrained=true"):
        _require_b14_contract(config)
