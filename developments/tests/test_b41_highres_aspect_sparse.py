from __future__ import annotations

import numpy as np
import torch

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b41_highres_aspect_sparse_mil import (
    B41_EXPERT58_ROOT,
    B41_IMAGE_SIZE,
    B41_RUN_ROOT,
    B41_RESIZE_POLICY,
    b41_preprocessing_state,
    preprocess_dense_triplets_b41,
    require_b41_aspect_contract,
    resize_triplets_aspect_preserving_pad,
)


def test_b41_uses_the_next_numbered_run_root() -> None:
    assert B41_RUN_ROOT == (
        "runs/076_Experiment_B41_native_aspect_90crop_sparse_mil/"
        "b41_highres_aspect_sparse_mil"
    )
    assert B41_EXPERT58_ROOT == f"{B41_RUN_ROOT}/expert58"


def test_b41_config_freezes_b37_controls_and_aspect_policy() -> None:
    config = dict(_read_config("config/b41_highres_aspect_sparse_448.yaml"))
    policy = require_b41_aspect_contract(config)
    assert policy["crop_fraction"] == 0.90
    assert config["b41_resize_policy"] == B41_RESIZE_POLICY
    bad = dict(config)
    bad["b41_resize_policy"] = "square_stretch"
    try:
        require_b41_aspect_contract(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("B41 accepted a direct-square resize policy")


def test_b41_576_by_1152_crop_fits_224_by_448_then_pads(monkeypatch) -> None:
    import rsna_knee.b41_highres_aspect_sparse_mil as module

    calls = []
    original = module.F.interpolate

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", wrapped)
    triplets = np.ones((2, 3, 576, 1152), dtype=np.float32)
    image = resize_triplets_aspect_preserving_pad(triplets)
    assert tuple(image.shape) == (2, 3, B41_IMAGE_SIZE, B41_IMAGE_SIZE)
    assert len(calls) == 1
    assert calls[0]["size"] == (224, 448)
    assert calls[0]["mode"] == "bilinear"
    assert calls[0]["align_corners"] is False
    assert calls[0]["antialias"] is True
    assert image[..., :112, :].eq(0).all()
    torch.testing.assert_close(
        image[..., 112:336, :],
        torch.ones_like(image[..., 112:336, :]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert image[..., 336:, :].eq(0).all()


def test_b41_square_crop_has_no_padding_bars() -> None:
    triplets = np.ones((1, 3, 576, 576), dtype=np.float32)
    image = resize_triplets_aspect_preserving_pad(triplets)
    assert tuple(image.shape) == (1, 3, B41_IMAGE_SIZE, B41_IMAGE_SIZE)
    torch.testing.assert_close(
        image,
        torch.ones_like(image),
        rtol=1e-6,
        atol=1e-6,
    )


def test_b41_preprocess_preserves_32_triplets_and_one_resize(monkeypatch) -> None:
    import rsna_knee.b41_highres_aspect_sparse_mil as module

    calls = []
    original = module.F.interpolate

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", wrapped)
    raw = np.arange(40 * 64 * 128, dtype=np.float32).reshape(40, 64, 128)
    image, position = preprocess_dense_triplets_b41(raw, gap=1, center_offset=0)
    assert tuple(image.shape) == (32, 3, B41_IMAGE_SIZE, B41_IMAGE_SIZE)
    assert tuple(position.shape) == (32,)
    assert len(calls) == 1
    assert calls[0]["size"][1] == B41_IMAGE_SIZE
    assert calls[0]["size"][0] < B41_IMAGE_SIZE


def test_b41_preprocessing_state_is_explicit_and_aspect_preserving() -> None:
    state = b41_preprocessing_state()
    assert state["crop_fraction"] == 0.90
    assert state["resize_policy"] == B41_RESIZE_POLICY
    assert state["preserves_in_plane_aspect_ratio"] is True
    assert state["deterministic_resize_count"] == 1
