from __future__ import annotations

import numpy as np
import torch

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b37_highres_sparse_training import (
    B37_CONSTRUCTION_SEED_OFFSET,
    B37_LOADER_SEED_OFFSET,
)
from rsna_knee.b42_constant_area_aspect_sparse_mil import (
    B42_EFFECTIVE_BATCH,
    B42_REFERENCE_AREA,
    B42_RESIZE_POLICY,
    B42_RUN_ROOT,
    B42_STRIDE_ALIGNMENT,
    b42_preprocessing_state,
    collate_b42,
    constant_area_shape,
    require_b42_contract,
    resize_triplets_constant_area,
)
from rsna_knee.b42_constant_area_aspect_sparse_training import (
    B42_CONSTRUCTION_SEED_OFFSET,
    B42_LOADER_SEED_OFFSET,
    _batch_scales,
)


def test_b42_uses_next_numbered_run_root_and_b37_random_streams() -> None:
    assert B42_RUN_ROOT == (
        "runs/077_Experiment_B42_constant_area_aspect_sparse_mil/"
        "b42_constant_area_aspect_sparse_mil"
    )
    assert B42_CONSTRUCTION_SEED_OFFSET == B37_CONSTRUCTION_SEED_OFFSET
    assert B42_LOADER_SEED_OFFSET == B37_LOADER_SEED_OFFSET


def test_b42_config_freezes_geometry_and_b37_controls() -> None:
    config = dict(_read_config("config/b42_constant_area_aspect_sparse.yaml"))
    crop = require_b42_contract(config)
    assert crop["crop_fraction"] == 0.90
    assert config["b42_resize_policy"] == B42_RESIZE_POLICY
    assert config["b42_reference_area"] == 448 * 448
    assert config["b42_stride_alignment"] == 32
    assert config["b42_effective_batch"] == 2

    bad = dict(config)
    bad["b42_reference_area"] = 384 * 384
    try:
        require_b42_contract(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("B42 accepted a changed reference pixel area")


def test_b42_two_to_one_shape_preserves_area_and_aspect() -> None:
    geometry = constant_area_shape(576, 1152)
    assert geometry["resized_height"] == 317
    assert geometry["resized_width"] == 634
    assert geometry["aligned_height"] == 320
    assert geometry["aligned_width"] == 640
    assert abs(geometry["anatomical_pixels"] - B42_REFERENCE_AREA) / B42_REFERENCE_AREA < 0.002
    assert geometry["aligned_height"] % B42_STRIDE_ALIGNMENT == 0
    assert geometry["aligned_width"] % B42_STRIDE_ALIGNMENT == 0
    assert geometry["pad_top"] + geometry["pad_bottom"] < B42_STRIDE_ALIGNMENT
    assert geometry["pad_left"] + geometry["pad_right"] < B42_STRIDE_ALIGNMENT


def test_b42_square_shape_is_exact_448_without_padding() -> None:
    geometry = constant_area_shape(576, 576)
    assert geometry["resized_height"] == 448
    assert geometry["resized_width"] == 448
    assert geometry["aligned_height"] == 448
    assert geometry["aligned_width"] == 448
    assert geometry["pad_top"] == 0
    assert geometry["pad_bottom"] == 0
    assert geometry["pad_left"] == 0
    assert geometry["pad_right"] == 0


def test_b42_transpose_geometry_is_symmetric() -> None:
    wide = constant_area_shape(576, 1152)
    tall = constant_area_shape(1152, 576)
    assert tall["resized_height"] == wide["resized_width"]
    assert tall["resized_width"] == wide["resized_height"]
    assert tall["aligned_height"] == wide["aligned_width"]
    assert tall["aligned_width"] == wide["aligned_height"]


def test_b42_resize_uses_one_interpolation_and_thin_reflection_padding(monkeypatch) -> None:
    import rsna_knee.b42_constant_area_aspect_sparse_mil as module

    calls = []
    original = module.F.interpolate

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", wrapped)
    triplets = np.ones((2, 3, 576, 1152), dtype=np.float32)
    image = resize_triplets_constant_area(triplets)
    assert tuple(image.shape) == (2, 3, 320, 640)
    assert len(calls) == 1
    assert calls[0]["size"] == (317, 634)
    assert calls[0]["mode"] == "bilinear"
    assert calls[0]["align_corners"] is False
    assert calls[0]["antialias"] is True
    torch.testing.assert_close(
        image,
        torch.ones_like(image),
        rtol=1e-6,
        atol=1e-6,
    )


def test_b42_collate_keeps_ragged_series_unstacked() -> None:
    items = [
        {"volumes": [torch.zeros(32, 3, 448, 448)]},
        {"volumes": [torch.zeros(32, 3, 320, 640)]},
    ]
    batch = collate_b42(items)
    assert isinstance(batch, list)
    assert len(batch) == B42_EFFECTIVE_BATCH
    assert tuple(batch[0]["volumes"][0].shape[-2:]) == (448, 448)
    assert tuple(batch[1]["volumes"][0].shape[-2:]) == (320, 640)


def test_b42_sequential_scales_reconstruct_batch_denominator() -> None:
    multiplier = torch.tensor([2.0, 1.0] + [1.0] * 10)
    w1 = torch.tensor([1.0, 1.0] + [0.0] * 10)
    w2 = torch.tensor([0.5, 3.0] + [0.0] * 10)
    items = [{"weight": w1}, {"weight": w2}]
    scales = _batch_scales(items, multiplier)
    mass1 = 1.0 * 2.0 + 1.0
    mass2 = 0.5 * 2.0 + 3.0
    total = mass1 + mass2
    assert np.isclose(scales[0], mass1 / total)
    assert np.isclose(scales[1], mass2 / total)
    assert np.isclose(sum(scales), 1.0)


def test_b42_preprocessing_state_records_full_occupancy_policy() -> None:
    state = b42_preprocessing_state()
    assert state["resize_policy"] == B42_RESIZE_POLICY
    assert state["preserves_in_plane_aspect_ratio"] is True
    assert state["reference_pixel_area"] == 448 * 448
    assert state["stride_alignment"] == 32
    assert state["padding"]["mode"] == "reflect"
    assert state["padding"]["square_padding"] is False
    assert state["ragged_series_encoding"] is True
