from __future__ import annotations

import numpy as np
import torch

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b35_target_spatial_residual import b35_centers
from rsna_knee.b36_sparse_mil import B36SparseMILHead
from rsna_knee.b37_highres_sparse_mil import (
    B37_EXPERT58_ROOT,
    B37_GRID_SIZE,
    B37_IMAGE_SIZE,
    B37_RUN_ROOT,
    B37_TOP_K,
    preprocess_dense_triplets_b37,
    require_b37_sparse_contract,
)
from rsna_knee.b37_highres_sparse_training import (
    _format_memory_state,
    _largest_series_indices,
    _memory_state,
)
from rsna_knee.runtime import resolve_runtime


def test_b37_uses_numbered_run_root() -> None:
    assert B37_RUN_ROOT == (
        "runs/071_Experiment_B37_highres_448_sparse_mil/"
        "b37_highres_sparse_mil"
    )
    assert B37_EXPERT58_ROOT == f"{B37_RUN_ROOT}/expert58"


def test_b37_preflight_selects_largest_series_batch() -> None:
    class Dataset:
        study_uids = ["three", "fourteen-b", "seven", "fourteen-a"]
        series_records = {
            "three": [None] * 3,
            "fourteen-b": [None] * 14,
            "seven": [None] * 7,
            "fourteen-a": [None] * 14,
        }

        def __len__(self):
            return len(self.study_uids)

    assert _largest_series_indices(Dataset(), 2) == (3, 1)


def test_b37_runtime_disables_host_memory_multiplication() -> None:
    config = _read_config("config/b37_highres_sparse_448.yaml")
    assert config["b37_micro_batch"] == 2
    assert config["num_workers"] == 0
    assert config["pin_memory"] is False
    assert config["series_cache_mb_per_worker"] == 0


def test_b37_memory_telemetry_is_finite(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    runtime = resolve_runtime({"device": "cpu", "num_workers": 0})
    state = _memory_state(runtime)
    assert set(state) == {
        "rss_gib",
        "rss_peak_gib",
        "system_available_gib",
        "cuda_allocated_gib",
        "cuda_reserved_gib",
        "cuda_peak_allocated_gib",
        "cuda_peak_reserved_gib",
    }
    assert all(np.isfinite(value) and value >= 0 for value in state.values())
    assert "rss=" in _format_memory_state(state)


def _config() -> dict:
    return {
        "b7_image_size": 448,
        "b37_grid_size": 6,
        "b37_top_k": 8,
        "b37_temperature": 1.0,
        "b37_local_aux_weight": 1.0,
        "b37_encoder_trainable_stages": 1,
        "b37_encoder_lr_scale": 0.05,
        "b37_encoder_chunk_size": 4,
        "b20_crop_focus_enabled": True,
        "b20_crop_focus_version": "joint_focus_center_crop_only_v1",
        "b20_crop_focus_crop_fraction": 0.90,
    }


def test_b37_contract_is_frozen() -> None:
    policy = require_b37_sparse_contract(_config())
    assert policy["crop_fraction"] == 0.90
    bad = _config()
    bad["b37_grid_size"] = 5
    try:
        require_b37_sparse_contract(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("B37 accepted a non-frozen grid size")


def test_b37_preprocess_is_32_triplets_at_448(monkeypatch) -> None:
    import rsna_knee.b37_highres_sparse_mil as module

    calls = []
    original = module.F.interpolate

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", wrapped)
    raw = np.arange(40 * 512 * 512, dtype=np.float32).reshape(40, 512, 512)
    image, position = preprocess_dense_triplets_b37(raw, gap=1, center_offset=0)
    assert tuple(image.shape) == (32, 3, B37_IMAGE_SIZE, B37_IMAGE_SIZE)
    assert tuple(position.shape) == (32,)
    assert len(calls) == 1
    assert calls[0]["size"] == (448, 448)
    assert calls[0]["mode"] == "bilinear"
    assert calls[0]["align_corners"] is False
    assert calls[0]["antialias"] is True


def test_b37_first_16_centers_equal_historical_b34_centers() -> None:
    centers, _ = b35_centers(40, gap=1, center_offset=0)
    historical, _ = b35_centers(
        40,
        gap=1,
        center_offset=0,
        base_slices=16,
        dense_slices=16,
    )
    assert np.array_equal(centers[:16], historical)


def test_b37_sparse_head_uses_6x6_and_top8() -> None:
    head = B36SparseMILHead(
        dim=32,
        grid_size=B37_GRID_SIZE,
        top_k=B37_TOP_K,
        temperature=1.0,
        token_dropout=0.0,
    )
    spatial = torch.randn(1, 1, 32, B37_GRID_SIZE * B37_GRID_SIZE, 32)
    present = torch.ones(1, 1)
    meta = torch.tensor([[[1, 2, 2]]], dtype=torch.long)
    position = torch.linspace(0, 1, 32).reshape(1, 1, 32)
    logits, indices, values = head(spatial, present, meta, position)
    assert tuple(logits.shape) == (1, 12)
    assert tuple(indices.shape) == (1, 12, B37_TOP_K)
    assert tuple(values.shape) == (1, 12, B37_TOP_K)
    assert head.n_regions == 36
