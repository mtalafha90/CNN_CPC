from __future__ import annotations

import numpy as np

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b35_target_spatial_residual import b35_centers
from rsna_knee.b38_highres_global import (
    B38_EXPERT58_ROOT,
    B38_IMAGE_SIZE,
    B38_N_SLICES,
    B38_RUN_ROOT,
    preprocess_global_triplets_b38,
    require_b38_global_contract,
)
from rsna_knee.b38_highres_global_training import (
    _format_memory_state,
    _largest_series_indices,
    _memory_state,
)
from rsna_knee.runtime import resolve_runtime


def test_b38_uses_its_permanent_numbered_run_root() -> None:
    assert B38_RUN_ROOT == (
        "runs/073_Experiment_B38_highres_448_global_tail_ablation/"
        "b38_highres_global_tail"
    )
    assert B38_EXPERT58_ROOT == f"{B38_RUN_ROOT}/expert58"


def test_b38_preflight_selects_largest_series_batch() -> None:
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


def test_b38_runtime_disables_host_memory_multiplication() -> None:
    config = _read_config("config/b38_highres_global_448.yaml")
    assert config["b38_micro_batch"] == 2
    assert config["num_workers"] == 0
    assert config["pin_memory"] is False
    assert config["series_cache_mb_per_worker"] == 0


def test_b38_memory_telemetry_is_finite(monkeypatch) -> None:
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
        "b7_n_slices": 16,
        "b38_tail_reference_lr": 1e-4,
        "b38_encoder_trainable_stages": 1,
        "b38_encoder_lr_scale": 0.05,
        "b38_encoder_chunk_size": 4,
        "b20_crop_focus_enabled": True,
        "b20_crop_focus_version": "joint_focus_center_crop_only_v1",
        "b20_crop_focus_crop_fraction": 0.90,
    }


def test_b38_contract_is_global_only_and_frozen() -> None:
    policy = require_b38_global_contract(_config())
    assert policy["crop_fraction"] == 0.90

    bad_slices = _config()
    bad_slices["b7_n_slices"] = 32
    try:
        require_b38_global_contract(bad_slices)
    except ValueError:
        pass
    else:
        raise AssertionError("B38 accepted non-historical slice centres")

    sparse_carryover = _config()
    sparse_carryover["b37_top_k"] = 8
    try:
        require_b38_global_contract(sparse_carryover)
    except ValueError:
        pass
    else:
        raise AssertionError("B38 accepted a sparse-MIL configuration field")


def test_b38_preprocess_is_16_historical_triplets_at_448(monkeypatch) -> None:
    import rsna_knee.b38_highres_global as module

    calls = []
    original = module.F.interpolate

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", wrapped)
    raw = np.arange(40 * 24 * 32, dtype=np.float32).reshape(40, 24, 32)
    image, position = preprocess_global_triplets_b38(
        raw,
        gap=1,
        center_offset=0,
    )
    expected_centres, expected_position = b35_centers(
        40,
        gap=1,
        center_offset=0,
        base_slices=B38_N_SLICES,
        dense_slices=B38_N_SLICES,
    )
    assert tuple(image.shape) == (
        B38_N_SLICES,
        3,
        B38_IMAGE_SIZE,
        B38_IMAGE_SIZE,
    )
    assert tuple(position.shape) == (B38_N_SLICES,)
    assert np.array_equal(position, expected_position)
    assert tuple(expected_centres.shape) == (B38_N_SLICES,)
    assert len(calls) == 1
    assert calls[0]["size"] == (448, 448)
    assert calls[0]["mode"] == "bilinear"
    assert calls[0]["align_corners"] is False
    assert calls[0]["antialias"] is True
