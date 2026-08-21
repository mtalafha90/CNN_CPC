from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee import b37_highres as b37


def _volume(depth: int = 24, height: int = 80, width: int = 100) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[None, :, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, None, :]
    z = np.linspace(0.0, 0.2, depth, dtype=np.float32)[:, None, None]
    return (x + y + z).astype(np.float32)


def test_b37_native_crop_shape_is_exact_90_percent() -> None:
    x = np.zeros((16, 3, 512, 512), dtype=np.float32)
    cropped = b37._native_center_crop_4d(x, 0.90)
    assert cropped.shape == (16, 3, 461, 461)


def test_b37_preprocessor_outputs_fixed_288_triplets() -> None:
    out = b37.preprocess_triplets_b37(
        _volume(),
        n_slices=16,
        image_size=288,
        gap=1,
    )
    assert out.shape == (16, 3, 288, 288)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_b37_deterministic_preprocessing_calls_resize_once(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    original = b37.F.interpolate

    def counted(*args, **kwargs):
        size = kwargs.get("size", args[1] if len(args) > 1 else None)
        calls.append(tuple(int(x) for x in size))
        return original(*args, **kwargs)

    monkeypatch.setattr(b37.F, "interpolate", counted)
    out = b37.preprocess_triplets_b37(_volume(), image_size=288, gap=1)
    assert out.shape[-2:] == (288, 288)
    assert calls == [(288, 288)]


def test_b37_normalizes_full_volume_before_native_crop(monkeypatch) -> None:
    seen: dict[str, tuple[int, ...]] = {}
    original = b37._normalise_volume

    def recorded(v):
        seen["shape"] = tuple(np.asarray(v).shape)
        return original(v)

    monkeypatch.setattr(b37, "_normalise_volume", recorded)
    source = _volume(depth=20, height=90, width=110)
    b37.preprocess_triplets_b37(source, n_slices=8, image_size=288, gap=1)
    assert seen["shape"] == source.shape


def test_b37_contract_rejects_historical_224_resolution() -> None:
    with pytest.raises(ValueError, match="b7_image_size=288"):
        b37.require_b37_preprocessing_contract(
            {
                "b7_image_size": 224,
                "b20_crop_focus_version": "joint_focus_center_crop_only_v1",
                "b20_crop_focus_crop_fraction": 0.90,
            }
        )


def test_b37_state_records_single_resize_and_no_post_resize_crop() -> None:
    state = b37.b37_preprocessing_state()
    assert state["image_size"] == 288
    assert state["crop_fraction"] == pytest.approx(0.90)
    assert state["deterministic_resize_count"] == 1
    assert state["historical_b20_post_resize_crop_applied"] is False
    assert state["normalization_stage"] == "full_native_volume_before_crop"
