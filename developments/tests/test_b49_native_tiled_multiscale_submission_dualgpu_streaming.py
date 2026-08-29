from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.b35_target_spatial_residual import b35_centers
from rsna_knee.b49_native_tiled_multiscale_mil import full_fov_context_from_normalized
from rsna_knee.b49_native_tiled_multiscale_submission_dualgpu_streaming import (
    B49_PRELOADED_NORMALIZED_SOURCE_KEY,
    b49_streamed_study_metadata,
    build_b49_streamed_view,
    preloaded_or_disk_b49_source_normalized,
)


def _records() -> list[dict]:
    return [
        {
            "study_uid": "study-a",
            "series_uid": "series-a",
            "plane_id": 1,
            "fluid_id": 2,
            "fat_id": 1,
        },
        {
            "study_uid": "study-a",
            "series_uid": "series-b",
            "plane_id": 3,
            "fluid_id": 0,
            "fat_id": 2,
        },
    ]


def _normalized(seed: int, frames: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(frames, height, width)).astype(np.float32)


def test_b49_streamed_view_matches_frozen_context_and_centres():
    records = _records()
    normalized = [_normalized(4, 19, 73, 121), _normalized(5, 27, 121, 73)]
    context, sources, position = build_b49_streamed_view(
        normalized,
        records,
        ["/tmp/series-a", "/tmp/series-b"],
        study_uid="study-a",
        gap=1,
        center_offset=-1,
    )

    assert len(context) == len(records) == len(sources)
    assert tuple(position.shape) == (2, 32)
    for series_index, native in enumerate(normalized):
        centres, expected_position = b35_centers(len(native), gap=1, center_offset=-1)
        expected_context = full_fov_context_from_normalized(native, centres, gap=1)
        assert torch.equal(context[series_index], expected_context)
        assert torch.equal(position[series_index], torch.from_numpy(expected_position))
        assert sources[series_index]["centres"] == centres.tolist()
        assert sources[series_index]["study_uid"] == "study-a"
        assert sources[series_index]["slice_positions"] == pytest.approx(expected_position.tolist())
        assert sources[series_index][B49_PRELOADED_NORMALIZED_SOURCE_KEY] is native


def test_b49_streamed_metadata_is_tta_invariant_and_complete():
    present, meta = b49_streamed_study_metadata(_records())
    assert torch.equal(present, torch.ones(2, dtype=torch.float32))
    assert torch.equal(meta, torch.tensor([[1, 2, 1], [3, 0, 2]], dtype=torch.long))


def test_preloaded_b49_source_reuses_exact_array_and_checks_geometry():
    native = _normalized(9, 5, 41, 67)
    source = {
        "native_height": 41,
        "native_width": 67,
        B49_PRELOADED_NORMALIZED_SOURCE_KEY: native,
    }
    assert preloaded_or_disk_b49_source_normalized(source) is native
    source["native_width"] = 66
    with pytest.raises(RuntimeError, match="geometry"):
        preloaded_or_disk_b49_source_normalized(source)


def test_b49_streamed_view_rejects_changed_gap():
    with pytest.raises(ValueError, match="triplet gap"):
        build_b49_streamed_view(
            [_normalized(13, 7, 33, 35)],
            _records()[:1],
            ["/tmp/series-a"],
            study_uid="study-a",
            gap=2,
            center_offset=0,
        )
