from __future__ import annotations

import numpy as np

from rsna_knee.b35_target_spatial_residual import b35_centers
from rsna_knee.b44_frozen_64_center_coverage_audit import (
    B44_DENSE_SLICES,
    b44_nested_centers_64,
)


def test_b44_first_32_centers_are_exact_historical_b42_centers():
    for n_frames in (5, 17, 32, 61, 127):
        for offset in (-1, 0, 1):
            historical, historical_position = b35_centers(
                n_frames,
                gap=1,
                center_offset=offset,
                dense_slices=32,
            )
            centers64, position64 = b44_nested_centers_64(
                n_frames,
                gap=1,
                center_offset=offset,
            )
            assert centers64.shape == (B44_DENSE_SLICES,)
            assert position64.shape == (B44_DENSE_SLICES,)
            assert np.array_equal(centers64[:32], historical)
            assert np.array_equal(position64[:32], historical_position)


def test_b44_added_centers_stay_inside_scan_and_preserve_first16_base_path():
    n_frames = 101
    base16, base16_position = b35_centers(
        n_frames,
        gap=1,
        center_offset=0,
        dense_slices=16,
    )
    centers64, position64 = b44_nested_centers_64(
        n_frames,
        gap=1,
        center_offset=0,
    )
    assert np.array_equal(centers64[:16], base16)
    assert np.array_equal(position64[:16], base16_position)
    assert int(centers64.min()) >= 0
    assert int(centers64.max()) < n_frames
    assert np.all(position64 >= 0.0)
    assert np.all(position64 <= 1.0)
    # On a sufficiently long series the appended set must add coverage beyond
    # the historical 32-centre prefix rather than only duplicating it.
    assert len(set(centers64[32:].tolist()).difference(set(centers64[:32].tolist()))) > 0
