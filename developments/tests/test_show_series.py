"""Rendering a study must show what the model sees, not a prettier version.

The viewer exists so a disagreement between a report and an expert label can be
judged by looking. That only works if the picture is honest: the same reader,
the same normalisation, and slices spread through the volume rather than
whichever end the loader happened to start at.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.show_series import (
    DEFAULT_COLUMNS,
    _slice_indices,
    choose_series,
    montage,
    to_png_bytes,
)


def _series(name, *, plane=None, fluid=None, slices=20):
    return {
        "series": name,
        "path": None,
        "slices": slices,
        "plane": plane,
        "weighting": None,
        "fluid_sensitive": fluid,
    }


# --- choosing which series to show -------------------------------------------


def test_a_sagittal_fluid_sensitive_series_is_preferred():
    """Menisci and cruciates are read on sagittal fluid-sensitive images."""
    chosen = choose_series(
        [
            _series("axial", plane="Axial", fluid=True),
            _series("sag_t1", plane="Sagittal", fluid=False),
            _series("sag_fluid", plane="Sagittal", fluid=True),
        ]
    )
    assert chosen["series"] == "sag_fluid"


def test_sagittal_is_preferred_when_none_is_fluid_sensitive():
    chosen = choose_series(
        [_series("axial", plane="Axial"), _series("sag", plane="Sagittal")]
    )
    assert chosen["series"] == "sag"


def test_the_longest_series_wins_when_no_plane_is_known():
    chosen = choose_series([_series("short", slices=5), _series("long", slices=40)])
    assert chosen["series"] == "long"


def test_the_longest_sagittal_wins_among_several():
    chosen = choose_series(
        [
            _series("sag_short", plane="Sagittal", slices=11),
            _series("sag_long", plane="Sagittal", slices=33),
        ]
    )
    assert chosen["series"] == "sag_long"


def test_a_plane_recorded_in_lower_case_still_counts():
    chosen = choose_series(
        [_series("axial", plane="Axial", slices=99), _series("sag", plane="sagittal")]
    )
    assert chosen["series"] == "sag"


# --- which slices ------------------------------------------------------------


def test_slices_are_spread_across_the_whole_volume():
    """Taking the first N would show one end of the knee and miss the joint."""
    # linspace(0, 39, 5) is 0, 9.75, 19.5, 29.25, 39; numpy rounds half to even.
    assert _slice_indices(40, 5) == [0, 10, 20, 29, 39]


def test_a_short_series_shows_every_slice():
    assert _slice_indices(3, 12) == [0, 1, 2]


def test_the_indices_stay_inside_the_volume():
    for depth in (1, 2, 7, 33, 320):
        indices = _slice_indices(depth, 12)
        assert min(indices) >= 0
        assert max(indices) < depth
        assert indices == sorted(indices)


# --- tiling ------------------------------------------------------------------


def test_a_montage_lays_slices_out_in_reading_order():
    volume = np.stack([np.full((2, 2), value, dtype=np.float32) for value in (1, 2, 3, 4)])
    tiled = montage(volume, columns=2)

    assert tiled.shape == (4, 4)
    assert tiled[0, 0] == 1 and tiled[0, 2] == 2
    assert tiled[2, 0] == 3 and tiled[2, 2] == 4


def test_a_short_last_row_is_padded_not_dropped():
    volume = np.ones((3, 2, 2), dtype=np.float32)
    tiled = montage(volume, columns=2)

    assert tiled.shape == (4, 4)
    assert tiled[2, 2] == 0  # the empty fourth cell


def test_more_columns_than_slices_makes_one_row():
    volume = np.ones((2, 3, 3), dtype=np.float32)
    assert montage(volume, columns=DEFAULT_COLUMNS).shape == (3, 6)


def test_an_empty_volume_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        montage(np.zeros((0, 4, 4), dtype=np.float32))


def test_a_two_dimensional_array_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        montage(np.zeros((4, 4), dtype=np.float32))


# --- turning floats into a picture -------------------------------------------


def test_the_darkest_and_brightest_pixels_reach_the_ends_of_the_range():
    image = np.array([[-3.0, 0.0, 7.0]], dtype=np.float32)
    # 0.0 sits 3/10 of the way from -3 to 7: 76.5, rounded half to even.
    assert to_png_bytes(image).tolist() == [[0, 76, 255]]


def test_a_flat_image_does_not_divide_by_zero():
    assert to_png_bytes(np.full((2, 2), 5.0, dtype=np.float32)).tolist() == [[0, 0], [0, 0]]


def test_an_all_nan_image_renders_black_rather_than_raising():
    assert to_png_bytes(np.full((2, 2), np.nan, dtype=np.float32)).max() == 0


def test_a_nan_among_real_pixels_does_not_poison_the_scale():
    image = np.array([[0.0, np.nan, 10.0]], dtype=np.float32)
    rendered = to_png_bytes(image)

    assert rendered[0, 0] == 0
    assert rendered[0, 2] == 255
    assert rendered.dtype == np.uint8
