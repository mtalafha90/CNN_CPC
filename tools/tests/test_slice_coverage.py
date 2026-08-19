"""The coverage maths must be right, because a decision rests on it."""

from __future__ import annotations

import pytest

from tools.slice_coverage import FOCAL_SPANS, SAMPLED_SLICES, _hit_chance


def test_a_thin_series_is_read_completely():
    """With fewer slices than we sample, nothing can be stepped over."""
    for frames in (1, 8, 16):
        assert _hit_chance(frames, span=4) == 1.0


def test_a_thick_series_can_step_over_a_small_structure():
    assert _hit_chance(200, span=4) < 0.5
    assert _hit_chance(300, span=4) < _hit_chance(200, span=4)


def test_a_larger_structure_is_easier_to_see():
    assert _hit_chance(200, span=8) > _hit_chance(200, span=4)


def test_the_chance_is_a_probability():
    for frames in (1, 50, 500, 5000):
        for span in FOCAL_SPANS:
            assert 0.0 <= _hit_chance(frames, span) <= 1.0


def test_the_boundary_is_where_the_gap_equals_the_structure():
    """At exactly one structure-width per gap, coverage is just complete."""
    frames = SAMPLED_SLICES * 4
    assert _hit_chance(frames, span=4) == pytest.approx(1.0)
    assert _hit_chance(frames + SAMPLED_SLICES, span=4) < 1.0
