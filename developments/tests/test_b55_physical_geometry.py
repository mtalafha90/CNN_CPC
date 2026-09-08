"""B55's geometry: a physical crop, one canonical side, and 336 instead of 448.

The test that earns its place hardest is the one pairing the slice-order
reversal with the position basis. Reversing a sagittal stack without reversing
the positions that describe it would tell the model that slice 0 is anterior
when it is now posterior -- silently, on every right knee, for the whole run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.b42_constant_area_aspect_sparse_mil import (
    B42_REFERENCE_AREA,
    preprocess_dense_triplets_b42,
)
from rsna_knee.b55_physical_geometry import (
    B55_REFERENCE_AREA,
    B55_REFERENCE_SIDE,
    preprocess_dense_triplets_b55,
    read_series_geometry,
)
from rsna_knee.physical_crop import CROP_MM


def _raw(frames=40, h=400, w=360, seed=0):
    return np.random.RandomState(seed).rand(frames, h, w).astype(np.float32)


def _run(**kwargs):
    defaults = {"pixel_spacing_mm": 0.4, "laterality": "L", "plane": "sagittal"}
    return preprocess_dense_triplets_b55(_raw(), **{**defaults, **kwargs})


# --- the resolution -----------------------------------------------------------


def test_the_reference_area_is_336_not_448():
    assert B55_REFERENCE_SIDE == 336
    assert B55_REFERENCE_AREA == 336 * 336
    assert B55_REFERENCE_AREA < B42_REFERENCE_AREA


def test_the_output_is_smaller_than_b42s():
    b55, _, _ = _run()
    b42, _ = preprocess_dense_triplets_b42(_raw())

    assert b55.shape[-1] * b55.shape[-2] < b42.shape[-1] * b42.shape[-2]


def test_the_reference_area_is_a_parameter():
    """So the resolution can be moved without editing the module."""
    small, _, _ = _run(reference_area=224 * 224)
    large, _, _ = _run(reference_area=448 * 448)

    assert small.shape[-1] * small.shape[-2] < large.shape[-1] * large.shape[-2]


def test_the_slice_count_and_channels_are_unchanged():
    """Slice choice is the frozen B35 contract; B55 does not touch it."""
    b55, _, _ = _run()
    b42, _ = preprocess_dense_triplets_b42(_raw())

    assert b55.shape[0] == b42.shape[0] == 32
    assert b55.shape[1] == 3


# --- the physical crop --------------------------------------------------------


def test_the_crop_is_physical_when_the_spacing_is_usable():
    """0.6 mm/px wants 217 px, which fits inside this 400x360 fixture."""
    _, _, audit = _run(pixel_spacing_mm=0.6)

    assert audit["mode"] == "physical"
    assert audit["wanted_pixels"] == int(round(CROP_MM / 0.6))


def test_two_spacings_retain_the_same_millimetres():
    """Both must fit, or the comparison measures the fixture's edges instead."""
    _, _, fine = _run(pixel_spacing_mm=0.6)
    _, _, coarse = _run(pixel_spacing_mm=0.9)

    assert fine["mode"] == coarse["mode"] == "physical"
    assert fine["retained_mm"] == pytest.approx(coarse["retained_mm"], rel=0.01)
    assert fine["cropped_hw"] != coarse["cropped_hw"]


def test_a_missing_spacing_falls_back_to_the_fraction():
    _, _, audit = _run(pixel_spacing_mm=float("nan"))
    assert audit["mode"] == "fallback"


def test_the_fallback_fraction_is_b42s_own():
    """So a series without geometry gets exactly the old behaviour."""
    _, _, audit = _run(pixel_spacing_mm=float("nan"), fallback_fraction=0.90)
    assert audit["cropped_hw"] == [360, 324]


# --- laterality, and the position basis that must follow it -------------------


def test_a_canonical_side_changes_nothing():
    _, _, audit = _run(laterality="L")
    assert audit["applied"] == "none"


def test_a_sagittal_right_knee_reverses_the_slice_order():
    _, _, audit = _run(laterality="R", plane="sagittal")
    assert audit["applied"] == "slice_order_reversed"


def test_the_position_basis_follows_the_reversed_slices():
    """The one that would fail silently.

    Reversing the stack without reversing the positions tells the model that
    slice 0 is anterior when it has just become posterior.
    """
    _, forward, _ = _run(laterality="L", plane="sagittal")
    _, reversed_, audit = _run(laterality="R", plane="sagittal")

    assert audit["applied"] == "slice_order_reversed"
    assert np.array_equal(reversed_, forward[::-1])


def test_the_position_basis_is_untouched_by_an_in_plane_mirror():
    """A coronal mirror changes the picture, not which slice is which."""
    _, forward, _ = _run(laterality="L", plane="coronal")
    _, mirrored, audit = _run(laterality="R", plane="coronal")

    assert audit["applied"] == "width_mirrored"
    assert np.array_equal(mirrored, forward)


def test_the_returned_position_is_contiguous():
    """torch.from_numpy refuses a negative-stride array."""
    _, position, _ = _run(laterality="R", plane="sagittal")
    assert position.flags["C_CONTIGUOUS"]
    torch.from_numpy(position)  # must not raise


def test_a_coronal_right_knee_produces_a_different_image():
    """The mirror reaches the pixels.

    Deliberately not asserted as exact equality with `flip(left)`:
    `resize_triplets_constant_area` reflection-pads to the encoder stride, and
    that padding is applied to one side only, so a mirror does not survive it
    bit-for-bit. The mirror itself is checked exactly in
    `test_physical_crop.py`, before any resize; here the claim is only that it
    happened.
    """
    left, _, _ = _run(laterality="L", plane="coronal")
    right, _, audit = _run(laterality="R", plane="coronal")

    assert audit["applied"] == "width_mirrored"
    assert not torch.equal(left, right)
    assert left.shape == right.shape
    # The same pixels, rearranged: the sorted contents must be near-identical.
    assert torch.allclose(
        left.flatten().sort().values, right.flatten().sort().values, atol=1e-3
    )


def test_an_unknown_laterality_leaves_the_series_alone():
    _, _, audit = _run(laterality=None)
    assert audit["applied"] == "none"


# --- the audit ----------------------------------------------------------------


def test_every_call_reports_what_it_did():
    _, _, audit = _run(pixel_spacing_mm=0.3, laterality="R", plane="axial")

    assert audit["mode"] in {"physical", "clipped", "fallback"}
    assert audit["applied"] == "width_mirrored"
    assert audit["pixel_spacing_mm"] == pytest.approx(0.3)


def test_a_bad_gap_is_refused():
    with pytest.raises(ValueError, match="gap must be positive"):
        _run(gap=0)


# --- reading the headers ------------------------------------------------------


def test_an_empty_directory_yields_no_geometry(tmp_path):
    directory = tmp_path / "series"
    directory.mkdir()
    geometry = read_series_geometry(str(directory))

    assert not np.isfinite(geometry["pixel_spacing_mm"])
    assert geometry["laterality"] is None


def test_unreadable_files_do_not_raise(tmp_path):
    """A series with pixels but broken headers must still train."""
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "1.dcm").write_bytes(b"not a dicom file")

    geometry = read_series_geometry(str(directory))
    assert not np.isfinite(geometry["pixel_spacing_mm"])


def test_the_read_is_cached(tmp_path):
    """Three TTA offsets must not pay for the headers three times."""
    directory = tmp_path / "cached"
    directory.mkdir()
    (directory / "1.dcm").write_bytes(b"x")

    before = read_series_geometry.cache_info().hits
    read_series_geometry(str(directory))
    read_series_geometry(str(directory))

    assert read_series_geometry.cache_info().hits > before


# --- what B55 does not change -------------------------------------------------


def test_b42s_own_preprocessing_is_untouched():
    """B55 is a separate path; the frozen one must behave as it always did."""
    a, pa = preprocess_dense_triplets_b42(_raw())
    b, pb = preprocess_dense_triplets_b42(_raw())

    assert torch.equal(a, b)
    assert np.array_equal(pa, pb)
    assert a.shape[-1] * a.shape[-2] > B55_REFERENCE_AREA * 0.8
