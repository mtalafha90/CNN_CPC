"""A fixed physical crop, and pointing every knee the same way.

The test that matters most is the sagittal one. Mirroring a sagittal knee
horizontally is the obvious implementation, it looks right, and it would swap
anterior and posterior on the largest plane in this corpus -- consistently, on
every right knee, for the whole run. Worse than not normalising at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.physical_crop import (
    CANONICAL_SIDE,
    CROP_MM,
    crop_pixels,
    crop_to_millimetres,
    laterality_from_positions,
    normalise_laterality,
    resolve_laterality,
    summarise,
)


def _volume(slices=4, h=200, w=180):
    """[S, C, H, W] with every voxel distinct, so any move is detectable."""
    return np.arange(slices * 3 * h * w, dtype=np.float32).reshape(slices, 3, h, w)


# --- the physical crop --------------------------------------------------------


def test_the_crop_is_a_fixed_number_of_millimetres():
    assert crop_pixels(0.5, 130.0) == 260
    assert crop_pixels(1.0, 130.0) == 130
    assert crop_pixels(0.26, 130.0) == 500


def test_two_scanners_see_the_same_anatomy():
    """The whole point: same millimetres, different pixel counts."""
    fine, coarse = _volume(h=600, w=600), _volume(h=300, w=300)

    a, audit_a = crop_to_millimetres(fine, 0.25)
    b, audit_b = crop_to_millimetres(coarse, 0.50)

    assert audit_a["retained_mm"] == pytest.approx(audit_b["retained_mm"], rel=1e-6)
    assert a.shape[-1] != b.shape[-1], "different pixel counts, same millimetres"


def test_an_unusable_spacing_falls_back_and_says_so():
    """A wrong number is worse than the old behaviour; both are recorded."""
    for bad in (0.0, -1.0, float("nan"), 99.0):
        out, audit = crop_to_millimetres(_volume(), bad)
        assert audit["mode"] == "fallback"
        assert out.shape[-2:] == (180, 162)


def test_a_crop_larger_than_the_image_is_named_clipped():
    """The failure the 160 mm figure would have hit in ~60% of series.

    130 mm at 1 mm/px wants 130 pixels from a 100-pixel image, so nothing can
    be removed and the run must be able to see that it happened.
    """
    out, audit = crop_to_millimetres(_volume(h=100, w=100), 1.0)

    assert audit["mode"] == "clipped"
    assert audit["wanted_pixels"] == 130
    assert out.shape[-2:] == (100, 100), "nothing could be removed"


def test_a_crop_that_only_just_fits_is_not_called_clipped():
    """The boundary, so the two modes cannot drift into each other."""
    _, audit = crop_to_millimetres(_volume(h=130, w=130), 1.0)

    assert audit["mode"] == "physical"
    assert audit["cropped_hw"] == [130, 130]


def test_a_crop_that_fits_is_named_physical():
    _, audit = crop_to_millimetres(_volume(h=600, w=600), 0.25)
    assert audit["mode"] == "physical"


def test_the_crop_is_centred():
    out, _ = crop_to_millimetres(_volume(h=200, w=200), 1.0)
    assert out.shape[-2:] == (130, 130)
    expected = _volume(h=200, w=200)[..., 35:165, 35:165]
    assert np.array_equal(out, expected)


def test_the_crop_does_not_interpolate():
    """Every retained value must be a value that was there before."""
    source = _volume(h=300, w=300)
    out, _ = crop_to_millimetres(source, 0.5)
    assert np.isin(out, source).all()


def test_a_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"\[S, C, H, W\]"):
        crop_to_millimetres(np.zeros((4, 200, 200)), 0.5)


def test_the_default_is_the_measured_figure():
    assert CROP_MM == 130.0


# --- which knee ---------------------------------------------------------------


def test_positive_x_is_the_left_knee():
    """DICOM's patient frame puts +x toward the patient's left."""
    assert laterality_from_positions([[62.0, 0, 0], [64.0, 0, 1]]) == "L"


def test_negative_x_is_the_right_knee():
    assert laterality_from_positions([[-62.0, 0, 0], [-64.0, 0, 1]]) == "R"


def test_a_stack_near_the_midline_is_refused():
    """Guessing is worse than leaving the study alone."""
    assert laterality_from_positions([[1.0, 0, 0], [-1.0, 0, 1]]) is None


def test_the_median_decides_a_stack_that_crosses_the_midline():
    positions = [[-5.0, 0, 0], [40.0, 0, 1], [45.0, 0, 2], [50.0, 0, 3]]
    assert laterality_from_positions(positions) == "L"


def test_missing_or_malformed_positions_are_refused():
    for bad in (None, [], np.zeros((0, 3)), [[1.0, 2.0]]):
        assert laterality_from_positions(bad) is None


def test_the_tag_wins_over_the_derivation():
    resolved = resolve_laterality("R", [[62.0, 0, 0]])

    assert resolved["laterality"] == "R"
    assert resolved["source"] == "tag"
    assert resolved["agree"] is False


def test_the_derivation_is_used_when_the_tag_is_absent():
    """The tag is missing on roughly half of studies."""
    resolved = resolve_laterality(None, [[-62.0, 0, 0]])

    assert resolved["laterality"] == "R"
    assert resolved["source"] == "positions"


def test_disagreement_is_recorded_rather_than_hidden():
    """So a run can measure whether the derivation is trustworthy."""
    assert resolve_laterality("L", [[62.0, 0, 0]])["agree"] is True
    assert resolve_laterality("L", [[-62.0, 0, 0]])["agree"] is False
    assert resolve_laterality("L", None)["agree"] is None


def test_neither_source_gives_nothing():
    resolved = resolve_laterality(None, None)
    assert resolved["laterality"] is None
    assert resolved["source"] is None


# --- the plane-aware normalisation, which is the delicate part ----------------


def test_a_canonical_knee_is_untouched():
    volume = _volume()
    out, audit = normalise_laterality(volume, CANONICAL_SIDE, "coronal")

    assert np.array_equal(out, volume)
    assert audit["applied"] == "none"


def test_a_coronal_right_knee_is_mirrored_in_the_image():
    volume = _volume()
    out, audit = normalise_laterality(volume, "R", "coronal")

    assert audit["applied"] == "width_mirrored"
    assert np.array_equal(out, volume[..., ::-1])


def test_an_axial_right_knee_is_mirrored_in_the_image():
    out, audit = normalise_laterality(_volume(), "R", "axial")
    assert audit["applied"] == "width_mirrored"


def test_a_sagittal_right_knee_reverses_the_slice_order_instead():
    """The one that matters.

    A sagittal image's axes are anterior-posterior and superior-inferior; there
    is no left-right axis in the picture. Mirroring the width would swap
    anterior and posterior on every right knee, consistently, all run.
    """
    volume = _volume()
    out, audit = normalise_laterality(volume, "R", "sagittal")

    assert audit["applied"] == "slice_order_reversed"
    assert np.array_equal(out, volume[::-1])
    assert not np.array_equal(out, volume[..., ::-1]), (
        "a sagittal series must not be mirrored in the image plane"
    )


def test_the_in_plane_content_of_a_sagittal_series_is_preserved():
    """Each slice's picture is untouched; only their order changes."""
    volume = _volume()
    out, _ = normalise_laterality(volume, "R", "sagittal")

    for index in range(volume.shape[0]):
        assert np.array_equal(out[index], volume[volume.shape[0] - 1 - index])


def test_an_unknown_plane_does_nothing():
    volume = _volume()
    out, audit = normalise_laterality(volume, "R", None)

    assert np.array_equal(out, volume)
    assert "unknown plane" in audit["reason"]


def test_an_unknown_laterality_does_nothing():
    volume = _volume()
    out, audit = normalise_laterality(volume, None, "coronal")

    assert np.array_equal(out, volume)
    assert audit["reason"] == "laterality unknown"


def test_normalisation_is_its_own_inverse():
    """Applying it twice returns the original, for both plane behaviours."""
    volume = _volume()
    for plane in ("coronal", "sagittal"):
        once, _ = normalise_laterality(volume, "R", plane)
        twice, _ = normalise_laterality(once, "R", plane)
        assert np.array_equal(twice, volume), plane


def test_the_result_is_contiguous():
    """Negative strides break downstream torch.from_numpy calls."""
    for plane in ("coronal", "sagittal"):
        out, _ = normalise_laterality(_volume(), "R", plane)
        assert out.flags["C_CONTIGUOUS"], plane


def test_case_is_not_significant():
    volume = _volume()
    out, audit = normalise_laterality(volume, "r", "Sagittal")
    assert audit["applied"] == "slice_order_reversed"


# --- the audit ----------------------------------------------------------------


def test_the_summary_counts_every_outcome():
    records = [
        {"mode": "physical", "applied": "width_mirrored", "agree": True},
        {"mode": "physical", "applied": "none", "agree": True},
        {"mode": "fallback", "applied": "slice_order_reversed", "agree": False},
        {"mode": "clipped", "applied": "none"},
    ]
    result = summarise(records)

    assert result["series"] == 4
    assert result["crop_mode"] == {"clipped": 1, "fallback": 1, "physical": 2}
    assert result["laterality_applied"]["none"] == 2
    assert result["laterality_tag_vs_derived_agreement"] == pytest.approx(2 / 3)
    assert result["laterality_comparable_series"] == 3


def test_the_summary_survives_having_nothing_to_compare():
    result = summarise([{"mode": "fallback", "applied": "none"}])
    assert result["laterality_tag_vs_derived_agreement"] is None
