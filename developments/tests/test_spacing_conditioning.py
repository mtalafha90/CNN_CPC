"""Telling the model how thick a series is.

Three properties carry the whole design and are pinned here:

* zero at initialisation, so switching this on cannot change where a run starts;
* switchable, so one trained checkpoint yields its own ablation;
* silent on a series with no measurable spacing, rather than inventing one.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.spacing_conditioning import (
    SPACING_BASIS,
    SPACING_MAX_MM,
    SPACING_MIN_MM,
    SpacingConditioning,
    normalised_log_spacing,
    resolve_spacing,
    spacing_basis,
    spacing_lookup,
    triplet_depth_mm,
)


# --- the basis ----------------------------------------------------------------


def test_the_window_brackets_the_measured_corpus():
    """0.80 mm to 8.33 mm was the measured range; neither end may clip."""
    assert SPACING_MIN_MM < 0.80
    assert SPACING_MAX_MM > 8.33


def test_the_scale_is_logarithmic_not_linear():
    """Doubling the spacing is the same step wherever it happens."""
    z = normalised_log_spacing(torch.tensor([0.8, 1.6, 3.2, 6.4]))
    steps = torch.diff(z)
    assert torch.allclose(steps, steps[0].expand_as(steps), atol=1e-6)


def test_the_ends_of_the_window_map_to_zero_and_one():
    z = normalised_log_spacing(torch.tensor([SPACING_MIN_MM, SPACING_MAX_MM]))
    assert z[0].item() == pytest.approx(0.0)
    assert z[1].item() == pytest.approx(1.0)


def test_a_spacing_outside_the_window_is_clamped_not_wrapped():
    z = normalised_log_spacing(torch.tensor([0.01, 500.0]))
    assert z[0].item() == pytest.approx(0.0)
    assert z[1].item() == pytest.approx(1.0)


def test_a_missing_spacing_is_not_treated_as_a_thin_series():
    """NaN must not silently become the smallest spacing in the window."""
    z = normalised_log_spacing(torch.tensor([float("nan"), 0.0, -1.0]))
    assert torch.all(z == 0.0)


def test_the_basis_has_the_declared_width():
    assert spacing_basis(torch.tensor([3.3])).shape == (1, SPACING_BASIS)


def test_the_basis_keeps_the_leading_shape():
    assert spacing_basis(torch.zeros(2, 5)).shape == (2, 5, SPACING_BASIS)


def test_two_different_spacings_get_different_descriptions():
    thin = spacing_basis(torch.tensor([0.6]))
    thick = spacing_basis(torch.tensor([6.0]))
    assert not torch.allclose(thin, thick)


def test_the_triplet_depth_is_twice_the_gap_times_the_spacing():
    assert triplet_depth_mm([3.3], gap=1)[0] == pytest.approx(6.6)
    assert triplet_depth_mm([3.3], gap=2)[0] == pytest.approx(13.2)


# --- the module ---------------------------------------------------------------


def test_it_contributes_nothing_at_initialisation():
    """The whole point: a run that switches this on starts where one without it does."""
    module = SpacingConditioning(16)
    out = module(torch.tensor([0.6, 3.3, 8.0]))

    assert out.shape == (3, 16)
    assert torch.all(out == 0.0)


def test_it_contributes_something_once_trained():
    module = SpacingConditioning(16)
    with torch.no_grad():
        module.projection.weight.normal_()
    assert module(torch.tensor([3.3])).abs().sum() > 0


def test_switching_it_off_zeroes_a_trained_module():
    """This is the free ablation: one checkpoint, both arms."""
    module = SpacingConditioning(16)
    with torch.no_grad():
        module.projection.weight.normal_()

    live = module(torch.tensor([0.6, 6.0]))
    module.set_enabled(False)
    ablated = module(torch.tensor([0.6, 6.0]))

    assert live.abs().sum() > 0
    assert torch.all(ablated == 0.0)
    assert ablated.shape == live.shape


def test_switching_it_back_on_restores_it():
    module = SpacingConditioning(16)
    with torch.no_grad():
        module.projection.weight.normal_()
    before = module(torch.tensor([3.3]))
    assert torch.allclose(module.set_enabled(False).set_enabled(True)(torch.tensor([3.3])), before)


def test_a_series_with_no_spacing_contributes_nothing():
    """Not a guessed spacing — nothing, like a padded series."""
    module = SpacingConditioning(16)
    with torch.no_grad():
        module.projection.weight.normal_()
    out = module(torch.tensor([float("nan"), 3.3]))

    assert torch.all(out[0] == 0.0)
    assert out[1].abs().sum() > 0


def test_it_matches_the_series_shape_the_model_uses():
    """[batch, series] in, [batch, series, d] out, ready to add to features."""
    module = SpacingConditioning(8)
    assert module(torch.full((4, 6), 3.3)).shape == (4, 6, 8)


def test_a_gradient_reaches_the_projection():
    module = SpacingConditioning(4)
    module(torch.tensor([3.3])).sum().backward()
    assert module.projection.weight.grad is not None
    assert torch.isfinite(module.projection.weight.grad).all()


def test_the_width_must_be_positive():
    with pytest.raises(ValueError, match="d_model"):
        SpacingConditioning(0)


# --- where the number comes from ----------------------------------------------


def _geometry_csv(path, rows):
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": series,
                "slice_spacing_mm": spacing,
            }
            for study, series, spacing in rows
        ]
    ).to_csv(path, index=False)
    return path


def test_the_lookup_reads_the_scans_table(tmp_path):
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", 3.3), ("a", "s2", 0.6)])
    lookup = spacing_lookup(path)

    assert lookup[("a", "s1")] == pytest.approx(3.3)
    assert lookup[("a", "s2")] == pytest.approx(0.6)


def test_a_table_without_the_spacing_column_is_refused(tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"], "SeriesInstanceUID": ["s1"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="slice_geometry_scan"):
        spacing_lookup(path)


def test_the_table_is_preferred_when_it_has_a_value(tmp_path):
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", 3.3)])
    assert resolve_spacing("a", "s1", lookup=spacing_lookup(path)) == pytest.approx(3.3)


def test_a_series_absent_from_the_table_falls_through_to_the_headers(tmp_path):
    """Which is what a submission set does, having no table at all."""
    from test_slice_geometry_scan import _write_series

    folder = tmp_path / "series"
    _write_series(folder, 8, pitch=4.0)
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", 3.3)])

    value = resolve_spacing(
        "b", "s2", lookup=spacing_lookup(path), series_dir=folder
    )
    assert value == pytest.approx(4.0)


def test_a_blank_row_in_the_table_also_falls_through(tmp_path):
    from test_slice_geometry_scan import _write_series

    folder = tmp_path / "series"
    _write_series(folder, 8, pitch=4.0)
    path = _geometry_csv(tmp_path / "g.csv", [("a", "s1", None)])

    value = resolve_spacing(
        "a", "s1", lookup=spacing_lookup(path), series_dir=folder
    )
    assert value == pytest.approx(4.0)


def test_no_table_and_no_directory_means_no_spacing():
    assert math.isnan(resolve_spacing("a", "s1"))


def test_an_unreadable_directory_means_no_spacing(tmp_path):
    assert math.isnan(resolve_spacing("a", "s1", series_dir=tmp_path / "nowhere"))


def test_the_resolved_value_flows_through_to_a_zero_contribution():
    """End to end: an unresolvable series must not perturb the model."""
    module = SpacingConditioning(8)
    with torch.no_grad():
        module.projection.weight.normal_()
    spacing = torch.tensor([resolve_spacing("a", "s1")], dtype=torch.float32)
    assert torch.all(module(spacing) == 0.0)
    assert np.isnan(spacing.item())
