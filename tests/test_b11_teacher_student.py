import numpy as np

from rsna_knee.b11_pseudo_labels import (
    B11_PSEUDO_BASE_WEIGHT,
    B11_PSEUDO_MASS_CAP_FRACTION,
    combine_b6_and_teacher,
)
from rsna_knee.constants import TARGETS


def _arrays(n=8):
    y = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros_like(y)
    mean = np.full_like(y, 0.5)
    spread = np.zeros_like(y)
    return y, w, mean, spread


def test_b11_never_overwrites_b6_cell():
    y, w, mean, spread = _arrays(4)
    y[0, 0] = 0.85
    w[0, 0] = 0.50
    mean[0, 0] = 0.99
    combined_y, combined_w, pseudo_w, summary = combine_b6_and_teacher(y, w, mean, spread)
    assert combined_y[0, 0] == np.float32(0.85)
    assert combined_w[0, 0] == np.float32(0.50)
    assert pseudo_w[0, 0] == 0.0
    assert summary["per_target"][TARGETS[0]]["pseudo_cells"] == 0


def test_b11_requires_confidence_and_tta_consistency():
    y, w, mean, spread = _arrays(5)
    # Give the target non-zero B6 mass so accepted pseudo labels receive weight.
    y[4, 0] = 0.05
    w[4, 0] = 1.0
    mean[0, 0] = 0.95
    spread[0, 0] = 0.01  # accept
    mean[1, 0] = 0.89
    spread[1, 0] = 0.01  # confidence fail
    mean[2, 0] = 0.02
    spread[2, 0] = 0.06  # consistency fail
    mean[3, 0] = 0.05
    spread[3, 0] = 0.02  # accept
    _, combined_w, pseudo_w, summary = combine_b6_and_teacher(y, w, mean, spread)
    assert pseudo_w[0, 0] > 0
    assert pseudo_w[1, 0] == 0
    assert pseudo_w[2, 0] == 0
    assert pseudo_w[3, 0] > 0
    assert combined_w[4, 0] == 1.0
    assert summary["per_target"][TARGETS[0]]["pseudo_cells"] == 2


def test_b11_pseudo_mass_is_capped_relative_to_b6():
    n = 20
    y, w, mean, spread = _arrays(n)
    # B6 base mass = 4.0 for target 0.
    y[:4, 0] = 0.05
    w[:4, 0] = 1.0
    # 16 otherwise-unsupervised cells would have raw pseudo mass 3.2.
    mean[4:, 0] = 0.99
    _, _, pseudo_w, summary = combine_b6_and_teacher(y, w, mean, spread)
    applied_mass = float(pseudo_w[:, 0].sum())
    expected_cap = B11_PSEUDO_MASS_CAP_FRACTION * 4.0
    assert np.isclose(applied_mass, expected_cap, atol=1e-6)
    assert applied_mass < B11_PSEUDO_BASE_WEIGHT * 16
    assert summary["per_target"][TARGETS[0]]["pseudo_scale"] < 1.0


def test_b11_can_activate_previously_unsupervised_study():
    y, w, mean, spread = _arrays(3)
    # Supply B6 mass for the target from study 0, leaving study 1 initially inactive.
    y[0, 0] = 0.05
    w[0, 0] = 1.0
    mean[1, 0] = 0.99
    _, combined_w, pseudo_w, summary = combine_b6_and_teacher(y, w, mean, spread)
    assert w[1].sum() == 0
    assert pseudo_w[1].sum() > 0
    assert combined_w[1].sum() > 0
    assert summary["newly_activated_studies"] == 1
