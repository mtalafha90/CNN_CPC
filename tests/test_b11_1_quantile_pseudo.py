import numpy as np

from rsna_knee.b11_1_quantile_pseudo import (
    B11_1_HIGH_TARGET,
    B11_1_LOW_TARGET,
    combine_b6_and_quantile_teacher,
)
from rsna_knee.constants import TARGETS


def _arrays(n=200):
    y = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros_like(y)
    mean = np.zeros_like(y)
    spread = np.full_like(y, 0.01)
    for j in range(len(TARGETS)):
        # Deliberately different absolute calibration by target while preserving rank.
        lo = 0.02 + 0.04 * j
        hi = min(0.98, lo + 0.35)
        mean[:, j] = np.linspace(lo, hi, n, dtype=np.float32)
    return y, w, mean, spread


def test_quantile_policy_uses_both_tails_despite_calibration_shift():
    y, w, mean, spread = _arrays()
    # Give each target enough B6 mass for the pseudo cap not to bind.
    w[:100, :] = 1.0
    y[:100, :] = 0.05
    combined_y, combined_w, pseudo_w, summary, thresholds = combine_b6_and_quantile_teacher(
        y, w, mean, spread
    )
    assert summary["pseudo_cells"] > 0
    for target in TARGETS:
        row = summary["per_target"][target]
        assert row["pseudo_low_cells"] > 0
        assert row["pseudo_high_cells"] > 0
        assert thresholds[target][0] < thresholds[target][1]
    assert np.any(combined_y == B11_1_LOW_TARGET)
    assert np.any(combined_y == B11_1_HIGH_TARGET)
    assert np.any(pseudo_w > 0)
    assert np.all(combined_w >= w)


def test_b6_cells_are_never_overwritten():
    y, w, mean, spread = _arrays()
    y[0, :] = 0.85
    w[0, :] = 0.5
    original = y[0].copy()
    combined_y, combined_w, pseudo_w, _, _ = combine_b6_and_quantile_teacher(y, w, mean, spread)
    assert np.allclose(combined_y[0], original)
    assert np.allclose(combined_w[0], w[0])
    assert np.allclose(pseudo_w[0], 0.0)


def test_unstable_tail_cells_are_rejected():
    y, w, mean, spread = _arrays()
    w[:100, :] = 1.0
    # Make every unsupervised candidate unstable.
    spread[100:, :] = 0.20
    _, _, pseudo_w, summary, _ = combine_b6_and_quantile_teacher(y, w, mean, spread)
    assert np.allclose(pseudo_w, 0.0)
    assert summary["pseudo_cells"] == 0
    assert summary["viability_passed"] is False
