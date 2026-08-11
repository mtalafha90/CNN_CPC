"""Tests for the weak-label validation surface."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.weak_validation import (
    evaluate_on_weak_surface,
    format_weak_report,
    make_weak_holdout,
    resolution_estimate,
    weak_macro_auc,
)


def test_holdout_is_roughly_the_requested_size():
    uids = [f"s{i}" for i in range(1000)]
    mask = make_weak_holdout(uids, holdout_fraction=0.2, seed=1)
    assert 0.18 <= mask.mean() <= 0.22


def test_holdout_is_reproducible_and_seed_sensitive():
    uids = [f"s{i}" for i in range(500)]
    assert np.array_equal(
        make_weak_holdout(uids, seed=7), make_weak_holdout(uids, seed=7)
    )
    assert not np.array_equal(
        make_weak_holdout(uids, seed=7), make_weak_holdout(uids, seed=8)
    )


def test_duplicate_reports_never_straddle_the_split():
    """Teacher labels leak if the same report appears on both sides."""
    uids = [f"s{i}" for i in range(600)]
    groups = [f"g{i // 3}" for i in range(600)]  # three studies share each report

    mask = make_weak_holdout(uids, report_groups=groups, holdout_fraction=0.25, seed=3)

    for group in set(groups):
        members = mask[np.asarray(groups) == group]
        assert members.all() or (~members).all()


def test_invalid_fraction_is_rejected():
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b"], holdout_fraction=0.0)
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b"], holdout_fraction=1.0)


def test_misaligned_groups_are_rejected():
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b", "c"], report_groups=["g1", "g2"])


# --- Scoring ---------------------------------------------------------------


def _surface(n=200, n_targets=12, seed=0, signal=0.5):
    rng = np.random.default_rng(seed)
    truth = (rng.random((n, n_targets)) < 0.4).astype(float)
    soft = np.where(truth > 0.5, 0.85, 0.05)
    weights = np.ones((n, n_targets))
    predictions = truth * signal + rng.random((n, n_targets)) * (1 - signal)
    return soft, predictions, weights, truth


def test_perfect_agreement_scores_one():
    soft, _, weights, truth = _surface()
    score, _ = weak_macro_auc(soft, truth, weights)
    assert score == pytest.approx(1.0)


def test_unweighted_cells_are_excluded_not_treated_as_negatives():
    """Zero weight means 'unlabelled'. Counting it as 0 would invent negatives."""
    soft, predictions, weights, _ = _surface(n=100)
    weights[:, 0] = 0.0  # target 0 entirely unlabelled

    _, per_target = weak_macro_auc(soft, predictions, weights)

    assert np.isnan(per_target[0])
    assert np.isfinite(per_target[1])


def test_shape_mismatch_is_rejected():
    soft, predictions, weights, _ = _surface(n=50)
    with pytest.raises(ValueError):
        weak_macro_auc(soft, predictions[:10], weights)


def test_evaluate_reports_cell_counts_and_the_caveat():
    soft, predictions, weights, _ = _surface(n=300)

    payload = evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=100)

    assert payload["surface"] == "weak_b6_holdout"
    assert payload["labelled_cells"] == 300 * 12
    assert payload["positive_cells"] + payload["negative_cells"] == payload["labelled_cells"]
    assert "not with expert truth" in payload["measures"]
    assert set(payload["cells_per_target"]) == set(TARGETS)


def test_weak_surface_interval_is_far_tighter_than_gold():
    """The whole point: 3,120 studies resolve what 58 cannot."""
    small_soft, small_pred, small_w, _ = _surface(n=58, seed=1, signal=0.3)
    big_soft, big_pred, big_w, _ = _surface(n=3120, seed=1, signal=0.3)

    small = evaluate_on_weak_surface(small_soft, small_pred, small_w, n_bootstrap=200)
    big = evaluate_on_weak_surface(big_soft, big_pred, big_w, n_bootstrap=200)

    small_width = small["ci_upper"] - small["ci_lower"]
    big_width = big["ci_upper"] - big["ci_lower"]
    assert big_width < small_width / 3


def test_report_states_the_bias_caveat():
    soft, predictions, weights, _ = _surface(n=120)
    text = format_weak_report(evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=50))

    assert "RANK" in text
    assert "0.606" in text
    assert "58 gold" in text


# --- Resolution arithmetic -------------------------------------------------


def test_resolution_improves_with_the_square_root_of_n():
    gold = resolution_estimate(58)
    weak = resolution_estimate(3120)

    assert gold["estimated_ci_width"] == pytest.approx(0.115)
    # sqrt(3120/58) ~ 7.3x tighter
    assert weak["estimated_ci_width"] < 0.02
    assert weak["versus_gold_58"] > 6.0


def test_resolution_rejects_an_empty_surface():
    with pytest.raises(ValueError):
        resolution_estimate(0)
