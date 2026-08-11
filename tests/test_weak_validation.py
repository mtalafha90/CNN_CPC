"""Tests for the leakage-safe B6 weak validation surface."""
from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.weak_validation import (
    compare_on_weak_surface,
    evaluate_on_weak_surface,
    format_weak_report,
    make_weak_holdout,
    rough_resolution_estimate,
    weak_macro_auc,
)


def test_report_groups_are_mandatory():
    with pytest.raises(ValueError, match="report_groups is required"):
        make_weak_holdout(["a", "b", "c"], None)


def test_holdout_is_roughly_requested_size_and_group_safe():
    uids = [f"s{i}" for i in range(1000)]
    groups = [f"g{i // 3}" for i in range(999)] + ["last"]
    mask = make_weak_holdout(uids, groups, holdout_fraction=0.2, seed=1)
    assert 0.19 <= mask.mean() <= 0.21
    groups_array = np.asarray(groups)
    for group in set(groups):
        members = mask[groups_array == group]
        assert members.all() or (~members).all()


def test_holdout_is_reproducible_and_seed_sensitive():
    uids = [f"s{i}" for i in range(500)]
    groups = [f"g{i // 2}" for i in range(500)]
    assert np.array_equal(
        make_weak_holdout(uids, groups, seed=7),
        make_weak_holdout(uids, groups, seed=7),
    )
    assert not np.array_equal(
        make_weak_holdout(uids, groups, seed=7),
        make_weak_holdout(uids, groups, seed=8),
    )


def test_invalid_fraction_and_misaligned_groups_are_rejected():
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b"], ["g1", "g2"], holdout_fraction=0.0)
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b"], ["g1", "g2"], holdout_fraction=1.0)
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b", "c"], ["g1", "g2"])


# --- Sparse weak scoring ---------------------------------------------------


def _surface(n=200, n_targets=12, seed=0, signal=0.5, missing_fraction=0.55):
    rng = np.random.default_rng(seed)
    truth = (rng.random((n, n_targets)) < 0.4).astype(float)
    soft = np.where(truth > 0.5, 0.85, 0.05)
    weights = np.ones((n, n_targets), dtype=float)
    missing = rng.random((n, n_targets)) < missing_fraction
    weights[missing] = 0.0
    predictions = truth * signal + rng.random((n, n_targets)) * (1 - signal)
    return soft, predictions, weights, truth


def test_perfect_agreement_scores_one_on_labelled_cells():
    soft, _, weights, truth = _surface(missing_fraction=0.4)
    score, _ = weak_macro_auc(soft, truth, weights)
    assert score == pytest.approx(1.0)


def test_unweighted_cells_are_excluded_not_treated_as_negatives():
    soft, predictions, weights, _ = _surface(n=100)
    weights[:, 0] = 0.0
    _, per_target = weak_macro_auc(soft, predictions, weights)
    assert np.isnan(per_target[0])
    assert np.isfinite(per_target[1])


def test_shape_mismatch_is_rejected():
    soft, predictions, weights, _ = _surface(n=50)
    with pytest.raises(ValueError):
        weak_macro_auc(soft, predictions[:10], weights)


def test_evaluate_reports_actual_sparse_cell_counts_and_caveat():
    soft, predictions, weights, _ = _surface(n=300, missing_fraction=0.65)
    payload = evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=100)
    assert payload["surface"] == "weak_b6_holdout"
    assert payload["labelled_cells"] == int((weights > 0).sum())
    assert payload["labelled_cells"] < 300 * 12
    assert payload["positive_cells"] + payload["negative_cells"] == payload["labelled_cells"]
    assert "not expert truth" in payload["measures"]
    assert set(payload["cells_per_target"]) == set(TARGETS)


def test_paired_weak_comparison_uses_same_sparse_surface():
    soft, pred_a, weights, truth = _surface(n=500, seed=4, signal=0.25, missing_fraction=0.5)
    rng = np.random.default_rng(5)
    pred_b = truth * 0.75 + rng.random(truth.shape) * 0.25
    result = compare_on_weak_surface(
        soft,
        pred_a,
        pred_b,
        weights,
        n_bootstrap=300,
        seed=6,
    )
    assert result["raw_difference_b_minus_a"] > 0
    assert result["probability_b_better"] > 0.9
    assert result["surface"] == "weak_b6_holdout"


def test_report_states_training_exclusion_and_independence_caveat():
    soft, predictions, weights, _ = _surface(n=120)
    text = format_weak_report(
        evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=50)
    )
    assert "holdout studies" in text
    assert "58-study gold" in text
    assert "independent signal" in text


# --- Resolution planning --------------------------------------------------


def test_rough_resolution_uses_actual_holdout_size_not_all_active_studies():
    rough = rough_resolution_estimate(624)
    # 20% of 3,120 is about 624; the simple study-count heuristic is ~0.035,
    # not the ~0.015 previously claimed from incorrectly using all 3,120.
    assert 0.03 < rough["rough_ci_width_from_study_count_only"] < 0.04
    assert "empirical bootstrap" in rough["warning"]


def test_resolution_rejects_empty_surface():
    with pytest.raises(ValueError):
        rough_resolution_estimate(0)
