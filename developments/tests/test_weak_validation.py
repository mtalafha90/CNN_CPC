"""Tests for the leakage-safe B6 weak validation surface."""
from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.weak_validation import (
    WEAK_HOLDOUT_V2,
    compare_on_weak_surface,
    evaluate_on_weak_surface,
    format_weak_report,
    make_stratified_weak_holdout,
    make_weak_holdout,
    rough_resolution_estimate,
    weak_macro_auc,
)


def test_report_groups_are_mandatory():
    with pytest.raises(ValueError, match="report_groups is required"):
        make_weak_holdout(["a", "b", "c"], None)


def test_historical_v1_holdout_is_group_safe():
    uids = [f"s{i}" for i in range(1000)]
    groups = [f"g{i // 3}" for i in range(999)] + ["last"]
    mask = make_weak_holdout(uids, groups, holdout_fraction=0.2, seed=1)
    assert 0.19 <= mask.mean() <= 0.21
    groups_array = np.asarray(groups)
    for group in set(groups):
        members = mask[groups_array == group]
        assert members.all() or (~members).all()


def _balanced_surface(n=800, seed=0):
    rng = np.random.default_rng(seed)
    truth = (rng.random((n, 12)) < 0.45).astype(float)
    # Make Synovitis deliberately rare-negative, mirroring the real B6 issue.
    truth[:, 8] = 1.0
    truth[:20, 8] = 0.0
    soft = np.where(truth > 0.5, 0.85, 0.05)
    weights = np.ones_like(soft)
    return soft, weights, truth


def test_v2_stratification_enforces_rare_class_floor_and_is_reproducible():
    n = 800
    uids = [f"s{i}" for i in range(n)]
    groups = [f"g{i}" for i in range(n)]
    soft, weights, truth = _balanced_surface(n=n, seed=4)

    mask_a, diagnostics_a = make_stratified_weak_holdout(
        uids,
        groups,
        soft,
        weights,
        holdout_fraction=0.2,
        seed=2026,
        min_class_count=4,
        n_candidates=512,
    )
    mask_b, diagnostics_b = make_stratified_weak_holdout(
        uids,
        groups,
        soft,
        weights,
        holdout_fraction=0.2,
        seed=2026,
        min_class_count=4,
        n_candidates=512,
    )

    assert np.array_equal(mask_a, mask_b)
    assert diagnostics_a == diagnostics_b
    assert 0.19 <= mask_a.mean() <= 0.21
    # Synovitis negatives are the rare class: at least four must land on each side.
    syn_neg = truth[:, 8] == 0
    assert int((mask_a & syn_neg).sum()) >= 4
    assert int(((~mask_a) & syn_neg).sum()) >= 4
    assert diagnostics_a["feasible_candidates"] > 0


def test_v2_stratification_never_splits_report_groups():
    n = 600
    uids = [f"s{i}" for i in range(n)]
    groups = [f"g{i // 2}" for i in range(n)]
    soft, weights, _ = _balanced_surface(n=n, seed=8)
    mask, _ = make_stratified_weak_holdout(
        uids,
        groups,
        soft,
        weights,
        holdout_fraction=0.2,
        seed=5,
        min_class_count=4,
        n_candidates=512,
    )
    groups_array = np.asarray(groups)
    for group in set(groups):
        members = mask[groups_array == group]
        assert members.all() or (~members).all()


def test_invalid_fraction_and_misaligned_groups_are_rejected():
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b"], ["g1", "g2"], holdout_fraction=0.0)
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b"], ["g1", "g2"], holdout_fraction=1.0)
    with pytest.raises(ValueError):
        make_weak_holdout(["a", "b", "c"], ["g1", "g2"])


# --- Sparse weak scoring ---------------------------------------------------


def _surface(n=300, n_targets=12, seed=0, signal=0.5, missing_fraction=0.45):
    rng = np.random.default_rng(seed)
    truth = (rng.random((n, n_targets)) < 0.4).astype(float)
    # Guarantee both classes for every target before sparsification.
    truth[:20, :] = 0.0
    truth[20:40, :] = 1.0
    soft = np.where(truth > 0.5, 0.85, 0.05)
    weights = np.ones((n, n_targets), dtype=float)
    missing = rng.random((n, n_targets)) < missing_fraction
    weights[missing] = 0.0
    # Keep the guaranteed anchors labelled.
    weights[:40, :] = 1.0
    predictions = truth * signal + rng.random((n, n_targets)) * (1 - signal)
    return soft, predictions, weights, truth


def test_perfect_agreement_scores_one_on_labelled_cells():
    soft, _, weights, truth = _surface(missing_fraction=0.4)
    score, per_target = weak_macro_auc(soft, truth, weights)
    assert score == pytest.approx(1.0)
    assert np.isfinite(per_target).all()


def test_unweighted_target_makes_strict_macro_undefined():
    soft, predictions, weights, _ = _surface(n=100)
    weights[:, 0] = 0.0
    score, per_target = weak_macro_auc(soft, predictions, weights)
    assert np.isnan(score)
    assert np.isnan(per_target[0])
    assert np.isfinite(per_target[1])


def test_shape_mismatch_is_rejected():
    soft, predictions, weights, _ = _surface(n=50)
    with pytest.raises(ValueError):
        weak_macro_auc(soft, predictions[:10], weights)


def test_evaluate_reports_actual_sparse_cell_counts_and_strict_contract():
    soft, predictions, weights, _ = _surface(n=300, missing_fraction=0.65)
    payload = evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=100)
    assert payload["surface"] == WEAK_HOLDOUT_V2
    assert payload["labelled_cells"] == int((weights > 0).sum())
    assert payload["labelled_cells"] < 300 * 12
    assert payload["positive_cells"] + payload["negative_cells"] == payload["labelled_cells"]
    assert payload["strict_all_12_targets"] is True
    assert payload["n_valid_replicates"] <= payload["n_bootstrap"]
    assert "not expert truth" in payload["measures"]
    assert set(payload["cells_per_target"]) == set(TARGETS)


def test_strict_bootstrap_discards_replicates_missing_rare_class():
    n = 300
    rng = np.random.default_rng(9)
    truth = (rng.random((n, 12)) < 0.5).astype(float)
    truth[:, 8] = 1.0
    truth[0, 8] = 0.0  # only one Synovitis negative
    soft = np.where(truth > 0.5, 0.85, 0.05)
    weights = np.ones_like(soft)
    predictions = truth * 0.6 + rng.random(truth.shape) * 0.4

    payload = evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=400, seed=12)
    assert payload["strict_all_12_targets"] is True
    # Roughly exp(-1) of study bootstraps omit the only negative; exact value varies.
    assert payload["valid_replicate_fraction"] < 0.8


def test_paired_weak_comparison_uses_same_strict_sparse_surface():
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
    assert result["surface"] == WEAK_HOLDOUT_V2
    assert result["strict_all_12_targets"] is True


def test_report_states_training_exclusion_and_independence_caveat():
    soft, predictions, weights, _ = _surface(n=120)
    text = format_weak_report(
        evaluate_on_weak_surface(soft, predictions, weights, n_bootstrap=50)
    )
    assert "v2 holdout" in text
    assert "58-study gold" in text
    assert "independent signal" in text
    assert "all 12 targets" in text


# --- Resolution planning --------------------------------------------------


def test_rough_resolution_uses_actual_holdout_size_not_all_active_studies():
    rough = rough_resolution_estimate(624)
    assert 0.03 < rough["rough_ci_width_from_study_count_only"] < 0.04
    assert "empirical strict" in rough["warning"]


def test_resolution_rejects_empty_surface():
    with pytest.raises(ValueError):
        rough_resolution_estimate(0)
