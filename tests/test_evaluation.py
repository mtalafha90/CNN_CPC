"""Tests for gold-only AUC evaluation and bootstrap uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.evaluation import (
    bootstrap_macro_auc,
    compare_runs,
    fast_auc,
    load_oof,
    macro_auc_from_arrays,
)


def test_fast_auc_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(80) < 0.3).astype(float)
        if y.sum() in (0, len(y)):
            continue
        p = rng.random(80)
        assert fast_auc(y, p) == pytest.approx(sklearn_metrics.roc_auc_score(y, p), abs=1e-9)


def test_fast_auc_handles_ties():
    assert fast_auc(np.array([0.0, 1, 0, 1]), np.full(4, 0.5)) == pytest.approx(0.5)


def test_fast_auc_undefined_for_single_class():
    assert np.isnan(fast_auc(np.zeros(10), np.random.default_rng(0).random(10)))


def test_perfect_and_inverted_predictions():
    y = np.array([[0.0], [0], [1], [1]])
    assert macro_auc_from_arrays(y, np.array([[0.1], [0.2], [0.8], [0.9]]))[0] == pytest.approx(1.0)
    assert macro_auc_from_arrays(y, np.array([[0.9], [0.8], [0.2], [0.1]]))[0] == pytest.approx(0.0)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random((58, 12)) < 0.3).astype(float)
    p = y * 0.6 + rng.random((58, 12)) * 0.4
    result = bootstrap_macro_auc(y, p, n_bootstrap=300, seed=7)
    assert result.lower <= result.macro_auc <= result.upper
    assert result.n_studies == 58


def _noisy_predictions(y: np.ndarray, rng, signal: float = 0.3) -> np.ndarray:
    return y * signal + rng.random(y.shape) * (1.0 - signal)


def test_interval_is_wide_at_this_sample_size():
    rng = np.random.default_rng(2)
    y = (rng.random((58, 12)) < 0.25).astype(float)
    p = _noisy_predictions(y, rng)
    result = bootstrap_macro_auc(y, p, n_bootstrap=500, seed=3)
    assert result.upper - result.lower > 0.02


def test_more_studies_give_a_tighter_interval():
    rng = np.random.default_rng(4)
    big_y = (rng.random((600, 12)) < 0.3).astype(float)
    big_p = _noisy_predictions(big_y, rng)
    wide = bootstrap_macro_auc(big_y[:58], big_p[:58], n_bootstrap=400, seed=5)
    narrow = bootstrap_macro_auc(big_y, big_p, n_bootstrap=400, seed=5)
    assert (narrow.upper - narrow.lower) < (wide.upper - wide.lower)


def test_single_class_targets_are_reported_not_invented():
    y = np.zeros((20, 12))
    y[:, 0] = (np.arange(20) % 2).astype(float)
    p = np.random.default_rng(6).random((20, 12))
    result = bootstrap_macro_auc(y, p, n_bootstrap=100, seed=1)
    assert result.per_target_defined[TARGETS[0]] is True
    assert result.per_target_defined[TARGETS[1]] is False
    assert np.isnan(result.per_target[TARGETS[1]])
    assert "undefined" in result.summary()


def test_nan_gold_cells_are_ignored():
    rng = np.random.default_rng(8)
    y = (rng.random((40, 12)) < 0.4).astype(float)
    p = y * 0.7 + rng.random((40, 12)) * 0.3
    y[::2, 3] = np.nan
    result = bootstrap_macro_auc(y, p, n_bootstrap=100, seed=2)
    assert np.isfinite(result.macro_auc)


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(9)
    y = (rng.random((50, 12)) < 0.3).astype(float)
    p = rng.random((50, 12))
    first = bootstrap_macro_auc(y, p, n_bootstrap=200, seed=42)
    second = bootstrap_macro_auc(y, p, n_bootstrap=200, seed=42)
    assert first.lower == second.lower and first.upper == second.upper


def test_paired_comparison_detects_a_better_run():
    rng = np.random.default_rng(10)
    y = (rng.random((58, 12)) < 0.3).astype(float)
    weak = rng.random((58, 12))
    strong = y * 0.8 + rng.random((58, 12)) * 0.2
    comparison = compare_runs(y, weak, strong, n_bootstrap=200, seed=11)
    assert comparison["median_difference"] > 0
    assert comparison["probability_b_better"] > 0.9


def test_paired_comparison_of_identical_runs_is_centred_on_zero():
    rng = np.random.default_rng(12)
    y = (rng.random((58, 12)) < 0.3).astype(float)
    p = rng.random((58, 12))
    comparison = compare_runs(y, p, p, n_bootstrap=100, seed=13)
    assert comparison["median_difference"] == pytest.approx(0.0)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        bootstrap_macro_auc(np.empty((0, 12)), np.empty((0, 12)))


def _write_data(tmp_path: Path, n: int = 20):
    rng = np.random.default_rng(14)
    train = pd.DataFrame({"StudyInstanceUID": [f"s{i}" for i in range(n)], "Report": [""] * n})
    for target in TARGETS:
        values = (rng.random(n) < 0.3).astype(float)
        values[n // 2:] = np.nan
        train[target] = values
    train.to_csv(tmp_path / "train.csv", index=False)

    oof = pd.DataFrame({"StudyInstanceUID": [f"s{i}" for i in range(n)]})
    for target in TARGETS:
        oof[target] = rng.random(n)
    oof.to_csv(tmp_path / "oof.csv", index=False)
    return tmp_path / "train.csv", tmp_path / "oof.csv"


def test_load_oof_keeps_only_gold_studies(tmp_path: Path):
    train_csv, oof_csv = _write_data(tmp_path, n=20)
    y_true, y_pred, uids = load_oof(train_csv, [oof_csv])
    assert len(uids) == 10
    assert y_true.shape == y_pred.shape == (10, 12)


def test_load_oof_rejects_studies_repeated_across_files(tmp_path: Path):
    train_csv, oof_csv = _write_data(tmp_path, n=20)
    with pytest.raises(ValueError, match="multiple OOF files"):
        load_oof(train_csv, [oof_csv, oof_csv])


def test_load_oof_rejects_a_mismatched_run(tmp_path: Path):
    train_csv, _ = _write_data(tmp_path, n=20)
    other = pd.DataFrame({"StudyInstanceUID": ["zzz"], **{t: [0.5] for t in TARGETS}})
    other.to_csv(tmp_path / "other.csv", index=False)
    with pytest.raises(ValueError, match="no gold-labelled studies"):
        load_oof(train_csv, [tmp_path / "other.csv"])
