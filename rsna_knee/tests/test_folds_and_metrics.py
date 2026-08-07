"""Tests for fold assignment and the metric implementations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsnaknee.folds import multilabel_stratified_group_kfold  # noqa: E402
from rsnaknee.metrics import (  # noqa: E402
    binary_log_loss,
    macro_auc,
    per_label_auc,
    weighted_auc,
)


def test_groups_never_span_folds() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random((300, 5)) < 0.2).astype(float)
    groups = np.repeat(np.arange(100), 3)  # three exams per patient

    folds = multilabel_stratified_group_kfold(labels, groups, n_splits=5, seed=1)

    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1


def test_rare_labels_are_spread_across_folds() -> None:
    """A label with ten positives should not land entirely in one fold."""
    labels = np.zeros((500, 3))
    labels[:10, 0] = 1          # rare
    labels[:250, 1] = 1         # common
    rng = np.random.default_rng(2)
    labels[:, 2] = (rng.random(500) < 0.5).astype(float)

    folds = multilabel_stratified_group_kfold(labels, None, n_splits=5, seed=3)

    positives_per_fold = [labels[folds == f, 0].sum() for f in range(5)]
    assert min(positives_per_fold) >= 1
    assert max(positives_per_fold) <= 4


def test_fold_sizes_are_balanced() -> None:
    rng = np.random.default_rng(4)
    labels = (rng.random((1000, 4)) < 0.3).astype(float)

    folds = multilabel_stratified_group_kfold(labels, None, n_splits=5, seed=5)

    counts = np.bincount(folds, minlength=5)
    assert counts.min() >= 150 and counts.max() <= 250


def test_auc_of_perfect_and_inverted_predictions() -> None:
    y_true = np.array([[0], [0], [1], [1]], dtype=float)
    assert macro_auc(y_true, np.array([[0.1], [0.2], [0.8], [0.9]])) == pytest.approx(1.0)
    assert macro_auc(y_true, np.array([[0.9], [0.8], [0.2], [0.1]])) == pytest.approx(0.0)


def test_auc_handles_ties() -> None:
    """All-equal scores must give exactly 0.5, not an arbitrary value."""
    y_true = np.array([[0], [1], [0], [1]], dtype=float)
    y_score = np.full((4, 1), 0.5)
    assert macro_auc(y_true, y_score) == pytest.approx(0.5)


def test_single_class_label_is_ignored_not_scored_as_half() -> None:
    y_true = np.array([[0, 0], [0, 1], [0, 1], [0, 0]], dtype=float)
    y_score = np.array([[0.2, 0.1], [0.3, 0.9], [0.4, 0.8], [0.5, 0.2]])

    scores = per_label_auc(y_true, y_score)

    assert np.isnan(scores[0])            # no positives at all
    assert scores[1] == pytest.approx(1.0)
    assert macro_auc(y_true, y_score) == pytest.approx(1.0)


def test_auc_matches_sklearn() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    y_true = (rng.random((200, 3)) < 0.3).astype(float)
    y_score = rng.random((200, 3))

    ours = per_label_auc(y_true, y_score)
    theirs = [
        sklearn_metrics.roc_auc_score(y_true[:, i], y_score[:, i]) for i in range(3)
    ]

    assert ours == pytest.approx(theirs, abs=1e-9)


def test_weighted_auc_respects_weights() -> None:
    y_true = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y_score = np.array([[0.1, 0.9], [0.9, 0.8], [0.2, 0.7], [0.8, 0.1]])

    unweighted = weighted_auc(y_true, y_score)
    heavy_on_first = weighted_auc(y_true, y_score, np.array([10.0, 1.0]))

    assert heavy_on_first > unweighted


def test_log_loss_is_lower_for_confident_correct_predictions() -> None:
    y_true = np.array([[1], [0]], dtype=float)
    assert binary_log_loss(y_true, np.array([[0.99], [0.01]])) < binary_log_loss(
        y_true, np.array([[0.6], [0.4]])
    )
