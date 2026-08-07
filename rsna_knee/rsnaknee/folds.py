"""Cross-validation splitting for multi-label data with grouping.

Two constraints apply at once:

* **Grouping** — a patient (or a site, if you prefer a harder validation) must
  never appear in both training and validation, otherwise the score is
  optimistic.
* **Multi-label stratification** — some findings are rare, and a fold that
  happens to contain no positives for a finding makes its AUC undefined and
  the mean score noisy.

``scikit-learn`` has no splitter that does both, so this module implements the
standard greedy assignment: process groups from rarest label content outwards,
and place each group in whichever fold is furthest below its quota.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

LOGGER = get_logger()


def multilabel_stratified_group_kfold(
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Assign every row to a fold.

    Parameters
    ----------
    labels:
        Binary matrix of shape ``[n_samples, n_labels]``. Missing values should
        already be filled with zero.
    groups:
        Group identifier per row. When ``None`` each row is its own group.
    n_splits:
        Number of folds.
    seed:
        Controls the tie-breaking shuffle.

    Returns
    -------
    Integer array of fold indices, one per row.
    """
    labels = np.nan_to_num(np.asarray(labels, dtype=np.float64))
    n_samples, n_labels = labels.shape
    if groups is None:
        groups = np.arange(n_samples)
    groups = np.asarray(groups)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)

    # Per-group label counts and sizes.
    group_counts = np.zeros((n_groups, n_labels), dtype=np.float64)
    group_sizes = np.zeros(n_groups, dtype=np.float64)
    for row in range(n_samples):
        group_counts[inverse[row]] += labels[row]
        group_sizes[inverse[row]] += 1.0

    # Rare labels first: they are the ones that break if we get greedy late.
    label_totals = group_counts.sum(axis=0)
    label_order = np.argsort(label_totals)
    label_weights = np.zeros(n_labels, dtype=np.float64)
    label_weights[label_order] = np.linspace(n_labels, 1.0, n_labels)
    label_weights /= label_weights.sum()

    # Visit the groups carrying the most rare-label mass first, so the rare
    # positives are placed while every fold is still empty enough to take them.
    priority = group_counts @ label_weights
    rng = np.random.default_rng(seed)
    jitter = rng.random(n_groups) * 1e-6
    group_order = np.argsort(-(priority + jitter))

    fold_counts = np.zeros((n_splits, n_labels), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    target_counts = np.maximum(label_totals / n_splits, 1.0)
    target_size = max(group_sizes.sum() / n_splits, 1.0)

    group_fold = np.zeros(n_groups, dtype=np.int64)
    for group in group_order:
        # Greedy load balancing: place the group where it raises the squared
        # load least. Squaring is what makes this balance — the emptiest fold
        # always has the lowest marginal cost, so folds fill evenly instead of
        # one filling first. Loads are divided by their target so a label with
        # ten positives counts as much as one with a thousand.
        load = (fold_counts + group_counts[group]) / target_counts
        label_cost = (load**2) @ label_weights
        size_cost = ((fold_sizes + group_sizes[group]) / target_size) ** 2
        fold = int(np.argmin(label_cost + 0.5 * size_cost))
        group_fold[group] = fold
        fold_counts[fold] += group_counts[group]
        fold_sizes[fold] += group_sizes[group]

    return group_fold[inverse]


def add_folds(
    frame: pd.DataFrame,
    label_columns: list[str],
    group_column: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a fold column added."""
    frame = frame.copy()
    labels = frame[label_columns].to_numpy(dtype=np.float64)
    groups = frame[group_column].to_numpy() if group_column and group_column in frame else None
    frame[fold_column] = multilabel_stratified_group_kfold(
        labels, groups, n_splits=n_splits, seed=seed
    )

    positives = frame.groupby(fold_column)[label_columns].sum()
    empty = (positives == 0).sum().sum()
    if empty:
        LOGGER.warning(
            "%d label/fold combinations have no positives; consider fewer folds", int(empty)
        )
    LOGGER.info("Fold sizes: %s", frame[fold_column].value_counts().sort_index().to_dict())
    return frame
