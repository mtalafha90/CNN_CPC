"""Evaluation metrics.

The organisers have not published the exact scoring function in any source
reachable from this environment, so the pipeline reports the family of metrics
that RSNA challenges normally use — per-label AUC, macro mean AUC, and a
weighted variant — and lets you select whichever turns out to be official via
``metric_name`` in the config. Swapping the target metric is then a one-line
change rather than a rewrite.
"""

from __future__ import annotations

import numpy as np

from .utils import get_logger

LOGGER = get_logger()


def _auc_single(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC for one label, computed from ranks so ties are handled correctly.

    Returns ``nan`` when the label has only one class present, which keeps a
    degenerate fold from dragging the mean towards 0.5.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # Average the ranks inside each group of tied scores.
    sorted_scores = y_score[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_label_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """AUC for every label, as an array that may contain ``nan``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    return np.array(
        [_auc_single(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])],
        dtype=np.float64,
    )


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Unweighted mean AUC across labels, ignoring degenerate labels."""
    scores = per_label_auc(y_true, y_score)
    valid = scores[np.isfinite(scores)]
    return float(valid.mean()) if valid.size else float("nan")


def weighted_auc(
    y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Mean AUC with per-label weights, as used by several RSNA challenges."""
    scores = per_label_auc(y_true, y_score)
    if weights is None:
        return macro_auc(y_true, y_score)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(scores)
    if not valid.any():
        return float("nan")
    return float(np.average(scores[valid], weights=weights[valid]))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Mean binary cross entropy, a useful secondary signal for calibration."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    mask = np.isfinite(y_true)
    losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(losses[mask].mean()) if mask.any() else float("nan")


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str] | None = None,
    weights: np.ndarray | None = None,
) -> dict:
    """Compute the full metric report for a set of predictions."""
    scores = per_label_auc(y_true, y_score)
    names = label_names or [f"label_{i}" for i in range(len(scores))]
    report = {
        "macro_auc": macro_auc(y_true, y_score),
        "weighted_auc": weighted_auc(y_true, y_score, weights),
        "log_loss": binary_log_loss(y_true, y_score),
        "per_label_auc": {name: float(score) for name, score in zip(names, scores)},
    }
    return report


def log_report(report: dict, prefix: str = "") -> None:
    """Print a metric report in a readable order, worst labels first."""
    LOGGER.info(
        "%smacro AUC %.5f | weighted AUC %.5f | log loss %.5f",
        f"{prefix} " if prefix else "",
        report["macro_auc"],
        report["weighted_auc"],
        report["log_loss"],
    )
    ordered = sorted(report["per_label_auc"].items(), key=lambda kv: kv[1])
    LOGGER.info("  weakest labels: %s", ", ".join(f"{k}={v:.3f}" for k, v in ordered[:5]))
