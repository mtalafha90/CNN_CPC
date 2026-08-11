"""A high-power secondary validation surface built from weak labels.

The 58 gold studies give a 95% CI of roughly +/-0.06 on macro AUC. Seventeen
sequential decisions have now been taken on that surface, and every remaining
candidate change is smaller than its resolution. Structure cannot be chosen
empirically on 58 studies: the comparison returns noise.

The B6 export carries confident labels for ~14,000 cells across 3,120 studies.
Holding a slice of that corpus out of training gives a validation surface two
orders of magnitude larger, and the interval shrinks roughly as 1/sqrt(n) — from
about +/-0.06 to about +/-0.015.

**What this surface does and does not measure.** It measures agreement with the
report teacher, not with truth. The teacher's own gold-audited specificity is
0.606, so the absolute number here is biased and is not comparable to a gold
score or a leaderboard score. What it does support is *ranking*: if structure A
beats structure B by a margin this surface can resolve, A is genuinely better at
the task the teacher defines, and that is the task the model is trained on.

The intended protocol is therefore two-stage:

    weak holdout   ->  rank many candidate structures     (high power, biased)
    58 gold        ->  confirm the single chosen winner    (low power, unbiased)

This keeps the gold surface for confirmation instead of spending its limited
resolution on search.
"""

from __future__ import annotations

import numpy as np

from .constants import TARGETS
from .evaluation import bootstrap_macro_auc, macro_auc_from_arrays


def make_weak_holdout(
    study_uids,
    report_groups=None,
    holdout_fraction: float = 0.2,
    seed: int = 2026,
) -> np.ndarray:
    """Split the weak corpus into training and validation study sets.

    Returns a boolean mask that is ``True`` for holdout studies.

    Grouping matters as much here as on the gold surface: duplicate normalised
    reports must not straddle the split, or the teacher labels leak and the
    surface flatters any model that memorises report-correlated appearance.
    """
    uids = np.asarray([str(u) for u in study_uids])
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0,1)")

    groups = np.asarray([str(g) for g in report_groups]) if report_groups is not None else uids
    if groups.shape != uids.shape:
        raise ValueError("report_groups must align with study_uids")

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_holdout = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout_groups = set(shuffled[:n_holdout].tolist())

    return np.array([g in holdout_groups for g in groups], dtype=bool)


def weak_macro_auc(
    weak_targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    positive_threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Macro AUC against confident weak labels only.

    Cells with zero weight are ``uncertain`` or ``unmentioned`` — they carry no
    label and are excluded rather than being read as negatives. The soft targets
    (0.85 / 0.05) are thresholded back to binary because AUC needs a hard class.
    """
    weak_targets = np.asarray(weak_targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (weak_targets.shape == predictions.shape == weights.shape):
        raise ValueError("weak targets, predictions and weights must share a shape")

    binary = np.where(weights > 0, (weak_targets >= positive_threshold).astype(float), np.nan)
    return macro_auc_from_arrays(binary, predictions)


def evaluate_on_weak_surface(
    weak_targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
    positive_threshold: float = 0.5,
) -> dict:
    """Score predictions on the weak surface, with an interval and cell counts."""
    weak_targets = np.asarray(weak_targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    binary = np.where(weights > 0, (weak_targets >= positive_threshold).astype(float), np.nan)

    result = bootstrap_macro_auc(binary, predictions, n_bootstrap=n_bootstrap, seed=seed)
    payload = result.to_dict()
    payload.update(
        {
            "surface": "weak_b6_holdout",
            "measures": "agreement with the report teacher, not with expert truth",
            "labelled_cells": int((weights > 0).sum()),
            "positive_cells": int(((weights > 0) & (weak_targets >= positive_threshold)).sum()),
            "negative_cells": int(((weights > 0) & (weak_targets < positive_threshold)).sum()),
            "cells_per_target": {
                target: int((weights[:, j] > 0).sum())
                for j, target in enumerate(TARGETS[: weights.shape[1]])
            },
        }
    )
    return payload


def resolution_estimate(n_studies: int, reference_n: int = 58, reference_width: float = 0.115) -> dict:
    """Estimate the interval width a surface of this size supports.

    Bootstrap width scales roughly as 1/sqrt(n), so this gives an honest sense of
    what a comparison on this surface can and cannot distinguish before spending
    a training run on it.
    """
    if n_studies < 1:
        raise ValueError("n_studies must be positive")
    width = reference_width * np.sqrt(reference_n / n_studies)
    return {
        "n_studies": int(n_studies),
        "estimated_ci_width": float(width),
        "smallest_resolvable_difference": float(width),
        "versus_gold_58": float(reference_width / width) if width > 0 else float("inf"),
    }


def format_weak_report(payload: dict) -> str:
    """Render a weak-surface result with its caveat attached, not buried."""
    lines = [
        f"weak-surface macro AUC {payload['macro_auc']:.4f} "
        f"[{payload['ci_lower']:.4f}, {payload['ci_upper']:.4f}]",
        f"  studies {payload['n_studies']}, labelled cells {payload['labelled_cells']} "
        f"({payload['positive_cells']} positive / {payload['negative_cells']} negative)",
        "",
        "This measures agreement with the B6 report teacher, whose gold-audited",
        "specificity is 0.606. Use it to RANK structures, never as an estimate of",
        "gold or hidden-test performance. Confirm the winner on the 58 gold studies.",
    ]
    weakest = sorted(
        ((k, v) for k, v in payload["per_target_auc"].items() if np.isfinite(v)),
        key=lambda kv: kv[1],
    )[:4]
    if weakest:
        lines += ["", "weakest targets: " + ", ".join(f"{k}={v:.3f}" for k, v in weakest)]
    return "\n".join(lines)
