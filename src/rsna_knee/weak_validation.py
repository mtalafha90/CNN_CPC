"""Leakage-safe weak-label validation for pre-B15 model ranking.

The weak surface measures agreement with frozen B6 v1.2.1 report labels, not
expert truth. It is therefore a biased development/ranking aid only.

Two split generations are distinguished explicitly:

- v1: one random report-group-safe 20% split. It is retained only as historical
  context and is superseded for model selection because the realised Synovitis
  holdout had 70 positive cells and only 1 negative cell.
- v2: deterministic search over report-group-safe candidate splits, using only
  frozen B6 labels to balance all 24 target/class cell counts while enforcing a
  rare-class floor where globally feasible. v2 must be frozen before any B15 or
  matched B13-control training.

Every model scored on v2 must exclude every v2 holdout StudyInstanceUID from
training. Existing B13/B14 checkpoints were trained on all 3,120 active B6
studies and are not valid weak-holdout models.

Weak-surface bootstrap is strict: a replicate is usable only when all 12 target
AUCs are defined, so the estimand never silently changes from a 12-target macro
to an 11-target (or smaller) macro.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    prepare_b7_supervision,
)
from .constants import TARGETS
from .data import add_report_groups, load_train_csv
from .evaluation import macro_auc_from_arrays

WEAK_HOLDOUT_V2 = "weak_b6_holdout_v2"
WEAK_HOLDOUT_V1 = "weak_b6_holdout_v1"
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_MIN_CLASS_COUNT = 4
DEFAULT_SEARCH_CANDIDATES = 4096
DEFAULT_SEED = 2026


def make_weak_holdout(
    study_uids,
    report_groups,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Historical v1 random report-group-safe split.

    This helper is retained for reproducibility/tests only. New model selection
    must use :func:`make_stratified_weak_holdout` through ``freeze_weak_holdout``.
    """
    uids = np.asarray([str(u) for u in study_uids])
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0,1)")
    if report_groups is None:
        raise ValueError("report_groups is required for leakage-safe weak holdout")
    groups = np.asarray([str(g) for g in report_groups])
    if groups.shape != uids.shape:
        raise ValueError("report_groups must align with study_uids")
    if len(uids) < 2:
        raise ValueError("weak holdout requires at least two studies")

    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 2:
        raise ValueError("weak holdout requires at least two report groups")

    target_n = max(1, int(round(len(uids) * holdout_fraction)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    selected: set[str] = set()
    selected_n = 0
    size_by_group = {str(g): int(c) for g, c in zip(unique, counts)}
    for index in order.tolist():
        group = str(unique[index])
        if selected_n >= target_n:
            break
        selected.add(group)
        selected_n += size_by_group[group]

    mask = np.asarray([group in selected for group in groups], dtype=bool)
    if mask.all() or (~mask).all():
        raise ValueError("weak holdout split is degenerate")
    return mask


def _binary_weak_targets(
    weak_targets: np.ndarray,
    weights: np.ndarray,
    *,
    positive_threshold: float = 0.5,
) -> np.ndarray:
    weak_targets = np.asarray(weak_targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weak_targets.shape != weights.shape or weak_targets.ndim != 2:
        raise ValueError("weak targets and weights must share a 2D shape")
    return np.where(
        weights > 0,
        (weak_targets >= positive_threshold).astype(float),
        np.nan,
    )


def _class_features(binary: np.ndarray) -> np.ndarray:
    """Return per-study [target-positive, target-negative] indicator counts."""
    binary = np.asarray(binary, dtype=np.float64)
    if binary.ndim != 2:
        raise ValueError("binary weak targets must be 2D")
    pos = np.isfinite(binary) & (binary == 1)
    neg = np.isfinite(binary) & (binary == 0)
    return np.concatenate([pos.astype(np.int64), neg.astype(np.int64)], axis=1)


def _class_names(n_targets: int) -> list[str]:
    names = TARGETS[:n_targets]
    return [f"{name}:positive" for name in names] + [f"{name}:negative" for name in names]


def _aggregate_groups(groups: np.ndarray, features: np.ndarray):
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(unique)).astype(np.int64)
    counts = np.zeros((len(unique), features.shape[1]), dtype=np.int64)
    np.add.at(counts, inverse, features)
    return unique, inverse, sizes, counts


def _desired_holdout_counts(
    global_counts: np.ndarray,
    holdout_fraction: float,
    min_class_count: int,
) -> np.ndarray:
    desired = np.asarray(global_counts, dtype=float) * float(holdout_fraction)
    floor = int(min_class_count)
    if floor > 0:
        feasible = np.asarray(global_counts, dtype=int) >= 2 * floor
        desired[feasible] = np.clip(
            desired[feasible],
            floor,
            np.asarray(global_counts, dtype=float)[feasible] - floor,
        )
    return desired


def _candidate_is_feasible(
    holdout_counts: np.ndarray,
    global_counts: np.ndarray,
    min_class_count: int,
) -> bool:
    holdout_counts = np.asarray(holdout_counts, dtype=int)
    global_counts = np.asarray(global_counts, dtype=int)
    train_counts = global_counts - holdout_counts
    floor = int(min_class_count)
    for h, t, total in zip(holdout_counts.tolist(), train_counts.tolist(), global_counts.tolist()):
        if total <= 0:
            return False
        required = floor if total >= 2 * floor else 1
        if h < required or t < required:
            return False
    return True


def _candidate_score(
    holdout_counts: np.ndarray,
    desired_counts: np.ndarray,
    holdout_n: int,
    target_n: int,
) -> float:
    desired = np.maximum(np.asarray(desired_counts, dtype=float), 1.0)
    rel = (np.asarray(holdout_counts, dtype=float) - desired_counts) / desired
    class_mse = float(np.mean(rel**2))
    class_max = float(np.max(np.abs(rel)))
    size_rel = float((int(holdout_n) - int(target_n)) / max(int(target_n), 1))
    return class_mse + 0.25 * class_max**2 + 0.50 * size_rel**2


def make_stratified_weak_holdout(
    study_uids,
    report_groups,
    weak_targets: np.ndarray,
    weights: np.ndarray,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    n_candidates: int = DEFAULT_SEARCH_CANDIDATES,
    positive_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Choose a deterministic report-group-safe multilabel/class-balanced split.

    Candidate splits are generated only from report groups and frozen B6 labels;
    no gold labels, model predictions, or model performance enter the search.
    Among feasible candidates, the selected split minimizes deviation from the
    requested holdout size and from the requested fraction of every positive and
    negative target class. A minimum class count is enforced in both holdout and
    weak-train partitions whenever the global class count makes that possible.
    """
    uids = np.asarray([str(u) for u in study_uids])
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0,1)")
    if report_groups is None:
        raise ValueError("report_groups is required for leakage-safe weak holdout")
    groups = np.asarray([str(g) for g in report_groups])
    if groups.shape != uids.shape:
        raise ValueError("report_groups must align with study_uids")
    if int(min_class_count) < 1:
        raise ValueError("min_class_count must be >=1")
    if int(n_candidates) < 1:
        raise ValueError("n_candidates must be >=1")

    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    if binary.shape[0] != len(uids):
        raise ValueError("weak targets/weights must align with study_uids")
    features = _class_features(binary)
    global_counts = features.sum(axis=0).astype(np.int64)
    if np.any(global_counts <= 1):
        bad = [
            name for name, value in zip(_class_names(binary.shape[1]), global_counts.tolist())
            if value <= 1
        ]
        raise ValueError("cannot split weak class on both sides: " + ", ".join(bad))

    unique_groups, inverse, group_sizes, group_counts = _aggregate_groups(groups, features)
    if len(unique_groups) < 2:
        raise ValueError("weak holdout requires at least two report groups")

    target_n = max(1, int(round(len(uids) * holdout_fraction)))
    desired = _desired_holdout_counts(global_counts, holdout_fraction, min_class_count)
    rng = np.random.default_rng(seed)

    best = None
    feasible_candidates = 0
    for attempt in range(int(n_candidates)):
        order = rng.permutation(len(unique_groups))
        cumulative = np.cumsum(group_sizes[order])
        crossing = int(np.searchsorted(cumulative, target_n, side="left"))
        cut_options = {max(0, min(crossing, len(order) - 1))}
        if crossing > 0:
            cut_options.add(crossing - 1)

        for cutoff in sorted(cut_options):
            selected_idx = order[: cutoff + 1]
            holdout_n = int(group_sizes[selected_idx].sum())
            if holdout_n <= 0 or holdout_n >= len(uids):
                continue
            holdout_counts = group_counts[selected_idx].sum(axis=0).astype(np.int64)
            if not _candidate_is_feasible(holdout_counts, global_counts, min_class_count):
                continue
            feasible_candidates += 1
            score = _candidate_score(holdout_counts, desired, holdout_n, target_n)
            tie = (
                score,
                abs(holdout_n - target_n),
                attempt,
                cutoff,
            )
            if best is None or tie < best[0]:
                best = (tie, selected_idx.copy(), holdout_counts.copy(), holdout_n)

    if best is None:
        raise ValueError(
            "no feasible stratified weak holdout found with the frozen search policy; "
            "increase --search-candidates before any model training, without consulting gold/model performance"
        )

    _, selected_idx, holdout_counts, holdout_n = best
    selected_groups = set(unique_groups[selected_idx].astype(str).tolist())
    mask = np.asarray([group in selected_groups for group in groups], dtype=bool)
    if int(mask.sum()) != int(holdout_n):
        raise RuntimeError("internal weak-holdout study count mismatch")
    if mask.all() or (~mask).all():
        raise RuntimeError("stratified weak holdout is degenerate")

    train_counts = global_counts - holdout_counts
    diagnostics = {
        "strategy": "report_group_multilabel_class_random_search_v2",
        "search_candidates": int(n_candidates),
        "feasible_candidates": int(feasible_candidates),
        "min_class_count_requested": int(min_class_count),
        "target_holdout_studies": int(target_n),
        "selected_holdout_studies": int(holdout_n),
        "class_balance_score": float(best[0][0]),
        "class_names": _class_names(binary.shape[1]),
        "global_class_counts": [int(x) for x in global_counts.tolist()],
        "desired_holdout_class_counts": [float(x) for x in desired.tolist()],
        "selected_holdout_class_counts": [int(x) for x in holdout_counts.tolist()],
        "selected_train_class_counts": [int(x) for x in train_counts.tolist()],
    }
    return mask, diagnostics


def _strict_macro_auc_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return NaN macro unless every target AUC is defined."""
    point, per_target = macro_auc_from_arrays(y_true, y_score)
    if len(per_target) == 0 or not np.isfinite(per_target).all():
        return float("nan"), per_target
    return float(np.mean(per_target)), per_target


def _strict_bootstrap_macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("strict weak bootstrap requires equal 2D arrays")
    if y_true.shape[1] != len(TARGETS):
        raise ValueError("strict weak macro requires all 12 targets")
    point, per_target = _strict_macro_auc_from_arrays(y_true, y_score)
    if not np.isfinite(point):
        raise ValueError("full weak holdout does not define AUC for all 12 targets")

    rng = np.random.default_rng(seed)
    replicates = np.empty(int(n_bootstrap), dtype=float)
    replicates.fill(np.nan)
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        replicates[b], _ = _strict_macro_auc_from_arrays(y_true[idx], y_score[idx])
    valid = replicates[np.isfinite(replicates)]
    if not len(valid):
        raise ValueError("strict weak bootstrap produced zero all-12-target replicates")
    tail = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.percentile(valid, [100 * tail, 100 * (1.0 - tail)])
    return {
        "macro_auc": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence_level),
        "n_studies": int(len(y_true)),
        "n_bootstrap": int(n_bootstrap),
        "n_valid_replicates": int(len(valid)),
        "valid_replicate_fraction": float(len(valid) / int(n_bootstrap)),
        "strict_all_12_targets": True,
        "per_target_auc": {
            name: float(value) for name, value in zip(TARGETS, per_target)
        },
        "per_target_defined": {
            name: bool(np.isfinite(value)) for name, value in zip(TARGETS, per_target)
        },
    }


def weak_macro_auc(
    weak_targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    positive_threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Strict 12-target macro AUC against labelled B6 cells only."""
    predictions = np.asarray(predictions, dtype=np.float64)
    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    if predictions.shape != binary.shape:
        raise ValueError("weak targets, predictions and weights must share a shape")
    return _strict_macro_auc_from_arrays(binary, predictions)


def evaluate_on_weak_surface(
    weak_targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = DEFAULT_SEED,
    positive_threshold: float = 0.5,
) -> dict:
    """Score one model with a strict all-12-target study bootstrap."""
    predictions = np.asarray(predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    if predictions.shape != binary.shape:
        raise ValueError("weak targets, predictions and weights must share a shape")

    payload = _strict_bootstrap_macro_auc(
        binary,
        predictions,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )
    payload.update(
        {
            "surface": WEAK_HOLDOUT_V2,
            "measures": "agreement with the B6 report teacher, not expert truth",
            "labelled_cells": int((weights > 0).sum()),
            "positive_cells": int(((weights > 0) & (np.asarray(weak_targets) >= positive_threshold)).sum()),
            "negative_cells": int(((weights > 0) & (np.asarray(weak_targets) < positive_threshold)).sum()),
            "cells_per_target": {
                target: int((weights[:, j] > 0).sum())
                for j, target in enumerate(TARGETS)
            },
            "selection_policy": (
                "ranking/development only; evaluated checkpoints must exclude these "
                "holdout studies from training"
            ),
        }
    )
    return payload


def compare_on_weak_surface(
    weak_targets: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    weights: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = DEFAULT_SEED,
    positive_threshold: float = 0.5,
) -> dict:
    """Strict aligned B-minus-A comparison on the same sparse weak holdout."""
    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    predictions_a = np.asarray(predictions_a, dtype=np.float64)
    predictions_b = np.asarray(predictions_b, dtype=np.float64)
    if binary.shape != predictions_a.shape or binary.shape != predictions_b.shape:
        raise ValueError("paired weak comparison arrays must share a shape")
    if binary.shape[1] != len(TARGETS):
        raise ValueError("strict weak comparison requires all 12 targets")

    point_a, _ = _strict_macro_auc_from_arrays(binary, predictions_a)
    point_b, _ = _strict_macro_auc_from_arrays(binary, predictions_b)
    if not np.isfinite(point_a) or not np.isfinite(point_b):
        raise ValueError("full weak holdout must define all 12 target AUCs for both models")

    rng = np.random.default_rng(seed)
    differences = np.empty(int(n_bootstrap), dtype=float)
    differences.fill(np.nan)
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, len(binary), size=len(binary))
        score_a, _ = _strict_macro_auc_from_arrays(binary[idx], predictions_a[idx])
        score_b, _ = _strict_macro_auc_from_arrays(binary[idx], predictions_b[idx])
        if np.isfinite(score_a) and np.isfinite(score_b):
            differences[b] = score_b - score_a
    valid = differences[np.isfinite(differences)]
    if not len(valid):
        raise ValueError("paired weak bootstrap produced zero all-12-target replicates")

    return {
        "surface": WEAK_HOLDOUT_V2,
        "macro_auc_a": float(point_a),
        "macro_auc_b": float(point_b),
        "raw_difference_b_minus_a": float(point_b - point_a),
        "median_difference": float(np.median(valid)),
        "ci_lower": float(np.percentile(valid, 2.5)),
        "ci_upper": float(np.percentile(valid, 97.5)),
        "probability_b_better": float((valid > 0).mean()),
        "n_bootstrap": int(n_bootstrap),
        "n_valid_replicates": int(len(valid)),
        "valid_replicate_fraction": float(len(valid) / int(n_bootstrap)),
        "strict_all_12_targets": True,
        "measures": "paired agreement with the B6 report teacher, not expert truth",
    }


def rough_resolution_estimate(
    n_studies: int,
    reference_n: int = 58,
    reference_width: float = 0.115,
) -> dict:
    """Study-count-only planning heuristic; empirical strict bootstrap is authoritative."""
    if n_studies < 1:
        raise ValueError("n_studies must be positive")
    width = reference_width * np.sqrt(reference_n / n_studies)
    return {
        "n_studies": int(n_studies),
        "rough_ci_width_from_study_count_only": float(width),
        "warning": (
            "B6 is sparse and class-imbalanced; use the empirical strict all-12-target "
            "bootstrap on the frozen holdout for model-selection decisions"
        ),
    }


def _manifest_sha256(manifest: pd.DataFrame) -> str:
    rows = manifest[["StudyInstanceUID", "report_group", "split"]].sort_values(
        "StudyInstanceUID"
    )
    text = rows.to_json(orient="records", force_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_weak_holdout(
    config: dict,
    *,
    b6_root: str | Path,
    out_root: str | Path = "runs/weak_holdout_v2",
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    n_candidates: int = DEFAULT_SEARCH_CANDIDATES,
) -> dict:
    """Freeze v2 before any B15 or matched B13-control training."""
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, y, w, supervision = prepare_b7_supervision(train, b6_frame)
    if len(uids) != 3120 or int(supervision.get("usable_cells", -1)) != 14123:
        raise ValueError("weak holdout must be frozen on the exact retained B6 surface")

    grouped = add_report_groups(train[["StudyInstanceUID", "Report"]])
    group_map = dict(
        zip(grouped["StudyInstanceUID"].astype(str), grouped["report_group"].astype(str))
    )
    report_groups = [group_map[str(uid)] for uid in uids]
    holdout, split_diagnostics = make_stratified_weak_holdout(
        uids,
        report_groups,
        y,
        w,
        holdout_fraction=holdout_fraction,
        seed=seed,
        min_class_count=min_class_count,
        n_candidates=n_candidates,
    )

    split = np.where(holdout, "holdout", "train")
    manifest = pd.DataFrame(
        {
            "StudyInstanceUID": [str(uid) for uid in uids],
            "report_group": report_groups,
            "split": split,
            "labelled_cells": (w > 0).sum(axis=1).astype(int),
            "positive_cells": ((w > 0) & (y >= 0.5)).sum(axis=1).astype(int),
            "negative_cells": ((w > 0) & (y < 0.5)).sum(axis=1).astype(int),
        }
    )

    train_groups = set(manifest.loc[~holdout, "report_group"])
    holdout_groups = set(manifest.loc[holdout, "report_group"])
    overlap = train_groups.intersection(holdout_groups)
    if overlap:
        raise ValueError(f"report-group leakage across weak split: {len(overlap)} group(s)")

    per_target: dict[str, dict] = {}
    floor_failures: list[str] = []
    for j, target in enumerate(TARGETS):
        labelled_all = w[:, j] > 0
        pos_all = int((labelled_all & (y[:, j] >= 0.5)).sum())
        neg_all = int((labelled_all & (y[:, j] < 0.5)).sum())
        labelled = holdout & labelled_all
        positives = int((labelled & (y[:, j] >= 0.5)).sum())
        negatives = int((labelled & (y[:, j] < 0.5)).sum())
        train_positives = pos_all - positives
        train_negatives = neg_all - negatives
        required_pos = int(min_class_count) if pos_all >= 2 * int(min_class_count) else 1
        required_neg = int(min_class_count) if neg_all >= 2 * int(min_class_count) else 1
        if positives < required_pos or train_positives < required_pos:
            floor_failures.append(f"{target}:positive")
        if negatives < required_neg or train_negatives < required_neg:
            floor_failures.append(f"{target}:negative")
        per_target[target] = {
            "global_positive_cells": pos_all,
            "global_negative_cells": neg_all,
            "holdout_labelled_cells": int(labelled.sum()),
            "holdout_positive_cells": positives,
            "holdout_negative_cells": negatives,
            "train_positive_cells": train_positives,
            "train_negative_cells": train_negatives,
        }
    if floor_failures:
        raise RuntimeError("v2 class-floor contract failed for: " + ", ".join(floor_failures))

    payload = {
        "surface": WEAK_HOLDOUT_V2,
        "status": "FROZEN before B15/control training",
        "supersedes": WEAK_HOLDOUT_V1,
        "v1_rejection_reason": (
            "historical v1 realised Synovitis 70 positive / 1 negative in holdout; "
            "insufficient for stable 12-target bootstrap"
        ),
        "b6_version": b6_audit.get("b6_version"),
        "seed": int(seed),
        "requested_holdout_fraction": float(holdout_fraction),
        "active_studies": int(len(uids)),
        "train_studies": int((~holdout).sum()),
        "holdout_studies": int(holdout.sum()),
        "actual_holdout_fraction": float(holdout.mean()),
        "train_report_groups": int(len(train_groups)),
        "holdout_report_groups": int(len(holdout_groups)),
        "report_group_overlap": 0,
        "all_usable_cells": int((w > 0).sum()),
        "holdout_usable_cells": int((w[holdout] > 0).sum()),
        "holdout_positive_cells": int(((w[holdout] > 0) & (y[holdout] >= 0.5)).sum()),
        "holdout_negative_cells": int(((w[holdout] > 0) & (y[holdout] < 0.5)).sum()),
        "per_target": per_target,
        "split_diagnostics": split_diagnostics,
        "manifest_sha256": _manifest_sha256(manifest),
        "gold_studies_in_surface": 0,
        "uses_gold_labels": False,
        "uses_model_predictions": False,
        "measurement": "teacher agreement only; not expert truth",
        "bootstrap_contract": (
            "study bootstrap; replicate usable only when all 12 target AUCs are defined"
        ),
        "training_contract": (
            "every model scored on this surface must be trained with all v2 holdout "
            "StudyInstanceUID values excluded"
        ),
        "rough_resolution": rough_resolution_estimate(int(holdout.sum())),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "weak_holdout_manifest.csv", index=False)
    (out / "weak_holdout.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def format_weak_report(payload: dict) -> str:
    """Render a weak-surface result with strict-bootstrap caveats attached."""
    lines = [
        f"weak-surface macro AUC {payload['macro_auc']:.4f} "
        f"[{payload['ci_lower']:.4f}, {payload['ci_upper']:.4f}]",
        f"  studies {payload['n_studies']}, labelled cells {payload['labelled_cells']} "
        f"({payload['positive_cells']} positive / {payload['negative_cells']} negative)",
        f"  strict bootstrap {payload['n_valid_replicates']}/{payload['n_bootstrap']} "
        "replicates define all 12 targets",
        "",
        "This measures agreement with the B6 report teacher, not expert truth.",
        "Use it only to rank models trained with every v2 holdout study excluded.",
        "The reused 58-study gold surface is development confirmation only;",
        "hidden competition evaluation remains the independent signal.",
    ]
    weakest = sorted(
        ((k, v) for k, v in payload["per_target_auc"].items() if np.isfinite(v)),
        key=lambda kv: kv[1],
    )[:4]
    if weakest:
        lines += ["", "weakest targets: " + ", ".join(f"{k}={v:.3f}" for k, v in weakest)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the stratified report-group-safe B6 weak holdout v2"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-class-count", type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument("--search-candidates", type=int, default=DEFAULT_SEARCH_CANDIDATES)
    parser.add_argument("--out-root", default="runs/weak_holdout_v2")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    payload = freeze_weak_holdout(
        config,
        b6_root=args.b6_root,
        out_root=args.out_root,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        min_class_count=args.min_class_count,
        n_candidates=args.search_candidates,
    )
    print(json.dumps(payload, indent=2))
    print(Path(args.out_root) / "weak_holdout_manifest.csv")
    print(Path(args.out_root) / "weak_holdout.json")


if __name__ == "__main__":
    main()
