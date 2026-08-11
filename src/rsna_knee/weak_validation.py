"""Leakage-safe secondary validation utilities built from frozen B6 weak labels.

This surface measures agreement with the B6 report teacher, not expert truth. It
is therefore a biased development/ranking aid only. The split must be frozen
*before* any candidate model is trained, and every model compared on it must be
trained without the holdout studies.

The canonical workflow is:

    freeze report-group-safe weak holdout
    -> train B13-control and candidate on the same weak-train partition
    -> compare them with an aligned bootstrap on the weak holdout
    -> use the repeatedly reused 58-study gold surface only for one development
       confirmation, not as independent validation
    -> use the hidden competition test/leaderboard as the independent signal
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
from .evaluation import bootstrap_macro_auc, compare_runs, macro_auc_from_arrays


def make_weak_holdout(
    study_uids,
    report_groups,
    holdout_fraction: float = 0.2,
    seed: int = 2026,
) -> np.ndarray:
    """Return a deterministic report-group-safe holdout mask.

    ``report_groups`` is mandatory. Falling back to study UID groups would make
    the no-duplicate-report guarantee silently disappear.
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


def weak_macro_auc(
    weak_targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    positive_threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Macro AUC against labelled B6 cells only; zero-weight cells stay missing."""
    predictions = np.asarray(predictions, dtype=np.float64)
    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    if predictions.shape != binary.shape:
        raise ValueError("weak targets, predictions and weights must share a shape")
    return macro_auc_from_arrays(binary, predictions)


def evaluate_on_weak_surface(
    weak_targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
    positive_threshold: float = 0.5,
) -> dict:
    """Score one model on the actual sparse weak holdout with empirical bootstrap."""
    predictions = np.asarray(predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    if predictions.shape != binary.shape:
        raise ValueError("weak targets, predictions and weights must share a shape")

    result = bootstrap_macro_auc(binary, predictions, n_bootstrap=n_bootstrap, seed=seed)
    payload = result.to_dict()
    payload.update(
        {
            "surface": "weak_b6_holdout",
            "measures": "agreement with the B6 report teacher, not expert truth",
            "labelled_cells": int((weights > 0).sum()),
            "positive_cells": int(((weights > 0) & (np.asarray(weak_targets) >= positive_threshold)).sum()),
            "negative_cells": int(((weights > 0) & (np.asarray(weak_targets) < positive_threshold)).sum()),
            "cells_per_target": {
                target: int((weights[:, j] > 0).sum())
                for j, target in enumerate(TARGETS[: weights.shape[1]])
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
    seed: int = 2026,
    positive_threshold: float = 0.5,
) -> dict:
    """Aligned model-B minus model-A comparison on the same sparse weak holdout."""
    binary = _binary_weak_targets(
        weak_targets,
        weights,
        positive_threshold=positive_threshold,
    )
    predictions_a = np.asarray(predictions_a, dtype=np.float64)
    predictions_b = np.asarray(predictions_b, dtype=np.float64)
    if binary.shape != predictions_a.shape or binary.shape != predictions_b.shape:
        raise ValueError("paired weak comparison arrays must share a shape")

    point_a, _ = macro_auc_from_arrays(binary, predictions_a)
    point_b, _ = macro_auc_from_arrays(binary, predictions_b)
    payload = compare_runs(
        binary,
        predictions_a,
        predictions_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    payload.update(
        {
            "surface": "weak_b6_holdout",
            "macro_auc_a": float(point_a),
            "macro_auc_b": float(point_b),
            "raw_difference_b_minus_a": float(point_b - point_a),
            "measures": "paired agreement with the B6 report teacher, not expert truth",
        }
    )
    return payload


def rough_resolution_estimate(
    n_studies: int,
    reference_n: int = 58,
    reference_width: float = 0.115,
) -> dict:
    """A study-count-only heuristic; never substitute it for the empirical CI.

    B6 is sparse and target-specific, so 1/sqrt(n) scaling cannot predict the
    actual weak-surface macro-AUC interval reliably. This helper is retained only
    as a rough planning number and is deliberately labelled as such.
    """
    if n_studies < 1:
        raise ValueError("n_studies must be positive")
    width = reference_width * np.sqrt(reference_n / n_studies)
    return {
        "n_studies": int(n_studies),
        "rough_ci_width_from_study_count_only": float(width),
        "warning": (
            "B6 is sparse; use the empirical bootstrap on the frozen holdout for "
            "all model-selection decisions"
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
    out_root: str | Path = "runs/weak_holdout_v1",
    holdout_fraction: float = 0.2,
    seed: int = 2026,
) -> dict:
    """Freeze the leakage-safe B6 holdout that future training must respect."""
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, y, w, supervision = prepare_b7_supervision(train, b6_frame)
    if len(uids) != 3120 or int(supervision.get("usable_cells", -1)) != 14123:
        raise ValueError("weak holdout must be frozen on the exact retained B6 surface")

    grouped = add_report_groups(train[["StudyInstanceUID", "Report"]])
    group_map = dict(zip(grouped["StudyInstanceUID"].astype(str), grouped["report_group"].astype(str)))
    report_groups = [group_map[str(uid)] for uid in uids]
    holdout = make_weak_holdout(
        uids,
        report_groups,
        holdout_fraction=holdout_fraction,
        seed=seed,
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
    missing_class_targets: list[str] = []
    for j, target in enumerate(TARGETS):
        labelled = holdout & (w[:, j] > 0)
        positives = int((labelled & (y[:, j] >= 0.5)).sum())
        negatives = int((labelled & (y[:, j] < 0.5)).sum())
        per_target[target] = {
            "labelled_cells": int(labelled.sum()),
            "positive_cells": positives,
            "negative_cells": negatives,
        }
        if positives == 0 or negatives == 0:
            missing_class_targets.append(target)
    if missing_class_targets:
        raise ValueError(
            "frozen weak holdout lacks both classes for: "
            + ", ".join(missing_class_targets)
            + ". Choose a split design before training; do not search using gold performance."
        )

    payload = {
        "surface": "weak_b6_holdout_v1",
        "status": "frozen before candidate training",
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
        "per_target_holdout": per_target,
        "manifest_sha256": _manifest_sha256(manifest),
        "gold_studies_in_surface": 0,
        "uses_gold_labels": False,
        "measurement": "teacher agreement only; not expert truth",
        "training_contract": (
            "every model scored on this surface must be trained with all holdout "
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
    """Render a weak-surface result with the bias/leakage caveat attached."""
    lines = [
        f"weak-surface macro AUC {payload['macro_auc']:.4f} "
        f"[{payload['ci_lower']:.4f}, {payload['ci_upper']:.4f}]",
        f"  studies {payload['n_studies']}, labelled cells {payload['labelled_cells']} "
        f"({payload['positive_cells']} positive / {payload['negative_cells']} negative)",
        "",
        "This measures agreement with the B6 report teacher, not expert truth.",
        "Use it only to rank models that were trained with these holdout studies",
        "excluded. The reused 58-study gold surface is development confirmation",
        "only; hidden competition evaluation remains the independent signal.",
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
        description="Freeze a report-group-safe B6 weak validation holdout"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-root", default="runs/weak_holdout_v1")
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
    )
    print(json.dumps(payload, indent=2))
    print(Path(args.out_root) / "weak_holdout_manifest.csv")
    print(Path(args.out_root) / "weak_holdout.json")


if __name__ == "__main__":
    main()
