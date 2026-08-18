"""Pre-evaluation Phase-9 v2 competition-aligned AUC addendum.

This addendum is frozen after both fixed-E2 Phase-9 v2 checkpoints were written
but before any PV2-holdout predictions or metrics were generated/inspected.
It does not change the original Phase-9 v2 primary endpoint: macro original-B6-
weighted soft-label BCE remains primary.  The addendum adds a key secondary
paired macro ROC-AUC readout because the Kaggle competition metric is macro AUC.

The AUC truth is still the original frozen B6 positive/negated state on the same
499-study PV2 holdout, not expert truth and not Kaggle hidden truth.  Therefore
this is competition-aligned in metric form only; PV2 remains weak-label and
historically exposed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import _read_config
from .constants import TARGETS
from .evaluation import macro_auc_from_arrays
from .phase9_matched_supervision_v2_eval import evaluate_phase9_v2
from .phase9_v2_supervision import load_phase9_v2_holdout

PHASE9_V2_AUC_ADDENDUM_VERSION = "1.0.0"
PHASE9_V2_AUC_BOOTSTRAP_SEED_OFFSET = 40_304


def hard_truth_from_b6(targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Map usable B6 positive/negated cells to 1/0 and all other cells to NaN."""
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.shape != w.shape or y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError("Phase 9 v2 AUC truth arrays must have shape [N,12]")
    out = np.full(y.shape, np.nan, dtype=np.float64)
    active = w > 0
    out[active] = (y[active] > 0.5).astype(np.float64)
    return out


def paired_bootstrap_macro_auc_difference(
    truth: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict:
    """Study-cluster paired bootstrap of candidate-control macro AUC.

    The point estimate and every accepted bootstrap replicate require AUC to be
    defined for all 12 targets.  Replicates that lose one class for any target
    after study resampling are discarded rather than silently changing the macro
    estimand from 12 targets to a smaller target subset.
    """
    y = np.asarray(truth, dtype=np.float64)
    a = np.asarray(control, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if y.shape != a.shape or y.shape != b.shape or y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError("Phase 9 v2 paired AUC arrays must all have shape [N,12]")
    if y.shape[0] == 0 or int(n_bootstrap) < 1:
        raise ValueError("Phase 9 v2 paired AUC requires studies and bootstrap replicates")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Phase 9 v2 paired AUC predictions must be finite")

    control_macro, control_per = macro_auc_from_arrays(y, a)
    candidate_macro, candidate_per = macro_auc_from_arrays(y, b)
    if not np.all(np.isfinite(control_per)) or not np.all(np.isfinite(candidate_per)):
        raise RuntimeError("Phase 9 v2 full PV2 holdout does not define AUC for all 12 targets")

    per_target = {}
    for j, target in enumerate(TARGETS):
        active = np.isfinite(y[:, j])
        hard = y[active, j]
        n_pos = int(np.sum(hard == 1.0))
        n_neg = int(np.sum(hard == 0.0))
        per_target[target] = {
            "active_cells": int(active.sum()),
            "positive_cells": n_pos,
            "negative_cells": n_neg,
            "control_auc": float(control_per[j]),
            "candidate_auc": float(candidate_per[j]),
            "candidate_minus_control": float(candidate_per[j] - control_per[j]),
            "role": "descriptive_only_no_targetwise_tuning",
        }

    rng = np.random.default_rng(int(seed))
    diffs: list[float] = []
    n = y.shape[0]
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        control_rep, control_rep_per = macro_auc_from_arrays(y[idx], a[idx])
        candidate_rep, candidate_rep_per = macro_auc_from_arrays(y[idx], b[idx])
        if not np.all(np.isfinite(control_rep_per)) or not np.all(np.isfinite(candidate_rep_per)):
            continue
        diffs.append(float(candidate_rep - control_rep))

    if not diffs:
        raise RuntimeError("Phase 9 v2 paired AUC bootstrap produced no strict all-target replicates")
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "metric": "macro ROC AUC across all 12 original-B6 positive/negated targets",
        "difference_definition": "candidate_macro_auc - control_macro_auc",
        "higher_is_better": True,
        "control_macro_auc": float(control_macro),
        "candidate_macro_auc": float(candidate_macro),
        "point_difference": float(candidate_macro - control_macro),
        "median_bootstrap_difference": float(np.median(arr)),
        "ci_lower": float(np.quantile(arr, 0.025)),
        "ci_upper": float(np.quantile(arr, 0.975)),
        "probability_candidate_better": float(np.mean(arr > 0.0)),
        "n_requested_replicates": int(n_bootstrap),
        "n_valid_replicates": int(arr.size),
        "bootstrap_unit": "StudyInstanceUID",
        "strict_all_12_targets_per_replicate": True,
        "per_target": per_target,
    }


def evaluate_phase9_v2_with_auc_addendum(
    config: dict,
    *,
    b6_root: str | Path,
    parent_pv1_manifest_path: str | Path,
    pv2_manifest_path: str | Path,
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    out_root: str | Path = "runs/phase9_matched_supervision_v2/eval",
    n_bootstrap: int = 5000,
) -> dict:
    """Run the frozen Phase-9 v2 evaluator and append the pre-frozen AUC readout."""
    result = evaluate_phase9_v2(
        config,
        b6_root=b6_root,
        parent_pv1_manifest_path=parent_pv1_manifest_path,
        pv2_manifest_path=pv2_manifest_path,
        control_checkpoint=control_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        out_root=out_root,
        n_bootstrap=n_bootstrap,
    )

    out = Path(out_root)
    paired_path = out / "paired_pv2_predictions.csv"
    if not paired_path.is_file():
        raise FileNotFoundError(f"Phase 9 v2 base evaluator did not write {paired_path}")
    frame = pd.read_csv(paired_path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)

    root = Path(config["data_root"])
    from .data import load_train_csv

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    holdout = load_phase9_v2_holdout(
        train,
        b6_root=b6_root,
        parent_pv1_manifest_path=parent_pv1_manifest_path,
        pv2_manifest_path=pv2_manifest_path,
    )
    uids = [str(x) for x in holdout["uids"]]
    if frame["StudyInstanceUID"].tolist() != uids:
        raise RuntimeError("Phase 9 v2 AUC addendum prediction/holdout UID order mismatch")

    targets = np.asarray(holdout["targets"], dtype=np.float64)
    weights = np.asarray(holdout["weights"], dtype=np.float64)
    hard_truth = hard_truth_from_b6(targets, weights)
    control = frame[[f"{target}__control" for target in TARGETS]].to_numpy(np.float64)
    candidate = frame[[f"{target}__candidate" for target in TARGETS]].to_numpy(np.float64)

    auc = paired_bootstrap_macro_auc_difference(
        hard_truth,
        control,
        candidate,
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 2026)) + PHASE9_V2_AUC_BOOTSTRAP_SEED_OFFSET,
    )

    # Cross-check the AUC point estimates against the base evaluator's secondary metric.
    for arm, point in (("control", auc["control_macro_auc"]), ("candidate", auc["candidate_macro_auc"])):
        secondary = result["metrics"][arm]["secondary"]
        if int(secondary.get("n_defined_targets", -1)) != len(TARGETS):
            raise RuntimeError(f"Phase 9 v2 {arm} base evaluator did not define all 12 AUC targets")
        observed = float(secondary["macro_auc_defined_targets"])
        if not np.isclose(observed, point, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Phase 9 v2 {arm} AUC addendum disagrees with base evaluator point estimate")

    addendum = {
        "addendum_version": PHASE9_V2_AUC_ADDENDUM_VERSION,
        "frozen_timing": (
            "defined after both fixed-E2 Phase-9 v2 checkpoints were written and before any PV2-holdout "
            "predictions or metrics were generated/inspected"
        ),
        "original_primary_unchanged": True,
        "original_primary": "macro of per-target original-B6-weighted soft-label BCE",
        "role": "key competition-aligned secondary inferential readout",
        "competition_alignment_scope": (
            "metric form only: macro ROC AUC across 12 targets; truth remains original B6 weak states, "
            "not expert or Kaggle hidden labels"
        ),
        "paired_macro_auc_bootstrap": auc,
        "governance": (
            "Do not retroactively replace the frozen BCE primary, do not use per-target deltas for rescue filtering "
            "or model mixing, and do not promote from this historically exposed weak-label surface alone."
        ),
    }
    result["competition_aligned_auc_addendum"] = addendum
    (out / "auc_addendum.json").write_text(json.dumps(addendum, indent=2), encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "control_macro_auc": auc["control_macro_auc"],
                "candidate_macro_auc": auc["candidate_macro_auc"],
                "candidate_minus_control_auc": {
                    "point": auc["point_difference"],
                    "median_bootstrap": auc["median_bootstrap_difference"],
                    "ci_lower": auc["ci_lower"],
                    "ci_upper": auc["ci_upper"],
                    "probability_candidate_better": auc["probability_candidate_better"],
                    "valid_replicates": auc["n_valid_replicates"],
                },
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate frozen Phase-9 v2 with pre-evaluation macro-AUC addendum")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--parent-pv1-manifest", required=True)
    ap.add_argument("--pv2-manifest", required=True)
    ap.add_argument("--control-checkpoint", required=True)
    ap.add_argument("--candidate-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision_v2/eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_phase9_v2_with_auc_addendum(
        config,
        b6_root=args.b6_root,
        parent_pv1_manifest_path=args.parent_pv1_manifest,
        pv2_manifest_path=args.pv2_manifest,
        control_checkpoint=args.control_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
