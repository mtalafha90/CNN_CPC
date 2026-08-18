"""Post-hoc descriptive Phase-9 v2 target analysis.

This module consumes only already-generated Phase-9 v2 paired holdout predictions
and frozen B6 / Phase-8 supervision artifacts. It performs no training, model
selection, checkpoint loading, target filtering, or rescue-policy changes.

Outputs:
- per-target paired study bootstrap for candidate-control AUC;
- macro leave-one-target-out influence diagnostics;
- exact Phase-7/8 added supervision counts by target, derived by comparing the
  frozen control and candidate supervision tables;
- a descriptive association table between supervision additions and AUC shifts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .evaluation import fast_auc
from .phase9_supervision import load_frozen_phase8_export
from .b7_weak_supervision import B7_MIN_CONFIDENCE, load_frozen_b6_export

ANALYSIS_VERSION = "1.0.0"


def _paired_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID"}
    for target in TARGETS:
        required.update(
            {
                f"{target}__target",
                f"{target}__weight",
                f"{target}__control",
                f"{target}__candidate",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"paired prediction file missing columns: {missing}")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("paired prediction file contains duplicate StudyInstanceUID values")
    return frame


def _hard_truth(target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    out = np.full(target.shape, np.nan, dtype=np.float64)
    active = np.isfinite(target) & np.isfinite(weight) & (weight > 0)
    out[active & (target > 0.5)] = 1.0
    out[active & (target < 0.5)] = 0.0
    return out


def paired_target_bootstrap(
    truth: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if not (truth.shape == control.shape == candidate.shape) or truth.ndim != 1:
        raise ValueError("paired target bootstrap expects equal 1D arrays")
    point_control = fast_auc(truth, control)
    point_candidate = fast_auc(truth, candidate)
    if not np.isfinite(point_control) or not np.isfinite(point_candidate):
        raise ValueError("target AUC is undefined on the full holdout")

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(truth)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        a = fast_auc(truth[idx], control[idx])
        b = fast_auc(truth[idx], candidate[idx])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(float(b - a))
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("no valid paired target bootstrap replicates")
    return {
        "control_auc": float(point_control),
        "candidate_auc": float(point_candidate),
        "point_difference": float(point_candidate - point_control),
        "median_bootstrap_difference": float(np.median(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "probability_candidate_better": float(np.mean(arr > 0)),
        "n_requested_replicates": int(n_bootstrap),
        "n_valid_replicates": int(arr.size),
    }


def _state_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for target in TARGETS:
        state = frame[f"{target}__state"].fillna("").astype(str).str.lower()
        conf = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").fillna(0.0)
        pos = (state == "positive") & (conf >= B7_MIN_CONFIDENCE)
        neg = (state == "negated") & (conf >= B7_MIN_CONFIDENCE)
        out[target] = {
            "positive": int(pos.sum()),
            "negative": int(neg.sum()),
            "usable": int((pos | neg).sum()),
        }
    return out


def added_supervision_by_target(b6_root: str | Path, phase8_root: str | Path) -> pd.DataFrame:
    b6, _, _ = load_frozen_b6_export(b6_root)
    phase8, _, _ = load_frozen_phase8_export(phase8_root)
    b6 = b6.copy()
    phase8 = phase8.copy()
    b6["StudyInstanceUID"] = b6["StudyInstanceUID"].astype(str)
    phase8["StudyInstanceUID"] = phase8["StudyInstanceUID"].astype(str)
    if set(b6["StudyInstanceUID"]) != set(phase8["StudyInstanceUID"]):
        raise RuntimeError("B6 and Phase-8 report-only populations differ")
    b6 = b6.sort_values("StudyInstanceUID").reset_index(drop=True)
    phase8 = phase8.sort_values("StudyInstanceUID").reset_index(drop=True)
    base = _state_counts(b6)
    cand = _state_counts(phase8)
    rows = []
    for target in TARGETS:
        bp, bn = base[target]["positive"], base[target]["negative"]
        cp, cn = cand[target]["positive"], cand[target]["negative"]
        if cp < bp or cn < bn:
            raise RuntimeError(f"Phase-8 reduced frozen B6 counts for {target}")
        rows.append(
            {
                "target": target,
                "original_positive": bp,
                "original_negative": bn,
                "original_usable": bp + bn,
                "added_positive": cp - bp,
                "added_negative": cn - bn,
                "added_usable": (cp + cn) - (bp + bn),
                "candidate_positive": cp,
                "candidate_negative": cn,
                "candidate_usable": cp + cn,
            }
        )
    table = pd.DataFrame(rows)
    if int(table["added_usable"].sum()) != 3901:
        raise RuntimeError("derived Phase-8 added-cell total is not 3,901")
    if int(table["added_positive"].sum()) != 2719 or int(table["added_negative"].sum()) != 1182:
        raise RuntimeError("derived Phase-8 class totals changed")
    return table


def analyse(
    *,
    paired_predictions: str | Path,
    b6_root: str | Path,
    phase8_root: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict:
    frame = _paired_frame(paired_predictions)
    additions = added_supervision_by_target(b6_root, phase8_root)
    add_map = additions.set_index("target")

    per_target_rows = []
    point_deltas = []
    for j, target in enumerate(TARGETS):
        truth = _hard_truth(
            frame[f"{target}__target"].to_numpy(float),
            frame[f"{target}__weight"].to_numpy(float),
        )
        control = frame[f"{target}__control"].to_numpy(float)
        candidate = frame[f"{target}__candidate"].to_numpy(float)
        active = np.isfinite(truth)
        pos = int(np.sum(truth[active] == 1))
        neg = int(np.sum(truth[active] == 0))
        boot = paired_target_bootstrap(
            truth,
            control,
            candidate,
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 + j,
        )
        delta = float(boot["point_difference"])
        point_deltas.append(delta)
        added = add_map.loc[target]
        per_target_rows.append(
            {
                "target": target,
                "holdout_active_cells": int(active.sum()),
                "holdout_positive_cells": pos,
                "holdout_negative_cells": neg,
                **boot,
                "added_usable_cells": int(added["added_usable"]),
                "added_positive_cells": int(added["added_positive"]),
                "added_negative_cells": int(added["added_negative"]),
                "added_positive_fraction": (
                    float(added["added_positive"] / added["added_usable"])
                    if int(added["added_usable"]) else float("nan")
                ),
                "role": "descriptive_only_no_targetwise_tuning",
            }
        )

    per_target = pd.DataFrame(per_target_rows)
    macro_delta = float(np.mean(point_deltas))
    influence_rows = []
    for _, row in per_target.iterrows():
        without = float(
            (macro_delta * len(TARGETS) - float(row["point_difference"])) / (len(TARGETS) - 1)
        )
        influence_rows.append(
            {
                "target": row["target"],
                "target_delta_auc": float(row["point_difference"]),
                "macro_delta_all_targets": macro_delta,
                "macro_delta_without_target": without,
                "target_contribution_to_macro_sum": float(row["point_difference"] / len(TARGETS)),
                "sign_flip_if_removed": bool(np.sign(without) != np.sign(macro_delta) and without != 0),
            }
        )
    influence = pd.DataFrame(influence_rows).sort_values(
        "target_contribution_to_macro_sum", ascending=False
    )

    # Descriptive association only: no inferential claim and no policy change.
    x = per_target["added_usable_cells"].to_numpy(float)
    y = per_target["point_difference"].to_numpy(float)
    x_pos_frac = per_target["added_positive_fraction"].to_numpy(float)
    corr_added = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    finite = np.isfinite(x_pos_frac) & np.isfinite(y)
    corr_posfrac = (
        float(np.corrcoef(x_pos_frac[finite], y[finite])[0, 1])
        if finite.sum() > 1 and np.std(x_pos_frac[finite]) > 0 and np.std(y[finite]) > 0
        else float("nan")
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    per_target.to_csv(out / "per_target_bootstrap.csv", index=False)
    additions.to_csv(out / "rescued_supervision_by_target.csv", index=False)
    influence.to_csv(out / "macro_target_influence.csv", index=False)

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "post-hoc descriptive analysis on frozen Phase-9 v2 predictions",
        "training_or_model_selection_performed": False,
        "n_holdout_studies": int(len(frame)),
        "n_bootstrap_per_target": int(n_bootstrap),
        "macro_point_difference_reconstructed": macro_delta,
        "largest_positive_target": str(per_target.loc[per_target["point_difference"].idxmax(), "target"]),
        "largest_negative_target": str(per_target.loc[per_target["point_difference"].idxmin(), "target"]),
        "targets_point_improved": int((per_target["point_difference"] > 0).sum()),
        "targets_point_worsened": int((per_target["point_difference"] < 0).sum()),
        "association_descriptive_only": {
            "pearson_added_usable_cells_vs_delta_auc": corr_added,
            "pearson_added_positive_fraction_vs_delta_auc": corr_posfrac,
            "warning": "12 target-level observations only; exploratory/descriptive, not causal or inferential",
        },
        "governance": (
            "Do not use target bootstrap intervals, influence, or rescue-density associations to remove translated cells, "
            "change target weights, create target-specific mixtures, retune B34, or revise the Phase-8 merge."
        ),
    }
    (out / "target_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser("Phase-9 v2 descriptive target analysis")
    ap.add_argument("--paired-predictions", default="runs/phase9_matched_supervision_v2/eval/paired_pv2_predictions.csv")
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--phase8-root", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision_v2/target_analysis")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    result = analyse(
        paired_predictions=args.paired_predictions,
        b6_root=args.b6_root,
        phase8_root=args.phase8_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
