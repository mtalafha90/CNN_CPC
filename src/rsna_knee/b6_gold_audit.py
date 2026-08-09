"""Fixed one-time audit of B6 weak labels against the 58 gold studies.

This module does not fit, calibrate, optimize, or change B6. It measures the
already-frozen high-confidence structured report labels against gold labels and
exports exact mismatches for inspection before B7 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .report_labels import STATE_NEGATED, STATE_POSITIVE


def _safe_div(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced = None
    if recall is not None and specificity is not None:
        balanced = float((recall + specificity) / 2.0)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_positive": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }


def _mean_defined(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def run_b6_gold_audit(
    train_csv: str | Path,
    structured_csv: str | Path,
    *,
    out_root: str | Path = "runs/b6_report_labels_v121/gold_audit",
    min_confidence: float = 0.75,
) -> dict:
    """Evaluate fixed B6 high-confidence cells on gold without fitting anything."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")

    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", "Report", *TARGETS]].copy()
    if gold.empty:
        raise ValueError("train.csv contains no gold studies")

    structured = pd.read_csv(structured_csv)
    required = {"StudyInstanceUID"}
    for target in TARGETS:
        required.update(
            {
                target,
                f"{target}__confidence",
                f"{target}__state",
                f"{target}__reason",
                f"{target}__evidence",
            }
        )
    missing = sorted(required.difference(structured.columns))
    if missing:
        raise ValueError(f"structured_labels.csv missing columns: {missing}")

    structured = structured.copy()
    structured["StudyInstanceUID"] = structured["StudyInstanceUID"].astype(str)
    if structured["StudyInstanceUID"].duplicated().any():
        raise ValueError("structured_labels.csv contains duplicate StudyInstanceUID values")

    truth = gold.rename(columns={target: f"{target}__truth" for target in TARGETS})
    merged = truth.merge(structured, on="StudyInstanceUID", how="left", validate="one_to_one")
    if len(merged) != len(gold):
        raise AssertionError("gold merge changed row count")

    missing_structured = int(merged[f"{TARGETS[0]}__state"].isna().sum())
    if missing_structured:
        raise ValueError(f"structured_labels.csv is missing {missing_structured} gold study rows")

    target_metrics: dict[str, dict] = {}
    usable_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    pooled_true: list[bool] = []
    pooled_pred: list[bool] = []

    for target in TARGETS:
        truth_col = f"{target}__truth"
        state_col = f"{target}__state"
        conf_col = f"{target}__confidence"
        reason_col = f"{target}__reason"
        evidence_col = f"{target}__evidence"

        y_numeric = pd.to_numeric(merged[truth_col], errors="coerce")
        defined = y_numeric.notna()
        unexpected = sorted(set(y_numeric.loc[defined].astype(float).unique()).difference({0.0, 1.0}))
        if unexpected:
            raise ValueError(f"gold target {target!r} contains non-binary values: {unexpected}")

        usable = (
            defined
            & merged[state_col].isin([STATE_POSITIVE, STATE_NEGATED])
            & pd.to_numeric(merged[conf_col], errors="coerce").ge(min_confidence)
        )
        y_true = y_numeric.loc[usable].astype(int).to_numpy(dtype=bool)
        y_pred = merged.loc[usable, state_col].eq(STATE_POSITIVE).to_numpy(dtype=bool)
        metrics = _binary_metrics(y_true, y_pred)

        n_defined = int(defined.sum())
        n_usable = int(usable.sum())
        metrics.update(
            {
                "n_gold_defined": n_defined,
                "n_usable": n_usable,
                "coverage": float(n_usable / n_defined) if n_defined else 0.0,
                "predicted_positive": int(np.sum(y_pred)),
                "predicted_negative": int(np.sum(~y_pred)),
                "gold_positive_within_usable": int(np.sum(y_true)),
                "gold_negative_within_usable": int(np.sum(~y_true)),
            }
        )
        target_metrics[target] = metrics

        pooled_true.extend(y_true.tolist())
        pooled_pred.extend(y_pred.tolist())

        for index in merged.index[usable]:
            state = str(merged.at[index, state_col])
            pred = int(state == STATE_POSITIVE)
            true = int(float(merged.at[index, truth_col]))
            row = {
                "StudyInstanceUID": str(merged.at[index, "StudyInstanceUID"]),
                "target": target,
                "truth": true,
                "predicted": pred,
                "state": state,
                "confidence": float(merged.at[index, conf_col]),
                "correct": bool(pred == true),
                "reason": str(merged.at[index, reason_col]),
                "evidence": str(merged.at[index, evidence_col]),
                "report": str(merged.at[index, "Report"]),
            }
            usable_rows.append(row)
            if pred != true:
                mismatch_rows.append(row)

    pooled = _binary_metrics(np.asarray(pooled_true, dtype=bool), np.asarray(pooled_pred, dtype=bool))
    pooled.update(
        {
            "n_usable_cells": int(len(pooled_true)),
            "possible_gold_cells": int(sum(item["n_gold_defined"] for item in target_metrics.values())),
            "coverage": float(
                len(pooled_true) / max(1, sum(item["n_gold_defined"] for item in target_metrics.values()))
            ),
        }
    )

    macro = {
        "precision_positive": _mean_defined(
            [item["precision_positive"] for item in target_metrics.values()]
        ),
        "recall_sensitivity": _mean_defined(
            [item["recall_sensitivity"] for item in target_metrics.values()]
        ),
        "specificity": _mean_defined([item["specificity"] for item in target_metrics.values()]),
        "negative_predictive_value": _mean_defined(
            [item["negative_predictive_value"] for item in target_metrics.values()]
        ),
        "accuracy": _mean_defined([item["accuracy"] for item in target_metrics.values()]),
        "balanced_accuracy": _mean_defined(
            [item["balanced_accuracy"] for item in target_metrics.values()]
        ),
    }

    policy_path = Path(structured_csv).with_name("policy.json")
    b6_version = None
    if policy_path.exists():
        try:
            b6_version = json.loads(policy_path.read_text(encoding="utf-8")).get("version")
        except (OSError, json.JSONDecodeError):
            b6_version = None

    payload = {
        "experiment": "B6_gold_audit",
        "b6_version": b6_version,
        "n_gold_studies": int(len(gold)),
        "min_confidence": float(min_confidence),
        "fitting_or_calibration_performed": False,
        "threshold_search_performed": False,
        "parser_change_after_this_audit_allowed": False,
        "targets": target_metrics,
        "macro_over_targets": macro,
        "pooled_over_usable_cells": pooled,
        "n_mismatches": int(len(mismatch_rows)),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gold_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(usable_rows).to_csv(out / "gold_usable_cells.csv", index=False)
    pd.DataFrame(mismatch_rows).to_csv(out / "gold_mismatches.csv", index=False)

    print(json.dumps(payload, indent=2))
    print(out / "gold_audit.json")
    print(out / "gold_usable_cells.csv")
    print(out / "gold_mismatches.csv")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b6-audit")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--structured", required=True)
    parser.add_argument("--out-root", default="runs/b6_report_labels_v121/gold_audit")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    args = parser.parse_args()
    run_b6_gold_audit(
        args.train_csv,
        args.structured,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
