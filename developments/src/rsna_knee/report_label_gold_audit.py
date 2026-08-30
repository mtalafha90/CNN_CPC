"""Measure any B6- or B23-shaped report export against the 58 expert studies.

Three experiments now show the same shape: a gain measured against report-derived
labels that does not reach expert truth. B40 improved every training loss while
Expert-58 macro AUC fell. B50's adapted hierarchy gained `+0.011221` on 548
unseen-scanner studies scored with report labels, and `-0.002432` on Expert-58.
B51, the production version of that same change, came out `-0.011785` against
B42 on Expert-58 with 3 of 12 targets improved.

That pattern points at the teacher rather than the student. If the labels the
model is trained on disagree with experts on a large share of cells, then a model
that fits those labels better is not necessarily a model that scores better, and
architecture work is being spent on the wrong axis.

`b6_gold_audit` already measures this, but only for B6: it requires a
`__reason` column that B23 does not write, and it is documented as a fixed
one-time audit of B6 which should not be edited. This module measures the same
quantities, using the same metric functions imported from it, for either
exporter -- so B6 and B23 can be compared on identical terms.

What it reports, per target and pooled:

    coverage      how many expert cells the export even makes a call on
    precision     of the cells it calls positive, how many experts agree
    sensitivity   of the expert positives it covers, how many it finds
    specificity   of the expert negatives it covers, how many it clears

Only `positive` and `negated` states at or above `min_confidence` are scored.
`uncertain` and `unmentioned` are not predictions -- report silence is not a
negative -- so they lower coverage rather than counting as errors.

This is a measurement, not a gate. It fits nothing and changes no export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b6_gold_audit import _binary_metrics, _mean_defined
from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .report_labels import STATE_NEGATED, STATE_POSITIVE

REPORT_LABEL_GOLD_AUDIT_VERSION = "report_label_gold_audit_v1"

# Recorded for B6 v1.2.1 in b23_llm_labels.py's own docstring, which is the
# comparison B23 was built to beat.
B6_V121_REFERENCE = {
    "recall_sensitivity": 0.9749,
    "specificity": 0.6061,
    "precision_positive": 0.6905,
    "coverage": 0.3606,
}


def _load_structured(path: Path) -> pd.DataFrame:
    """Read an export's structured labels, accepting B6's and B23's shapes."""
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID"}
    for target in TARGETS:
        required.update({target, f"{target}__confidence", f"{target}__state"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing[:6]}")

    frame = frame.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path.name} lists the same study more than once")
    return frame


def audit_export_against_gold(
    train_csv: str | Path,
    structured_csv: str | Path,
    *,
    label: str,
    min_confidence: float = 0.75,
    out_root: str | Path | None = None,
) -> dict:
    """Score one export's definite calls against the 58 expert studies."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")

    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    if gold.empty:
        raise ValueError("train.csv contains no expert-labelled studies")
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)

    structured = _load_structured(Path(structured_csv))
    truth = gold.rename(columns={target: f"{target}__truth" for target in TARGETS})
    merged = truth.merge(structured, on="StudyInstanceUID", how="left", validate="one_to_one")

    covered = int(merged[f"{TARGETS[0]}__state"].notna().sum())
    if covered == 0:
        raise ValueError(
            f"{Path(structured_csv).name} contains none of the {len(gold)} expert "
            "studies. training_targets.csv excludes them by design -- this audit "
            "needs structured_labels.csv, which keeps them for exactly this purpose."
        )

    target_metrics: dict[str, dict] = {}
    pooled_true: list[bool] = []
    pooled_pred: list[bool] = []
    mismatches: list[dict] = []

    for target in TARGETS:
        truth_col, state_col = f"{target}__truth", f"{target}__state"
        conf_col = f"{target}__confidence"

        y_numeric = pd.to_numeric(merged[truth_col], errors="coerce")
        defined = y_numeric.notna()
        unexpected = sorted(
            set(y_numeric.loc[defined].astype(float).unique()).difference({0.0, 1.0})
        )
        if unexpected:
            raise ValueError(f"expert target {target!r} has non-binary values: {unexpected}")

        # Only definite calls are predictions. Silence and hedging are not errors.
        usable = (
            defined
            & merged[state_col].isin([STATE_POSITIVE, STATE_NEGATED])
            & pd.to_numeric(merged[conf_col], errors="coerce").ge(min_confidence)
        )
        y_true = y_numeric.loc[usable].astype(int).to_numpy(dtype=bool)
        y_pred = merged.loc[usable, state_col].eq(STATE_POSITIVE).to_numpy(dtype=bool)

        metrics = _binary_metrics(y_true, y_pred)
        n_defined, n_usable = int(defined.sum()), int(usable.sum())
        metrics.update(
            {
                "n_gold_defined": n_defined,
                "n_usable": n_usable,
                "coverage": float(n_usable / n_defined) if n_defined else 0.0,
                "gold_positive_within_usable": int(np.sum(y_true)),
                "predicted_positive": int(np.sum(y_pred)),
            }
        )
        target_metrics[target] = metrics

        pooled_true.extend(y_true.tolist())
        pooled_pred.extend(y_pred.tolist())

        for index in merged.index[usable]:
            predicted = bool(merged.at[index, state_col] == STATE_POSITIVE)
            actual = bool(float(merged.at[index, truth_col]))
            if predicted != actual:
                mismatches.append(
                    {
                        "StudyInstanceUID": str(merged.at[index, "StudyInstanceUID"]),
                        "target": target,
                        "expert": int(actual),
                        "report_label": int(predicted),
                        "state": str(merged.at[index, state_col]),
                        "confidence": float(merged.at[index, conf_col]),
                    }
                )

    possible = int(sum(item["n_gold_defined"] for item in target_metrics.values()))
    pooled = _binary_metrics(
        np.asarray(pooled_true, dtype=bool), np.asarray(pooled_pred, dtype=bool)
    )
    pooled.update(
        {
            "n_usable_cells": len(pooled_true),
            "possible_gold_cells": possible,
            "coverage": float(len(pooled_true) / possible) if possible else 0.0,
        }
    )

    macro = {
        name: _mean_defined([item[name] for item in target_metrics.values()])
        for name in (
            "precision_positive",
            "recall_sensitivity",
            "specificity",
            "negative_predictive_value",
            "accuracy",
            "balanced_accuracy",
        )
    }
    macro["coverage"] = _mean_defined([item["coverage"] for item in target_metrics.values()])

    result = {
        "version": REPORT_LABEL_GOLD_AUDIT_VERSION,
        "label": label,
        "structured_csv": str(Path(structured_csv).resolve()),
        "n_gold_studies": int(len(gold)),
        "gold_studies_present_in_export": covered,
        "min_confidence": float(min_confidence),
        "pooled": pooled,
        "macro": macro,
        "per_target": target_metrics,
        "n_mismatched_cells": len(mismatches),
        "b6_v121_reference": B6_V121_REFERENCE,
        "note": (
            "Coverage is the share of expert cells the export makes any definite "
            "call on; silence and hedging lower it rather than counting as errors. "
            "Precision is how often a reported positive matches the expert. This "
            "audit fits nothing and changes no export."
        ),
    }

    if out_root is not None:
        out = Path(out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gold_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if mismatches:
            pd.DataFrame(mismatches).to_csv(out / "mismatches.csv", index=False)
    return result


def _print_summary(result: dict) -> None:
    macro, pooled = result["macro"], result["pooled"]
    reference = result["b6_v121_reference"]
    print()
    print(f"{result['label']} against {result['n_gold_studies']} expert studies")
    print(f"{'':<16}{'this export':>13}{'B6 v1.2.1':>12}")
    for name, key in (
        ("coverage", "coverage"),
        ("precision", "precision_positive"),
        ("sensitivity", "recall_sensitivity"),
        ("specificity", "specificity"),
    ):
        value = macro.get(key)
        shown = f"{value:.4f}" if value is not None else "n/a"
        print(f"{name:<16}{shown:>13}{reference[key]:>12.4f}")
    print()
    print(
        f"pooled: {pooled['n_usable_cells']:,} of {pooled['possible_gold_cells']:,} "
        f"expert cells called ({pooled['coverage']:.1%}), "
        f"{result['n_mismatched_cells']:,} disagree with the expert"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a B6 or B23 report-label export against the expert studies"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--structured", required=True, help="structured_labels.csv from the export"
    )
    parser.add_argument("--label", default="export", help="name for this export in the output")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    result = audit_export_against_gold(
        args.train_csv,
        args.structured,
        label=args.label,
        min_confidence=args.min_confidence,
        out_root=args.out_root,
    )
    print(json.dumps(result["macro"], indent=2))
    _print_summary(result)


if __name__ == "__main__":
    main()


__all__ = ["B6_V121_REFERENCE", "audit_export_against_gold"]
