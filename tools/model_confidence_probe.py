"""Does the LLM's own confidence know which of its labels are wrong?

The teacher behind B51 calls 66% of expert cells and gets about a third of its
positives wrong, with sensitivity 0.977 against specificity 0.499 -- it says
"positive" far too readily. Raising precision normally means re-running the
labeller, which is expensive.

But B23 already records `<target>__model_confidence`: the model's own stated
confidence for every cell. By policy it is "stored as a diagnostic column only
and never thresholds supervision", and it has never been checked. If it
separates correct labels from wrong ones, precision can be bought with a
threshold on a file that already exists. If it does not, the policy of ignoring
it was right and that avenue is closed.

This measures exactly that, on the 58 expert studies:

    AUC        how well model_confidence ranks correct labels above wrong ones
    sweep      precision and coverage at each candidate threshold

An AUC near 0.5 means the confidence is noise. The honest outcome is reported
either way; a lever that does not exist is worth knowing about before more work
is spent assuming it does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "developments" / "src"))

from rsna_knee.constants import TARGETS  # noqa: E402
from rsna_knee.data import gold_mask, load_train_csv  # noqa: E402
from rsna_knee.evaluation import fast_auc  # noqa: E402
from rsna_knee.report_labels import STATE_NEGATED, STATE_POSITIVE  # noqa: E402

COMMITTED = (STATE_POSITIVE, STATE_NEGATED)


def probe(
    train_csv: str | Path,
    structured_csv: str | Path,
    *,
    min_confidence: float = 0.75,
) -> dict:
    """Score model_confidence as a predictor of a label being right."""
    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)

    structured = pd.read_csv(structured_csv)
    structured["StudyInstanceUID"] = structured["StudyInstanceUID"].astype(str)
    missing = [
        f"{target}__model_confidence"
        for target in TARGETS
        if f"{target}__model_confidence" not in structured.columns
    ]
    if missing:
        raise ValueError(
            f"{Path(structured_csv).name} has no model_confidence columns "
            f"(e.g. {missing[0]}). Only a B23 export records them; B6 does not."
        )

    merged = gold.rename(columns={t: f"{t}__truth" for t in TARGETS}).merge(
        structured, on="StudyInstanceUID", how="left", validate="one_to_one"
    )

    confidences: list[float] = []
    correct: list[int] = []
    positive_call: list[int] = []
    for target in TARGETS:
        truth = pd.to_numeric(merged[f"{target}__truth"], errors="coerce")
        usable = (
            truth.notna()
            & merged[f"{target}__state"].isin(COMMITTED)
            & pd.to_numeric(merged[f"{target}__confidence"], errors="coerce").ge(
                min_confidence
            )
        )
        for index in merged.index[usable]:
            predicted = merged.at[index, f"{target}__state"] == STATE_POSITIVE
            confidences.append(float(merged.at[index, f"{target}__model_confidence"]))
            correct.append(int(bool(predicted) == bool(truth.at[index])))
            positive_call.append(int(predicted))

    confidence = np.asarray(confidences, dtype=np.float64)
    is_correct = np.asarray(correct, dtype=np.int64)
    is_positive = np.asarray(positive_call, dtype=bool)

    if confidence.size == 0:
        raise ValueError("no committed cells overlap the expert studies")

    # fast_auc returns NaN when every label is correct or every one is wrong;
    # that is "cannot tell", not a score, so it is carried through as None.
    raw = fast_auc(is_correct, confidence)
    auc = None if not np.isfinite(raw) else float(raw)
    sweep = []
    for threshold in sorted({round(float(v), 2) for v in np.unique(confidence)}):
        keep = confidence >= threshold
        if not keep.any():
            continue
        kept_positive = keep & is_positive
        sweep.append(
            {
                "threshold": threshold,
                "cells_kept": int(keep.sum()),
                "coverage_of_called_cells": float(keep.mean()),
                "accuracy": float(is_correct[keep].mean()),
                "precision_positive": (
                    float(is_correct[kept_positive].mean())
                    if kept_positive.any() else None
                ),
                "positive_cells_kept": int(kept_positive.sum()),
            }
        )

    return {
        "structured_csv": str(Path(structured_csv).resolve()),
        "n_cells": int(confidence.size),
        "baseline_accuracy": float(is_correct.mean()),
        "baseline_precision_positive": float(is_correct[is_positive].mean()),
        "confidence_auc_for_correctness": auc,
        "confidence_distribution": {
            "min": float(confidence.min()),
            "median": float(np.median(confidence)),
            "max": float(confidence.max()),
            "distinct_values": int(np.unique(confidence).size),
        },
        "sweep": sweep,
        "verdict": (
            "informative" if auc is not None and auc >= 0.60
            else "uninformative" if auc is not None
            else "undefined"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether the LLM's own confidence predicts label correctness"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--structured", required=True, help="a B23 structured_labels.csv")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    result = probe(args.train_csv, args.structured, min_confidence=args.min_confidence)

    print(f"cells on the expert surface   : {result['n_cells']}")
    print(f"distinct confidence values    : {result['confidence_distribution']['distinct_values']}")
    print(f"baseline accuracy             : {result['baseline_accuracy']:.4f}")
    print(f"baseline positive precision   : {result['baseline_precision_positive']:.4f}")
    auc = result["confidence_auc_for_correctness"]
    print(f"AUC, confidence -> correctness: {auc:.4f}" if auc is not None else "AUC: undefined")
    print(f"verdict                       : {result['verdict']}")
    print()
    print(f"{'threshold':>10}{'kept':>8}{'coverage':>10}{'accuracy':>10}{'pos prec':>10}")
    for row in result["sweep"]:
        precision = row["precision_positive"]
        shown = f"{precision:.4f}" if precision is not None else "     n/a"
        print(
            f"{row['threshold']:>10.2f}{row['cells_kept']:>8}"
            f"{row['coverage_of_called_cells']:>10.3f}{row['accuracy']:>10.4f}{shown:>10}"
        )

    if result["verdict"] == "uninformative":
        print()
        print(
            "The confidence does not separate correct labels from wrong ones. "
            "Thresholding on it would drop cells without improving precision, "
            "and the policy of ignoring it was right."
        )

    if args.out_root:
        out = Path(args.out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_confidence_probe.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()


__all__ = ["probe"]
