"""Does anything in the export separate the teacher's right answers from its wrong ones?

The confidence column is a constant. Every committed cell carries exactly 0.90
and every silent one 0.0, so `conf >= 0.75` asks "is this cell answered" and
nothing else. Training uses it that way and no other way: the weight a cell gets
is a flat POSITIVE_WEIGHT or NEGATIVE_WEIGHT, chosen by state, never scaled by
confidence. The column looks like a quality score and is not one.

This measures what could go there instead. Three signals travel with a cell
already, and each is scored the same way -- against the 58 expert studies, by
what it would cost in coverage and buy in accuracy.

```text
who answered      base cells carry no __model_confidence, filled cells do
what was said     positive or negated
who else agreed   the LLM's own verdict on a cell the parser answered
```

## The third one is the interesting one

The first is descriptive. The second is already spent: the negated-only rule
came from it, and it is why the current teacher exists.

The third has never been tried. A base cell the LLM independently contradicts is
not the same as one it corroborates, and **dropping a contradicted cell is not
overriding it**. That distinction matters because overriding is the thing that
has been measured and refused: B23 replaced the parser's calls and lost
specificity; B24X put a number on it and found replacement worth nothing at all
(95% CI [-0.0100, +0.0035]). Neither tested removal.

## What this must not become

58 studies, and about 313 cells once silence is excluded. Slice that into
buckets and the buckets are small; a filter that removes eight errors has removed
eight plus or minus three. **This surface is a veto, not a search space.**

Use it to find an effect large and principled enough to survive being wrong
about its exact size -- the negated-only rule qualified, at 97.8% against 62.8%,
a gap no plausible sampling error closes. Do not use it to choose a threshold.
Tuning a cut-off here spends the only expert-truth proxy the project has and
buys a number that will not reproduce.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .report_labels import STATE_NEGATED, STATE_POSITIVE

AUDIT_VERSION = "teacher_confidence_v1"

COMMITTED_STATES = (STATE_POSITIVE, STATE_NEGATED)
MIN_CONFIDENCE = 0.75

# Verdicts a second labeller can return on a cell the first one answered.
CORROBORATED = "corroborated"
CONTRADICTED = "contradicted"
UNWITNESSED = "unwitnessed"


def _read_structured(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "structured_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. This audit needs structured_labels.csv, which keeps "
            "the 58 expert studies; training_targets.csv drops them by design"
        )
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} lists a study more than once")
    missing = [
        column
        for target in TARGETS
        for column in (f"{target}__state", f"{target}__confidence")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing[:6])}")
    return frame


def cell_table(
    *,
    train_csv: str | Path,
    teacher: str | Path,
    witness: str | Path | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> pd.DataFrame:
    """One row per expert cell the teacher answers, with everything known about it.

    Columns: study, target, state, correct, source, model_confidence, witness.
    """
    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    if gold.empty:
        raise ValueError("train.csv contains no expert-labelled studies")
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)

    teacher_frame = _read_structured(teacher)
    witness_frame = _read_structured(witness) if witness is not None else None

    truth = gold.rename(columns={target: f"{target}__truth" for target in TARGETS})
    merged = truth.merge(
        teacher_frame, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    if merged[f"{TARGETS[0]}__state"].isna().all():
        raise ValueError(
            "the teacher export contains none of the expert studies; it must be "
            "structured_labels.csv, not training_targets.csv"
        )
    if witness_frame is not None:
        witness_frame = witness_frame.set_index("StudyInstanceUID")

    rows: list[dict] = []
    for target in TARGETS:
        truth_value = pd.to_numeric(merged[f"{target}__truth"], errors="coerce")
        state = merged[f"{target}__state"].astype(str)
        confidence = pd.to_numeric(
            merged[f"{target}__confidence"], errors="coerce"
        ).fillna(0.0)
        model_column = f"{target}__model_confidence"
        model = (
            pd.to_numeric(merged[model_column], errors="coerce")
            if model_column in merged.columns
            else pd.Series(np.nan, index=merged.index)
        )

        # Only definite calls are predictions. Silence and hedging are not errors.
        usable = (
            truth_value.notna()
            & state.isin(COMMITTED_STATES)
            & confidence.ge(min_confidence)
        )
        for index in merged.index[usable]:
            uid = str(merged.at[index, "StudyInstanceUID"])
            said_positive = state.at[index] == STATE_POSITIVE
            model_value = float(model.at[index]) if pd.notna(model.at[index]) else None
            rows.append(
                {
                    "StudyInstanceUID": uid,
                    "target": target,
                    "state": state.at[index],
                    "correct": bool(said_positive == bool(float(truth_value.at[index]))),
                    # The parser has no self-report, so the filler's own number is
                    # absent exactly on the cells the parser produced.
                    "source": "filled" if model_value is not None else "base",
                    "model_confidence": model_value,
                    "witness": _witness_verdict(
                        witness_frame, uid, target, said_positive, min_confidence
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "StudyInstanceUID",
            "target",
            "state",
            "correct",
            "source",
            "model_confidence",
            "witness",
        ],
    )


def _witness_verdict(
    witness: pd.DataFrame | None,
    uid: str,
    target: str,
    said_positive: bool,
    min_confidence: float,
) -> str | None:
    """What a second labeller says about a cell the first one answered."""
    if witness is None or uid not in witness.index:
        return None
    state = str(witness.at[uid, f"{target}__state"])
    confidence = pd.to_numeric(
        pd.Series([witness.at[uid, f"{target}__confidence"]]), errors="coerce"
    ).fillna(0.0).iloc[0]
    if state not in COMMITTED_STATES or confidence < min_confidence:
        return UNWITNESSED
    return CORROBORATED if (state == STATE_POSITIVE) == said_positive else CONTRADICTED


def _accuracy(subset: pd.DataFrame) -> dict:
    cells = int(len(subset))
    wrong = int((~subset["correct"]).sum())
    correct = cells - wrong
    rate = correct / cells if cells else 0.0
    # A binomial standard error, so a bucket cannot be read as more precise than
    # its size allows. Small buckets are the whole hazard on 58 studies.
    error = math.sqrt(rate * (1.0 - rate) / cells) if cells else 0.0
    return {
        "cells": cells,
        "wrong": wrong,
        "accuracy": rate,
        "standard_error": error,
        "error_rate": (wrong / cells) if cells else 0.0,
    }


def discrimination(subset: pd.DataFrame, column: str) -> float | None:
    """AUC of a per-cell score against whether the cell is right. 0.5 is useless."""
    scored = subset.loc[subset[column].notna()]
    right = scored.loc[scored["correct"], column].to_numpy(float)
    wrong = scored.loc[~scored["correct"], column].to_numpy(float)
    if not len(right) or not len(wrong):
        return None
    comparisons = (right[:, None] > wrong[None, :]).sum() + 0.5 * (
        right[:, None] == wrong[None, :]
    ).sum()
    return float(comparisons / (len(right) * len(wrong)))


def audit(
    *,
    train_csv: str | Path,
    teacher: str | Path,
    witness: str | Path | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    out_json: str | Path | None = None,
) -> dict:
    cells = cell_table(
        train_csv=train_csv,
        teacher=teacher,
        witness=witness,
        min_confidence=min_confidence,
    )
    result = {
        "version": AUDIT_VERSION,
        "teacher": str(teacher),
        "witness": None if witness is None else str(witness),
        "min_confidence": float(min_confidence),
        "overall": _accuracy(cells),
        "by_source": {
            source: _accuracy(group) for source, group in cells.groupby("source")
        },
        "by_state": {
            state: _accuracy(group) for state, group in cells.groupby("state")
        },
        "model_confidence_auc": discrimination(cells, "model_confidence"),
        "model_confidence_is_constant": bool(
            cells["model_confidence"].notna().any()
            and cells["model_confidence"].dropna().nunique() == 1
        ),
    }
    if witness is not None:
        result["by_witness"] = {
            verdict: _accuracy(group) for verdict, group in cells.groupby("witness")
        }
        kept = cells.loc[cells["witness"] != CONTRADICTED]
        result["filters"] = {
            "baseline": _accuracy(cells),
            "drop_contradicted": _accuracy(kept),
        }
    return _finish(result, out_json)


def _finish(result: dict, out_json: str | Path | None) -> dict:
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _line(name: str, item: dict) -> str:
    return (
        f"  {name:<26}{item['cells']:>7,}{item['wrong']:>8,}"
        f"{item['error_rate'] * 100:>8.1f}%"
        f"    {item['accuracy']:.3f} +/- {item['standard_error']:.3f}"
    )


def _report(result: dict) -> None:
    print()
    print(f"  {'':<26}{'cells':>7}{'wrong':>8}{'err':>9}    accuracy")
    print(_line("all cells answered", result["overall"]))

    print()
    print("  who answered the cell")
    for source, item in sorted(result["by_source"].items()):
        print(_line(source, item))

    print()
    print("  what was said")
    for state, item in sorted(result["by_state"].items()):
        print(_line(state, item))

    if result.get("by_witness"):
        print()
        print("  what the second labeller said")
        for verdict, item in sorted(result["by_witness"].items()):
            print(_line(verdict, item))

        print()
        print("  what a filter would cost and buy")
        for name, item in result["filters"].items():
            print(_line(name, item))

    auc = result["model_confidence_auc"]
    print()
    if result["model_confidence_is_constant"]:
        print(
            "  The filler's own confidence is a constant here, so it cannot "
            "discriminate."
        )
    elif auc is None:
        print("  The filler's own confidence is absent, or every cell agrees.")
    else:
        print(f"  The filler's own confidence separates right from wrong at {auc:.3f} AUC.")

    print()
    print(
        "  58 studies. Read the standard errors: this surface is a veto on an\n"
        "  effect large enough to survive them, not a place to choose a threshold."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Find whether anything in the export predicts which cells are wrong"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--teacher", required=True, help="structured_labels.csv from the merged export"
    )
    parser.add_argument(
        "--witness",
        default=None,
        help=(
            "a second labeller's structured_labels.csv, to say which of the "
            "teacher's cells it corroborates and which it contradicts"
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--out-json", default=None)
    parser.add_argument(
        "--out-cells", default=None, help="write the per-cell table to this CSV"
    )
    args = parser.parse_args()

    if args.out_cells is not None:
        cells = cell_table(
            train_csv=args.train_csv,
            teacher=args.teacher,
            witness=args.witness,
            min_confidence=args.min_confidence,
        )
        path = Path(args.out_cells)
        path.parent.mkdir(parents=True, exist_ok=True)
        cells.to_csv(path, index=False)

    _report(
        audit(
            train_csv=args.train_csv,
            teacher=args.teacher,
            witness=args.witness,
            min_confidence=args.min_confidence,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()
