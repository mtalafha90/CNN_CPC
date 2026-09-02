"""Why the frozen parser calls 31% of its positives wrong, and whether that is fixable.

The teacher's errors are located precisely. Of 57 disagreements with the 58
experts, 52 are positive calls and 55 come from B6 rather than the LLM filler.
No confidence scheme reaches them: the parser is a regular expression and has no
confidence to record. What is left is the parser's own decisions.

## The question that decides whether point 2 is worth any more effort

Every one of those calls has a cause, and B6 already records it. Each cell
carries the rule that fired and the clause that triggered it:

```text
__reason      explicit_pathology_mention, explicit_negation, uncertainty_scope,
              conflicting_definite_evidence, explicit_structural_abnormality, ...
__evidence    up to 360 characters of the clause the rule matched
```

So a wrong positive is one of two completely different things:

```text
the clause does NOT assert the finding
    -> the parser misread it
    -> a better parser or labeller fixes it

the clause DOES assert the finding, and the expert still says no
    -> the report and the expert disagree
    -> nothing that reads the report can fix it, ever
```

The second is a ceiling, not a defect. The report was written by a radiologist
reading the same images and recording something different from the expert
labeller -- a different threshold for "abnormal", a finding mentioned in passing,
a hedge the writer meant loosely. **If most of the 52 are that, the teacher is
already near the limit of what report text can support, and further labelling
work is wasted.** If most are misreads, there is real room.

Nobody has separated them. This does.

## What it produces

An aggregate map of where the errors sit -- by rule, by target, by state -- and
a per-cell file carrying the evidence clause so each one can actually be read
and judged. The aggregate says where to look; only reading settles which of the
two cases a cell is.

## Governance

Reason codes are a pre-existing categorical property of the parser, not a
threshold discovered here, and this module fits nothing and changes no export.
But accuracy per reason on 58 studies is still 58 studies: a code carrying nine
cells says almost nothing, and every bucket reports its standard error so that
is visible. Use it to find where to read, not to decide what to drop.

The per-cell file contains **raw competition report text** and is local-only.
Do not commit it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .report_labels import STATE_NEGATED, STATE_POSITIVE

ANATOMY_VERSION = "parser_error_anatomy_v1"

COMMITTED_STATES = (STATE_POSITIVE, STATE_NEGATED)
MIN_CONFIDENCE = 0.75

# How much of the report to carry beside the clause, so a reader can see whether
# the parser picked the wrong sentence out of a report that says otherwise.
REPORT_EXCERPT = 2000


def _read_export(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "structured_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. This needs B6's own structured_labels.csv: it is the "
            "only export carrying __reason and __evidence, which is the whole point"
        )
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} lists a study more than once")
    missing = [
        column
        for target in TARGETS
        for column in (f"{target}__state", f"{target}__confidence", f"{target}__reason")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{path} is missing {', '.join(missing[:4])}. A merged export drops "
            "__reason; point this at the B6 export itself"
        )
    return frame


def cell_table(
    *,
    train_csv: str | Path,
    b6_export: str | Path,
    min_confidence: float = MIN_CONFIDENCE,
) -> pd.DataFrame:
    """One row per expert cell the parser makes a definite call on."""
    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", "Report", *TARGETS]].copy()
    if gold.empty:
        raise ValueError("train.csv contains no expert-labelled studies")
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)

    export = _read_export(b6_export)
    truth = gold.rename(columns={target: f"{target}__truth" for target in TARGETS})
    merged = truth.merge(export, on="StudyInstanceUID", how="left", validate="one_to_one")
    if merged[f"{TARGETS[0]}__state"].isna().all():
        raise ValueError(
            "the export contains none of the expert studies; it must be "
            "structured_labels.csv, which keeps them, not training_targets.csv"
        )

    rows: list[dict] = []
    for target in TARGETS:
        truth_value = pd.to_numeric(merged[f"{target}__truth"], errors="coerce")
        state = merged[f"{target}__state"].astype(str)
        confidence = pd.to_numeric(
            merged[f"{target}__confidence"], errors="coerce"
        ).fillna(0.0)
        evidence_column = f"{target}__evidence"
        usable = (
            truth_value.notna()
            & state.isin(COMMITTED_STATES)
            & confidence.ge(min_confidence)
        )
        for index in merged.index[usable]:
            said_positive = state.at[index] == STATE_POSITIVE
            expert_positive = bool(float(truth_value.at[index]))
            rows.append(
                {
                    "StudyInstanceUID": str(merged.at[index, "StudyInstanceUID"]),
                    "target": target,
                    "parser_said": state.at[index],
                    "expert_said": "positive" if expert_positive else "negative",
                    "correct": bool(said_positive == expert_positive),
                    "reason": str(merged.at[index, f"{target}__reason"]),
                    "evidence": (
                        str(merged.at[index, evidence_column])
                        if evidence_column in merged.columns
                        else ""
                    ),
                    "report": str(merged.at[index, "Report"])[:REPORT_EXCERPT],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "StudyInstanceUID",
            "target",
            "parser_said",
            "expert_said",
            "correct",
            "reason",
            "evidence",
            "report",
        ],
    )


def _accuracy(subset: pd.DataFrame) -> dict:
    cells = int(len(subset))
    wrong = int((~subset["correct"]).sum())
    rate = (cells - wrong) / cells if cells else 0.0
    return {
        "cells": cells,
        "wrong": wrong,
        "accuracy": rate,
        "error_rate": (wrong / cells) if cells else 0.0,
        # 58 studies sliced by rule leaves small buckets. Make that visible.
        "standard_error": math.sqrt(rate * (1.0 - rate) / cells) if cells else 0.0,
    }


def _grouped(cells: pd.DataFrame, column: str) -> dict:
    return {
        str(key): _accuracy(group)
        for key, group in sorted(
            cells.groupby(column), key=lambda item: -int((~item[1]["correct"]).sum())
        )
    }


def anatomy(
    *,
    train_csv: str | Path,
    b6_export: str | Path,
    min_confidence: float = MIN_CONFIDENCE,
    out_root: str | Path | None = None,
) -> dict:
    cells = cell_table(
        train_csv=train_csv, b6_export=b6_export, min_confidence=min_confidence
    )
    positives = cells.loc[cells["parser_said"] == STATE_POSITIVE]
    result = {
        "version": ANATOMY_VERSION,
        "b6_export": str(b6_export),
        "min_confidence": float(min_confidence),
        "overall": _accuracy(cells),
        "positive_calls": _accuracy(positives),
        "negated_calls": _accuracy(cells.loc[cells["parser_said"] == STATE_NEGATED]),
        "by_reason": _grouped(cells, "reason"),
        "by_reason_positive_calls_only": _grouped(positives, "reason"),
        "by_target": _grouped(cells, "target"),
    }

    if out_root is not None:
        out = Path(out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        wrong = cells.loc[~cells["correct"]].sort_values(["reason", "target"])
        wrong.to_csv(out / "disagreements.csv", index=False)
        cells.to_csv(out / "all_expert_cells.csv", index=False)
        result["out_root"] = str(out)
        result["disagreements_written"] = int(len(wrong))
    return result


def _line(name: str, item: dict) -> str:
    return (
        f"  {name:<38}{item['cells']:>6,}{item['wrong']:>7,}"
        f"{item['error_rate'] * 100:>8.1f}%   +/-{item['standard_error']:.3f}"
    )


def _report(result: dict) -> None:
    print()
    print(f"  {'':<38}{'cells':>6}{'wrong':>7}{'err':>9}")
    print(_line("every definite call", result["overall"]))
    print(_line("positive calls", result["positive_calls"]))
    print(_line("negated calls", result["negated_calls"]))

    print()
    print("  positive calls, by the rule that fired   (worst first)")
    for reason, item in result["by_reason_positive_calls_only"].items():
        print(_line(reason, item))

    print()
    print("  every call, by finding   (worst first)")
    for target, item in result["by_target"].items():
        print(_line(target, item))

    if "out_root" in result:
        print()
        print(
            f"  {result['disagreements_written']} disagreements written to "
            f"{result['out_root']}/disagreements.csv"
        )
        print(
            "  It carries raw report text. Local only -- do not commit it.\n"
            "\n"
            "  Read the evidence column. Each row is one of two things, and they\n"
            "  need opposite responses:\n"
            "    the clause does not assert the finding  -> the parser misread it\n"
            "    the clause asserts it and the expert disagrees -> a ceiling"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Split the parser's wrong calls into misreads and report/expert disagreements"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--b6-export",
        required=True,
        help="B6's own structured_labels.csv, the only one carrying __reason",
    )
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    _report(
        anatomy(
            train_csv=args.train_csv,
            b6_export=args.b6_export,
            min_confidence=args.min_confidence,
            out_root=args.out_root,
        )
    )


if __name__ == "__main__":
    main()
