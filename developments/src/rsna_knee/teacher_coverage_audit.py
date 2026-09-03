"""Where a teacher is still silent, and how much the translation rescue would fill.

Two separate repairs were built on top of frozen B6 v1.2.1 and were never
combined:

```text
B6 v1.2.1  --+--> B23 fill (LLM answers cells B6 left silent)      <- trained on
             |
             +--> Phase 7/8 (translate the report, re-run B6)      <- measured,
                                                                      never used
```

The fill merge is what the current runs train on. The translation rescue is a
completed, frozen audit that recovered 3,901 cells from 1,053 of the 1,229
studies B6 could not read at all. Nothing has ever measured what one leaves for
the other to do.

That is the question here, and it has to be asked before anything is merged,
because the answer decides whether the merge is worth doing. The LLM filler ran
over the whole report-only population, including those 1,229 studies. If it
already reached most of them, the rescue adds little and the cost is a fresh
teacher nobody can audit. If it did not, the rescue is the only thing that
touches them.

## The two kinds of cell it would add

A rescued cell is reported under one of two headings, because they are not the
same decision:

```text
study is still completely silent in this export
    -> filling it is exactly the frozen Phase-8 policy

study was silent under B6 but the LLM has since answered part of it
    -> filling the rest is a NEW policy, never frozen and never measured
```

Phase 6 froze "no translated cell may enter a B6-active study". These studies
were not B6-active; they became active through the filler. Whether that clause
reaches them is a judgement, not a lookup, so this audit refuses to make it and
reports the two totals apart.

## What this cannot tell you

Not one of the 58 expert studies is in the 1,229 -- they are gold, and the
rescue population is report-only by construction. So the expert audit score of
any rescued teacher is **identical** to the score without it, to the last
decimal. This change cannot be judged on that surface. Only the hidden test, or
a matched training pair, can judge it, and the one matched pair that has been
run (Phase 9 v2) came back inconclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .constants import TARGETS

AUDIT_VERSION = "teacher_coverage_v2"

# What B6 writes into __evidence when the legacy OA path decides a cell:
# a marker naming the rule, not a quotation from the report.
OA_CONTEXT_MARKER = "oa_context_parser"

COMMITTED_STATES = ("positive", "negated")
MIN_CONFIDENCE = 0.75


def _committed(frame: pd.DataFrame, target: str, *, min_confidence: float) -> pd.Series:
    """Cells this teacher stands behind, by the same rule every merge step uses."""
    state = frame[f"{target}__state"].astype(str)
    confidence = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").fillna(0.0)
    return state.isin(COMMITTED_STATES) & confidence.ge(min_confidence)


def read_teacher(path: str | Path) -> pd.DataFrame:
    """Read an export directory or a single labels CSV.

    `structured_labels.csv` carries the 58 gold studies, `training_targets.csv`
    does not. Both are accepted; gold rows are counted separately rather than
    dropped, because a coverage number that quietly mixes them is misleading in
    the direction that flatters the teacher.
    """
    path = Path(path)
    if path.is_dir():
        for name in ("structured_labels.csv", "training_targets.csv"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            raise FileNotFoundError(
                f"{path} holds neither structured_labels.csv nor training_targets.csv"
            )
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher export: {path}")

    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} lists a study more than once")
    missing = [
        column
        for target in TARGETS
        for column in (f"{target}__confidence", f"{target}__state")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing[:6])}")
    return frame


def read_recovered_cells(phase7_root: str | Path) -> pd.DataFrame:
    """Read the frozen Phase-7 recovered cells, long form: one row per cell."""
    path = Path(phase7_root)
    if path.is_dir():
        path = path / "recovered_cells.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path.parent} exists but holds no recovered_cells.csv; "
                "it is not a Phase-7 rescue directory"
            )
    elif not path.exists():
        raise FileNotFoundError(
            f"nothing at {path}. Phase 7 completed, so its output is somewhere: "
            "find it with  find . -name recovered_cells.csv"
        )
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase-7 recovered cells: {path}")

    recovered = pd.read_csv(path)
    required = {"StudyInstanceUID", "target", "state"}
    absent = required.difference(recovered.columns)
    if absent:
        raise ValueError(f"{path} is missing columns: {sorted(absent)}")
    recovered["StudyInstanceUID"] = recovered["StudyInstanceUID"].astype(str)
    recovered["target"] = recovered["target"].astype(str)
    if recovered.duplicated(["StudyInstanceUID", "target"]).any():
        raise ValueError(f"{path} names the same study and target twice")
    unknown = sorted(set(recovered["target"]).difference(TARGETS))
    if unknown:
        raise ValueError(f"{path} contains unknown target(s): {unknown}")
    return recovered


def provenance(
    teacher: pd.DataFrame,
    parser: pd.DataFrame,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict:
    """For every cell the teacher answers, who decided it and what backs it.

    Three kinds of answered cell, and they are not equally accountable:

    ```text
    parser, quoted     the rule matched text, and the clause is recorded
    parser, unquoted   the legacy OA path decided it and wrote a rule name
                       where a quotation would go
    filled             the parser was silent, so the filler answered and there
                       is no clause to record
    ```

    Attribution is exact rather than inferred: the merge preserves every parser
    call and writes only where the parser was silent, so a cell the parser
    committed is the parser's and any other answered cell is the filler's.
    """
    is_gold = (
        teacher["is_gold"].astype(bool)
        if "is_gold" in teacher.columns
        else pd.Series(False, index=teacher.index)
    )
    rows = teacher.loc[~is_gold].set_index("StudyInstanceUID")
    parser_rows = parser.set_index("StudyInstanceUID").reindex(rows.index)

    per_target: dict[str, dict] = {}
    reasons: dict[str, int] = {}
    for target in TARGETS:
        answered = _committed(rows, target, min_confidence=min_confidence)
        by_parser = _committed(parser_rows, target, min_confidence=min_confidence).fillna(False)

        evidence = (
            parser_rows[f"{target}__evidence"].fillna("").astype(str)
            if f"{target}__evidence" in parser_rows.columns
            else pd.Series("", index=rows.index)
        )
        quoted = evidence.str.strip().ne("") & evidence.ne(OA_CONTEXT_MARKER)

        from_parser = answered & by_parser
        per_target[target] = {
            "answered": int(answered.sum()),
            "parser_quoted": int((from_parser & quoted).sum()),
            "parser_unquoted": int((from_parser & ~quoted).sum()),
            "filled": int((answered & ~by_parser).sum()),
        }
        if f"{target}__reason" in parser_rows.columns:
            reason = parser_rows[f"{target}__reason"].fillna("").astype(str)
            for name, count in reason.loc[from_parser & ~quoted].value_counts().items():
                reasons[str(name)] = reasons.get(str(name), 0) + int(count)

    total = {
        key: sum(item[key] for item in per_target.values())
        for key in ("answered", "parser_quoted", "parser_unquoted", "filled")
    }
    answered = total["answered"]
    return {
        **total,
        "no_clause": total["parser_unquoted"] + total["filled"],
        "no_clause_fraction": (
            (total["parser_unquoted"] + total["filled"]) / answered if answered else 0.0
        ),
        "unquoted_parser_reasons": dict(
            sorted(reasons.items(), key=lambda pair: -pair[1])
        ),
        "per_target": per_target,
        "min_confidence": float(min_confidence),
    }


def coverage(frame: pd.DataFrame, *, min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Count what this teacher answers, per study and per target."""
    is_gold = (
        frame["is_gold"].astype(bool)
        if "is_gold" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    report_only = frame.loc[~is_gold]

    per_target = {
        target: int(_committed(report_only, target, min_confidence=min_confidence).sum())
        for target in TARGETS
    }
    answered = sum(
        _committed(report_only, target, min_confidence=min_confidence).astype(int)
        for target in TARGETS
    )
    possible = len(report_only) * len(TARGETS)
    return {
        "studies": int(len(report_only)),
        "gold_studies_carried": int(is_gold.sum()),
        "possible_cells": int(possible),
        "answered_cells": int(answered.sum()),
        "answered_fraction": (float(answered.sum()) / possible) if possible else 0.0,
        "active_studies": int((answered > 0).sum()),
        "silent_studies": int((answered == 0).sum()),
        "per_target": per_target,
        "min_confidence": float(min_confidence),
    }


def rescue_headroom(
    frame: pd.DataFrame,
    recovered: pd.DataFrame,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict:
    """How many Phase-7 cells land where this teacher is still silent.

    Split by whether the study is silent altogether or only partly, because the
    frozen policy covers the first and says nothing about the second.
    """
    is_gold = (
        frame["is_gold"].astype(bool)
        if "is_gold" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    report_only = frame.loc[~is_gold].set_index("StudyInstanceUID")

    answered_per_study = sum(
        _committed(report_only, target, min_confidence=min_confidence).astype(int)
        for target in TARGETS
    )
    silent_study = set(answered_per_study.index[answered_per_study == 0])

    known = recovered["StudyInstanceUID"].isin(report_only.index)
    absent_uids = sorted(set(recovered.loc[~known, "StudyInstanceUID"]))
    usable = recovered.loc[known]

    rows = []
    for row in usable.itertuples(index=False):
        uid, target = str(row.StudyInstanceUID), str(row.target)
        state = str(report_only.at[uid, f"{target}__state"])
        confidence = pd.to_numeric(
            pd.Series([report_only.at[uid, f"{target}__confidence"]]), errors="coerce"
        ).fillna(0.0).iloc[0]
        already = state in COMMITTED_STATES and confidence >= min_confidence
        rows.append(
            {
                "StudyInstanceUID": uid,
                "target": target,
                "state": str(row.state),
                "cell_already_answered": bool(already),
                "study_wholly_silent": uid in silent_study,
            }
        )
    landing = pd.DataFrame(
        rows,
        columns=[
            "StudyInstanceUID",
            "target",
            "state",
            "cell_already_answered",
            "study_wholly_silent",
        ],
    )
    new = landing.loc[~landing["cell_already_answered"]]
    wholly = new.loc[new["study_wholly_silent"]]
    partly = new.loc[~new["study_wholly_silent"]]

    def _split(subset: pd.DataFrame) -> dict:
        return {
            "cells": int(len(subset)),
            "positive": int((subset["state"] == "positive").sum()),
            "negated": int((subset["state"] == "negated").sum()),
            "studies": int(subset["StudyInstanceUID"].nunique()),
            # Which findings the cells pile into, because the aggregate hides
            # the risk. B26 fell over on Synovitis alone.
            "per_target": {
                target: {
                    "positive": int(
                        ((subset["target"] == target) & (subset["state"] == "positive")).sum()
                    ),
                    "negated": int(
                        ((subset["target"] == target) & (subset["state"] == "negated")).sum()
                    ),
                }
                for target in TARGETS
            },
        }

    return {
        "recovered_cells_offered": int(len(recovered)),
        "recovered_studies_offered": int(recovered["StudyInstanceUID"].nunique()),
        "studies_absent_from_teacher": absent_uids[:5],
        "studies_absent_count": len(absent_uids),
        "cells_the_teacher_already_answers": int(landing["cell_already_answered"].sum()),
        "frozen_policy": _split(wholly),
        "new_policy": _split(partly),
        "silent_studies_that_would_become_active": int(
            wholly["StudyInstanceUID"].nunique()
        ),
        "silent_studies_left_after_rescue": int(
            len(silent_study) - wholly["StudyInstanceUID"].nunique()
        ),
    }


def _report(result: dict) -> None:
    before = result["coverage"]
    print()
    print("What the teacher answers now")
    print(f"  report-only studies        {before['studies']:>8,}")
    print(f"  cells it could answer      {before['possible_cells']:>8,}")
    print(
        f"  cells it does answer       {before['answered_cells']:>8,}"
        f"   {before['answered_fraction'] * 100:5.1f}%"
    )
    print(f"  studies with any answer    {before['active_studies']:>8,}")
    print(f"  studies with none at all   {before['silent_studies']:>8,}")
    if before["gold_studies_carried"]:
        print(
            f"  gold studies carried       {before['gold_studies_carried']:>8,}"
            "   (excluded from every count above)"
        )

    kinds = result.get("provenance")
    if kinds:
        print()
        print("  Where each answered cell came from, and what backs it")
        print(
            f"    parser, clause recorded    {kinds['parser_quoted']:>8,}"
            f"{kinds['parser_quoted'] / kinds['answered'] * 100:>7.1f}%"
        )
        print(
            f"    parser, no clause         {kinds['parser_unquoted']:>8,}"
            f"{kinds['parser_unquoted'] / kinds['answered'] * 100:>7.1f}%"
        )
        print(
            f"    filled, no clause exists  {kinds['filled']:>8,}"
            f"{kinds['filled'] / kinds['answered'] * 100:>7.1f}%"
        )
        print(
            f"    -------------------------------------------\n"
            f"    no clause at all          {kinds['no_clause']:>8,}"
            f"{kinds['no_clause_fraction'] * 100:>7.1f}%"
        )
        if kinds["unquoted_parser_reasons"]:
            print()
            print("    the parser's unquoted calls, by rule")
            for name, count in kinds["unquoted_parser_reasons"].items():
                print(f"      {name:<38}{count:>8,}")
        print()
        print(f"  {'target':<20}{'answered':>10}{'quoted':>9}{'unquoted':>10}{'filled':>9}")
        for target, item in sorted(
            kinds["per_target"].items(), key=lambda pair: -pair[1]["filled"]
        ):
            print(
                f"  {target:<20}{item['answered']:>10,}{item['parser_quoted']:>9,}"
                f"{item['parser_unquoted']:>10,}{item['filled']:>9,}"
            )
        print(
            "\n  A filled cell has no clause by construction: the filler was asked\n"
            "  precisely because the parser found nothing. An unquoted parser cell\n"
            "  is different -- the parser committed without recording why."
        )

    headroom = result.get("rescue_headroom")
    if not headroom:
        print()
        print("Pass --phase7-root to see what the translation rescue would add.")
        return

    frozen, fresh = headroom["frozen_policy"], headroom["new_policy"]
    print()
    print("What the translation rescue would add on top")
    print(f"  cells it offers            {headroom['recovered_cells_offered']:>8,}")
    print(
        f"  already answered here      "
        f"{headroom['cells_the_teacher_already_answers']:>8,}   (no change)"
    )
    print()
    print("  into studies that are still wholly silent   -- the frozen Phase-8 policy")
    print(
        f"    cells   {frozen['cells']:>6,}"
        f"   positive {frozen['positive']:>5,}   negated {frozen['negated']:>5,}"
        f"   studies {frozen['studies']:>5,}"
    )
    print("  into studies the filler already reached      -- a NEW, unmeasured policy")
    print(
        f"    cells   {fresh['cells']:>6,}"
        f"   positive {fresh['positive']:>5,}   negated {fresh['negated']:>5,}"
        f"   studies {fresh['studies']:>5,}"
    )
    print()
    print(
        f"  studies with nothing at all: {before['silent_studies']:,}"
        f" -> {headroom['silent_studies_left_after_rescue']:,}"
    )
    print()
    print("  where the cells land, by finding      frozen pile      new pile")
    print(f"  {'':<18}{'pos':>7}{'neg':>7}   {'pos':>7}{'neg':>7}")
    for target in TARGETS:
        a, b = frozen["per_target"][target], fresh["per_target"][target]
        if not (a["positive"] or a["negated"] or b["positive"] or b["negated"]):
            continue
        print(
            f"  {target:<18}{a['positive']:>7,}{a['negated']:>7,}   "
            f"{b['positive']:>7,}{b['negated']:>7,}"
        )
    if headroom["studies_absent_count"]:
        print(
            f"  WARNING {headroom['studies_absent_count']:,} rescued studies are not "
            "in this export at all"
        )


def audit(
    *,
    export: str | Path,
    phase7_root: str | Path | None = None,
    b6_export: str | Path | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    out_json: str | Path | None = None,
) -> dict:
    frame = read_teacher(export)
    result = {
        "version": AUDIT_VERSION,
        "export": str(export),
        "coverage": coverage(frame, min_confidence=min_confidence),
    }
    if b6_export is not None:
        result["b6_export"] = str(b6_export)
        result["provenance"] = provenance(
            frame, read_teacher(b6_export), min_confidence=min_confidence
        )
    if phase7_root is not None:
        recovered = read_recovered_cells(phase7_root)
        result["phase7_root"] = str(phase7_root)
        result["rescue_headroom"] = rescue_headroom(
            frame, recovered, min_confidence=min_confidence
        )
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        "Measure a teacher's coverage, and what the translation rescue would add"
    )
    parser.add_argument(
        "--export", help="teacher export directory or labels CSV"
    )
    parser.add_argument(
        "--phase7-root",
        default=None,
        help="Phase-7 rescue directory, or its recovered_cells.csv",
    )
    # Every other audit in this family takes --teacher. This one took --export
    # alone, which is a needless way to lose a command; both are accepted and
    # --export stays because a runbook already records it.
    parser.add_argument(
        "--teacher",
        dest="export",
        help="a teacher structured_labels.csv; the same thing as --export",
    )
    parser.add_argument(
        "--b6-export",
        default=None,
        help=(
            "B6's structured_labels.csv, to say which cells the parser decided "
            "and which of those recorded a clause"
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    if not args.export:
        parser.error("one of --teacher or --export is required")

    result = audit(
        export=args.export,
        phase7_root=args.phase7_root,
        b6_export=args.b6_export,
        min_confidence=args.min_confidence,
        out_json=args.out_json,
    )
    _report(result)


if __name__ == "__main__":
    main()
