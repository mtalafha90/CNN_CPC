"""One study, end to end: its report, its labels, and where each label came from.

Every number in this project is an aggregate. 18.2% wrong, 0.678 macro, 34,010
cells. None of them lets anyone look at a single knee and ask the obvious
question: **does this label actually describe this case?**

The evidence that labels are joined to the right studies is already strong, and
it is indirect. If reports were attached to the wrong study the teacher would
agree with the experts at chance rather than 81.8%; if images were attached to
the wrong labels the model would score 0.5 on the hidden test rather than 0.716.
Neither of those can be seen by looking, and neither tells you anything about
the case in front of you.

This prints one card:

```text
which series the study has, and what they are
each of the twelve findings:
    what the teacher says
    what the expert says, when the study is one of the 58
    which labeller decided it -- the frozen parser or the LLM
    the clause the parser matched, when it was the parser
the report text the whole thing was derived from
```

Read down the card and the join is visible: the clause quoted beside ACL either
does talk about the cruciate ligament in that report, or it does not.

## What "from" can and cannot tell you

A cell shows `parser` when it carries no `__model_confidence`, because the
regular expression has no self-report and the LLM always records one. That makes
the column exact for a merged export and blank for a raw parser export, where
every cell is the parser's by construction.

It does not attribute the *evidence*: the clause comes from the B6 export, so a
cell the LLM filled has no clause to show. That is correct rather than missing —
the LLM was asked the question precisely because the parser found nothing.

## This prints patient data

The report text and the study identifier both. Local only.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv

CARD_VERSION = "study_card_v1"

MIN_CONFIDENCE = 0.75
COMMITTED = ("positive", "negated")

# Enough of the clause to judge it, short enough to sit in a column.
CLAUSE_WIDTH = 96


def _read(path: str | Path, name: str) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "structured_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    return frame.set_index("StudyInstanceUID")


def _row(frame: pd.DataFrame, study: str, name: str) -> pd.Series | None:
    if frame is None:
        return None
    if study not in frame.index:
        raise KeyError(f"{name} has no row for study {study}")
    return frame.loc[study]


def card(
    *,
    data_root: str | Path,
    study: str,
    teacher: str | Path,
    b6_export: str | Path | None = None,
    split: str = "train",
    min_confidence: float = MIN_CONFIDENCE,
) -> dict:
    """Assemble everything known about one study, from every source at once."""
    root = Path(data_root)
    train = load_train_csv(root / "train.csv")
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    is_gold = dict(zip(train["StudyInstanceUID"], gold_mask(train)))
    if str(study) not in is_gold:
        raise KeyError(f"train.csv has no study {study}")

    study = str(study)
    truth_row = train.loc[train["StudyInstanceUID"].eq(study)].iloc[0]
    teacher_row = _row(_read(teacher, "teacher export"), study, "the teacher export")
    parser_row = (
        _row(_read(b6_export, "B6 export"), study, "the B6 export")
        if b6_export is not None
        else None
    )

    findings = []
    for target in TARGETS:
        state = str(teacher_row.get(f"{target}__state", ""))
        confidence = pd.to_numeric(
            pd.Series([teacher_row.get(f"{target}__confidence")]), errors="coerce"
        ).fillna(0.0).iloc[0]
        answered = state in COMMITTED and confidence >= min_confidence

        model_confidence = teacher_row.get(f"{target}__model_confidence")
        has_column = f"{target}__model_confidence" in teacher_row.index
        source = ""
        if answered and has_column:
            source = "parser" if pd.isna(model_confidence) else "LLM"
        elif answered:
            source = "parser"

        expert = pd.to_numeric(pd.Series([truth_row.get(target)]), errors="coerce").iloc[0]
        findings.append(
            {
                "target": target,
                "teacher": state if answered else "-",
                "expert": (
                    "-" if pd.isna(expert) else ("positive" if float(expert) else "negative")
                ),
                "agrees": (
                    None
                    if not answered or pd.isna(expert)
                    else bool((state == "positive") == bool(float(expert)))
                ),
                "from": source,
                "evidence": (
                    str(parser_row.get(f"{target}__evidence", "") or "")
                    if parser_row is not None
                    else ""
                ),
                "reason": (
                    str(parser_row.get(f"{target}__reason", "") or "")
                    if parser_row is not None
                    else ""
                ),
            }
        )

    return {
        "version": CARD_VERSION,
        "study": study,
        "is_gold": bool(is_gold[study]),
        "report": str(truth_row.get("Report", "") or ""),
        "series": _series_of(root, study, split),
        "findings": findings,
        "answered": sum(1 for item in findings if item["teacher"] != "-"),
        "disagreements": sum(1 for item in findings if item["agrees"] is False),
    }


def _series_of(root: Path, study: str, split: str) -> list[dict]:
    """The study's series and what the CSV says each one is."""
    path = root / "train_series.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    rows = frame.loc[frame["StudyInstanceUID"].eq(study)]
    found = []
    for row in rows.itertuples(index=False):
        name = str(row.SeriesInstanceUID)
        folder = root / f"{split}_series" / study / name
        found.append(
            {
                "series": name,
                "plane": str(getattr(row, "Anatomical_Plane", "") or "?"),
                "fluid_sensitive": str(getattr(row, "Fluid_Sensitive", "") or "?"),
                "slices": (
                    sum(1 for p in folder.iterdir() if p.is_file())
                    if folder.is_dir()
                    else 0
                ),
            }
        )
    return found


def _report(result: dict) -> None:
    print()
    print(f"  study   {result['study']}")
    print(
        f"  labels  {'EXPERT-LABELLED (one of the 58)' if result['is_gold'] else 'report-derived only'}"
    )
    print()
    for item in result["series"]:
        print(
            f"    {item['series'][-24:]:<26}{item['plane']:<12}"
            f"fluid={item['fluid_sensitive']:<8}{item['slices']:>4} slices"
        )
    if not result["series"]:
        print("    (no series rows found)")

    print()
    header = f"  {'finding':<18}{'teacher':<11}{'expert':<11}{'from':<9}evidence the parser matched"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for item in result["findings"]:
        mark = {True: " ", False: "!", None: " "}[item["agrees"]]
        clause = item["evidence"][:CLAUSE_WIDTH]
        print(
            f" {mark}{item['target']:<18}{item['teacher']:<11}{item['expert']:<11}"
            f"{item['from']:<9}{clause}"
        )

    print()
    print(
        f"  {result['answered']} of 12 findings answered"
        + (
            f", {result['disagreements']} disagree with the expert  (marked !)"
            if result["is_gold"]
            else ""
        )
    )
    print()
    print("  report")
    for line in result["report"].splitlines() or [""]:
        print(textwrap.fill(line, 92, initial_indent="    ", subsequent_indent="    "))
    print()
    print("  Patient data. Local only -- do not commit it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Show one study's report, labels, and where each label came from"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--study", required=True, help="StudyInstanceUID")
    parser.add_argument("--teacher", required=True, help="a teacher structured_labels.csv")
    parser.add_argument(
        "--b6-export",
        default=None,
        help="B6's structured_labels.csv, to show the clause behind each parser call",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    args = parser.parse_args()

    _report(
        card(
            data_root=args.data_root,
            study=args.study,
            teacher=args.teacher,
            b6_export=args.b6_export,
            split=args.split,
            min_confidence=args.min_confidence,
        )
    )


if __name__ == "__main__":
    main()
