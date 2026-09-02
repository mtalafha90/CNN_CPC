"""How many positive calls the parser makes on a clause that plainly negates.

Reading the parser's wrong calls on the 58 experts turned up one unambiguous
defect among a great many judgement calls:

```text
Baker's    parser positive    expert negative
           evidence: "baker cyst: none"
```

The report says the cyst is absent. The parser called it present. B6's negation
patterns handle prose negation -- "no", "without", "is not" -- but a structured
report writes its findings as a list, and `finding: none` negates by layout
rather than by grammar. Nothing in the frozen lexicon sees a colon.

That is a real bug rather than a disagreement about what counts as abnormal, and
it is the only one of its kind found so far. This measures how far it reaches.

## Why it is worth counting before anything else

The 58 expert studies show one instance. One instance is not a reason to touch a
frozen parser. But B6 records the clause behind every call, so the same shape
can be counted across all 4,407 studies without any expert labels at all -- and
that count is what says whether this is a stray or a systematic hole.

The distinction matters because the fix is not free: **B6 v1.2.1 is frozen**, and
every checkpoint records the supervision it was trained on. Changing the parser
means a new version, a new export, and a new teacher, which is a declared
experiment rather than an edit.

## What it does not measure

Whether the parser is *right* -- there are no labels here. A matching cell is
one where the evidence clause contains a list-style negation and the parser
still said positive. On the one case where expert truth exists, the parser was
wrong. On the rest, this reports the shape and nothing more.

The patterns are deliberately narrow and listed in the open below. A wide
pattern would find matches everywhere and prove nothing.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .constants import TARGETS
from .report_labels import STATE_POSITIVE

SCAN_VERSION = "negation_gap_v1"

# List-style negation: a finding, a colon or dash, then a word meaning absent.
#
# The vocabulary is the multilingual set B6 already carries for prose negation,
# so this adds no new language coverage -- only the layout the frozen patterns
# cannot see. Kept narrow on purpose: a loose pattern would match everywhere.
ABSENT_WORDS = (
    "none",
    "no",
    "nil",
    "absent",
    "negative",
    "not present",
    "normal",
    "intact",
    "unremarkable",
    "yok",  # Turkish
    "nema",  # South-Slavic
    "sin",  # Spanish
    "geen",  # Dutch
    "kein",  # German
)

LIST_NEGATION = re.compile(
    r"[:\-–—]\s*(?:" + "|".join(re.escape(word) for word in ABSENT_WORDS) + r")\b",
    re.IGNORECASE,
)


def _read_export(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "structured_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. This needs B6's own structured_labels.csv, the only "
            "export carrying __evidence"
        )
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    missing = [
        column
        for target in TARGETS
        for column in (f"{target}__state", f"{target}__evidence")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{path} is missing {', '.join(missing[:4])}. A merged export drops "
            "__evidence; point this at the B6 export itself"
        )
    return frame


def matches(clause: str) -> bool:
    """Whether a clause negates by layout rather than by grammar."""
    return bool(LIST_NEGATION.search(str(clause)))


def scan(*, b6_export: str | Path, out_root: str | Path | None = None) -> dict:
    """Count positive calls whose own evidence contains a list-style negation."""
    frame = _read_export(b6_export)
    is_gold = (
        frame["is_gold"].astype(bool)
        if "is_gold" in frame.columns
        else pd.Series(False, index=frame.index)
    )

    rows: list[dict] = []
    per_target: dict[str, dict] = {}
    for target in TARGETS:
        state = frame[f"{target}__state"].astype(str)
        evidence = frame[f"{target}__evidence"].fillna("").astype(str)
        positive = state.eq(STATE_POSITIVE)
        hit = positive & evidence.map(matches)
        per_target[target] = {
            "positive_calls": int(positive.sum()),
            "list_negated": int(hit.sum()),
            "share": float(hit.sum() / positive.sum()) if int(positive.sum()) else 0.0,
        }
        for index in frame.index[hit]:
            rows.append(
                {
                    "StudyInstanceUID": str(frame.at[index, "StudyInstanceUID"]),
                    "target": target,
                    "is_gold": bool(is_gold.at[index]),
                    "evidence": evidence.at[index],
                }
            )

    found = pd.DataFrame(rows, columns=["StudyInstanceUID", "target", "is_gold", "evidence"])
    positive_total = sum(item["positive_calls"] for item in per_target.values())
    result = {
        "version": SCAN_VERSION,
        "b6_export": str(b6_export),
        "studies": int(len(frame)),
        "positive_calls": positive_total,
        "list_negated_calls": int(len(found)),
        "share_of_positive_calls": (
            float(len(found) / positive_total) if positive_total else 0.0
        ),
        "studies_affected": int(found["StudyInstanceUID"].nunique()),
        "in_gold_studies": int(found["is_gold"].sum()),
        "patterns": list(ABSENT_WORDS),
        "by_target": per_target,
    }
    if out_root is not None:
        out = Path(out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        found.to_csv(out / "list_negated_positives.csv", index=False)
        result["out_root"] = str(out)
    return result


def _report(result: dict) -> None:
    print()
    print(f"  studies scanned              {result['studies']:>8,}")
    print(f"  positive calls               {result['positive_calls']:>8,}")
    print(
        f"  of those, list-negated       {result['list_negated_calls']:>8,}"
        f"   {result['share_of_positive_calls'] * 100:5.2f}%"
    )
    print(f"  studies affected             {result['studies_affected']:>8,}")
    print(f"  among the 58 expert studies  {result['in_gold_studies']:>8,}")

    print()
    print(f"  {'target':<20}{'positive':>10}{'negated':>9}{'share':>9}")
    for target, item in sorted(
        result["by_target"].items(), key=lambda pair: -pair[1]["list_negated"]
    ):
        if not item["list_negated"]:
            continue
        print(
            f"  {target:<20}{item['positive_calls']:>10,}{item['list_negated']:>9,}"
            f"{item['share'] * 100:>8.1f}%"
        )

    print()
    print(
        "  These are cells where the parser said positive and its own evidence\n"
        "  contains a list-style negation. There are no labels here: this is the\n"
        "  size of the shape, not proof that every one is wrong."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Count positive calls whose evidence negates by layout rather than grammar"
    )
    parser.add_argument(
        "--b6-export", required=True, help="B6's structured_labels.csv, which carries __evidence"
    )
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()
    _report(scan(b6_export=args.b6_export, out_root=args.out_root))


if __name__ == "__main__":
    main()
