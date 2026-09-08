"""Rebuild a teacher export with the competition's severity rubric applied.

`severity_rubric` decides one cell at a time. This is the step that walks a
real export, applies it, and writes a new labels root the trainer can consume
unchanged.

## The wide table, and the mistake it invites

`training_targets.csv` is one row per study with three columns per target:

```text
<target>                the label the model trains on   1.0 / 0.0 / blank
<target>__state         positive / negated / uncertain
<target>__confidence
```

`severity_rubric` moves the **state**. If only the state moved, the run would
train on an unchanged `1.0` while its audit claimed the cell was negative --
a change that looks applied, changes nothing, and is exactly the shape of the
B52 augmentation flag that set fields nobody read.

So the value column is rewritten with the state, and a test asserts the two can
never disagree after a rebuild.

## What it does not touch

Gold rows, if any are carried for audit, are excluded from
`training_targets.csv` at the write step exactly as the merge does -- the 58
expert studies stay out of gradients. Confidence is left alone: the rubric
changes what the answer is, not how sure the reader was of the text.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .constants import TARGETS
from .report_labels import STATE_NEGATED, STATE_POSITIVE
from .severity_rubric import (
    DOWNGRADE_STATE,
    SEVERITY_VERSION,
    apply_rubric,
    write_audit,
)

B55_TEACHER_VERSION = "b55_rubric_teacher_v1"

#: The label value each committed state trains as.
STATE_VALUE = {STATE_POSITIVE: 1.0, STATE_NEGATED: 0.0}


def teacher_anchors() -> dict[str, tuple[str, ...]]:
    """The phrases that mean each finding, taken from the teacher's own lists.

    `LEXICON` carries multilingual literal phrases for all twelve targets;
    `V13_PATTERNS` adds regular expressions for the three OA compartments,
    which is where v1.3 did its work. Both are used rather than a third
    vocabulary invented here -- the gate has to look where the teacher looked,
    or it grades cells the teacher never made.
    """
    from .b6_v13_report_labels import V13_PATTERNS
    from .report_labels import LEXICON

    return {
        target: tuple(LEXICON.get(target, ())) + tuple(V13_PATTERNS.get(target, ()))
        for target in TARGETS
    }


def melt_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Wide export to one row per (study, target), which the rubric expects."""
    rows = []
    for _, row in frame.iterrows():
        for target in TARGETS:
            rows.append(
                {
                    "StudyInstanceUID": str(row["StudyInstanceUID"]),
                    "target": target,
                    "state": str(row.get(f"{target}__state", "")),
                }
            )
    return pd.DataFrame(rows)


def write_states(frame: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Put the decided states back, and move the label value with them.

    A state without its value is a change that does not reach training.
    """
    out = frame.copy()
    lookup = {
        (row["StudyInstanceUID"], row["target"]): row["state"]
        for _, row in states.iterrows()
    }
    for index, row in out.iterrows():
        uid = str(row["StudyInstanceUID"])
        for target in TARGETS:
            state = lookup.get((uid, target))
            if state is None:
                continue
            out.at[index, f"{target}__state"] = state
            # Every state, not only the committed ones. An uncertain cell has
            # no training value, and leaving the old 1.0 behind would be the
            # exact state-and-value disagreement this module exists to prevent
            # -- reappearing in the branch that was made configurable.
            out.at[index, target] = STATE_VALUE.get(state, float("nan"))
    return out


def rebuild(
    labels_root: str | Path,
    reports: dict[str, str],
    anchors: dict[str, tuple[str, ...]],
    out_root: str | Path,
    *,
    downgrade_to: str = DOWNGRADE_STATE,
) -> dict:
    """Apply the rubric to an export and write a new one beside it.

    The source is never modified. `policy.json` and `audit.json` are carried
    across so `load_fill_merged_export` still accepts the result, with the
    rubric recorded as an added section rather than by rewriting their claims:
    the fill merge's own certifications are still true of these rows.
    """
    source = Path(labels_root)
    destination = Path(out_root)
    destination.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source / "training_targets.csv")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)

    states = melt_states(frame)
    graded, audit = apply_rubric(
        states, reports, anchors, downgrade_to=downgrade_to
    )
    rebuilt = write_states(frame, graded)

    rebuilt.to_csv(destination / "training_targets.csv", index=False)
    for name in ("policy.json", "audit.json"):
        shutil.copyfile(source / name, destination / name)

    # The per-cell evidence goes to the CSV, not into audit.json where several
    # thousand rows would bury the summary a person actually reads.
    changes = audit.pop("changes", [])

    export_audit = json.loads((destination / "audit.json").read_text(encoding="utf-8"))
    export_audit["severity_rubric"] = {
        "version": SEVERITY_VERSION,
        "teacher_version": B55_TEACHER_VERSION,
        "source_labels_root": str(source.resolve()),
        **audit,
    }
    (destination / "audit.json").write_text(
        json.dumps(export_audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_audit(audit, destination / "rubric_changes.csv", changes)
    return audit


def report(audit: dict) -> None:
    """What a person reads before deciding to train on this."""
    print()
    print(f"  positive cells before      {audit['positive_cells_before']:,}")
    print(f"  downgraded to {audit['downgrade_state']:<14}{audit['cells_downgraded']:,}")
    print(
        f"  fraction of positives      "
        f"{audit['fraction_of_positives_downgraded']:.1%}"
    )
    print()
    print("  by target")
    for target, count in audit["by_target"].items():
        print(f"    {target:<20}{count:>6,}")
    print()
    if audit["targets_never_gated"]:
        print(f"  never gated: {', '.join(audit['targets_never_gated'])}")
    print()
    print(
        "  Read rubric_changes.csv before training on this. Each row carries the\n"
        "  sentence that caused it. A large fraction is either the rubric working\n"
        "  or the vocabulary over-firing, and only the sentences tell you which."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the competition's severity rubric to a teacher export"
    )
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--reports-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--downgrade-to",
        default=DOWNGRADE_STATE,
        help="what a sub-threshold positive becomes; the rubric says negated",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.reports_csv)
    column = next(
        (c for c in ("report", "report_text", "text") if c in frame.columns), None
    )
    if column is None:
        raise ValueError(
            f"{args.reports_csv} has no report text column "
            f"(looked for report, report_text, text)"
        )
    reports = {
        str(uid): str(text)
        for uid, text in zip(frame["StudyInstanceUID"], frame[column])
        if isinstance(text, str)
    }
    print(f"[B55 teacher] {len(reports):,} reports", flush=True)

    anchors = teacher_anchors()
    thin = [t for t, terms in anchors.items() if len(terms) < 2]
    if thin:
        raise ValueError(
            f"these targets have almost no anchor vocabulary: {thin}. The gate "
            "would silently never fire for them."
        )
    audit = rebuild(
        args.labels_root,
        reports,
        anchors,
        args.out_root,
        downgrade_to=args.downgrade_to,
    )
    report(audit)


if __name__ == "__main__":
    main()


__all__ = [
    "B55_TEACHER_VERSION",
    "STATE_VALUE",
    "melt_states",
    "rebuild",
    "report",
    "write_states",
]
