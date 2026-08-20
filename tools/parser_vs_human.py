"""Compare the frozen parser with a person reading the same reports.

`tools.label_audit` scores the parser against the 58 expert-annotated studies
and reports that a parser "yes" matches the expert 69% of the time. That number
mixes two different failures and cannot separate them:

    the parser misread the report
    the report itself disagreed with the images

Only the first is fixable by changing the parser, and the audit cannot say how
much of the 31% is which.

Giving the parser and a person the *same reports* removes the second failure
entirely. Both read identical text, so every disagreement is a parsing mistake
and nothing else. That makes this the direct measurement of parser quality, and
the audit the measurement of the whole label pipeline.

The hand labels are read from their own file rather than from `train.csv`.
Filling target columns into `train.csv` turns those studies into gold ones,
which removes them from the training population and silently adds them to the
58-study validation surface -- so the numbers stop meaning what they did.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model._implementation import ensure_developments_source, read_config

ensure_developments_source()

# Below this the parser is treated as having said nothing, matching training.
MIN_CONFIDENCE = 0.75


def read_hand_labels(path: str | Path, targets) -> pd.DataFrame:
    """Load a hand-labelled sheet: a UID column plus any of the 12 findings."""
    frame = pd.read_csv(path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError(f"{path} needs a StudyInstanceUID column")
    present = [t for t in targets if t in frame.columns]
    if not present:
        raise ValueError(
            f"{path} carries none of the 12 finding columns; expected some of: "
            f"{', '.join(targets)}"
        )
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} lists the same study more than once")

    labelled = frame[present].notna().any(axis=1)
    if not labelled.any():
        raise ValueError(f"{path} has finding columns but every cell is empty")
    return frame.loc[labelled, ["StudyInstanceUID", *present]].reset_index(drop=True)


def compare(human: pd.DataFrame, parsed: pd.DataFrame, targets) -> dict:
    """Score the parser's definite answers against the person's, per finding."""
    rows: list[dict] = []
    disagreements: list[dict] = []

    for target in targets:
        if target not in human.columns:
            continue
        truth = human[target].to_numpy(dtype=float)
        state = parsed[f"{target}__state"].to_numpy()
        confidence = parsed[f"{target}__confidence"].to_numpy(dtype=float)

        known = ~np.isnan(truth)
        usable = known & (confidence >= MIN_CONFIDENCE)
        said_yes = usable & (state == "positive")
        said_no = usable & (state == "negated")
        # Silence is every cell that produced no definite answer, not only the
        # low-confidence ones: `uncertain` and `unmentioned` supervise nothing
        # either, however confidently the parser reached them. Defining it as
        # "not usable" would drop those cells out of the accounting entirely.
        silent = known & ~(said_yes | said_no)

        row = {
            "target": target,
            "hand_labelled": int(known.sum()),
            "said_yes": int(said_yes.sum()),
            "yes_agreed": int((truth[said_yes] == 1.0).sum()) if said_yes.any() else 0,
            "said_no": int(said_no.sum()),
            "no_agreed": int((truth[said_no] == 0.0).sum()) if said_no.any() else 0,
            "said_nothing": int(silent.sum()),
            "missed_positives": (
                int((truth[silent] == 1.0).sum()) if silent.any() else 0
            ),
        }
        row["yes_agreement"] = (
            row["yes_agreed"] / row["said_yes"] if row["said_yes"] else None
        )
        row["no_agreement"] = (
            row["no_agreed"] / row["said_no"] if row["said_no"] else None
        )
        rows.append(row)

        for index in np.where(said_yes & (truth != 1.0))[0]:
            disagreements.append(
                {
                    "study": human["StudyInstanceUID"].iloc[index],
                    "target": target,
                    "parser": "yes",
                    "person": "no",
                }
            )
        for index in np.where(said_no & (truth != 0.0))[0]:
            disagreements.append(
                {
                    "study": human["StudyInstanceUID"].iloc[index],
                    "target": target,
                    "parser": "no",
                    "person": "yes",
                }
            )

    said_yes = sum(r["said_yes"] for r in rows)
    said_no = sum(r["said_no"] for r in rows)
    silent = sum(r["said_nothing"] for r in rows)
    missed = sum(r["missed_positives"] for r in rows)
    return {
        "studies": int(len(human)),
        "min_confidence": MIN_CONFIDENCE,
        "overall": {
            "said_yes": said_yes,
            "yes_agreement": (
                sum(r["yes_agreed"] for r in rows) / said_yes if said_yes else None
            ),
            "said_no": said_no,
            "no_agreement": (
                sum(r["no_agreed"] for r in rows) / said_no if said_no else None
            ),
            "said_nothing": silent,
            "missed_positives": missed,
            "recall_of_positives_stated_in_reports": (
                sum(r["yes_agreed"] for r in rows)
                / (sum(r["yes_agreed"] for r in rows) + missed)
                if (sum(r["yes_agreed"] for r in rows) + missed)
                else None
            ),
        },
        "per_target": rows,
        "disagreements": disagreements,
    }


def _percent(value) -> str:
    return "     -" if value is None else f"{value * 100:5.1f}%"


def _report(result: dict) -> None:
    overall = result["overall"]
    print(f"\n{result['studies']} hand-labelled studies, parser against the person\n")
    print(f"  parser said yes     {overall['said_yes']:5d}   "
          f"person agreed {_percent(overall['yes_agreement'])}")
    print(f"  parser said no      {overall['said_no']:5d}   "
          f"person agreed {_percent(overall['no_agreement'])}")
    print(f"  parser said nothing {overall['said_nothing']:5d}   "
          f"of which the person found {overall['missed_positives']} positive")
    if overall["recall_of_positives_stated_in_reports"] is not None:
        print(
            "\n  of the positives a person could read in these reports, the parser "
            f"caught {_percent(overall['recall_of_positives_stated_in_reports']).strip()}"
        )

    print("\n  per finding:")
    print(f"    {'finding':<18}{'yes':>6}{'agree':>8}{'no':>6}{'agree':>8}{'silent':>8}{'missed':>8}")
    for row in result["per_target"]:
        print(
            f"    {row['target']:<18}{row['said_yes']:>6}{_percent(row['yes_agreement']):>8}"
            f"{row['said_no']:>6}{_percent(row['no_agreement']):>8}"
            f"{row['said_nothing']:>8}{row['missed_positives']:>8}"
        )

    if result["disagreements"]:
        print(f"\n  {len(result['disagreements'])} outright disagreement(s), "
              "where both read the same words and drew opposite conclusions:")
        for item in result["disagreements"][:20]:
            print(
                f"    {item['target']:<18} parser said {item['parser']:<3} "
                f"person said {item['person']:<3} {item['study'][:40]}..."
            )
        if len(result["disagreements"]) > 20:
            print(f"    ... {len(result['disagreements']) - 20} more, see the JSON")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the frozen parser against a person reading the same reports"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--hand-labels",
        required=True,
        help="CSV of your own labels: StudyInstanceUID plus any of the 12 findings",
    )
    parser.add_argument("--out", default="runs/parser_vs_human.json")
    args = parser.parse_args()

    from rsna_knee.b6_report_labels import build_b6_frame
    from rsna_knee.constants import TARGETS
    from rsna_knee.data import load_train_csv

    config = read_config(args.config)
    root = Path(args.data_root).resolve()
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

    human = read_hand_labels(args.hand_labels, TARGETS)
    known = set(train["StudyInstanceUID"])
    missing = [u for u in human["StudyInstanceUID"] if u not in known]
    if missing:
        raise SystemExit(
            f"{len(missing)} hand-labelled study/studies are not in train.csv, "
            f"first: {missing[0]}"
        )

    subset = (
        train.set_index("StudyInstanceUID")
        .loc[human["StudyInstanceUID"]]
        .reset_index()
    )
    print(f"running the frozen parser on {len(subset)} report(s)...")
    parsed = build_b6_frame(subset).reset_index(drop=True)

    result = compare(human, parsed, TARGETS)
    _report(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
