"""Check the report parser against the expert labels, per finding and per state.

Training treats the parser's two definite answers very differently:

    "yes"  ->  target 0.85,  weight 0.50
    "no"   ->  target 0.05,  weight 1.00

A "no" therefore counts twice as much as a "yes". That is only sound if "no" is
at least as trustworthy, and nobody has checked. Published work on rule-based
report parsers points the other way: negation is usually their weakest skill,
far behind simply spotting that a finding was mentioned.

The 58 expert-labelled studies can settle it. They are far too small to rank
models -- one target's AUC there carries an error of about +/-0.16 -- but this
asks something much easier of them: when the parser says "no", how often is the
expert's answer also "no"? That is one proportion per finding, not a ranking.

Nothing here trains or selects anything. The expert labels stay out of the
gradient; they are read only to score the labels themselves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model._implementation import ensure_developments_source, read_config

ensure_developments_source()

MIN_CELLS_TO_REPORT = 3


def _audit_target(frame: pd.DataFrame, target: str) -> dict:
    """Score the parser's definite answers for one finding against the expert."""
    truth = frame[f"{target}__truth"].to_numpy()
    state = frame[f"{target}__state"].to_numpy()
    confidence = frame[f"{target}__confidence"].to_numpy(dtype=float)

    usable = confidence >= 0.75
    said_yes = usable & (state == "positive")
    said_no = usable & (state == "negated")

    def agreement(mask: np.ndarray, expected: float) -> tuple[int, int]:
        if not mask.any():
            return 0, 0
        return int((truth[mask] == expected).sum()), int(mask.sum())

    yes_right, yes_total = agreement(said_yes, 1.0)
    no_right, no_total = agreement(said_no, 0.0)

    silent = ~usable
    silent_positive = int((truth[silent] == 1.0).sum()) if silent.any() else 0

    return {
        "said_yes": yes_total,
        "yes_correct": yes_right,
        "yes_accuracy": (yes_right / yes_total) if yes_total else None,
        "said_no": no_total,
        "no_correct": no_right,
        "no_accuracy": (no_right / no_total) if no_total else None,
        "said_nothing": int(silent.sum()),
        "said_nothing_but_positive": silent_positive,
        "expert_positive_rate": float((truth == 1.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the report parser against experts")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="runs/label_audit.json")
    args = parser.parse_args()

    from rsna_knee.b6_report_labels import build_b6_frame
    from rsna_knee.constants import TARGETS
    from rsna_knee.data import gold_mask, load_train_csv

    config = read_config(args.config)
    root = Path(args.data_root).resolve()
    train = load_train_csv(root / config.get("train_csv", "train.csv"))

    gold = train.loc[gold_mask(train)].copy()
    print(f"expert-labelled studies: {len(gold)}")
    if gold[list(TARGETS)].isna().any().any():
        raise SystemExit("expected every expert study to carry all 12 labels")

    print("running the frozen parser on their reports...")
    parsed = build_b6_frame(gold)

    frame = pd.DataFrame({"StudyInstanceUID": gold["StudyInstanceUID"].astype(str).to_numpy()})
    for target in TARGETS:
        frame[f"{target}__truth"] = gold[target].to_numpy(dtype=float)
        frame[f"{target}__state"] = parsed[f"{target}__state"].to_numpy()
        frame[f"{target}__confidence"] = parsed[f"{target}__confidence"].to_numpy(dtype=float)

    results = {t: _audit_target(frame, t) for t in TARGETS}

    print()
    print(f"{'finding':18s} {'says yes':>18s} {'says no':>18s} {'silent':>8s}")
    print(f"{'':18s} {'n':>6s} {'right':>11s} {'n':>6s} {'right':>11s}")
    for target in TARGETS:
        r = results[target]
        yes = f"{r['yes_accuracy']*100:5.0f}%" if r["yes_accuracy"] is not None else "    -"
        no = f"{r['no_accuracy']*100:5.0f}%" if r["no_accuracy"] is not None else "    -"
        flag = ""
        if (r["no_accuracy"] is not None and r["yes_accuracy"] is not None
                and r["said_no"] >= MIN_CELLS_TO_REPORT
                and r["no_accuracy"] < r["yes_accuracy"]):
            flag = "  <- 'no' is the weaker answer"
        print(f"{target:18s} {r['said_yes']:6d} {yes:>11s} "
              f"{r['said_no']:6d} {no:>11s} {r['said_nothing']:8d}{flag}")

    def pooled(kind: str) -> tuple[int, int]:
        right = sum(results[t][f"{kind}_correct"] for t in TARGETS)
        total = sum(results[t][f"said_{kind}"] for t in TARGETS)
        return right, total

    yes_right, yes_total = pooled("yes")
    no_right, no_total = pooled("no")
    print()
    print("across all findings")
    if yes_total:
        print(f"   parser says yes: right {yes_right}/{yes_total} = {yes_right/yes_total*100:.1f}%")
    if no_total:
        print(f"   parser says no : right {no_right}/{no_total} = {no_right/no_total*100:.1f}%")

    verdict = None
    if yes_total and no_total:
        if no_right / no_total < yes_right / yes_total:
            verdict = (
                "the parser's 'no' is less reliable than its 'yes', yet training "
                "weights 'no' twice as heavily; the weights are pointed the wrong way"
            )
        else:
            verdict = (
                "the parser's 'no' is at least as reliable as its 'yes', so the "
                "heavier weight on negatives is defensible"
            )
        print(f"\n{verdict}")

    payload = {
        "expert_studies": int(len(gold)),
        "weights_in_training": {"positive": 0.50, "negated": 1.00},
        "per_target": results,
        "pooled": {
            "yes_correct": yes_right, "yes_total": yes_total,
            "no_correct": no_right, "no_total": no_total,
        },
        "verdict": verdict,
        "caution": (
            "58 studies is small; per-finding counts are often single digits. "
            "Read the pooled figure and the direction, not one finding's percentage."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
