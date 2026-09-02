"""Is a target actually failing, or is 58 studies too few to tell?

ACL sits at 0.5478 and MCL at 0.5011 against expert truth, and the plan was to
treat that as the next experiment. Before spending a day of GPU on it, the
claim has to survive the question nobody has asked of it: **how many positives
does each target even have among the 58?**

An AUC is a rank statistic over positive-negative pairs. With eight positives
and fifty negatives it rests on 400 comparisons, and its standard error is
large enough that 0.55 and 0.70 are not distinguishable. Chasing a target that
is merely under-sampled would spend a training run learning nothing.

## What this computes

The Hanley-McNeil standard error, which is the right one for an AUC because it
accounts for the positive and negative counts separately rather than treating
the cells as independent draws:

```text
Q1 = A / (2 - A)                    probability two positives both outrank a negative
Q2 = 2A^2 / (1 + A)                 probability a positive outranks two negatives

SE = sqrt( [ A(1-A) + (np-1)(Q1 - A^2) + (nn-1)(Q2 - A^2) ] / (np * nn) )
```

From that, a 95% interval, and whether the target is distinguishable from 0.5
at all.

## The second surface

A target can be weak for two quite different reasons, and one number cannot
separate them:

```text
weak on BOTH the report surface and the expert surface
    -> the model genuinely cannot see this finding

strong on the report surface, weak on the expert surface
    -> the model learned the teacher faithfully, and the teacher disagrees
       with the expert about what this finding is
```

The second is what Contusion already looks like, and it is a ceiling rather
than a model defect. Passing `--history` reads the training run's own
per-target validation AUC -- measured on 548 report-labelled studies, so far
tighter than 58 -- and puts the two side by side.

## What it does not do

It fits nothing and selects nothing. It reads counts from `train.csv` and
numbers other runs already recorded. No threshold here touches training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv

DIAGNOSIS_VERSION = "target_diagnosis_v1"

# 1.96 sigma. The interval is normal-approximate, which is fine at these counts
# for saying "this does not exclude 0.5" and unreliable near 0 or 1.
Z95 = 1.959963985


def hanley_mcneil_se(auc: float, positives: int, negatives: int) -> float:
    """Standard error of an AUC given how many of each class it was measured on."""
    if positives <= 0 or negatives <= 0:
        return float("nan")
    a = float(auc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    variance = (
        a * (1.0 - a)
        + (positives - 1) * (q1 - a * a)
        + (negatives - 1) * (q2 - a * a)
    ) / (positives * negatives)
    return math.sqrt(max(variance, 0.0))


def expert_class_counts(train_csv: str | Path) -> dict[str, dict]:
    """Per target, how many of the 58 expert studies are positive and negative."""
    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train), TARGETS]
    if gold.empty:
        raise ValueError("train.csv contains no expert-labelled studies")

    counts: dict[str, dict] = {}
    for target in TARGETS:
        values = pd.to_numeric(gold[target], errors="coerce")
        defined = values.notna()
        positives = int((values.loc[defined] > 0.5).sum())
        negatives = int(defined.sum()) - positives
        counts[target] = {
            "positives": positives,
            "negatives": negatives,
            "labelled": int(defined.sum()),
            "pairs": positives * negatives,
        }
    return counts


def best_epoch_per_target(history_path: str | Path) -> dict[str, float]:
    """Per-target validation AUC from the epoch the run selected on."""
    path = Path(history_path)
    if path.is_dir():
        path = path / "history.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing training history: {path}")

    history = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in history if "validation_per_target_auc" in row]
    if not rows:
        raise ValueError(f"{path} records no validation_per_target_auc")

    def macro(row: dict) -> float:
        value = row.get("validation_macro_auc")
        if value is not None:
            return float(value)
        scores = [
            float(v)
            for v in row["validation_per_target_auc"].values()
            if v is not None and math.isfinite(float(v))
        ]
        return sum(scores) / len(scores) if scores else float("-inf")

    return dict(max(rows, key=macro)["validation_per_target_auc"])


def diagnose(
    *,
    train_csv: str | Path,
    expert_auc: dict[str, float] | None = None,
    history: str | Path | None = None,
    out_json: str | Path | None = None,
) -> dict:
    counts = expert_class_counts(train_csv)
    report_auc = best_epoch_per_target(history) if history is not None else {}

    rows: dict[str, dict] = {}
    for target in TARGETS:
        item = dict(counts[target])
        value = (expert_auc or {}).get(target)
        if value is not None:
            standard_error = hanley_mcneil_se(
                float(value), item["positives"], item["negatives"]
            )
            low = float(value) - Z95 * standard_error
            high = float(value) + Z95 * standard_error
            item.update(
                {
                    "expert_auc": float(value),
                    "standard_error": standard_error,
                    "ci_low": low,
                    "ci_high": high,
                    # The only honest verdict a 58-study surface supports.
                    "distinguishable_from_chance": bool(low > 0.5),
                }
            )
        if target in report_auc and report_auc[target] is not None:
            item["report_auc"] = float(report_auc[target])
            if value is not None:
                item["report_minus_expert"] = float(report_auc[target]) - float(value)
        rows[target] = item

    return _finish(
        {
            "version": DIAGNOSIS_VERSION,
            "expert_studies": max(item["labelled"] for item in counts.values()),
            "targets": rows,
        },
        out_json,
    )


def _finish(result: dict, out_json: str | Path | None) -> dict:
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _report(result: dict) -> None:
    rows = result["targets"]
    has_expert = any("expert_auc" in item for item in rows.values())
    has_report = any("report_auc" in item for item in rows.values())

    print()
    print(f"  the expert surface holds {result['expert_studies']} labelled studies")
    print()
    header = f"  {'target':<18}{'pos':>5}{'neg':>5}{'pairs':>7}"
    if has_expert:
        header += f"{'expert':>9}{'95% interval':>19}"
    if has_report:
        header += f"{'report':>9}{'gap':>8}"
    print(header)

    order = sorted(
        rows.items(), key=lambda pair: pair[1].get("expert_auc", pair[1]["positives"])
    )
    for target, item in order:
        line = (
            f"  {target:<18}{item['positives']:>5}{item['negatives']:>5}"
            f"{item['pairs']:>7}"
        )
        if has_expert and "expert_auc" in item:
            line += (
                f"{item['expert_auc']:>9.4f}"
                f"   [{item['ci_low']:.3f}, {item['ci_high']:.3f}]"
            )
        elif has_expert:
            line += " " * 28
        if has_report and "report_auc" in item:
            line += f"{item['report_auc']:>9.4f}"
            if "report_minus_expert" in item:
                line += f"{item['report_minus_expert']:>+8.3f}"
        print(line)

    if not has_expert:
        print()
        print("  Pass --expert-auc to weigh a measured AUC against these counts.")
        return

    resolved = [t for t, i in rows.items() if i.get("distinguishable_from_chance")]
    print()
    print(
        f"  Distinguishable from chance: {len(resolved)} of {len(rows)}"
        + (f"  ({', '.join(sorted(resolved))})" if resolved else "")
    )
    print(
        "\n  A target whose interval spans 0.5 is not evidence that the model is\n"
        "  blind to it. It is evidence that 58 studies cannot tell."
    )
    if has_report:
        print(
            "\n  Where the report AUC is high and the expert AUC is low, the model\n"
            "  learned its teacher and the teacher disagrees with the expert. That\n"
            "  is a ceiling, not a model defect, and no architecture change moves it."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Weigh each target's expert AUC against the counts it was measured on"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--expert-auc",
        default=None,
        help="JSON of target -> AUC, or an expert-audit summary containing one",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="a run's history.json, for per-target AUC on the report surface",
    )
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    expert = None
    if args.expert_auc is not None:
        payload = json.loads(Path(args.expert_auc).read_text(encoding="utf-8"))
        for key in ("per_target_auc", "per_target", "targets"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
        expert = {
            target: float(payload[target])
            for target in TARGETS
            if target in payload and payload[target] is not None
        }
        if not expert:
            raise ValueError(
                f"{args.expert_auc} carries no per-target AUC for these twelve targets"
            )

    _report(
        diagnose(
            train_csv=args.train_csv,
            expert_auc=expert,
            history=args.history,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()
