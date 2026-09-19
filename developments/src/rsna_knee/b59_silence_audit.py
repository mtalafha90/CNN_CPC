"""B59 — is report silence a negative? Measured on the 58 experts, and only there.

## The question

Two thirds of the teacher's grid is answered and one third is blank. The blank
third is not missing data: it is the two states the loss discards.

```text
positive     -> target 0.85, weight 0.50    used
negated      -> target 0.05, weight 1.00    used
uncertain    -> weight 0.00                 discarded
unmentioned  -> weight 0.00                 discarded
```

Converting `unmentioned` to a negative would take coverage from 65.2% to
essentially 100% at no compute cost. Nothing else moves the number materially:
reading harder has about one point left in it, and the confidence column is a
constant, so lowering the 0.75 threshold does nothing at all.

## Why the standing ban is worth re-examining

`B6_STRUCTURED_REPORT_LABELS.md` and `RAISING_AUC.md` both say, in bold, not to
assume `unmentioned = negative`. That was written on 2026-08-12 and it is good
medicine: absence of evidence is not evidence of absence.

On 2026-09-08 this project established what the hidden labels actually are --
image-derived by two MSK radiologists under a severity rubric in which
**borderline findings are graded NEGATIVE**. The target is not "is anything
wrong with this structure"; it is "would two specialists call this definite and
severe". A finding the reporting radiologist did not think worth one sentence is
unlikely to clear that bar.

So the rule that blocks the conversion was written before the project knew what
it was aiming at. That does not make the conversion right. It makes it
**unmeasured**, which is what this module fixes.

## Why this is not the same bet that failed five times

Every coverage increase this project has tried added *positives*: the LLM fill's
cells are 27.1% wrong against B6's own 21.9%; both translation-rescue piles were
about 97% positive; B26 added Synovitis cells and Synovitis got worse.

The teacher's measured fault is entirely one-directional:

```text
111 wrong cells = 106 false positives + 5 false negatives
sensitivity 0.9768    specificity 0.4988
```

It almost never misses a positive. It calls a negative positive half the time.
Silence-to-negative is the only coverage increase that adds almost entirely
**negatives**, so it is aimed at the defect rather than feeding it. That is an
argument for measuring it, not a prediction that it works.

## The ruler

**Expert-58 only.** This module has no 548-study path and will not grow one.
The report-derived validation surface scores against the same label process that
produced the teacher, so it cannot referee a change to that process -- and
coverage predicts its AUC at Pearson `-0.931`, which means raising coverage
pushes that number *down* whether or not the model improves. Reading this
change on the report surface would report a real gain as a failure.

Expert-58 labels all twelve findings on all 58 studies whatever the report
happened to say, so it carries no mention-selection at all. It is small and it
has been looked at many times. It is still the only instrument here that points
at the thing being maximised.

## What this module is not for

It measures a *labeller*, never a model. Auditing a labeller against gold is
defensible where selecting a checkpoint against gold is not: the question here
is "does report silence indicate absence more often than chance", not "is model
A better than model B by 0.002". Do not use any number this writes to pick a
checkpoint, an epoch, a blend or a submission.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import (
    B7_MIN_CONFIDENCE,
    B7_NEGATIVE_TARGET,
    B7_NEGATIVE_WEIGHT,
)
from .constants import TARGETS
from .data import gold_mask, load_train_csv

B59_VERSION = "b59_silence_audit_v1"

#: The four states a report cell can carry. `positive` and `negated` already
#: train; the other two are what this audit is about.
STATES = ("positive", "negated", "uncertain", "unmentioned")

#: The states a policy may convert. `uncertain` is included so the two are
#: measured side by side -- it is a different claim from silence and gets its
#: own verdict, never silence's.
CONVERTIBLE_STATES = ("unmentioned", "uncertain")

#: Two-sided 95%.
WILSON_Z = 1.959963984540054

# --- the decision rule, frozen before the audit is run ------------------------
#
# Every threshold below is declared here, in code, so that a later reader can
# see it was not chosen after seeing the table. Nothing in this module reads a
# result to set a threshold, and none of these may be swept.

#: Fewest gold cells in a state before that target gets a verdict at all. With
#: 58 studies a target has at most 58 cells, so a rarely-silent finding can
#: reach a confident-looking proportion on a handful of rows. This project has
#: already made that mistake once, on 45 OA cells, and the correction is
#: recorded in `TWO_THIRDS_OF_THE_TEACHER_HAS_NO_EVIDENCE`.
SILENCE_MIN_GOLD_CELLS = 15

#: Absolute ceiling on the manufactured false-negative rate. Even a finding
#: where silence beats its own base rate is refused above this, because the
#: cells being added are asserted negatives and a wrong one is a false negative
#: in a teacher that currently has five.
SILENCE_MAX_POSITIVE_RATE = 0.20

#: What a converted cell is worth. It carries the same target as an explicit
#: negation and one quarter of its weight, so four silent cells are needed to
#: carry the influence of one sentence that actually says "no". One value,
#: declared in advance, never swept.
SILENCE_NEGATIVE_TARGET = B7_NEGATIVE_TARGET
SILENCE_NEGATIVE_WEIGHT = 0.25

VERDICT_CONVERT = "convert"
VERDICT_NO_EVIDENCE = "no_evidence"
VERDICT_NOT_INFORMATIVE = "refuse_not_informative"
VERDICT_TOO_MANY_FALSE_NEGATIVES = "refuse_too_many_false_negatives"


def wilson_interval(successes: int, n: int, *, z: float = WILSON_Z) -> tuple[float, float]:
    """A binomial interval that stays inside [0,1] at tiny n and zero counts.

    The normal approximation is useless here: several targets will have a
    handful of gold cells and some will have no positives at all, where
    `p +/- z*sqrt(p(1-p)/n)` gives a zero-width interval around zero and would
    wave through a conversion on no evidence whatsoever.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > n:
        raise ValueError(f"{successes} successes out of {n} is not a proportion")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def gold_surface(train: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """The 58 expert studies and their truth, refused if they are not all there.

    `gold_mask` selects rows with any target populated. A study that somehow
    carried a partial expert row would quietly contribute fewer cells to some
    targets than others, so the per-target `n` is reported rather than assumed.
    """
    gold = train.loc[gold_mask(train)]
    if gold.empty:
        raise ValueError(
            "no expert-labelled studies in this train.csv. B59 measures on the "
            "58 gold studies and has no other surface."
        )
    uids = [str(uid) for uid in gold["StudyInstanceUID"]]
    truth = gold[list(TARGETS)].to_numpy(dtype=np.float64)
    return uids, truth


def state_columns(structured: pd.DataFrame, uids: list[str]) -> pd.DataFrame:
    """The labeller's rows for those studies, in that order.

    The teacher's `structured_labels.csv` holds all 4,407 studies; only
    `training_targets.csv` drops the gold ones. If the gold rows are absent this
    is the wrong file, and saying so here is cheaper than a confusing KeyError
    twenty lines later.
    """
    frame = structured.copy()
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError("the labeller export has no StudyInstanceUID column")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    frame = frame.set_index("StudyInstanceUID")
    missing = [uid for uid in uids if uid not in frame.index]
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(uids)} expert studies are absent from this "
            "export. B59 needs `structured_labels.csv`, which carries every study; "
            "`training_targets.csv` has the gold rows removed by design."
        )
    return frame.loc[uids]


def _state_vector(frame: pd.DataFrame, target: str) -> np.ndarray:
    column = f"{target}__state"
    if column not in frame.columns:
        raise ValueError(f"the export has no {column!r} column")
    values = frame[column].fillna("").astype(str).str.strip().str.lower().to_numpy()
    unknown = sorted(set(values).difference(STATES).difference({""}))
    if unknown:
        raise ValueError(f"target {target!r} carries unknown states {unknown}")
    return values


def state_truth_table(truth: np.ndarray, states: pd.DataFrame) -> pd.DataFrame:
    """P(expert positive | state) per target and state, with Wilson intervals.

    `prevalence` on every row is that target's expert positive rate across all
    58 studies. It is the comparator the decision rule uses: silence is only
    worth converting where it is measurably *less* likely to be positive than a
    study drawn at random. A state whose rate merely equals the base rate
    carries no information about absence, and asserting a negative there is
    manufacturing noise.
    """
    rows = []
    for j, target in enumerate(TARGETS):
        column = _state_vector(states, target)
        y = truth[:, j]
        labelled = np.isfinite(y)
        prevalence = float(np.mean(y[labelled] == 1)) if labelled.any() else float("nan")
        for state in STATES:
            mask = (column == state) & labelled
            n = int(mask.sum())
            positives = int(np.sum(y[mask] == 1))
            low, high = wilson_interval(positives, n)
            rows.append(
                {
                    "target": target,
                    "state": state,
                    "n": n,
                    "gold_positive": positives,
                    "gold_negative": n - positives,
                    "p_gold_positive": float(positives / n) if n else float("nan"),
                    "wilson_low": low,
                    "wilson_high": high,
                    "prevalence": prevalence,
                    "gold_cells": int(labelled.sum()),
                }
            )
    return pd.DataFrame(rows)


def decide_target(row: pd.Series) -> tuple[str, str]:
    """The frozen rule, applied to one target's row for one state.

    Three clauses, in order, each answering a different way the conversion can
    be wrong:

    1. **Is there evidence at all?** Fewer than `SILENCE_MIN_GOLD_CELLS` cells
       cannot support a per-target decision on a 58-study surface.
    2. **Does the state say anything?** The Wilson upper bound must sit below
       the target's own prevalence. Note the asymmetry: an interval upper bound
       is compared against a point estimate, because prevalence is itself
       measured on these same 58 studies and giving it an interval too would
       compare two overlapping samples. This is a screen, not a proof.
    3. **Is the cost acceptable?** Even an informative state is refused above
       `SILENCE_MAX_POSITIVE_RATE`, because converted cells are asserted
       negatives and every wrong one is a manufactured false negative.
    """
    n = int(row["n"])
    if n < SILENCE_MIN_GOLD_CELLS:
        return (
            VERDICT_NO_EVIDENCE,
            f"{n} gold cells, fewer than the {SILENCE_MIN_GOLD_CELLS} this rule requires",
        )
    high = float(row["wilson_high"])
    prevalence = float(row["prevalence"])
    if not high < prevalence:
        return (
            VERDICT_NOT_INFORMATIVE,
            f"upper bound {high:.3f} is not below the {prevalence:.3f} base rate, so "
            "this state does not indicate absence for this finding",
        )
    if high > SILENCE_MAX_POSITIVE_RATE:
        return (
            VERDICT_TOO_MANY_FALSE_NEGATIVES,
            f"upper bound {high:.3f} exceeds the {SILENCE_MAX_POSITIVE_RATE:.2f} ceiling "
            "on manufactured false negatives",
        )
    return (
        VERDICT_CONVERT,
        f"{int(row['gold_positive'])}/{n} gold positive, upper bound {high:.3f} below "
        f"both the {prevalence:.3f} base rate and the {SILENCE_MAX_POSITIVE_RATE:.2f} ceiling",
    )


def training_cells(
    structured: pd.DataFrame,
    train: pd.DataFrame,
    *,
    min_confidence: float = B7_MIN_CONFIDENCE,
) -> pd.DataFrame:
    """How large the intervention would be, per target, on the studies that train.

    The verdict is decided on 58 studies; the effect lands on 4,349. Those are
    different numbers and a policy should not be adopted without both -- a
    target can pass the rule and then turn out to move eleven cells.
    """
    gold_uids = set(str(uid) for uid in train.loc[gold_mask(train), "StudyInstanceUID"])
    frame = structured.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    report_only = frame.loc[~frame["StudyInstanceUID"].isin(gold_uids)]
    if report_only.empty:
        raise ValueError("this export contains no report-only studies to convert")

    rows = []
    for target in TARGETS:
        column = _state_vector(report_only, target)
        confidence = pd.to_numeric(
            report_only.get(f"{target}__confidence"), errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        answered = np.isin(column, ("positive", "negated")) & (confidence >= min_confidence)
        entry = {
            "target": target,
            "report_only_studies": int(len(report_only)),
            "answered_now": int(answered.sum()),
        }
        for state in CONVERTIBLE_STATES:
            entry[f"{state}_cells"] = int(np.sum(column == state))
        rows.append(entry)
    return pd.DataFrame(rows)


def audit(
    train_csv: str | Path,
    structured_csv: str | Path,
    *,
    min_confidence: float = B7_MIN_CONFIDENCE,
) -> dict:
    """The whole measurement: evidence, verdicts, and the size of the change."""
    train = load_train_csv(train_csv)
    structured = pd.read_csv(structured_csv)
    uids, truth = gold_surface(train)
    states = state_columns(structured, uids)

    evidence = state_truth_table(truth, states)
    sizes = training_cells(structured, train, min_confidence=min_confidence)

    decisions = []
    for state in CONVERTIBLE_STATES:
        for _, row in evidence.loc[evidence["state"] == state].iterrows():
            verdict, reason = decide_target(row)
            size = sizes.loc[sizes["target"] == row["target"]].iloc[0]
            decisions.append(
                {
                    "target": row["target"],
                    "state": state,
                    "verdict": verdict,
                    "reason": reason,
                    "gold_cells": int(row["n"]),
                    "gold_positive": int(row["gold_positive"]),
                    "p_gold_positive": float(row["p_gold_positive"]),
                    "wilson_low": float(row["wilson_low"]),
                    "wilson_high": float(row["wilson_high"]),
                    "prevalence": float(row["prevalence"]),
                    "training_cells_added": int(size[f"{state}_cells"])
                    if verdict == VERDICT_CONVERT
                    else 0,
                    "training_cells_available": int(size[f"{state}_cells"]),
                }
            )
    decision_frame = pd.DataFrame(decisions)

    answered_now = int(sizes["answered_now"].sum())
    grid = int(sizes["report_only_studies"].iloc[0]) * len(TARGETS)
    added = int(decision_frame["training_cells_added"].sum())
    converted = decision_frame.loc[decision_frame["verdict"] == VERDICT_CONVERT]

    summary = {
        "version": B59_VERSION,
        "surface": "expert_58_only",
        "gold_studies": len(uids),
        "gold_cells": int(np.isfinite(truth).sum()),
        "min_confidence": float(min_confidence),
        "rule": {
            "min_gold_cells": SILENCE_MIN_GOLD_CELLS,
            "max_positive_rate": SILENCE_MAX_POSITIVE_RATE,
            "negative_target": SILENCE_NEGATIVE_TARGET,
            "negative_weight": SILENCE_NEGATIVE_WEIGHT,
            "explicit_negative_weight": B7_NEGATIVE_WEIGHT,
        },
        "coverage": {
            "grid_cells": grid,
            "answered_now": answered_now,
            "coverage_now": answered_now / grid if grid else float("nan"),
            "cells_added": added,
            "coverage_after": (answered_now + added) / grid if grid else float("nan"),
        },
        "verdicts": {
            verdict: int((decision_frame["verdict"] == verdict).sum())
            for verdict in (
                VERDICT_CONVERT,
                VERDICT_NO_EVIDENCE,
                VERDICT_NOT_INFORMATIVE,
                VERDICT_TOO_MANY_FALSE_NEGATIVES,
            )
        },
        "targets_converted": sorted(
            f"{row.target}/{row.state}" for row in converted.itertuples()
        ),
    }
    return {
        "summary": summary,
        "evidence": evidence,
        "decisions": decision_frame,
        "sizes": sizes,
    }


def policy_from(decisions: pd.DataFrame) -> dict:
    """The machine-readable outcome: which target/state pairs may be converted.

    Written whether or not anything passed. An empty policy is a result -- it
    says silence carries no usable information on this teacher -- and recording
    it stops the question being reopened from memory later.
    """
    allowed: dict[str, list[str]] = {}
    for row in decisions.loc[decisions["verdict"] == VERDICT_CONVERT].itertuples():
        allowed.setdefault(str(row.state), []).append(str(row.target))
    return {
        "version": B59_VERSION,
        "negative_target": SILENCE_NEGATIVE_TARGET,
        "negative_weight": SILENCE_NEGATIVE_WEIGHT,
        "convert": {state: sorted(targets) for state, targets in sorted(allowed.items())},
    }


def apply_policy(
    targets_array: np.ndarray,
    weights_array: np.ndarray,
    structured_rows: pd.DataFrame,
    policy: dict,
) -> dict:
    """Write the converted cells into an existing B7 target/weight pair.

    Applied in place, and only where the weight is currently zero. A cell the
    teacher already answers is never touched, so this cannot overwrite a parser
    call -- the same guarantee `base_cells_overridden: 0` gives the LLM merge,
    and worth as little: it protects reproducibility, not accuracy.
    """
    if targets_array.shape != weights_array.shape:
        raise ValueError("target and weight arrays must have the same shape")
    if targets_array.shape[1] != len(TARGETS):
        raise ValueError(f"expected {len(TARGETS)} target columns")
    if len(structured_rows) != targets_array.shape[0]:
        raise ValueError(
            f"{len(structured_rows)} label rows against {targets_array.shape[0]} "
            "supervision rows; these must be the same studies in the same order"
        )
    if policy.get("version") != B59_VERSION:
        raise ValueError(f"not a {B59_VERSION} policy: {policy.get('version')!r}")

    value = float(policy["negative_target"])
    weight = float(policy["negative_weight"])
    written = {}
    for state, names in dict(policy.get("convert") or {}).items():
        if state not in CONVERTIBLE_STATES:
            raise ValueError(f"policy names an unconvertible state {state!r}")
        for target in names:
            if target not in TARGETS:
                raise ValueError(f"policy names an unknown target {target!r}")
            j = TARGETS.index(target)
            column = _state_vector(structured_rows, target)
            mask = (column == state) & (weights_array[:, j] <= 0)
            targets_array[mask, j] = value
            weights_array[mask, j] = weight
            written[f"{target}/{state}"] = int(mask.sum())
    return {"cells_written": int(sum(written.values())), "by_target": written}


def main() -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-b59-silence-audit",
        description=(
            "Measure whether report silence indicates absence, on the 58 expert "
            "studies. Read-only: it writes tables and a policy, never labels."
        ),
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--structured-csv",
        required=True,
        help="the teacher's structured_labels.csv, which carries the gold rows",
    )
    parser.add_argument("--out-root", default="runs/b59_silence_audit")
    parser.add_argument("--min-confidence", type=float, default=B7_MIN_CONFIDENCE)
    args = parser.parse_args()

    result = audit(
        args.train_csv, args.structured_csv, min_confidence=args.min_confidence
    )
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    result["evidence"].to_csv(out / "state_truth_by_target.csv", index=False)
    result["decisions"].to_csv(out / "silence_decision.csv", index=False)
    result["sizes"].to_csv(out / "training_cells.csv", index=False)
    (out / "silence_audit.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8"
    )
    policy = policy_from(result["decisions"])
    (out / "silence_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = result["summary"]
    coverage = summary["coverage"]
    print(f"[B59] {summary['gold_studies']} expert studies, {summary['gold_cells']} cells")
    for state in CONVERTIBLE_STATES:
        rows = result["decisions"].loc[result["decisions"]["state"] == state]
        print(f"\n[B59] {state}")
        for row in rows.itertuples():
            print(
                f"  {row.target:<18} n={row.gold_cells:<3} "
                f"p={row.p_gold_positive:.3f} "
                f"[{row.wilson_low:.3f},{row.wilson_high:.3f}] "
                f"base={row.prevalence:.3f}  {row.verdict}"
            )
    print(
        f"\n[B59] coverage {coverage['coverage_now']:.3f} -> {coverage['coverage_after']:.3f} "
        f"({coverage['cells_added']:,} cells added)"
    )
    print(f"[B59] wrote {out}")


if __name__ == "__main__":
    main()


__all__ = [
    "B59_VERSION",
    "CONVERTIBLE_STATES",
    "SILENCE_MAX_POSITIVE_RATE",
    "SILENCE_MIN_GOLD_CELLS",
    "SILENCE_NEGATIVE_TARGET",
    "SILENCE_NEGATIVE_WEIGHT",
    "STATES",
    "apply_policy",
    "audit",
    "decide_target",
    "gold_surface",
    "policy_from",
    "state_truth_table",
    "training_cells",
    "wilson_interval",
]
