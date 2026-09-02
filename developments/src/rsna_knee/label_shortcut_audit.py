"""What could predict a teacher label without looking at the anatomy.

Synovitis reaches 0.9954 on the run's 548-study validation surface. That is not
a model reading synovium. It is a subtle finding, the teacher answers only 723
positive cells for it across the whole population, and the same teacher is 28.6%
wrong about it against the experts. A near-perfect score on top of that
combination means something other than the joint lining is carrying the signal.

It matters beyond curiosity. Synovitis is a twelfth of the macro every run picks
its best epoch on, so an inflated twelfth partly steers epoch selection, and the
hidden test will not reward whatever is inflating it.

## Two things a model can read instead of anatomy

```text
the scanner        manufacturer, model, field strength, 3D or 2D, slice count
                   all of them visible in the pixels as noise, contrast and
                   field of view, and none of them anatomy

another finding    if Synovitis is labelled whenever Effusion is, then
                   predicting Effusion predicts Synovitis for free
```

Both are measured here the same way: how well one column alone ranks the
teacher's label, as an AUC. A column that reaches 0.9 is a shortcut. A column
at 0.55 is not.

## Why a permutation null, and not leave-one-out

Scoring a study by the positive rate of its own scanner group is circular: a
group of one scores perfectly, and many small groups score well on nothing.
Leave-one-out looks like the fix and is worse -- removing your own label from
the group rate makes a positive study score *below* its neighbours, so a
balanced group manufactures a perfect inverse ranking out of pure arithmetic.
The first version of this module did that and a test caught it.

So the rates are taken plainly, and the bias is measured instead of avoided.
The same statistic is recomputed on shuffled labels, many times, and the 95th
percentile of that null is the score a column of this shape reaches on noise
alone. A column counts as a shortcut only by clearing its own null.

## What a high number does and does not prove

It proves the label is predictable from that column on this population. It does
not prove the model used that route, and it does not prove the finding is
spurious -- some scanners genuinely serve populations with more disease, and a
site that images inflamed knees will have both more synovitis and its own
scanner. Confounding and shortcut look identical from here.

What it does establish is whether an explanation other than anatomy is
*available*. If none is, the 0.9954 needs a different account.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .report_labels import STATE_NEGATED, STATE_POSITIVE

AUDIT_VERSION = "label_shortcut_v2"

# Enough shuffles to place a 95th percentile without the run taking minutes.
DEFAULT_DRAWS = 100
DEFAULT_SEED = 2026

MIN_CONFIDENCE = 0.75

# Above this, a single column ranks the label about as well as the model does,
# and anatomy is no longer the only available explanation.
SHORTCUT_AUC = 0.80

# Columns that identify the study rather than describe it.
IDENTIFIER_COLUMNS = ("StudyInstanceUID", "SeriesInstanceUID")


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC, ties counting half. NaN when one class is absent."""
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if not positives.size or not negatives.size:
        return float("nan")
    # Averaging ranks within each tied group is what makes a tie contribute
    # half a pair rather than a whole one.
    ranks = pd.Series(np.concatenate([positives, negatives])).rank(method="average").to_numpy()
    positive_ranks = ranks[: positives.size].sum()
    return float(
        (positive_ranks - positives.size * (positives.size + 1) / 2.0)
        / (positives.size * negatives.size)
    )


def group_rates(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Score each study by the positive rate of its group, that study included.

    Circular on purpose. The permutation null below measures exactly how much
    that circularity is worth for this column's group sizes, and only the
    excess counts.
    """
    frame = pd.DataFrame({"label": labels, "group": pd.Series(groups).astype(str)})
    return frame.groupby("group")["label"].transform("mean").to_numpy(float)


def association(labels: np.ndarray, score: np.ndarray) -> float:
    """Direction-agnostic AUC: a column that ranks backwards predicts just as well."""
    value = auc(labels, score)
    return float("nan") if not np.isfinite(value) else max(value, 1.0 - value)


def permutation_ceiling(
    labels: np.ndarray,
    values: pd.Series,
    *,
    categorical: bool,
    draws: int,
    seed: int,
) -> float:
    """The association this column reaches on shuffled labels, at the 95th percentile."""
    if draws <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    reached = []
    for _ in range(int(draws)):
        rng.shuffle(shuffled)
        score = (
            group_rates(shuffled, values.to_numpy())
            if categorical
            else values.to_numpy(float)
        )
        value = association(shuffled, score)
        if np.isfinite(value):
            reached.append(value)
    return float(np.percentile(reached, 95)) if reached else float("nan")


def _prepare_column(values: pd.Series) -> tuple[pd.Series, bool] | None:
    """Return the column to score on and whether it is categorical, or None."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and numeric.nunique() > 2:
        return numeric, False
    filled = values.fillna("<missing>").astype(str)
    if filled.nunique() < 2:
        return None
    return filled, True


def _supervised(frame: pd.DataFrame, target: str, min_confidence: float) -> pd.Series:
    state = frame[f"{target}__state"].astype(str)
    confidence = pd.to_numeric(
        frame[f"{target}__confidence"], errors="coerce"
    ).fillna(0.0)
    return state.isin((STATE_POSITIVE, STATE_NEGATED)) & confidence.ge(min_confidence)


def audit(
    *,
    teacher: str | Path,
    study_table: str | Path | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    out_json: str | Path | None = None,
) -> dict:
    frame = _read_teacher(teacher)
    metadata = _read_study_table(study_table) if study_table is not None else None
    if metadata is not None:
        frame = frame.merge(metadata, on="StudyInstanceUID", how="left", validate="one_to_one")
        columns = [
            column
            for column in metadata.columns
            if column not in IDENTIFIER_COLUMNS
        ]
    else:
        columns = []

    results: dict[str, dict] = {}
    for target in TARGETS:
        usable = _supervised(frame, target, min_confidence)
        rows = frame.loc[usable]
        labels = rows[f"{target}__state"].eq(STATE_POSITIVE).to_numpy(float)
        item: dict = {
            "cells": int(usable.sum()),
            "positives": int(labels.sum()),
            "negatives": int(labels.size - labels.sum()),
        }
        if labels.size and 0 < labels.sum() < labels.size:
            item["metadata"] = _rank_columns(rows, labels, columns, draws, seed)
            item["sibling_targets"] = _rank_siblings(
                rows, labels, target, min_confidence, draws, seed
            )
            item["best"] = _best_of(item)
        results[target] = item

    return _finish(
        {
            "version": AUDIT_VERSION,
            "teacher": str(teacher),
            "study_table": None if study_table is None else str(study_table),
            "shortcut_threshold": SHORTCUT_AUC,
            "permutation_draws": int(draws),
            "targets": results,
        },
        out_json,
    )


def _entry(labels: np.ndarray, values: pd.Series, categorical: bool, draws: int, seed: int) -> dict | None:
    score = (
        group_rates(labels, values.to_numpy()) if categorical else values.to_numpy(float)
    )
    value = association(labels, score)
    if not np.isfinite(value):
        return None
    ceiling = permutation_ceiling(
        labels, values, categorical=categorical, draws=draws, seed=seed
    )
    return {
        "auc": value,
        "null_p95": ceiling,
        "excess": (value - ceiling) if np.isfinite(ceiling) else float("nan"),
        "clears_null": bool(np.isfinite(ceiling) and value > ceiling),
    }


def _rank_columns(
    rows: pd.DataFrame, labels: np.ndarray, columns: list[str], draws: int, seed: int
) -> dict:
    scored: dict[str, dict] = {}
    for column in columns:
        prepared = _prepare_column(rows[column])
        if prepared is None:
            continue
        values, categorical = prepared
        entry = _entry(labels, values, categorical, draws, seed)
        if entry is not None:
            scored[column] = entry
    return dict(sorted(scored.items(), key=lambda pair: -pair[1]["auc"]))


def _rank_siblings(
    rows: pd.DataFrame,
    labels: np.ndarray,
    target: str,
    min_confidence: float,
    draws: int,
    seed: int,
) -> dict:
    scored: dict[str, dict] = {}
    for other in TARGETS:
        if other == target:
            continue
        state = rows[f"{other}__state"].astype(str)
        confidence = pd.to_numeric(
            rows[f"{other}__confidence"], errors="coerce"
        ).fillna(0.0)
        committed = state.isin((STATE_POSITIVE, STATE_NEGATED)) & confidence.ge(min_confidence)
        # Silence is not a negative anywhere else in this project and is not one
        # here: an unanswered sibling cell scores between the two states.
        score = np.where(
            committed & state.eq(STATE_POSITIVE), 1.0, np.where(committed, 0.0, 0.5)
        )
        entry = _entry(labels, pd.Series(score), False, draws, seed)
        if entry is not None:
            scored[other] = entry
    return dict(sorted(scored.items(), key=lambda pair: -pair[1]["auc"]))


def _best_of(item: dict) -> dict:
    candidates = [
        ("metadata", name, entry)
        for name, entry in list(item.get("metadata", {}).items())[:1]
    ] + [
        ("sibling", name, entry)
        for name, entry in list(item.get("sibling_targets", {}).items())[:1]
    ]
    if not candidates:
        return {}
    kind, name, entry = max(candidates, key=lambda row: row[2]["auc"])
    return {
        "kind": kind,
        "column": name,
        **entry,
        # Both tests must pass: high enough to matter, and beyond what this
        # column's own group structure reaches on shuffled labels.
        "is_shortcut": bool(entry["auc"] >= SHORTCUT_AUC and entry["clears_null"]),
    }


def _read_teacher(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "structured_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher export: {path}")
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    missing = [
        column
        for target in TARGETS
        for column in (f"{target}__state", f"{target}__confidence")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing[:4])}")
    if "is_gold" in frame.columns:
        frame = frame.loc[~frame["is_gold"].astype(bool)].reset_index(drop=True)
    return frame


def _read_study_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "study_domain_table.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing per-study metadata: {path}. The domain audit writes "
            "study_domain_table.csv"
        )
    frame = pd.read_csv(path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError(f"{path} has no StudyInstanceUID column")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} lists a study more than once")
    return frame


def _finish(result: dict, out_json: str | Path | None) -> dict:
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _report(result: dict) -> None:
    rows = result["targets"]
    print()
    print(
        f"  {'target':<18}{'cells':>7}{'pos':>7}   "
        f"{'best predictor that is not anatomy':<38}{'AUC':>7}{'null':>7}{'excess':>8}"
    )
    for target, item in sorted(
        rows.items(), key=lambda pair: -pair[1].get("best", {}).get("excess", -9.0)
    ):
        best = item.get("best")
        if not best:
            print(f"  {target:<18}{item['cells']:>7}{item['positives']:>7}   (one class only)")
            continue
        flag = "  <-- shortcut" if best["is_shortcut"] else ""
        label = f"{best['kind']}: {best['column']}"
        print(
            f"  {target:<18}{item['cells']:>7}{item['positives']:>7}   "
            f"{label:<38}{best['auc']:>7.4f}{best['null_p95']:>7.3f}"
            f"{best['excess']:>+8.3f}{flag}"
        )

    flagged = [t for t, i in rows.items() if i.get("best", {}).get("is_shortcut")]
    print()
    if flagged:
        print(
            f"  {len(flagged)} target(s) at or above {result['shortcut_threshold']:.2f}: "
            f"{', '.join(sorted(flagged))}"
        )
    else:
        print(f"  Nothing reaches {result['shortcut_threshold']:.2f}.")
    print(
        f"\n  'null' is what this column reaches on shuffled labels at the 95th\n"
        f"  percentile, over {result['permutation_draws']} draws. A column with many small groups\n"
        "  scores well on noise alone, so only the excess above its own null counts."
    )
    print(
        "\n  A high number says an explanation other than anatomy is available,\n"
        "  not that the model took it, and not that the finding is spurious --\n"
        "  a site that images inflamed knees has both more synovitis and its own\n"
        "  scanner. Confounding and shortcut look identical from here."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Find what predicts a teacher label without reading the anatomy"
    )
    parser.add_argument("--teacher", required=True, help="a teacher structured_labels.csv")
    parser.add_argument(
        "--study-table",
        default=None,
        help="study_domain_table.csv from the domain audit, for scanner metadata",
    )
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help="label shuffles used to place each column's null ceiling",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    _report(
        audit(
            teacher=args.teacher,
            study_table=args.study_table,
            min_confidence=args.min_confidence,
            draws=args.draws,
            seed=args.seed,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()
