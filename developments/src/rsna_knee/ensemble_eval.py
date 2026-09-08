"""Measure what TTA is worth, and whether an ensemble of these models helps.

Two questions that share one answer path, because both are about combining
predictions and both must be measured before a submission slot is spent.

## The TTA question, and why it gates the other one

Every submission scores each study **three times**, at centre offsets
`(-1, 0, 1)`, and averages the probabilities. That costs three quarters of the
runtime budget:

```text
Kaggle ceiling                   9.00 h
guard                            8.25 h
one model, 3 offsets, ~1,300 studies, 2x T4    ~4.5 h
one model, 1 offset                            ~1.5 h
```

**Nobody here has measured what those two extra views buy.** B39 tried to widen
TTA to five offsets and died operationally before producing a score, so the
marginal worth of offset averaging is unknown in this repository.

It decides whether an ensemble is affordable at all:

```text
1 model  x 3 offsets   ~4.5 h    what runs today
3 models x 1 offset    ~4.5 h    the same cost, three independent models
```

## The ensemble question, and the reason it was blocked

There is no ensembling code in this repository, and until now there was nothing
worth ensembling: 086, 087 and B54 share a base checkpoint, an architecture, a
seed and a split, differing only in training population and two things measured
at zero. Rank-averaging near-identical models lands between them, not above.

B53 (augmentation) and B55 (physical crop, canonical laterality, 336) are the
first genuinely different checkpoints. Whether they are different *enough* is
measurable rather than arguable, so `member_agreement` reports it and the
report prints it next to the gain.

## Why ranks and not probabilities

The metric is macro ROC AUC, which depends only on the ordering of studies
within a target. Averaging raw probabilities lets a member with a different
calibration dominate one it merely disagrees with in scale; averaging **ranks**
removes calibration from the question entirely and leaves only the ordering the
metric reads. It also needs no weights to fit, so there is nothing to overfit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common_ruler_eval import score_one
from .b52_competition_training import macro_auc
from .loader_throughput import add_worker_argument

ENSEMBLE_VERSION = "ensemble_eval_v1"

#: What every submission currently pays for, from `B42_FAST_TTA_OFFSETS`.
SUBMISSION_TTA_OFFSETS = (-1, 0, 1)

#: Below this, three views are not buying their three-fold cost and the budget
#: is better spent on more models. Declared before the number is read.
TTA_WORTH_KEEPING = 0.005


def rank_normalise(probabilities: np.ndarray) -> np.ndarray:
    """Per-target ranks mapped onto [0, 1].

    Ties take their average rank, so two studies a member cannot separate stay
    unseparated rather than being ordered by array position.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected [studies, targets], got {values.shape}")
    studies = values.shape[0]
    if studies < 2:
        return np.zeros_like(values)

    ranked = np.empty_like(values)
    for column in range(values.shape[1]):
        order = values[:, column].argsort(kind="stable")
        positions = np.empty(studies, dtype=float)
        positions[order] = np.arange(studies, dtype=float)
        # Average the positions of equal values, so ties do not gain an order.
        unique, inverse = np.unique(values[:, column], return_inverse=True)
        if len(unique) < studies:
            sums = np.zeros(len(unique))
            counts = np.zeros(len(unique))
            np.add.at(sums, inverse, positions)
            np.add.at(counts, inverse, 1.0)
            positions = (sums / counts)[inverse]
        ranked[:, column] = positions / (studies - 1)
    return ranked


def rank_average(members: list[np.ndarray]) -> np.ndarray:
    """The ensemble prediction: mean of each member's per-target ranks."""
    if not members:
        raise ValueError("an ensemble needs at least one member")
    shapes = {np.asarray(m).shape for m in members}
    if len(shapes) != 1:
        raise ValueError(f"members disagree on shape: {sorted(shapes)}")
    return np.mean([rank_normalise(m) for m in members], axis=0)


def member_agreement(members: list[np.ndarray]) -> dict:
    """How much the members already say the same thing.

    An ensemble pays for disagreement. Members that rank studies identically
    average to themselves, so this is reported beside any gain rather than left
    for the gain to be explained by.
    """
    ranked = [rank_normalise(m) for m in members]
    pairs: list[float] = []
    for first in range(len(ranked)):
        for second in range(first + 1, len(ranked)):
            columns = [
                float(np.corrcoef(ranked[first][:, t], ranked[second][:, t])[0, 1])
                for t in range(ranked[first].shape[1])
                if np.std(ranked[first][:, t]) > 0 and np.std(ranked[second][:, t]) > 0
            ]
            if columns:
                pairs.append(float(np.mean(columns)))
    return {
        "pairs": len(pairs),
        "mean_rank_correlation": float(np.mean(pairs)) if pairs else float("nan"),
        "max_rank_correlation": float(np.max(pairs)) if pairs else float("nan"),
    }


def _macro(targets, weights, predictions) -> float:
    return float(macro_auc(targets, weights, predictions)["macro_auc"])


def measure_tta(checkpoint: str | Path, *, offsets=SUBMISSION_TTA_OFFSETS, **kwargs):
    """One offset against several, on the same studies with the same weights.

    Each offset is a separate single-view pass whose probabilities are then
    averaged -- which is exactly what the submission does, and needs no change
    to the model or the dataset to reproduce here.
    """
    views: dict[int, np.ndarray] = {}
    targets = weights = None
    for offset in offsets:
        result = score_one(
            checkpoint, center_offset=int(offset), with_predictions=True, **kwargs
        )
        views[int(offset)] = result["probabilities"]
        targets, weights = result["targets"], result["weights"]
        print(
            f"[tta] offset {offset:+d}  macro {result['macro_auc']:.6f}", flush=True
        )

    single = _macro(targets, weights, views[0])
    averaged = _macro(targets, weights, np.mean(list(views.values()), axis=0))
    delta = averaged - single
    return {
        "offsets": list(offsets),
        "single_offset_macro": single,
        "averaged_macro": averaged,
        "delta": delta,
        "cost_multiple": len(offsets),
        "worth_keeping": bool(delta >= TTA_WORTH_KEEPING),
        "threshold": TTA_WORTH_KEEPING,
        "per_offset_macro": {
            str(offset): _macro(targets, weights, view)
            for offset, view in views.items()
        },
    }


def measure_ensemble(checkpoints, **kwargs) -> dict:
    """Rank-average several checkpoints and compare against the best member."""
    members: list[np.ndarray] = []
    rows: list[dict] = []
    targets = weights = None

    for checkpoint in checkpoints:
        result = score_one(checkpoint, with_predictions=True, **kwargs)
        members.append(result["probabilities"])
        targets, weights = result["targets"], result["weights"]
        rows.append(
            {
                "checkpoint": result["checkpoint"],
                "experiment": result["experiment"],
                "macro_auc": result["macro_auc"],
            }
        )
        print(
            f"[ensemble] {result['experiment']} {result['macro_auc']:.6f}", flush=True
        )

    ensemble = _macro(targets, weights, rank_average(members))
    best = max(row["macro_auc"] for row in rows)
    return {
        "members": rows,
        "ensemble_macro": ensemble,
        "best_member_macro": best,
        "gain_over_best_member": ensemble - best,
        "agreement": member_agreement(members),
    }


def report(result: dict) -> None:
    """What a person reads before spending a submission slot."""
    if "offsets" in result:
        print()
        print(f"  TTA: {result['offsets']} against a single centre offset")
        for offset, macro in result["per_offset_macro"].items():
            print(f"    offset {offset:>3}          {macro:.6f}")
        print()
        print(f"    one offset            {result['single_offset_macro']:.6f}")
        print(f"    all averaged          {result['averaged_macro']:.6f}")
        print(f"    TTA is worth          {result['delta']:+.6f}")
        print(f"    for                   {result['cost_multiple']}x the runtime")
        print()
        print(
            f"  {'keep it' if result['worth_keeping'] else 'DROP IT'}: the threshold "
            f"declared before the number was {result['threshold']}"
        )
        if not result["worth_keeping"]:
            print(
                "  Dropping to one offset frees two thirds of the inference budget,\n"
                "  which is what makes a three-model ensemble affordable."
            )
        print()
        return

    print()
    print("  Ensemble, rank-averaged")
    for row in sorted(result["members"], key=lambda r: -r["macro_auc"]):
        print(f"    {str(row['experiment']):<28}{row['macro_auc']:.6f}")
    print()
    print(f"    best single member    {result['best_member_macro']:.6f}")
    print(f"    the ensemble          {result['ensemble_macro']:.6f}")
    print(f"    gain                  {result['gain_over_best_member']:+.6f}")
    print()
    agreement = result["agreement"]
    print(f"    mean rank correlation {agreement['mean_rank_correlation']:.4f}")
    print(f"    highest pair          {agreement['max_rank_correlation']:.4f}")
    print()
    print(
        "  An ensemble pays for disagreement. A correlation near 1 means the\n"
        "  members already say the same thing, and a small gain there is the\n"
        "  expected result rather than a disappointing one."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure what TTA is worth, and whether an ensemble helps"
    )
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--mode",
        choices=("tta", "ensemble"),
        required=True,
        help="tta takes one checkpoint; ensemble takes two or more",
    )
    parser.add_argument("--expected-supervision-cells", type=int, default=None)
    parser.add_argument("--out-json", default=None)
    add_worker_argument(parser)
    args = parser.parse_args()

    from .b52_competition_training import _read_config

    shared = dict(
        config=_read_config(args.config),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        surface_cells=args.expected_supervision_cells,
        num_workers=args.num_workers,
    )

    if args.mode == "tta":
        if len(args.checkpoint) != 1:
            raise SystemExit("--mode tta takes exactly one --checkpoint")
        result = measure_tta(args.checkpoint[0], **shared)
    else:
        if len(args.checkpoint) < 2:
            raise SystemExit("--mode ensemble needs at least two --checkpoint")
        result = measure_ensemble(args.checkpoint, **shared)

    result["version"] = ENSEMBLE_VERSION
    result["labels_root"] = str(Path(args.labels_root).resolve())
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report(result)


if __name__ == "__main__":
    main()


__all__ = [
    "ENSEMBLE_VERSION",
    "SUBMISSION_TTA_OFFSETS",
    "TTA_WORTH_KEEPING",
    "measure_ensemble",
    "measure_tta",
    "member_agreement",
    "rank_average",
    "rank_normalise",
    "report",
]
