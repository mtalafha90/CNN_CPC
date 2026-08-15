"""Audit class balance in a weak-supervision surface, before any training.

B25X diagnosed the campaign's one durable supervision defect: B6 supplied 322
positive and 13 negative Synovitis cells, a 96.1% one-sided training surface.
A head trained on that ranked worse than chance -- weak-v2 AUC `0.2370` --
under the current recipe. Filling the missing
negatives moved it to `0.9123`, and that single target accounted for 96.4% of
the entire 12-target macro gain. Across the other eleven targets the same
supervision change was worth `+0.0024`, i.e. nothing.

## Why this module exists

The obvious next move -- "fill Synovitis" -- looks like target-wise selection
chosen from a weak-v2 result, which the repository rightly prohibits. This
audit is what replaces that with a general rule.

It reads **training-label counts only**: no model, no prediction, no
evaluation surface. A target failing the test is identifiable before anything
is trained, and the rule applies uniformly to all twelve rather than naming
one.

**What this does and does not fix.** The thresholds below were chosen after
the B25X Synovitis diagnosis, so they do not retroactively make B25X a
prospective experiment -- that result stays exploratory. What the rule buys is
forward-looking: from here on, targets are selected for fill by a stated
policy rather than by their weak-v2 rank.

The result happens not to depend on where the thresholds sit. On the full
frozen B6 surface both criteria are sharply bimodal:

    majority share    Synovitis 95.9%, next-worst MCL 80.1%
    minority cells    Synovitis    17, next-fewest Fracture 203

so any threshold inside a 15.8-point band, or anywhere between 18 and 203
cells, selects exactly the same single target. Synovitis is not a marginal
call that a chosen cut-off produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS

# Beyond this share of a single class the target is treated as having
# insufficient minority-class support to rank reliably under the current
# recipe. Frozen from here forward; do not tune against a future outcome.
DEFAULT_IMBALANCE_THRESHOLD = 0.90
# Below this many minority-class cells the estimate is too fragile to trust
# even when the share looks acceptable.
DEFAULT_MIN_MINORITY_CELLS = 30


def balance_table(targets: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """Per-target positive/negative counts among cells that carry supervision."""
    targets = np.asarray(targets)
    weights = np.asarray(weights)
    if targets.shape != weights.shape:
        raise ValueError("targets and weights must align")
    if targets.shape[1] != len(TARGETS):
        raise ValueError(f"expected {len(TARGETS)} target columns")

    rows = []
    for j, target in enumerate(TARGETS):
        used = weights[:, j] > 0
        positive = int(np.sum(used & (targets[:, j] > 0.5)))
        negative = int(np.sum(used & (targets[:, j] < 0.5)))
        total = positive + negative
        majority = max(positive, negative)
        rows.append(
            {
                "target": target,
                "usable_cells": total,
                "positive": positive,
                "negative": negative,
                "majority_share": float(majority / total) if total else float("nan"),
                "minority_cells": min(positive, negative),
                "minority_class": "negative" if negative < positive else "positive",
            }
        )
    return pd.DataFrame(rows)


def flag_imbalanced(
    table: pd.DataFrame,
    *,
    threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    min_minority: int = DEFAULT_MIN_MINORITY_CELLS,
) -> pd.DataFrame:
    """Mark targets with insufficient minority-class support.

    Deliberately not called "unlearnable": B25X showed a practical failure
    under one specific recipe, not an impossibility result for binary
    classification. The claim is that the training surface gives the head too
    little of one class to rank reliably, which is weaker and defensible.
    """
    out = table.copy()
    out["fails_balance"] = out["majority_share"] >= float(threshold)
    out["fails_minority_count"] = out["minority_cells"] < int(min_minority)
    out["needs_fill"] = out["fails_balance"] | out["fails_minority_count"]
    return out


def audit_supervision_balance(
    config: dict,
    export_root: str | Path,
    *,
    labeller: str = "b6",
    threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    min_minority: int = DEFAULT_MIN_MINORITY_CELLS,
) -> dict:
    """Run the balance audit against a frozen supervision export."""
    from .b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
    from .data import load_train_csv

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if labeller == "b6":
        frame, _policy, _audit = load_frozen_b6_export(export_root)
    elif labeller == "b23":
        from .b23_llm_labels import load_frozen_b23_export

        frame, _policy, _audit = load_frozen_b23_export(export_root)
    else:
        raise ValueError("labeller must be 'b6' or 'b23'")

    _uids, y, w, summary = prepare_b7_supervision(train, frame)
    table = flag_imbalanced(
        balance_table(y, w), threshold=threshold, min_minority=min_minority
    )
    flagged = table.loc[table["needs_fill"], "target"].tolist()
    return {
        "labeller": labeller,
        "export_root": str(export_root),
        "threshold": float(threshold),
        "min_minority_cells": int(min_minority),
        "rule": (
            "computed from training-label counts only; no model, no prediction "
            "and no evaluation surface consulted. Thresholds were set after the "
            "B25X diagnosis, so they are binding forward rather than making any "
            "earlier result prospective."
        ),
        "targets_needing_fill": flagged,
        "n_targets_needing_fill": len(flagged),
        "total_usable_cells": int(summary.get("usable_cells", table["usable_cells"].sum())),
        "table": table.to_dict(orient="records"),
    }


def format_balance(payload: dict) -> str:
    table = pd.DataFrame(payload["table"])
    lines = [
        f"Supervision balance audit ({payload['labeller']})",
        f"  threshold {payload['threshold']:.0%} one class"
        f" | minimum {payload['min_minority_cells']} minority cells",
        "",
        f"  {'target':18} {'usable':>7} {'pos':>7} {'neg':>7} {'major':>7}  flag",
    ]
    for row in table.sort_values("majority_share", ascending=False).to_dict("records"):
        flag = "NEEDS FILL" if row["needs_fill"] else ""
        share = row["majority_share"]
        share_s = "  n/a" if not np.isfinite(share) else f"{share:6.1%}"
        lines.append(
            f"  {row['target']:18} {row['usable_cells']:7d} {row['positive']:7d} "
            f"{row['negative']:7d} {share_s}  {flag}"
        )
    lines.extend(
        [
            "",
            f"  targets needing fill: {payload['n_targets_needing_fill']} of {len(TARGETS)}",
        ]
    )
    if payload["targets_needing_fill"]:
        lines.append(f"    {', '.join(payload['targets_needing_fill'])}")
    lines.extend(
        [
            "",
            "  This is computed from training labels alone. It does not read any",
            "  model output or evaluation surface, so acting on it is not",
            "  outcome-driven target selection.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit weak-supervision class balance before training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--labeller", default="b6", choices=["b6", "b23"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_IMBALANCE_THRESHOLD)
    parser.add_argument("--min-minority", type=int, default=DEFAULT_MIN_MINORITY_CELLS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from .b7_weak_supervision import _read_config

    config = _read_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    payload = audit_supervision_balance(
        config,
        args.export_root,
        labeller=args.labeller,
        threshold=args.threshold,
        min_minority=args.min_minority,
    )
    print(format_balance(payload))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
