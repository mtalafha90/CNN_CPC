"""How far apart two matched arms' predictions actually are.

A matched experiment in this project trains two models that share everything --
the same frozen B34 base, the same data, the same optimiser, the same seed --
and differ in one mechanism inside the local sparse branch. The scored output is

    z = z_base + tanh(g) * z_local

so the base contribution is *identical* in both arms and cancels when the two
prediction files are compared. Whatever separates them comes entirely from the
local branch, scaled by a gate that the completed runs recorded as
`|tanh(g)|` around 0.02 to 0.05.

That matters for reading B48 and B49. Both reported candidate-minus-control
macro AUC differences with confidence intervals about two ten-thousandths wide
on 903 studies, where a genuine difference should carry an interval nearer a
hundredth. Before concluding that a mechanism does not work, it is worth
establishing whether the two arms produced meaningfully different predictions at
all -- because an AUC can only move when studies change places.

This tool answers that from the prediction CSVs the evaluation already wrote. It
needs no labels, no checkpoints and no GPU.

Per target it reports:

    mean |dp|         how far the two probabilities sit apart on average
    spearman          rank agreement between the arms
    discordant        the fraction of study pairs the two arms order differently

The last one is the one that matters. ROC AUC is a function of pair ordering
alone, so if two arms order essentially every pair the same way, their AUCs
cannot differ by much whatever the mechanism did upstream. A near-zero
discordant fraction means the experiment had very little to measure, which is a
statement about the measurement and not about the idea being tested.

No threshold and no verdict: what counts as "too similar" depends on the effect
size the experiment predeclared, which lives in its protocol document.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UID_COLUMN = "StudyInstanceUID"


def load_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if UID_COLUMN not in frame.columns:
        raise ValueError(f"{path} has no {UID_COLUMN} column")
    if frame[UID_COLUMN].duplicated().any():
        raise ValueError(f"{path} contains duplicate study rows")
    return frame.set_index(UID_COLUMN).sort_index()


def align(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the two arms on the studies and targets they share.

    A silent mismatch here would make every number below meaningless, so the
    shared set is intersected explicitly and its size is reported rather than
    assumed.
    """
    shared_uids = first.index.intersection(second.index)
    if len(shared_uids) == 0:
        raise ValueError("the two prediction files share no studies")
    shared_targets = [c for c in first.columns if c in second.columns]
    if not shared_targets:
        raise ValueError("the two prediction files share no target columns")
    return first.loc[shared_uids, shared_targets], second.loc[shared_uids, shared_targets]


def discordant_fraction(left: np.ndarray, right: np.ndarray) -> float:
    """The share of study pairs the two arms put in a different order.

    ROC AUC depends on nothing but pair ordering, so this is the quantity that
    bounds how far two arms' AUCs can possibly diverge. Pairs that either arm
    ties are counted as concordant, because a tie cannot flip a ranking.
    """
    n = len(left)
    if n < 2:
        return 0.0
    left_sign = np.sign(left[:, None] - left[None, :])
    right_sign = np.sign(right[:, None] - right[None, :])
    upper = np.triu_indices(n, k=1)
    ls, rs = left_sign[upper], right_sign[upper]
    disagree = (ls * rs) < 0
    return float(disagree.sum() / len(ls))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    lr = pd.Series(left).rank().to_numpy()
    rr = pd.Series(right).rank().to_numpy()
    if np.std(lr) == 0 or np.std(rr) == 0:
        return float("nan")
    return float(np.corrcoef(lr, rr)[0, 1])


def compare(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    left, right = align(first, second)
    rows = []
    for target in left.columns:
        a = left[target].to_numpy(float)
        b = right[target].to_numpy(float)
        delta = np.abs(a - b)
        rows.append(
            {
                "target": target,
                "mean_abs_delta": float(delta.mean()),
                "max_abs_delta": float(delta.max()),
                "spearman": _spearman(a, b),
                "discordant_pair_fraction": discordant_fraction(a, b),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "studies": int(len(left)),
        "targets": int(len(left.columns)),
        "per_target": frame,
        "mean_abs_delta": float(frame["mean_abs_delta"].mean()),
        "mean_discordant_pair_fraction": float(frame["discordant_pair_fraction"].mean()),
        "min_spearman": float(frame["spearman"].min()),
    }


def auc_difference_ceiling(discordant: float) -> float:
    """The largest AUC difference a given amount of reordering can produce.

    An ROC AUC is the share of positive/negative pairs a model orders correctly,
    so two models' AUCs can differ only on pairs they order differently. Pairs
    they agree on -- both right or both wrong -- cancel exactly.

    If the discordant pairs fall among positive/negative pairs at the same rate
    as among pairs generally, then the share of scoring pairs that could have
    flipped is the discordant fraction itself, and the AUC difference is largest
    when every one of them flips the same way. That makes the discordant
    fraction a direct ceiling on |AUC_candidate - AUC_control|.

    The uniformity assumption is what makes this an estimate rather than a proof.
    It could be beaten if a mechanism disturbed exactly the pairs that separate
    the classes and left every other pair alone, which is not how a small dense
    perturbation behaves. Treat it as the right order of magnitude.
    """
    return float(discordant)


def describe(result: dict) -> str:
    frame = result["per_target"]
    lines = [
        f"{result['studies']} studies x {result['targets']} targets",
        "",
        f"{'target':<18} {'mean |dp|':>11} {'max |dp|':>10} {'spearman':>9} "
        f"{'discordant':>11} {'max |dAUC|':>11}",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{row.target:<18} {row.mean_abs_delta:>11.6f} {row.max_abs_delta:>10.6f} "
            f"{row.spearman:>9.5f} {row.discordant_pair_fraction:>11.6f} "
            f"{auc_difference_ceiling(row.discordant_pair_fraction):>11.6f}"
        )
    mean_discordant = result["mean_discordant_pair_fraction"]
    lines.extend(
        [
            "",
            f"mean over targets   |dp| {result['mean_abs_delta']:.6f}   "
            f"discordant {mean_discordant:.6f}   "
            f"lowest spearman {result['min_spearman']:.5f}",
            "",
            "max |dAUC| is the discordant fraction: an AUC can only move on pairs",
            "the two arms order differently, so this bounds how far their AUCs",
            "could differ, assuming discordance falls on class-separating pairs at",
            "the same rate as on pairs generally.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two matched arms' prediction CSVs. The shared frozen base "
            "cancels, so what remains is the local branch's contribution."
        )
    )
    parser.add_argument("control", help="control arm prediction CSV")
    parser.add_argument("candidate", help="candidate arm prediction CSV")
    parser.add_argument("--csv", help="optional path to write the per-target table")
    args = parser.parse_args()

    result = compare(load_predictions(args.control), load_predictions(args.candidate))
    print(describe(result))
    if args.csv:
        result["per_target"].to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
