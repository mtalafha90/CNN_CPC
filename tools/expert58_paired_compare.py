"""Compare two saved Expert-58 prediction files, paired, without re-running a GPU.

The Expert-58 evaluator scores one endpoint against three fixed references:
`base_224`, B37 and B41. When it is used to score a *later* endpoint, the one
comparison that usually matters most is missing -- the endpoint's own control.
B51's control is B42, and the B42 evaluation never loaded B42 as a reference
because B42 was the thing being scored.

Both runs already wrote their per-study probabilities, so the comparison needs no
model, no DICOM decoding and no GPU. This reads the two CSVs, checks they cover
the same studies in the same order, and reports the paired bootstrap the
evaluator would have produced.

Read the answer against the surface. Expert-58 is 58 studies and resolves to
roughly +/-0.03; it also failed to order B37, B41 and B42 the way the hidden
test did -- all three tied at a displayed 0.714 while spanning 0.678 to 0.686
here. A delta from this tool is a description, never a promotion criterion.

    python -m tools.expert58_paired_compare \\
      --data-root  /path/to/rsna-knee-abnormality-detection \\
      --control    runs/077_.../expert58/b42_combined_predictions.csv \\
      --candidate  runs/085_.../expert58/b51_combined_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "developments" / "src"))

from rsna_knee.constants import TARGETS  # noqa: E402
from rsna_knee.data import gold_mask, load_train_csv  # noqa: E402
from rsna_knee.evaluation import compare_runs, macro_auc_from_arrays  # noqa: E402

EXPECTED_GOLD_STUDIES = 58
# What this surface can resolve, and what the mechanism under test is worth.
EXPERT58_RESOLUTION = 0.03
B50_UNSEEN_SCANNER_DELTA = 0.011221


def _load_predictions(path: Path, uids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    required = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != required:
        raise ValueError(f"unexpected columns in {path.name}")
    observed = frame["StudyInstanceUID"].astype(str).tolist()
    if observed != uids:
        raise RuntimeError(
            f"{path.name} does not cover the same studies in the same order as "
            "the expert surface; the pairing would be meaningless"
        )
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path.name} contains non-finite probabilities")
    return values


def compare(
    data_root: str | Path,
    control_csv: str | Path,
    candidate_csv: str | Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict:
    root = Path(data_root)
    train = load_train_csv(root / "train.csv")
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("this needs the complete reused 58-study expert surface")

    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)
    control = _load_predictions(Path(control_csv), uids)
    candidate = _load_predictions(Path(candidate_csv), uids)

    control_macro, control_auc = macro_auc_from_arrays(truth, control)
    candidate_macro, candidate_auc = macro_auc_from_arrays(truth, candidate)
    delta = float(candidate_macro - control_macro)

    per_target = {
        target: {
            "control": float(control_auc[index]),
            "candidate": float(candidate_auc[index]),
            "delta": float(candidate_auc[index] - control_auc[index]),
        }
        for index, target in enumerate(TARGETS)
    }
    improved = [name for name, row in per_target.items() if row["delta"] > 0]

    return {
        "control": str(Path(control_csv).resolve()),
        "candidate": str(Path(candidate_csv).resolve()),
        "n_studies": len(uids),
        "control_macro_auc": float(control_macro),
        "candidate_macro_auc": float(candidate_macro),
        "delta_macro_auc": delta,
        "targets_improved": f"{len(improved)}/{len(TARGETS)}",
        "targets_improved_names": improved,
        "paired_bootstrap": compare_runs(
            truth, control, candidate, n_bootstrap=int(n_bootstrap), seed=int(seed)
        ),
        "per_target": per_target,
        "surface_resolution": EXPERT58_RESOLUTION,
        "inside_resolution": bool(abs(delta) < EXPERT58_RESOLUTION),
        "interpretation": (
            f"58 studies resolve to about +/-{EXPERT58_RESOLUTION}. The mechanism "
            f"under test was measured at +{B50_UNSEEN_SCANNER_DELTA} on 548 "
            "unseen-scanner studies, which is roughly a third of what this "
            "surface can see. A delta inside the resolution is inconclusive, and "
            "this surface tied B37, B41 and B42 on the hidden test while "
            "separating them here."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Expert-58 comparison between two saved prediction files"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--control", required=True, help="the baseline predictions CSV")
    parser.add_argument("--candidate", required=True, help="the new predictions CSV")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    result = compare(
        args.data_root,
        args.control,
        args.candidate,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(result, indent=2))

    bootstrap = result["paired_bootstrap"]
    print()
    print(f"control   {result['control_macro_auc']:.6f}")
    print(f"candidate {result['candidate_macro_auc']:.6f}")
    print(f"delta     {result['delta_macro_auc']:+.6f}   ({result['targets_improved']} targets)")
    print(
        f"bootstrap median {bootstrap['median_difference']:+.6f} "
        f"CI [{bootstrap['ci_lower']:+.6f}, {bootstrap['ci_upper']:+.6f}] "
        f"P(candidate better) {bootstrap['probability_b_better']:.3f}"
    )
    if result["inside_resolution"]:
        print()
        print("INCONCLUSIVE: the delta is inside this surface's resolution.")


if __name__ == "__main__":
    main()


__all__ = ["compare"]
