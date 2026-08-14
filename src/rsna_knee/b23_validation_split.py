"""B23 large frozen development surface.

This is the half of B23 that unblocks measurement rather than supervision.

Every model comparison in this campaign has been made on 58 expert studies. The
B22 duration audit measured the cost of that directly: gold macro AUC moved
0.0439 across a single training run in which only the epoch count changed, which
exceeds the entire B13 -> B20 campaign gain of 0.0378. At that noise level the
surface cannot resolve the differences the campaign is trying to make.

A high-coverage report labeller changes the arithmetic. B6 keeps 14,123 of
52,188 possible cells; a labeller at ~90% coverage keeps roughly 47,000. That is
enough to hold out several hundred studies for development ranking and still
train on the rest, which lowers the macro-AUC standard error by roughly the
square root of the size ratio -- from about 0.025 at n=58 toward about 0.007 at
n=800.

The split is frozen exactly like `weak_b6_holdout_v2`: report-group safe,
stratified with a rare-class floor, chosen by a deterministic candidate search
that sees no gold labels and no model predictions, and pinned by a manifest
SHA-256. What this surface measures is agreement with the B23 labeller, not
expert truth -- and B15 and B21 both showed a weak-surface gain need not carry
to gold. Its role is to make near-neighbour comparisons *measurable*; the
labeller audit is what establishes that the labeller is worth agreeing with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import _read_config, prepare_b7_supervision
from .b23_llm_labels import B23_VERSION, load_frozen_b23_export
from .constants import TARGETS
from .data import add_report_groups, load_train_csv
from .weak_validation import make_stratified_weak_holdout

B23_HOLDOUT_SURFACE = "b23_llm_holdout_v1"
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_SEED = 2026
DEFAULT_MIN_CLASS_COUNT = 8
DEFAULT_SEARCH_CANDIDATES = 4096


def manifest_sha256(manifest: pd.DataFrame) -> str:
    """Stable digest over the ordered UID/split pairs."""
    ordered = manifest.sort_values("StudyInstanceUID", kind="mergesort")
    payload = "\n".join(
        f"{uid}\t{split}"
        for uid, split in zip(
            ordered["StudyInstanceUID"].astype(str), ordered["split"].astype(str)
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_b23_holdout(
    config: dict,
    *,
    b23_root: str | Path,
    out_root: str | Path = "runs/b23_holdout_v1",
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    n_candidates: int = DEFAULT_SEARCH_CANDIDATES,
) -> dict:
    """Freeze the large B23 development split before any model is trained on it."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0,1)")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b23_frame, b23_policy, b23_audit = load_frozen_b23_export(b23_root)
    uids, y, w, supervision = prepare_b7_supervision(train, b23_frame)

    active = np.asarray(w).sum(axis=1) > 0
    if int(active.sum()) < 2:
        raise ValueError("B23 supervision produced fewer than two active studies")
    uids = [str(uid) for uid, keep in zip(uids, active) if keep]
    y = np.asarray(y)[active]
    w = np.asarray(w)[active]

    grouped = add_report_groups(train[["StudyInstanceUID", "Report"]])
    group_map = dict(
        zip(grouped["StudyInstanceUID"].astype(str), grouped["report_group"].astype(str))
    )
    report_groups = [group_map[uid] for uid in uids]

    holdout, split_diagnostics = make_stratified_weak_holdout(
        uids,
        report_groups,
        y,
        w,
        holdout_fraction=holdout_fraction,
        seed=seed,
        min_class_count=min_class_count,
        n_candidates=n_candidates,
    )

    manifest = pd.DataFrame(
        {
            "StudyInstanceUID": uids,
            "report_group": report_groups,
            "split": np.where(holdout, "holdout", "train"),
            "labelled_cells": (w > 0).sum(axis=1).astype(int),
            "positive_cells": ((w > 0) & (y >= 0.5)).sum(axis=1).astype(int),
            "negative_cells": ((w > 0) & (y < 0.5)).sum(axis=1).astype(int),
        }
    )

    train_groups = set(manifest.loc[~holdout, "report_group"])
    holdout_groups = set(manifest.loc[holdout, "report_group"])
    overlap = train_groups.intersection(holdout_groups)
    if overlap:
        raise RuntimeError(f"report-group leakage across the B23 split: {len(overlap)} group(s)")

    digest = manifest_sha256(manifest)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "manifest.csv", index=False)

    payload = {
        "surface": B23_HOLDOUT_SURFACE,
        "b23_version": B23_VERSION,
        "b23_labeller_policy_version": str(b23_policy.get("version")),
        "b23_cell_coverage": float(b23_audit.get("cell_coverage", float("nan"))),
        "holdout_fraction_requested": float(holdout_fraction),
        "seed": int(seed),
        "min_class_count": int(min_class_count),
        "search_candidates": int(n_candidates),
        "active_studies": int(len(uids)),
        "train_studies": int((~holdout).sum()),
        "holdout_studies": int(holdout.sum()),
        "holdout_cells": int((w[holdout] > 0).sum()),
        "train_cells": int((w[~holdout] > 0).sum()),
        "report_group_overlap": 0,
        "manifest_sha256": digest,
        "supervision": supervision,
        "split_diagnostics": split_diagnostics,
        "gold_labels_used": False,
        "model_predictions_used": False,
        "measures": "agreement with the B23 labeller, not expert truth",
    }
    (out / "weak_holdout.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_frozen_b23_holdout(out_root: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load a frozen B23 split and verify its manifest digest still matches."""
    root = Path(out_root)
    payload_path = root / "weak_holdout.json"
    manifest_path = root / "manifest.csv"
    for path in (payload_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"B23 holdout is missing artifact: {path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    manifest["StudyInstanceUID"] = manifest["StudyInstanceUID"].astype(str)
    digest = manifest_sha256(manifest)
    if digest != str(payload.get("manifest_sha256")):
        raise ValueError("B23 holdout manifest SHA-256 does not match its frozen record")
    if str(payload.get("surface")) != B23_HOLDOUT_SURFACE:
        raise ValueError(f"expected surface {B23_HOLDOUT_SURFACE!r}")
    return payload, manifest


def expected_standard_error(n_studies: int, reference_se: float = 0.0250, reference_n: int = 58) -> float:
    """Rough macro-AUC SE scaling, anchored on the repo's own bootstrap width.

    The B13/B14/B15 gold bootstraps report 95% intervals about 0.098 wide at
    n=58, implying SE ~= 0.0250. AUC standard error falls roughly as 1/sqrt(n)
    at fixed class balance, so this gives a usable planning estimate for how
    large a development surface needs to be to resolve a given difference.
    """
    if n_studies < 1 or reference_n < 1:
        raise ValueError("study counts must be positive")
    return float(reference_se * np.sqrt(reference_n / n_studies))


def format_split(payload: dict) -> str:
    se = expected_standard_error(int(payload["holdout_studies"]))
    return "\n".join(
        [
            f"B23 development split ({payload['surface']})",
            f"  active studies      {payload['active_studies']}",
            f"  train / holdout     {payload['train_studies']} / {payload['holdout_studies']}",
            f"  holdout cells       {payload['holdout_cells']}",
            f"  report-group overlap {payload['report_group_overlap']}",
            f"  manifest SHA-256    {payload['manifest_sha256']}",
            "",
            f"  expected macro-AUC SE ~ {se:.4f} (versus ~0.0250 at n=58)",
            f"  resolvable difference ~ {2 * 1.96 * se:.4f} at 95% confidence",
            "",
            "  This surface measures agreement with the B23 labeller, not expert truth.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the large B23 development split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b23-root", required=True)
    parser.add_argument("--out-root", default="runs/b23_holdout_v1")
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-class-count", type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_SEARCH_CANDIDATES)
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    payload = freeze_b23_holdout(
        config,
        b23_root=args.b23_root,
        out_root=args.out_root,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        min_class_count=args.min_class_count,
        n_candidates=args.n_candidates,
    )
    print(format_split(payload))


if __name__ == "__main__":  # pragma: no cover
    main()
