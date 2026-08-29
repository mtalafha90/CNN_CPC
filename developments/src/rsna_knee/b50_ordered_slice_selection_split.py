"""Freeze B50's fresh scanner-grouped model-selection surface.

B48 and B49 already consumed their scanner-domain validation rows.  B50 must
not select an ordered-slice architecture on the same rows, so this module makes
a new surface from *only* the B48/B49 training side.  The old B48 validation
rows are carried forward as ``excluded_prior_surface`` and may not enter B50
training, validation, checkpoint selection, or threshold selection.

This is fresh as a model-selection surface, not a new medical dataset: the
candidate pool consists of report-only studies that were training rows in the
completed B48/B49 experiments.  The distinction is recorded explicitly rather
than overstating independence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import load_train_csv
from .domain_shift_split import (
    DEFAULT_HOLDOUT_FRACTION,
    DOMAIN_SPLIT_VERSION,
    build_study_profiles,
    choose_holdout_profiles,
    choose_seen_scanner_validation,
    verify_domain_split,
)

B50_SELECTION_SPLIT_VERSION = "b50_fresh_scanner_grouped_selection_split_v1"
B50_SELECTION_SPLIT_SALT = "CNN_CPC|B50|ordered-slice|fresh-selection-gate|2026-08-29"
B50_SELECTION_HOLDOUT_FRACTION = DEFAULT_HOLDOUT_FRACTION
B50_SELECTION_ROOT = (
    "runs/083_Experiment_B50_ordered_slice_sequence_mil/"
    "b50_ordered_slice_selection_split"
)

B50_PARENT_TRAIN_SPLIT = "train"
B50_PARENT_EXCLUDED_SPLITS = {
    "validation_seen_scanners",
    "holdout_unseen_scanners",
}
B50_SPLIT_TRAIN = "train"
B50_SPLIT_SEEN = "validation_seen_scanners"
B50_SPLIT_UNSEEN = "validation_unseen_scanners"
B50_SPLIT_EXCLUDED = "excluded_prior_surface"
B50_ALLOWED_SPLITS = {
    B50_SPLIT_TRAIN,
    B50_SPLIT_SEEN,
    B50_SPLIT_UNSEEN,
    B50_SPLIT_EXCLUDED,
}


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_b48_parent_split(root: str | Path) -> tuple[dict, pd.DataFrame, dict]:
    """Read and verify the spent B48/B49 domain split without modifying it."""
    directory = Path(root).resolve()
    payload_path = directory / "domain_split.json"
    rows_path = directory / "domain_split_by_study.csv"
    hash_path = directory / "domain_split.sha256"
    for path in (payload_path, rows_path, hash_path):
        if not path.is_file():
            raise FileNotFoundError(f"B50 requires the existing B48 split artefact: {path}")

    observed = _sha256_file(payload_path)
    recorded = hash_path.read_text(encoding="utf-8").strip()
    if observed != recorded:
        raise ValueError("B50 refuses a parent domain split whose SHA-256 does not match")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("version") != DOMAIN_SPLIT_VERSION:
        raise ValueError(
            "B50 requires the completed B48/B49 scanner-domain split version "
            f"{DOMAIN_SPLIT_VERSION!r}; got {payload.get('version')!r}"
        )

    rows = pd.read_csv(rows_path)
    required = {"StudyInstanceUID", "scanner_profile", "holdout", "split"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"B50 parent split rows are missing columns: {missing}")
    rows = rows.copy()
    rows["StudyInstanceUID"] = rows["StudyInstanceUID"].astype(str)
    rows["scanner_profile"] = rows["scanner_profile"].astype(str)
    rows["split"] = rows["split"].astype(str)
    if rows["StudyInstanceUID"].duplicated().any():
        raise ValueError("B50 parent split contains duplicate study UIDs")
    unknown = sorted(
        set(rows["split"]).difference(
            {B50_PARENT_TRAIN_SPLIT, *B50_PARENT_EXCLUDED_SPLITS}
        )
    )
    if unknown:
        raise ValueError(f"B50 parent split contains unknown labels: {unknown}")
    if not rows.loc[rows["split"].eq("holdout_unseen_scanners"), "holdout"].all():
        raise ValueError("B50 parent split has an unseen-scanner row without holdout=True")
    if rows.loc[~rows["split"].eq("holdout_unseen_scanners"), "holdout"].any():
        raise ValueError("B50 parent split has a non-holdout row with holdout=True")
    verify_domain_split(rows)
    return payload, rows, {"sha256": observed, "root": str(directory)}


def _position_by_uid(uids: list[str]) -> dict[str, int]:
    normalized = [str(uid) for uid in uids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("B50 supervision UID list contains duplicates")
    return {uid: index for index, uid in enumerate(normalized)}


def _side_summary(
    rows: pd.DataFrame,
    positions: dict[str, int],
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict:
    if rows.empty:
        usable = np.zeros(len(TARGETS), dtype=np.int64)
        positive = np.zeros(len(TARGETS), dtype=np.int64)
    else:
        index = np.asarray(
            [positions[str(uid)] for uid in rows["StudyInstanceUID"]], dtype=np.int64
        )
        usable_mask = np.asarray(weights[index] > 0, dtype=bool)
        positive_mask = np.asarray(targets[index] > 0.5, dtype=bool) & usable_mask
        usable = usable_mask.sum(axis=0)
        positive = positive_mask.sum(axis=0)
    negative = usable - positive
    measurable = [
        target
        for target, pos, neg in zip(TARGETS, positive.tolist(), negative.tolist())
        if int(pos) > 0 and int(neg) > 0
    ]
    return {
        "studies": int(len(rows)),
        "scanner_profiles": int(rows["scanner_profile"].nunique()) if len(rows) else 0,
        "usable_cells": int(usable.sum()),
        "positive_cells": {target: int(value) for target, value in zip(TARGETS, positive)},
        "negative_cells": {target: int(value) for target, value in zip(TARGETS, negative)},
        "measurable_targets": measurable,
        "unmeasurable_targets": [target for target in TARGETS if target not in measurable],
    }


def verify_b50_selection_split(rows: pd.DataFrame) -> None:
    """Enforce B50's boundary around rows spent by B48/B49."""
    required = {
        "StudyInstanceUID",
        "scanner_profile",
        "parent_b48_split",
        "b50_split",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"B50 selection split rows are missing columns: {missing}")
    if rows["StudyInstanceUID"].astype(str).duplicated().any():
        raise ValueError("B50 selection split contains duplicate study UIDs")
    labels = set(rows["b50_split"].astype(str))
    unknown = sorted(labels.difference(B50_ALLOWED_SPLITS))
    if unknown:
        raise ValueError(f"B50 selection split contains unknown labels: {unknown}")

    parent = rows["parent_b48_split"].astype(str)
    b50 = rows["b50_split"].astype(str)
    prior = ~parent.eq(B50_PARENT_TRAIN_SPLIT)
    if not b50.loc[prior].eq(B50_SPLIT_EXCLUDED).all():
        raise ValueError("B50 would reuse a B48/B49 validation row instead of excluding it")
    if b50.loc[~prior].eq(B50_SPLIT_EXCLUDED).any():
        raise ValueError("B50 left a parent-training row outside its fresh split")

    train_profiles = set(rows.loc[b50.eq(B50_SPLIT_TRAIN), "scanner_profile"].astype(str))
    seen_profiles = set(rows.loc[b50.eq(B50_SPLIT_SEEN), "scanner_profile"].astype(str))
    unseen_profiles = set(rows.loc[b50.eq(B50_SPLIT_UNSEEN), "scanner_profile"].astype(str))
    if not train_profiles or not seen_profiles or not unseen_profiles:
        raise ValueError("B50 requires non-empty train, seen, and unseen groups")
    if unseen_profiles.intersection(train_profiles) or unseen_profiles.intersection(seen_profiles):
        raise ValueError("B50 unseen-scanner profiles straddle training or seen validation")
    if not seen_profiles.issubset(train_profiles):
        missing_profiles = sorted(seen_profiles.difference(train_profiles))
        raise ValueError(
            "B50 seen-scanner validation contains profiles absent from B50 training: "
            f"{missing_profiles[:5]}"
        )


def build_b50_selection_rows(
    profiles: pd.DataFrame,
    uids: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    parent_rows: pd.DataFrame,
    *,
    holdout_fraction: float = B50_SELECTION_HOLDOUT_FRACTION,
    salt: str = B50_SELECTION_SPLIT_SALT,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Partition only B48's former training rows into B50 train/seen/unseen."""
    positions = _position_by_uid(uids)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if targets.shape != weights.shape or targets.shape != (len(positions), len(TARGETS)):
        raise ValueError("B50 targets/weights do not match the report-only UID surface")

    base = profiles.copy()
    required = {"StudyInstanceUID", "scanner_profile"}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"B50 profile table is missing columns: {missing}")
    base["StudyInstanceUID"] = base["StudyInstanceUID"].astype(str)
    base["scanner_profile"] = base["scanner_profile"].astype(str)
    if base["StudyInstanceUID"].duplicated().any():
        raise ValueError("B50 profile table contains duplicate study UIDs")
    if set(base["StudyInstanceUID"]) != set(positions):
        raise ValueError("B50 profile table and weak-supervision UID surface differ")

    parent = parent_rows.copy()
    required_parent = {"StudyInstanceUID", "scanner_profile", "split"}
    missing = sorted(required_parent.difference(parent.columns))
    if missing:
        raise ValueError(f"B50 parent rows are missing columns: {missing}")
    parent["StudyInstanceUID"] = parent["StudyInstanceUID"].astype(str)
    parent["scanner_profile"] = parent["scanner_profile"].astype(str)
    parent["split"] = parent["split"].astype(str)
    if parent["StudyInstanceUID"].duplicated().any():
        raise ValueError("B50 parent rows contain duplicate study UIDs")
    if set(parent["StudyInstanceUID"]) != set(positions):
        raise ValueError("B50 parent split and weak-supervision UID surface differ")

    profile_by_uid = base.set_index("StudyInstanceUID")["scanner_profile"]
    parent_profile = parent.set_index("StudyInstanceUID")["scanner_profile"]
    changed = [
        uid
        for uid in sorted(positions)
        if str(profile_by_uid.loc[uid]) != str(parent_profile.loc[uid])
    ]
    if changed:
        raise ValueError(
            "B50 header-derived scanner profiles differ from B48's frozen parent split; "
            f"first differing study: {changed[0]}"
        )

    parent_split = parent.set_index("StudyInstanceUID")["split"]
    pool = base.loc[
        base["StudyInstanceUID"].map(parent_split).eq(B50_PARENT_TRAIN_SPLIT)
    ].copy()
    if pool.empty or pool["scanner_profile"].nunique() < 2:
        raise ValueError("B50 parent-training pool cannot form a scanner-disjoint holdout")

    holdout_profiles = choose_holdout_profiles(
        pool,
        targets,
        weights,
        list(positions),
        holdout_fraction=float(holdout_fraction),
        salt=str(salt),
    )
    unseen_mask = pool["scanner_profile"].isin(set(holdout_profiles))
    seen_uids = choose_seen_scanner_validation(
        pool.loc[~unseen_mask],
        int(unseen_mask.sum()),
        salt=str(salt),
    )

    rows = base.copy()
    rows["parent_b48_split"] = rows["StudyInstanceUID"].map(parent_split)
    rows["b50_split"] = B50_SPLIT_EXCLUDED
    in_pool = rows["parent_b48_split"].eq(B50_PARENT_TRAIN_SPLIT)
    rows.loc[in_pool, "b50_split"] = B50_SPLIT_TRAIN
    rows.loc[
        in_pool & rows["scanner_profile"].isin(set(holdout_profiles)), "b50_split"
    ] = B50_SPLIT_UNSEEN
    rows.loc[
        in_pool & rows["StudyInstanceUID"].isin(seen_uids), "b50_split"
    ] = B50_SPLIT_SEEN
    verify_b50_selection_split(rows)

    summary = {
        split: _side_summary(
            rows.loc[rows["b50_split"].eq(split)], positions, targets, weights
        )
        for split in (
            B50_SPLIT_TRAIN,
            B50_SPLIT_SEEN,
            B50_SPLIT_UNSEEN,
            B50_SPLIT_EXCLUDED,
        )
    }
    primary = summary[B50_SPLIT_UNSEEN]
    comparable = sorted(
        set(primary["measurable_targets"]).intersection(
            summary[B50_SPLIT_SEEN]["measurable_targets"]
        ),
        key=TARGETS.index,
    )
    if primary["unmeasurable_targets"]:
        raise ValueError(
            "B50 fresh unseen-scanner gate cannot measure all 12 targets: "
            f"{primary['unmeasurable_targets']}"
        )
    if len(comparable) != len(TARGETS):
        raise ValueError(
            "B50 fresh seen/unseen comparator is not defined for all 12 targets; "
            f"comparable={comparable}"
        )
    summary.update(
        {
            "selection_pool_source": "parent_b48_train_rows_only",
            "selection_pool_studies": int(in_pool.sum()),
            "selection_pool_scanner_profiles": int(rows.loc[in_pool, "scanner_profile"].nunique()),
            "excluded_prior_surface_studies": int((~in_pool).sum()),
            "requested_holdout_fraction_of_selection_pool": float(holdout_fraction),
            "actual_unseen_fraction_of_selection_pool": float(
                primary["studies"] / max(int(in_pool.sum()), 1)
            ),
            "comparable_targets": comparable,
        }
    )
    return rows, summary, holdout_profiles


def build_b50_selection_split(
    *,
    data_root: str | Path,
    header_csv: str | Path,
    labels_root: str | Path,
    parent_domain_split: str | Path,
) -> tuple[dict, pd.DataFrame]:
    """Build B50's prospective selection surface without writing it yet."""
    from .phase9_supervision import (
        load_fill_merged_export,
        prepare_all_report_only_supervision,
    )

    root = Path(data_root).resolve()
    train_path = root / "train.csv"
    train = load_train_csv(train_path)
    header_path = Path(header_csv).resolve()
    labels_path = Path(labels_root).resolve()
    parent_payload, parent_rows, parent_meta = load_b48_parent_split(parent_domain_split)

    frame, _fill_policy, _fill_audit = load_fill_merged_export(labels_path)
    uids, targets, weights, _supervision = prepare_all_report_only_supervision(train, frame)
    profiles = build_study_profiles(train, pd.read_csv(header_path))
    rows, summary, holdout_profiles = build_b50_selection_rows(
        profiles,
        uids,
        targets,
        weights,
        parent_rows,
    )
    labels_csv = labels_path / "training_targets.csv"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"B50 labels root is missing {labels_csv.name}: {labels_csv}")

    payload = {
        "version": B50_SELECTION_SPLIT_VERSION,
        "status": "frozen_before_any_B50_model_is_scored_on_this_surface",
        "purpose": (
            "fresh B50 architecture-selection gate built only from B48/B49 parent "
            "training rows; prior B48/B49 validation rows are excluded entirely"
        ),
        "freshness_limit": (
            "The rows were B48/B49 training data, not a new medical cohort. They are "
            "fresh as a held-out B50 selection surface because B48/B49 never scored "
            "their models on these rows."
        ),
        "source_train_csv": str(train_path),
        "source_train_csv_sha256": _sha256_file(train_path),
        "source_header_csv": str(header_path),
        "source_header_csv_sha256": _sha256_file(header_path),
        "source_labels_root": str(labels_path),
        "source_training_targets_sha256": _sha256_file(labels_csv),
        "parent_b48_domain_split": {
            "root": parent_meta["root"],
            "sha256": parent_meta["sha256"],
            "version": parent_payload["version"],
        },
        "salt": B50_SELECTION_SPLIT_SALT,
        "requested_holdout_fraction": B50_SELECTION_HOLDOUT_FRACTION,
        "group_key": "manufacturer_family|manufacturer_model|field_strength_bin",
        "assignment_algorithm": (
            "deterministic greedy per-target prevalence balancing over whole scanner "
            "profiles, with SHA-256 over the frozen salt breaking ties"
        ),
        "unseen_scanner_profiles": holdout_profiles,
        "how_to_read_it": (
            "Train B50 only on b50_split=train. Compare the predeclared candidate and "
            "control on validation_unseen_scanners; validation_seen_scanners is a matched "
            "domain-gap comparator. excluded_prior_surface rows may not be used by B50."
        ),
        "summary": summary,
    }
    return payload, rows


def write_b50_selection_split(
    payload: dict,
    rows: pd.DataFrame,
    *,
    out_root: str | Path,
) -> Path:
    """Write once; a frozen B50 gate is never silently regenerated."""
    out = Path(out_root)
    names = (
        "b50_selection_split.json",
        "b50_selection_split_by_study.csv",
        "b50_selection_split.sha256",
    )
    existing = [out / name for name in names if (out / name).exists()]
    if existing:
        raise FileExistsError(
            "B50 selection gate already exists; do not regenerate it: "
            + ", ".join(str(path) for path in existing)
        )
    out.mkdir(parents=True, exist_ok=True)
    rows.sort_values("StudyInstanceUID").to_csv(
        out / "b50_selection_split_by_study.csv", index=False
    )
    path = out / "b50_selection_split.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "b50_selection_split.sha256").write_text(
        _sha256_file(path) + "\n", encoding="utf-8"
    )
    return path


def _report(payload: dict) -> None:
    summary = payload["summary"]
    print(
        "[B50 selection split] parent B48 SHA "
        f"{payload['parent_b48_domain_split']['sha256']}, "
        f"pool={summary['selection_pool_studies']} studies / "
        f"{summary['selection_pool_scanner_profiles']} profiles"
    )
    for split, label in (
        (B50_SPLIT_TRAIN, "train                 "),
        (B50_SPLIT_SEEN, "seen scanners         "),
        (B50_SPLIT_UNSEEN, "unseen scanners       "),
        (B50_SPLIT_EXCLUDED, "excluded prior surface"),
    ):
        side = summary[split]
        print(
            f"[B50 selection split] {label} {side['studies']:>5} studies / "
            f"{side['scanner_profiles']:>3} profiles / {side['usable_cells']:>6} cells"
        )
    print(
        "[B50 selection split] all 12 targets are measurable on the fresh unseen gate; "
        f"domain comparison is defined for {len(summary['comparable_targets'])}/12 targets"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze B50's fresh scanner-grouped architecture-selection gate"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--header-csv",
        required=True,
        help="existing header_by_series.csv from rsna_knee.dataset_header_audit",
    )
    parser.add_argument("--labels-root", required=True)
    parser.add_argument(
        "--parent-domain-split",
        required=True,
        help="the immutable B48/B49 DOMAIN_SPLIT_ROOT directory",
    )
    parser.add_argument("--out-root", default=B50_SELECTION_ROOT)
    args = parser.parse_args()

    payload, rows = build_b50_selection_split(
        data_root=args.data_root,
        header_csv=args.header_csv,
        labels_root=args.labels_root,
        parent_domain_split=args.parent_domain_split,
    )
    _report(payload)
    print(write_b50_selection_split(payload, rows, out_root=args.out_root))


if __name__ == "__main__":
    main()


__all__ = [
    "B50_ALLOWED_SPLITS",
    "B50_SELECTION_HOLDOUT_FRACTION",
    "B50_SELECTION_ROOT",
    "B50_SELECTION_SPLIT_SALT",
    "B50_SELECTION_SPLIT_VERSION",
    "build_b50_selection_rows",
    "build_b50_selection_split",
    "load_b48_parent_split",
    "verify_b50_selection_split",
    "write_b50_selection_split",
]
