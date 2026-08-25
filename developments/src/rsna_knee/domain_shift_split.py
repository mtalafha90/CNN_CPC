"""A validation surface held out by scanner, not by study identity.

The challenge data comes from many centres, and every split this project has
frozen so far was made by hashing the study UID. Hashing is honest -- it cannot
be steered by looking at labels -- but it scatters each scanner across both
sides of the split. A model that has learnt "this is what a Siemens 1.5T knee
looks like" therefore meets the same scanners in training and in validation, and
nothing currently measures what happens when it does not.

This module builds the missing surface. Studies are grouped by the scanner that
produced them, and whole groups go to one side or the other, so no scanner
profile is ever seen during training and then scored during validation. The
difference between the ordinary weak-label score and the score on this surface
is an estimate of how much of the model's performance is acquisition-specific.

Two limits, stated plainly rather than buried:

*   **This is a scanner proxy, not a site.** The official `train.csv` carries no
    institution column, and DICOM institution tags are frequently blank or
    anonymised, so the group key is manufacturer, model and field strength. Two
    hospitals with the same scanner land in one group; one hospital that runs
    two scanners is split across two. It is a lower bound on domain separation,
    not a true centre holdout.
*   **The labels are still report-derived.** This surface measures domain
    generalisation, not label quality. It cannot tell you whether the model is
    right, only whether it travels.

The split is frozen the same way B46's folds are: a fixed salt, a deterministic
assignment, and a recorded SHA-256. It is built before any model is scored on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGETS, gold_mask, load_train_csv
from .dataset_domain_intersection_audit import manufacturer_family

DOMAIN_SPLIT_VERSION = "official_scanner_domain_split_v1"
DOMAIN_SPLIT_SALT = "CNN_CPC|domain-split|scanner-grouped|2026-08-25"

DEFAULT_HOLDOUT_FRACTION = 0.20

# Above this share of all studies, one scanner profile is large enough that no
# grouped split can be both sizeable and balanced. Reported, not enforced: the
# concentration is itself the finding.
CONCENTRATION_WARNING_SHARE = 0.35

MISSING_PROFILE_FIELD = "Missing"


def field_strength_bin(value: object) -> str:
    """Bucket the field strength, which is recorded with real-world slop.

    Scanners report values such as 1.494 or 2.8936 rather than exactly 1.5 or
    3.0, so an equality test would scatter one scanner across several groups.
    """
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return MISSING_PROFILE_FIELD
    if not np.isfinite(strength) or strength <= 0:
        return MISSING_PROFILE_FIELD
    for nominal, label in ((0.3, "0.3T"), (1.0, "1.0T"), (1.5, "1.5T"), (3.0, "3T")):
        if abs(strength - nominal) <= 0.25:
            return label
    return f"{strength:.1f}T"


def _profile_field(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return MISSING_PROFILE_FIELD
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else MISSING_PROFILE_FIELD


def scanner_profile(manufacturer: object, model: object, field_strength: object) -> str:
    """The group key: one string naming the machine a series came from."""
    return "|".join(
        (
            manufacturer_family(manufacturer),
            _profile_field(model),
            field_strength_bin(field_strength),
        )
    )


def _dominant_profile(values: pd.Series) -> str:
    """The profile most of a study's series agree on, ties broken by name.

    A study is one visit to one scanner, so its series almost always agree. When
    they do not -- a stitched or re-imported series -- the study still has to go
    to exactly one side of the split, so one profile has to win.
    """
    counts = values.astype(str).value_counts()
    if counts.empty:
        return MISSING_PROFILE_FIELD
    top = int(counts.max())
    return sorted(counts[counts.eq(top)].index.astype(str))[0]


def build_study_profiles(train: pd.DataFrame, header: pd.DataFrame) -> pd.DataFrame:
    """One row per report-only study: its scanner profile and series count.

    The 58 official gold studies are excluded. They are the only clean target
    labels in the project and B46 is already using them; this surface is about
    the report-only population.
    """
    required = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "manufacturer",
        "manufacturer_model",
        "magnetic_field_strength_t",
    }
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"header audit CSV is missing columns: {missing}")

    frame = header.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    frame["SeriesInstanceUID"] = frame["SeriesInstanceUID"].astype(str)
    if frame[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
        raise ValueError("header audit contains duplicate study/series rows")

    frame["scanner_profile"] = [
        scanner_profile(row.manufacturer, row.manufacturer_model, row.magnetic_field_strength_t)
        for row in frame.itertuples()
    ]

    study_uids = train.loc[~gold_mask(train), "StudyInstanceUID"].astype(str)
    report_only = set(study_uids)
    frame = frame.loc[frame["StudyInstanceUID"].isin(report_only)]

    profiles = (
        frame.groupby("StudyInstanceUID", sort=True)
        .agg(
            scanner_profile=("scanner_profile", _dominant_profile),
            series_count=("SeriesInstanceUID", "size"),
            distinct_profiles=("scanner_profile", "nunique"),
        )
        .reset_index()
    )

    absent = report_only.difference(set(profiles["StudyInstanceUID"]))
    if absent:
        raise ValueError(
            f"{len(absent)} report-only studies have no header rows; "
            "run rsna_knee.dataset_header_audit over the whole data root first"
        )
    return profiles


def _profile_order(profile: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{profile}".encode("utf-8")).hexdigest()


def choose_holdout_profiles(
    profiles: pd.DataFrame,
    targets: np.ndarray,
    weights: np.ndarray,
    uids: list[str],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    salt: str = DOMAIN_SPLIT_SALT,
) -> list[str]:
    """Pick whole scanner profiles for the holdout, balanced on prevalence.

    Greedy and deterministic. At each step the profile that brings the holdout's
    per-target positive rate closest to the whole population's is added, with
    SHA-256 over the salt breaking ties, and the loop stops once the requested
    share of studies is reached.

    Balancing prevalence matters because the alternative -- taking profiles in
    hash order alone -- can easily produce a holdout in which a rare finding has
    no positive cases at all, and an AUC needs both classes to exist.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    index = {uid: position for position, uid in enumerate(uids)}
    rows = profiles.loc[profiles["StudyInstanceUID"].isin(index)]
    if rows.empty:
        raise ValueError("no study in the profile table has weak supervision")

    positions = np.array([index[uid] for uid in rows["StudyInstanceUID"]], dtype=np.int64)
    usable = weights[positions] > 0
    positive = (targets[positions] > 0.5) & usable

    members: dict[str, np.ndarray] = {}
    for profile, group in rows.groupby("scanner_profile", sort=True):
        members[str(profile)] = np.array(
            [index[uid] for uid in group["StudyInstanceUID"]], dtype=np.int64
        )

    total_studies = len(rows)
    wanted = max(1, int(round(total_studies * holdout_fraction)))

    with np.errstate(invalid="ignore", divide="ignore"):
        global_rate = np.where(usable.sum(axis=0) > 0, positive.sum(axis=0) / np.maximum(usable.sum(axis=0), 1), 0.0)

    row_of = {uid: position for position, uid in enumerate(rows["StudyInstanceUID"])}
    local = {
        profile: np.array([row_of[uid] for uid in group["StudyInstanceUID"]], dtype=np.int64)
        for profile, group in rows.groupby("scanner_profile", sort=True)
    }

    chosen: list[str] = []
    remaining = sorted(members, key=lambda profile: _profile_order(profile, salt))
    held_usable = np.zeros(len(TARGETS), dtype=np.int64)
    held_positive = np.zeros(len(TARGETS), dtype=np.int64)
    held_studies = 0

    while remaining and held_studies < wanted:
        best_profile, best_score = None, None
        for profile in remaining:
            take = local[profile]
            candidate_usable = held_usable + usable[take].sum(axis=0)
            candidate_positive = held_positive + positive[take].sum(axis=0)
            rate = np.where(
                candidate_usable > 0,
                candidate_positive / np.maximum(candidate_usable, 1),
                global_rate,
            )
            size_error = ((held_studies + len(take)) / wanted - 1.0) ** 2
            score = float(np.square(rate - global_rate).sum()) + size_error
            if best_score is None or score < best_score:
                best_profile, best_score = profile, score

        take = local[best_profile]
        held_usable = held_usable + usable[take].sum(axis=0)
        held_positive = held_positive + positive[take].sum(axis=0)
        held_studies += len(take)
        chosen.append(best_profile)
        remaining.remove(best_profile)

    if not remaining:
        raise ValueError(
            "the holdout would take every scanner profile; the requested "
            "fraction is too large for this population"
        )
    return sorted(chosen)


def choose_seen_scanner_validation(
    candidates: pd.DataFrame,
    wanted: int,
    *,
    salt: str = DOMAIN_SPLIT_SALT,
) -> set[str]:
    """Hold back studies from scanners that *do* stay in training.

    Without this there is nothing to compare the grouped holdout against. A
    score on unseen scanners is only interesting next to a score on seen ones
    from the same model: the difference between the two is the domain gap, and a
    single number on its own cannot distinguish "this model does not travel"
    from "this model is not very good".

    Selection is by UID hash, matching how every other split in this project is
    drawn, and deliberately not by scanner: these studies must come from the
    same machines training keeps.
    """
    if wanted <= 0:
        raise ValueError("the seen-scanner validation set must hold at least one study")
    if wanted >= len(candidates):
        raise ValueError(
            "the seen-scanner validation set would consume the whole training side"
        )
    ranked = sorted(
        candidates["StudyInstanceUID"].astype(str),
        key=lambda uid: hashlib.sha256(f"{salt}|seen|{uid}".encode("utf-8")).hexdigest(),
    )
    return set(ranked[:wanted])


def summarise_split(
    profiles: pd.DataFrame,
    holdout_profiles: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    uids: list[str],
    *,
    seen_validation_uids: set[str] | None = None,
) -> dict:
    """Describe the split, and say which targets it can actually measure."""
    index = {uid: position for position, uid in enumerate(uids)}
    rows = profiles.loc[profiles["StudyInstanceUID"].isin(index)].copy()
    rows["holdout"] = rows["scanner_profile"].isin(set(holdout_profiles))

    def side(mask: pd.Series) -> dict:
        positions = np.array([index[uid] for uid in rows.loc[mask, "StudyInstanceUID"]], dtype=np.int64)
        usable = weights[positions] > 0
        positive = (targets[positions] > 0.5) & usable
        return {
            "studies": int(mask.sum()),
            "scanner_profiles": int(rows.loc[mask, "scanner_profile"].nunique()),
            "usable_cells": int(usable.sum()),
            "positive_cells": {
                target: int(positive[:, column].sum()) for column, target in enumerate(TARGETS)
            },
            "negative_cells": {
                target: int((usable[:, column] & ~positive[:, column]).sum())
                for column, target in enumerate(TARGETS)
            },
        }

    seen = set(seen_validation_uids or ())
    is_seen = ~rows["holdout"] & rows["StudyInstanceUID"].isin(seen)

    held = side(rows["holdout"])
    seen_validation = side(is_seen)
    kept = side(~rows["holdout"] & ~is_seen)

    # An AUC needs at least one positive and one negative. A grouped holdout can
    # easily lose both for a rare finding, and a target that cannot be scored
    # should be named rather than silently averaged in as a missing value.
    def _measurable(group: dict) -> list[str]:
        return [
            target
            for target in TARGETS
            if group["positive_cells"][target] > 0 and group["negative_cells"][target] > 0
        ]

    measurable = _measurable(held)
    # The domain gap is only defined for targets both sides can score.
    comparable = (
        sorted(set(measurable).intersection(_measurable(seen_validation)), key=TARGETS.index)
        if seen
        else []
    )

    sizes = rows["scanner_profile"].value_counts()
    largest_share = float(sizes.iloc[0] / len(rows)) if len(sizes) else 0.0

    return {
        "holdout_unseen_scanners": held,
        "validation_seen_scanners": seen_validation,
        "training": kept,
        "holdout_study_fraction": float(held["studies"] / len(rows)) if len(rows) else 0.0,
        "measurable_targets": measurable,
        "unmeasurable_targets": [t for t in TARGETS if t not in measurable],
        "comparable_targets": comparable,
        "distinct_scanner_profiles": int(rows["scanner_profile"].nunique()),
        "largest_profile": str(sizes.index[0]) if len(sizes) else "",
        "largest_profile_share": largest_share,
        "concentration_warning": bool(largest_share > CONCENTRATION_WARNING_SHARE),
        "studies_with_mixed_series_profiles": int((rows["distinct_profiles"] > 1).sum()),
    }


def verify_domain_split(rows: pd.DataFrame) -> None:
    """The one property the whole surface rests on: no profile on both sides."""
    sides = rows.groupby("scanner_profile")["holdout"].nunique()
    straddling = sorted(sides[sides > 1].index.astype(str))
    if straddling:
        raise ValueError(
            f"{len(straddling)} scanner profile(s) appear in both training and "
            f"holdout, which defeats the split: {straddling[:5]}"
        )
    if "split" not in rows.columns:
        return

    allowed = {"train", "validation_seen_scanners", "holdout_unseen_scanners"}
    unknown = sorted(set(rows["split"].astype(str)).difference(allowed))
    if unknown:
        raise ValueError(f"unknown split labels: {unknown}")

    # The seen-scanner set is only a comparator if its scanners really are ones
    # training keeps. If a profile ended up entirely inside it, that profile is
    # unseen too and the comparison quietly becomes a second unseen holdout.
    trained = set(rows.loc[rows["split"].eq("train"), "scanner_profile"])
    seen = set(rows.loc[rows["split"].eq("validation_seen_scanners"), "scanner_profile"])
    orphaned = sorted(seen.difference(trained))
    if orphaned:
        raise ValueError(
            f"{len(orphaned)} profile(s) appear in the seen-scanner validation set "
            f"but never in training, so they are not 'seen': {orphaned[:5]}"
        )


def build_domain_split(
    *,
    data_root: str | Path,
    header_csv: str | Path,
    labels_root: str | Path,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    salt: str = DOMAIN_SPLIT_SALT,
) -> tuple[dict, pd.DataFrame]:
    """Build the frozen scanner-grouped split and its description."""
    from .phase9_supervision import (
        load_fill_merged_export,
        prepare_all_report_only_supervision,
    )

    root = Path(data_root).resolve()
    train_path = root / "train.csv"
    train = load_train_csv(train_path)
    header = pd.read_csv(header_csv)

    frame, _fill_policy, _fill_audit = load_fill_merged_export(labels_root)
    uids, targets, weights, _supervision = prepare_all_report_only_supervision(train, frame)

    profiles = build_study_profiles(train, header)
    holdout_profiles = choose_holdout_profiles(
        profiles,
        targets,
        weights,
        uids,
        holdout_fraction=holdout_fraction,
        salt=salt,
    )

    rows = profiles.loc[profiles["StudyInstanceUID"].isin(set(uids))].copy()
    rows["holdout"] = rows["scanner_profile"].isin(set(holdout_profiles))

    # Match the comparator's size to the holdout's, so the two scores carry
    # similar noise and their difference is not mostly a sample-size artefact.
    seen_uids = choose_seen_scanner_validation(
        rows.loc[~rows["holdout"]],
        int(rows["holdout"].sum()),
        salt=salt,
    )
    rows["split"] = np.where(
        rows["holdout"],
        "holdout_unseen_scanners",
        np.where(
            rows["StudyInstanceUID"].isin(seen_uids),
            "validation_seen_scanners",
            "train",
        ),
    )
    verify_domain_split(rows)

    summary = summarise_split(
        profiles,
        holdout_profiles,
        targets,
        weights,
        uids,
        seen_validation_uids=seen_uids,
    )
    payload = {
        "version": DOMAIN_SPLIT_VERSION,
        "status": "frozen_before_any_model_is_scored_on_this_surface",
        "source_train_csv": str(train_path),
        "source_train_csv_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "source_header_csv": str(Path(header_csv).resolve()),
        "source_labels_root": str(Path(labels_root).resolve()),
        "salt": salt,
        "requested_holdout_fraction": float(holdout_fraction),
        "group_key": "manufacturer_family|manufacturer_model|field_strength_bin",
        "group_key_is_a_scanner_proxy_not_a_site": (
            "train.csv has no institution column and DICOM institution tags are "
            "frequently blank or anonymised, so two centres running the same "
            "scanner model share one group. This is a lower bound on domain "
            "separation, not a true centre holdout."
        ),
        "assignment_algorithm": (
            "deterministic greedy per-target prevalence balancing over whole "
            "scanner profiles, with SHA-256 over the salt breaking ties"
        ),
        "holdout_scanner_profiles": holdout_profiles,
        "how_to_read_it": (
            "Train one model on the `train` studies only. Score it twice: on "
            "`validation_seen_scanners` (machines it trained on) and on "
            "`holdout_unseen_scanners` (machines it has never met). The gap "
            "between the two, over `comparable_targets`, is the estimate of how "
            "much of the model's performance is acquisition-specific. Neither "
            "number means much on its own."
        ),
        "summary": summary,
        "governance": (
            "Membership is frozen before any model is scored here. This surface "
            "carries report-derived labels, so it measures domain generalisation "
            "and not label quality. Do not select checkpoints, thresholds or "
            "architectures from it without declaring that in advance."
        ),
    }
    return payload, rows


def write_domain_split(
    payload: dict,
    rows: pd.DataFrame,
    *,
    out_root: str | Path,
) -> Path:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    rows.sort_values("StudyInstanceUID").to_csv(out / "domain_split_by_study.csv", index=False)
    path = out / "domain_split.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "domain_split.sha256").write_text(f"{payload_sha}\n", encoding="utf-8")
    return path


def _report(payload: dict) -> None:
    summary = payload["summary"]
    held = summary["holdout_unseen_scanners"]
    seen = summary["validation_seen_scanners"]
    kept = summary["training"]
    print(f"[domain split] {summary['distinct_scanner_profiles']} scanner profiles")
    for name, group in (
        ("train           ", kept),
        ("seen scanners   ", seen),
        ("unseen scanners ", held),
    ):
        print(
            f"[domain split] {name} {group['studies']:>5} studies / "
            f"{group['scanner_profiles']:>3} profiles / {group['usable_cells']:>6} cells"
        )
    print(
        f"[domain split] the holdout is {summary['holdout_study_fraction']:.1%} of studies"
    )
    if summary["unmeasurable_targets"]:
        print(
            "[domain split] NOT measurable on the unseen holdout (no positive or "
            f"no negative): {', '.join(summary['unmeasurable_targets'])}"
        )
    else:
        print("[domain split] all 12 targets are measurable on the unseen holdout")
    print(
        f"[domain split] the domain gap is defined for "
        f"{len(summary['comparable_targets'])} of {len(TARGETS)} targets"
    )
    if summary["concentration_warning"]:
        print(
            f"[domain split] WARNING one profile holds "
            f"{summary['largest_profile_share']:.1%} of studies "
            f"({summary['largest_profile']}); a grouped split cannot separate it"
        )
    if summary["studies_with_mixed_series_profiles"]:
        print(
            f"[domain split] {summary['studies_with_mixed_series_profiles']} "
            "studies have series from more than one profile; the dominant one was used"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a scanner-grouped weak-label validation surface"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--header-csv",
        required=True,
        help="header_by_series.csv from rsna_knee.dataset_header_audit",
    )
    parser.add_argument(
        "--labels-root",
        required=True,
        help="the fill-merged report-label export used for weak supervision",
    )
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--out-root", default="runs/domain_shift_split")
    args = parser.parse_args()

    payload, rows = build_domain_split(
        data_root=args.data_root,
        header_csv=args.header_csv,
        labels_root=args.labels_root,
        holdout_fraction=args.holdout_fraction,
    )
    _report(payload)
    print(write_domain_split(payload, rows, out_root=args.out_root))


if __name__ == "__main__":
    main()
