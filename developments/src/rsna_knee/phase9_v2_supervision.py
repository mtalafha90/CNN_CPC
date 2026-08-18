"""Phase 9 v2 supervision contract with the frozen PV2 validation holdout.

Phase 9 v1 exposed all 4,349 report-only studies in both gradients.  That made
its MRI exposure comparison honest, but it also consumed the 499-study PV2
validation surface.  V2 removes the exact same frozen PV2 validation UIDs from
both arms before downstream training and evaluates both arms on the untouched
original-B6 supervision for those studies.

PV2 is not independent clinical validation: it has historical downstream
exposure and the B16 encoder saw its reports.  Its valid role here is a fixed,
label-independent, no-Phase9-gradient weak-label readout for the causal question
of whether the Phase-8 supervision addition improves a fixed B34 training path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
from .phase9_supervision import load_phase9_arm_supervision
from .prospective_weak_v1 import validate_prospective_weak_v1_manifest
from .prospective_weak_v2 import (
    PV2_PARENT_PV1_SPLIT_SHA256,
    PV2_VALIDATION_STUDIES,
    PV2_VERSION,
    validate_prospective_weak_v2_manifest,
)

PHASE9_V2_VERSION = "phase9_matched_b34_b6_vs_phase8_pv2_holdout_v2"
PHASE9_V2_TRAIN_STUDIES = 3850
PHASE9_V2_HOLDOUT_STUDIES = PV2_VALIDATION_STUDIES
PHASE9_V2_HOLDOUT_SERIES = 2775
PHASE9_V2_TRAIN_SERIES = 21260
PHASE9_V2_BATCHES_BATCH2 = 1925
PHASE9_V2_PV2_SPLIT_SHA256 = "b53331ce314b2d2ccc68aea1737427c01bd0d916997e78fbefe88fec5cc95855"


def _sha_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(str(x) for x in values).encode("utf-8")).hexdigest()


def load_phase9_v2_holdout(
    train_df: pd.DataFrame,
    *,
    b6_root: str | Path,
    parent_pv1_manifest_path: str | Path,
    pv2_manifest_path: str | Path,
) -> dict:
    """Load and validate the exact frozen 499-study PV2 validation surface."""
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    active_uids, active_targets, active_weights, _ = prepare_b7_supervision(train_df, b6_frame)
    active_uids = [str(x) for x in active_uids]

    parent = json.loads(Path(parent_pv1_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v1_manifest(parent, active_uids)
    if str(parent.get("split_sha256", "")) != PV2_PARENT_PV1_SPLIT_SHA256:
        raise ValueError("Phase 9 v2 requires the exact frozen parent PV1 split")

    pv2 = json.loads(Path(pv2_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v2_manifest(pv2, parent, active_uids)
    if str(pv2.get("version", "")) != PV2_VERSION:
        raise ValueError("Phase 9 v2 requires the frozen PV2 version")
    if str(pv2.get("split_sha256", "")) != PHASE9_V2_PV2_SPLIT_SHA256:
        raise ValueError("Phase 9 v2 PV2 split fingerprint changed")

    holdout_uids = [str(x) for x in pv2["validation_uids"]]
    if len(holdout_uids) != PHASE9_V2_HOLDOUT_STUDIES:
        raise RuntimeError("Phase 9 v2 holdout count changed")
    row = {uid: i for i, uid in enumerate(active_uids)}
    try:
        idx = np.asarray([row[uid] for uid in holdout_uids], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Phase 9 v2 holdout UID missing from original B6-active surface: {exc}") from exc

    targets = active_targets[idx]
    weights = active_weights[idx]
    expected = pv2["post_assignment_supervision_audit"]["validation"]
    observed = {
        "usable_cells": int((weights > 0).sum()),
        "positive_cells": int(((weights > 0) & (targets > 0.5)).sum()),
        "negative_cells": int(((weights > 0) & (targets < 0.5)).sum()),
    }
    for key, value in observed.items():
        if value != int(expected[key]):
            raise RuntimeError(f"Phase 9 v2 PV2 holdout {key} changed: {value} != {expected[key]}")

    return {
        "uids": holdout_uids,
        "targets": targets,
        "weights": weights,
        "pv2_split_sha256": str(pv2["split_sha256"]),
        "pv2_validation_uid_sha256": str(pv2["validation_uid_sha256"]),
        "parent_pv1_split_sha256": str(parent["split_sha256"]),
        "supervision_audit": observed,
        "limitation": str(pv2["exposure_note"]),
    }


def prepare_phase9_v2_arm_supervision(
    train_df: pd.DataFrame,
    *,
    arm: str,
    b6_root: str | Path,
    phase8_root: str | Path,
    parent_pv1_manifest_path: str | Path,
    pv2_manifest_path: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, dict, dict, dict]:
    """Prepare one matched training arm after removing the exact PV2 holdout.

    The candidate holdout must reproduce original B6 targets/weights exactly.
    This proves the validation labels are not part of the Phase-8 treatment.
    """
    all_uids, all_targets, all_weights, all_summary, source = load_phase9_arm_supervision(
        train_df,
        arm=arm,
        b6_root=b6_root,
        phase8_root=phase8_root,
    )
    all_uids = [str(x) for x in all_uids]
    holdout = load_phase9_v2_holdout(
        train_df,
        b6_root=b6_root,
        parent_pv1_manifest_path=parent_pv1_manifest_path,
        pv2_manifest_path=pv2_manifest_path,
    )
    holdout_uids = [str(x) for x in holdout["uids"]]
    holdout_set = set(holdout_uids)
    if len(holdout_set) != PHASE9_V2_HOLDOUT_STUDIES:
        raise RuntimeError("Phase 9 v2 holdout contains duplicate UIDs")
    if not holdout_set.issubset(set(all_uids)):
        raise RuntimeError("Phase 9 v2 holdout is not a subset of the report-only population")

    row = {uid: i for i, uid in enumerate(all_uids)}
    hold_idx = np.asarray([row[uid] for uid in holdout_uids], dtype=np.int64)
    if not np.array_equal(all_targets[hold_idx], holdout["targets"]):
        raise RuntimeError(f"Phase 9 v2 {arm} holdout targets differ from frozen original B6")
    if not np.array_equal(all_weights[hold_idx], holdout["weights"]):
        raise RuntimeError(f"Phase 9 v2 {arm} holdout weights differ from frozen original B6")

    keep = np.asarray([uid not in holdout_set for uid in all_uids], dtype=bool)
    train_uids = [uid for uid, use in zip(all_uids, keep) if use]
    targets = all_targets[keep]
    weights = all_weights[keep]
    if len(train_uids) != PHASE9_V2_TRAIN_STUDIES:
        raise RuntimeError("Phase 9 v2 training study count changed")

    active = weights > 0
    summary = {
        "report_only_population": int(len(all_uids)),
        "training_studies": int(len(train_uids)),
        "held_out_pv2_studies": int(len(holdout_uids)),
        "active_training_studies": int((weights.sum(axis=1) > 0).sum()),
        "zero_weight_training_studies": int((weights.sum(axis=1) == 0).sum()),
        "usable_cells": int(active.sum()),
        "positive_cells": int((active & (targets > 0.5)).sum()),
        "negative_cells": int((active & (targets < 0.5)).sum()),
        "all_arm_summary_before_holdout": all_summary,
        "pv2_holdout_original_b6_labels_unchanged_in_arm": True,
        "training_uid_sha256": _sha_lines(train_uids),
    }
    source = dict(source)
    source.update(
        {
            "phase9_v2_version": PHASE9_V2_VERSION,
            "pv2_split_sha256": holdout["pv2_split_sha256"],
            "pv2_validation_uid_sha256": holdout["pv2_validation_uid_sha256"],
            "pv2_holdout_removed_before_gradients": True,
        }
    )
    return train_uids, targets, weights, summary, source, holdout
