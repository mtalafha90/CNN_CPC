"""Nested prospective weak-validation v2 for the post-PV1 B34 experiment.

PV2 is frozen only to prevent further direct optimization against the now-exposed
PV1 validation surface.  It is NOT independent clinical validation.  Its source
pool is the 2,496 studies that formed the PV1 training partition, and those
studies were historically used in downstream gradients before PV2 existed.
Therefore PV2 is a fresh *metric surface* for matched B34 retraining, but not a
population unseen by the historical development process.

Membership uses StudyInstanceUID hashing only.  B6 labels, expert labels, model
predictions, PV1 outcomes, B29 outcomes, and B31 counterfactual outcomes never
enter assignment.  The original 624-study PV1 validation partition is locked and
excluded from both PV2 training and PV2 validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .b7_weak_supervision import _read_config, load_frozen_b6_export, prepare_b7_supervision
from .constants import TARGETS
from .data import load_train_csv
from .prospective_weak_v1 import (
    PV1_TOTAL_STUDIES,
    PV1_TRAIN_STUDIES,
    PV1_VALIDATION_STUDIES,
    validate_prospective_weak_v1_manifest,
)

PV2_VERSION = "prospective_weak_nested_pv1train_hash_80_20_v1"
PV2_SALT = "CNN_CPC|prospective-weak-v2|parent-pv1-train|2026-08-17"
PV2_PARENT_PV1_SPLIT_SHA256 = "a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6"
PV2_SOURCE_STUDIES = PV1_TRAIN_STUDIES
PV2_TRAIN_STUDIES = 1997
PV2_VALIDATION_STUDIES = 499
PV2_LOCKED_PV1_VALIDATION_STUDIES = PV1_VALIDATION_STUDIES
PV2_VALIDATION_FRACTION_OF_SOURCE = PV2_VALIDATION_STUDIES / PV2_SOURCE_STUDIES


def _sha_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(str(x) for x in values).encode("utf-8")).hexdigest()


def _assignment_key(uid: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{PV2_SALT}\0{uid}".encode("utf-8")).hexdigest()
    return digest, str(uid)


def _partition_audit(uids: list[str], targets: np.ndarray, weights: np.ndarray) -> dict:
    if len(uids) != int(targets.shape[0]) or targets.shape != weights.shape:
        raise ValueError("PV2 partition arrays are not aligned")
    active = weights > 0
    positive = active & (targets > 0.5)
    negative = active & (targets < 0.5)
    per_target = {}
    for j, target in enumerate(TARGETS):
        per_target[target] = {
            "usable_cells": int(active[:, j].sum()),
            "positive_cells": int(positive[:, j].sum()),
            "negative_cells": int(negative[:, j].sum()),
            "weight_mass": float(weights[:, j].sum()),
        }
    return {
        "studies": int(len(uids)),
        "usable_cells": int(active.sum()),
        "positive_cells": int(positive.sum()),
        "negative_cells": int(negative.sum()),
        "per_target": per_target,
    }


def build_prospective_weak_v2_manifest(
    parent_pv1_manifest: dict,
    study_uids: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict:
    uids = [str(uid) for uid in study_uids]
    if len(uids) != PV1_TOTAL_STUDIES or len(set(uids)) != PV1_TOTAL_STUDIES:
        raise ValueError(f"PV2 requires the exact {PV1_TOTAL_STUDIES}-study active B6 surface")
    if targets.shape != (PV1_TOTAL_STUDIES, len(TARGETS)) or weights.shape != targets.shape:
        raise ValueError("PV2 requires the exact [3120,12] B6 supervision arrays")

    validate_prospective_weak_v1_manifest(parent_pv1_manifest, uids)
    parent_sha = str(parent_pv1_manifest.get("split_sha256", ""))
    if parent_sha != PV2_PARENT_PV1_SPLIT_SHA256:
        raise ValueError("PV2 requires the exact frozen PV1 parent split")

    source_uids = [str(x) for x in parent_pv1_manifest["training_uids"]]
    locked_pv1_validation_uids = [str(x) for x in parent_pv1_manifest["validation_uids"]]
    if len(source_uids) != PV2_SOURCE_STUDIES:
        raise RuntimeError("PV2 source study count changed")
    if len(locked_pv1_validation_uids) != PV2_LOCKED_PV1_VALIDATION_STUDIES:
        raise RuntimeError("PV2 locked PV1-validation study count changed")
    if set(source_uids).intersection(locked_pv1_validation_uids):
        raise RuntimeError("PV2 parent PV1 partitions overlap")

    # Assignment is UID-only. Weak labels are inspected only after membership is frozen.
    ranked = sorted(source_uids, key=_assignment_key)
    validation_members = set(ranked[:PV2_VALIDATION_STUDIES])
    validation_uids = sorted(validation_members)
    training_uids = sorted(uid for uid in source_uids if uid not in validation_members)
    if len(training_uids) != PV2_TRAIN_STUDIES or len(validation_uids) != PV2_VALIDATION_STUDIES:
        raise RuntimeError("PV2 study-count contract failed")
    if set(training_uids).intersection(validation_uids):
        raise RuntimeError("PV2 train/validation overlap detected")
    if set(training_uids).union(validation_uids) != set(source_uids):
        raise RuntimeError("PV2 no longer partitions the exact PV1 training pool")
    if set(locked_pv1_validation_uids).intersection(training_uids + validation_uids):
        raise RuntimeError("PV2 illegally reuses the locked PV1 validation partition")

    row = {uid: i for i, uid in enumerate(uids)}
    train_idx = np.asarray([row[uid] for uid in training_uids], dtype=np.int64)
    val_idx = np.asarray([row[uid] for uid in validation_uids], dtype=np.int64)
    locked_idx = np.asarray([row[uid] for uid in locked_pv1_validation_uids], dtype=np.int64)

    split_core = {
        "version": PV2_VERSION,
        "salt": PV2_SALT,
        "parent_pv1_split_sha256": parent_sha,
        "assignment": "sort only frozen PV1-training StudyInstanceUIDs by SHA256(salt\\0uid); first 499 validation",
        "labels_used_to_assign_split": False,
        "expert_labels_used_to_assign_split": False,
        "model_outputs_used_to_assign_split": False,
        "pv1_outcomes_used_to_assign_split": False,
        "b29_addendum_outcomes_used_to_assign_split": False,
        "b31_counterfactual_outcomes_used_to_assign_split": False,
        "parent_pv1_validation_locked": True,
        "parent_pv1_validation_reused": False,
        "study_level_split": True,
        "patient_level_separation_certified": False,
        "historical_b16_encoder_saw_pv2_validation_reports": True,
        "historical_downstream_models_saw_pv2_validation_in_gradients": True,
        "exposure_note": (
            "PV2 validation studies were part of historical downstream training before PV2 was defined. "
            "PV2 is therefore a newly hidden metric surface for matched B34 retraining, not an independent "
            "or historically untouched validation population."
        ),
        "selection_scope": (
            "internal post-PV1 fixed-encoder B34 mechanism testing only; not independent clinical validation, "
            "not encoder selection, and not evidence sufficient to replace the active historical model"
        ),
        "source_studies": PV2_SOURCE_STUDIES,
        "training_studies": PV2_TRAIN_STUDIES,
        "validation_studies": PV2_VALIDATION_STUDIES,
        "locked_parent_pv1_validation_studies": PV2_LOCKED_PV1_VALIDATION_STUDIES,
        "validation_fraction_of_source": PV2_VALIDATION_FRACTION_OF_SOURCE,
        "training_uids": training_uids,
        "validation_uids": validation_uids,
        "locked_parent_pv1_validation_uids": locked_pv1_validation_uids,
        "source_uid_sha256": _sha_lines(sorted(source_uids)),
        "training_uid_sha256": _sha_lines(training_uids),
        "validation_uid_sha256": _sha_lines(validation_uids),
        "locked_parent_pv1_validation_uid_sha256": _sha_lines(locked_pv1_validation_uids),
    }
    split_core["split_sha256"] = hashlib.sha256(
        json.dumps(split_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    split_core["post_assignment_supervision_audit"] = {
        "training": _partition_audit(training_uids, targets[train_idx], weights[train_idx]),
        "validation": _partition_audit(validation_uids, targets[val_idx], weights[val_idx]),
        "locked_parent_pv1_validation": _partition_audit(
            locked_pv1_validation_uids, targets[locked_idx], weights[locked_idx]
        ),
        "audit_role": "descriptive only; labels did not influence PV2 membership",
    }
    return split_core


def validate_prospective_weak_v2_manifest(
    manifest: dict,
    parent_pv1_manifest: dict,
    active_uids: list[str] | None = None,
) -> dict:
    if manifest.get("version") != PV2_VERSION or manifest.get("salt") != PV2_SALT:
        raise ValueError("not the frozen prospective weak-v2 split policy")
    if str(manifest.get("parent_pv1_split_sha256", "")) != PV2_PARENT_PV1_SPLIT_SHA256:
        raise ValueError("PV2 parent PV1 fingerprint changed")
    for key in (
        "labels_used_to_assign_split",
        "expert_labels_used_to_assign_split",
        "model_outputs_used_to_assign_split",
        "pv1_outcomes_used_to_assign_split",
        "b29_addendum_outcomes_used_to_assign_split",
        "b31_counterfactual_outcomes_used_to_assign_split",
        "parent_pv1_validation_reused",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"PV2 assignment/governance flag changed: {key}")
    if manifest.get("parent_pv1_validation_locked") is not True:
        raise ValueError("PV2 must lock the original PV1 validation partition")
    if manifest.get("historical_downstream_models_saw_pv2_validation_in_gradients") is not True:
        raise ValueError("PV2 must retain its historical downstream-exposure limitation")

    if active_uids is not None:
        validate_prospective_weak_v1_manifest(parent_pv1_manifest, active_uids)
    else:
        validate_prospective_weak_v1_manifest(parent_pv1_manifest)
    if str(parent_pv1_manifest.get("split_sha256", "")) != PV2_PARENT_PV1_SPLIT_SHA256:
        raise ValueError("PV2 validator received the wrong parent PV1 manifest")

    source = [str(x) for x in parent_pv1_manifest["training_uids"]]
    locked = [str(x) for x in parent_pv1_manifest["validation_uids"]]
    train = [str(x) for x in manifest.get("training_uids", [])]
    val = [str(x) for x in manifest.get("validation_uids", [])]
    manifest_locked = [str(x) for x in manifest.get("locked_parent_pv1_validation_uids", [])]

    if len(train) != PV2_TRAIN_STUDIES or len(val) != PV2_VALIDATION_STUDIES:
        raise ValueError("PV2 study counts changed")
    if train != sorted(train) or val != sorted(val) or manifest_locked != sorted(manifest_locked):
        raise ValueError("PV2 UID lists must remain lexicographically sorted")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("PV2 contains duplicate studies")
    if set(train).intersection(val):
        raise ValueError("PV2 train/validation overlap detected")
    if set(train).union(val) != set(source):
        raise ValueError("PV2 does not exactly partition the frozen PV1 training pool")
    if manifest_locked != locked:
        raise ValueError("PV2 locked PV1 validation list changed")
    if set(locked).intersection(train + val):
        raise ValueError("PV2 reused the locked PV1 validation partition")

    expected_ranked = sorted(source, key=_assignment_key)
    expected_val = sorted(expected_ranked[:PV2_VALIDATION_STUDIES])
    expected_train = sorted(expected_ranked[PV2_VALIDATION_STUDIES:])
    if train != expected_train or val != expected_val:
        raise ValueError("PV2 manifest does not reproduce frozen UID-hash assignment")

    if manifest.get("source_uid_sha256") != _sha_lines(sorted(source)):
        raise ValueError("PV2 source UID fingerprint changed")
    if manifest.get("training_uid_sha256") != _sha_lines(train):
        raise ValueError("PV2 training UID fingerprint changed")
    if manifest.get("validation_uid_sha256") != _sha_lines(val):
        raise ValueError("PV2 validation UID fingerprint changed")
    if manifest.get("locked_parent_pv1_validation_uid_sha256") != _sha_lines(locked):
        raise ValueError("PV2 locked PV1-validation fingerprint changed")
    return manifest


def create_manifest_from_files(
    config: dict,
    *,
    b6_root: str | Path,
    parent_pv1_manifest_path: str | Path,
    out_path: str | Path,
) -> Path:
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    uids, targets, weights, _ = prepare_b7_supervision(train, b6_frame)
    uids = [str(x) for x in uids]
    parent = json.loads(Path(parent_pv1_manifest_path).read_text(encoding="utf-8"))
    manifest = build_prospective_weak_v2_manifest(parent, uids, targets, weights)
    validate_prospective_weak_v2_manifest(manifest, parent, uids)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "version": manifest["version"],
        "split_sha256": manifest["split_sha256"],
        "parent_pv1_split_sha256": manifest["parent_pv1_split_sha256"],
        "training_studies": manifest["training_studies"],
        "validation_studies": manifest["validation_studies"],
        "locked_parent_pv1_validation_studies": manifest["locked_parent_pv1_validation_studies"],
        "labels_used_to_assign_split": False,
        "historical_downstream_exposure": True,
        "path": str(path),
    }, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser("Create frozen nested prospective weak-validation split v2")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--parent-pv1-manifest", required=True)
    ap.add_argument("--out", default="runs/prospective_weak_v2/split_manifest.json")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    create_manifest_from_files(
        config,
        b6_root=args.b6_root,
        parent_pv1_manifest_path=args.parent_pv1_manifest,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
