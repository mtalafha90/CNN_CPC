"""Prospective weak-validation split for post-B33 architecture development.

This module freezes a study-level 80/20 partition of the 3,120 active B6
weak-supervision studies. Assignment uses StudyInstanceUID hashing only: no B6
states, confidences, expert labels, model predictions, or previous development
scores enter split assignment.

The purpose is architecture selection after the repeatedly reused 58-study
expert surface became too exposed. This is still WEAK-LABEL validation, not
independent clinical validation. The historical B16 encoder was aligned on all
4,349 non-gold reports, so this split is valid only for comparing downstream
architectures while that exact encoder remains frozen and shared.
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

PV1_VERSION = "prospective_weak_study_hash_80_20_v1"
PV1_SALT = "CNN_CPC|prospective-weak-v1|2026-08-16"
PV1_TOTAL_STUDIES = 3120
PV1_TRAIN_STUDIES = 2496
PV1_VALIDATION_STUDIES = 624
PV1_VALIDATION_FRACTION = 0.20


def _sha_lines(values: list[str]) -> str:
    payload = "\n".join(str(x) for x in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assignment_key(uid: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{PV1_SALT}\0{uid}".encode("utf-8")).hexdigest()
    return digest, str(uid)


def _partition_audit(uids: list[str], targets: np.ndarray, weights: np.ndarray) -> dict:
    if len(uids) != int(targets.shape[0]) or targets.shape != weights.shape:
        raise ValueError("PV1 partition arrays are not aligned")
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


def build_prospective_weak_v1_manifest(
    study_uids: list[str], targets: np.ndarray, weights: np.ndarray
) -> dict:
    uids = [str(uid) for uid in study_uids]
    if len(uids) != PV1_TOTAL_STUDIES or len(set(uids)) != PV1_TOTAL_STUDIES:
        raise ValueError(f"PV1 requires exactly {PV1_TOTAL_STUDIES} unique active studies")
    if targets.shape != (PV1_TOTAL_STUDIES, len(TARGETS)) or weights.shape != targets.shape:
        raise ValueError("PV1 requires the exact [3120,12] B6 supervision arrays")

    # Membership is assigned from UID hashes only. Labels are inspected only
    # after assignment to document the resulting weak-label composition.
    ranked = sorted(uids, key=_assignment_key)
    val_members = set(ranked[:PV1_VALIDATION_STUDIES])
    validation_uids = sorted(val_members)
    training_uids = sorted(uid for uid in uids if uid not in val_members)
    if len(training_uids) != PV1_TRAIN_STUDIES or len(validation_uids) != PV1_VALIDATION_STUDIES:
        raise RuntimeError("PV1 80/20 study-count contract failed")

    row = {uid: i for i, uid in enumerate(uids)}
    train_idx = np.asarray([row[uid] for uid in training_uids], dtype=np.int64)
    val_idx = np.asarray([row[uid] for uid in validation_uids], dtype=np.int64)

    split_core = {
        "version": PV1_VERSION,
        "salt": PV1_SALT,
        "assignment": "sort active StudyInstanceUIDs by SHA256(salt\\0uid); first 624 validation",
        "labels_used_to_assign_split": False,
        "expert_labels_used_to_assign_split": False,
        "model_outputs_used_to_assign_split": False,
        "study_level_split": True,
        "patient_level_separation_certified": False,
        "patient_level_note": (
            "The frozen supervision surface is keyed by StudyInstanceUID. This policy certifies study-level "
            "separation only; it does not claim patient-identity grouping."
        ),
        "historical_b16_encoder_saw_validation_reports": True,
        "historical_b16_report_alignment_studies": 4349,
        "selection_scope": (
            "downstream architecture comparison only with the exact shared frozen B16 encoder; "
            "not valid for selecting a new encoder or representation-pretraining method"
        ),
        "total_studies": PV1_TOTAL_STUDIES,
        "training_studies": PV1_TRAIN_STUDIES,
        "validation_studies": PV1_VALIDATION_STUDIES,
        "validation_fraction": PV1_VALIDATION_FRACTION,
        "training_uids": training_uids,
        "validation_uids": validation_uids,
        "full_uid_sha256": _sha_lines(sorted(uids)),
        "training_uid_sha256": _sha_lines(training_uids),
        "validation_uid_sha256": _sha_lines(validation_uids),
    }
    split_core["split_sha256"] = hashlib.sha256(
        json.dumps(split_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    split_core["post_assignment_supervision_audit"] = {
        "training": _partition_audit(training_uids, targets[train_idx], weights[train_idx]),
        "validation": _partition_audit(validation_uids, targets[val_idx], weights[val_idx]),
        "audit_role": "descriptive only; weak labels did not influence membership",
    }
    return split_core


def validate_prospective_weak_v1_manifest(manifest: dict, active_uids: list[str] | None = None) -> dict:
    if manifest.get("version") != PV1_VERSION or manifest.get("salt") != PV1_SALT:
        raise ValueError("not the frozen prospective weak-v1 split policy")
    if manifest.get("labels_used_to_assign_split") is not False:
        raise ValueError("PV1 split unexpectedly certifies label-based assignment")
    if manifest.get("expert_labels_used_to_assign_split") is not False:
        raise ValueError("PV1 split unexpectedly certifies expert-label assignment")
    if manifest.get("model_outputs_used_to_assign_split") is not False:
        raise ValueError("PV1 split unexpectedly certifies model-output assignment")
    if manifest.get("historical_b16_encoder_saw_validation_reports") is not True:
        raise ValueError("PV1 must explicitly retain the historical B16 pretraining-overlap limitation")
    if int(manifest.get("historical_b16_report_alignment_studies", -1)) != 4349:
        raise ValueError("PV1 B16 report-alignment provenance changed")

    training_uids = [str(x) for x in manifest.get("training_uids", [])]
    validation_uids = [str(x) for x in manifest.get("validation_uids", [])]
    if len(training_uids) != PV1_TRAIN_STUDIES or len(validation_uids) != PV1_VALIDATION_STUDIES:
        raise ValueError("PV1 split study counts changed")
    if len(set(training_uids)) != len(training_uids) or len(set(validation_uids)) != len(validation_uids):
        raise ValueError("PV1 split contains duplicate studies")
    if set(training_uids).intersection(validation_uids):
        raise ValueError("PV1 train/validation overlap detected")
    if training_uids != sorted(training_uids) or validation_uids != sorted(validation_uids):
        raise ValueError("PV1 UID lists must remain lexicographically sorted")

    if manifest.get("training_uid_sha256") != _sha_lines(training_uids):
        raise ValueError("PV1 training UID fingerprint changed")
    if manifest.get("validation_uid_sha256") != _sha_lines(validation_uids):
        raise ValueError("PV1 validation UID fingerprint changed")

    if active_uids is not None:
        active = [str(x) for x in active_uids]
        if len(active) != PV1_TOTAL_STUDIES or len(set(active)) != PV1_TOTAL_STUDIES:
            raise ValueError("PV1 active supervision surface changed")
        if set(training_uids).union(validation_uids) != set(active):
            raise ValueError("PV1 split no longer partitions the active B6 surface exactly")
        expected = sorted(active, key=_assignment_key)
        expected_val = sorted(expected[:PV1_VALIDATION_STUDIES])
        expected_train = sorted(expected[PV1_VALIDATION_STUDIES:])
        if validation_uids != expected_val or training_uids != expected_train:
            raise ValueError("PV1 manifest does not reproduce frozen UID-hash assignment")
        if manifest.get("full_uid_sha256") != _sha_lines(sorted(active)):
            raise ValueError("PV1 full UID fingerprint changed")
    return manifest


def create_manifest_from_files(config: dict, *, b6_root: str | Path, out_path: str | Path) -> Path:
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    uids, targets, weights, _ = prepare_b7_supervision(train, b6_frame)
    manifest = build_prospective_weak_v1_manifest(uids, targets, weights)
    validate_prospective_weak_v1_manifest(manifest, uids)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "version": manifest["version"],
        "split_sha256": manifest["split_sha256"],
        "training_studies": manifest["training_studies"],
        "validation_studies": manifest["validation_studies"],
        "labels_used_to_assign_split": False,
        "historical_b16_encoder_saw_validation_reports": True,
        "selection_scope": manifest["selection_scope"],
        "path": str(path),
    }, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser("Create frozen prospective weak-validation split v1")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--out", default="runs/prospective_weak_v1/split_manifest.json")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    create_manifest_from_files(config, b6_root=args.b6_root, out_path=args.out)


if __name__ == "__main__":
    main()
