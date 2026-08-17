from __future__ import annotations

import copy

import numpy as np
import pytest

from rsna_knee.prospective_weak_v1 import build_prospective_weak_v1_manifest
from rsna_knee.prospective_weak_v2 import (
    PV2_LOCKED_PV1_VALIDATION_STUDIES,
    PV2_SOURCE_STUDIES,
    PV2_TRAIN_STUDIES,
    PV2_VALIDATION_STUDIES,
    build_prospective_weak_v2_manifest,
    validate_prospective_weak_v2_manifest,
)


def _surface():
    uids = [f"study-{i:04d}" for i in range(3120)]
    targets = np.full((3120, 12), 0.05, dtype=np.float32)
    targets[::2, :] = 0.85
    weights = np.ones((3120, 12), dtype=np.float32)
    return uids, targets, weights


def _parent_and_v2():
    uids, targets, weights = _surface()
    parent = build_prospective_weak_v1_manifest(uids, targets, weights)
    # Unit tests use synthetic UIDs, so replace only the parent fingerprint field
    # expected by PV2 while retaining the exact validated PV1 assignment. The
    # production creator requires the canonical real-data parent SHA.
    from rsna_knee import prospective_weak_v2 as pv2
    original = pv2.PV2_PARENT_PV1_SPLIT_SHA256
    try:
        pv2.PV2_PARENT_PV1_SPLIT_SHA256 = parent["split_sha256"]
        manifest = build_prospective_weak_v2_manifest(parent, uids, targets, weights)
        validate_prospective_weak_v2_manifest(manifest, parent, uids)
    finally:
        pv2.PV2_PARENT_PV1_SPLIT_SHA256 = original
    return uids, targets, weights, parent, manifest


def test_pv2_exact_nested_counts_and_locked_pv1_validation():
    _, _, _, parent, manifest = _parent_and_v2()
    assert len(parent["training_uids"]) == PV2_SOURCE_STUDIES == 2496
    assert len(manifest["training_uids"]) == PV2_TRAIN_STUDIES == 1997
    assert len(manifest["validation_uids"]) == PV2_VALIDATION_STUDIES == 499
    assert len(manifest["locked_parent_pv1_validation_uids"]) == PV2_LOCKED_PV1_VALIDATION_STUDIES == 624
    assert set(manifest["training_uids"]).isdisjoint(manifest["validation_uids"])
    assert set(manifest["training_uids"]).union(manifest["validation_uids"]) == set(parent["training_uids"])
    assert set(manifest["locked_parent_pv1_validation_uids"]).isdisjoint(manifest["training_uids"])
    assert set(manifest["locked_parent_pv1_validation_uids"]).isdisjoint(manifest["validation_uids"])


def test_pv2_membership_is_uid_only_and_deterministic():
    uids, targets, weights = _surface()
    parent = build_prospective_weak_v1_manifest(uids, targets, weights)
    from rsna_knee import prospective_weak_v2 as pv2
    original = pv2.PV2_PARENT_PV1_SPLIT_SHA256
    try:
        pv2.PV2_PARENT_PV1_SPLIT_SHA256 = parent["split_sha256"]
        a = build_prospective_weak_v2_manifest(parent, uids, targets, weights)
        altered_targets = 0.90 - targets
        altered_weights = weights * 0.37
        b = build_prospective_weak_v2_manifest(parent, uids, altered_targets, altered_weights)
    finally:
        pv2.PV2_PARENT_PV1_SPLIT_SHA256 = original
    assert a["training_uids"] == b["training_uids"]
    assert a["validation_uids"] == b["validation_uids"]
    assert a["split_sha256"] == b["split_sha256"]
    assert a["post_assignment_supervision_audit"] != b["post_assignment_supervision_audit"]


def test_pv2_tampering_is_rejected():
    uids, _, _, parent, manifest = _parent_and_v2()
    broken = copy.deepcopy(manifest)
    broken["validation_uids"][0], broken["training_uids"][0] = (
        broken["training_uids"][0], broken["validation_uids"][0]
    )
    broken["validation_uids"].sort(); broken["training_uids"].sort()
    from rsna_knee import prospective_weak_v2 as pv2
    original = pv2.PV2_PARENT_PV1_SPLIT_SHA256
    try:
        pv2.PV2_PARENT_PV1_SPLIT_SHA256 = parent["split_sha256"]
        with pytest.raises(ValueError):
            validate_prospective_weak_v2_manifest(broken, parent, uids)
    finally:
        pv2.PV2_PARENT_PV1_SPLIT_SHA256 = original


def test_pv2_explicitly_records_historical_exposure_limitation():
    _, _, _, _, manifest = _parent_and_v2()
    assert manifest["historical_b16_encoder_saw_pv2_validation_reports"] is True
    assert manifest["historical_downstream_models_saw_pv2_validation_in_gradients"] is True
    assert manifest["parent_pv1_validation_locked"] is True
    assert manifest["parent_pv1_validation_reused"] is False
