from __future__ import annotations

import copy

import numpy as np
import pytest

from rsna_knee.prospective_weak_v1 import (
    PV1_TRAIN_STUDIES,
    PV1_VALIDATION_STUDIES,
    build_prospective_weak_v1_manifest,
    validate_prospective_weak_v1_manifest,
)
from rsna_knee.prospective_weak_v1_eval import (
    PV1_EVAL_BATCH_SIZE,
    PV1_EVAL_NUM_WORKERS,
    PV1_EVAL_PERSISTENT_WORKERS,
    PV1_EVAL_PREFETCH_FACTOR,
    PV1_EVAL_SERIES_CACHE_MB,
    low_memory_eval_config,
    macro_weighted_soft_bce,
    paired_bootstrap_loss_difference,
    weak_state_auc,
)


def _surface():
    uids = [f"study-{i:04d}" for i in range(3120)]
    targets = np.full((3120, 12), 0.05, dtype=np.float32)
    targets[::2, :] = 0.85
    weights = np.ones((3120, 12), dtype=np.float32)
    return uids, targets, weights


def test_pv1_split_is_exact_disjoint_and_deterministic():
    uids, targets, weights = _surface()
    a = build_prospective_weak_v1_manifest(uids, targets, weights)
    b = build_prospective_weak_v1_manifest(list(reversed(uids)), targets[::-1], weights[::-1])
    assert len(a["training_uids"]) == PV1_TRAIN_STUDIES
    assert len(a["validation_uids"]) == PV1_VALIDATION_STUDIES
    assert set(a["training_uids"]).isdisjoint(a["validation_uids"])
    assert a["training_uids"] == b["training_uids"]
    assert a["validation_uids"] == b["validation_uids"]
    validate_prospective_weak_v1_manifest(a, uids)


def test_pv1_membership_does_not_depend_on_labels():
    uids, targets, weights = _surface()
    a = build_prospective_weak_v1_manifest(uids, targets, weights)
    altered_targets = 0.90 - targets
    altered_weights = weights * 0.37
    b = build_prospective_weak_v1_manifest(uids, altered_targets, altered_weights)
    assert a["training_uids"] == b["training_uids"]
    assert a["validation_uids"] == b["validation_uids"]
    assert a["split_sha256"] == b["split_sha256"]
    assert a["post_assignment_supervision_audit"] != b["post_assignment_supervision_audit"]


def test_pv1_manifest_tampering_is_rejected():
    uids, targets, weights = _surface()
    manifest = build_prospective_weak_v1_manifest(uids, targets, weights)
    broken = copy.deepcopy(manifest)
    broken["validation_uids"][0], broken["training_uids"][0] = (
        broken["training_uids"][0],
        broken["validation_uids"][0],
    )
    broken["validation_uids"].sort()
    broken["training_uids"].sort()
    with pytest.raises(ValueError):
        validate_prospective_weak_v1_manifest(broken, uids)


def test_pv1_primary_loss_prefers_better_probabilities():
    targets = np.tile(np.asarray([[0.85, 0.05]], dtype=np.float64), (20, 6))
    weights = np.ones_like(targets)
    good = np.where(targets > 0.5, 0.82, 0.08)
    bad = np.full_like(targets, 0.5)
    good_loss = macro_weighted_soft_bce(targets, weights, good)["macro_weighted_soft_bce"]
    bad_loss = macro_weighted_soft_bce(targets, weights, bad)["macro_weighted_soft_bce"]
    assert good_loss < bad_loss


def test_pv1_weak_auc_reports_all_defined_targets():
    n = 40
    targets = np.full((n, 12), 0.05, dtype=np.float64)
    targets[: n // 2] = 0.85
    weights = np.ones_like(targets)
    pred = np.where(targets > 0.5, 0.9, 0.1)
    result = weak_state_auc(targets, weights, pred)
    assert result["n_defined_targets"] == 12
    assert result["macro_auc_defined_targets"] == pytest.approx(1.0)


def test_pv1_paired_bootstrap_sign_is_candidate_minus_reference():
    n = 80
    targets = np.full((n, 12), 0.05, dtype=np.float64)
    targets[: n // 2] = 0.85
    weights = np.ones_like(targets)
    reference = np.full_like(targets, 0.5)
    candidate = np.where(targets > 0.5, 0.82, 0.08)
    result = paired_bootstrap_loss_difference(
        targets,
        weights,
        reference,
        candidate,
        n_bootstrap=200,
        seed=17,
    )
    assert result["median_difference"] < 0
    assert result["probability_candidate_better"] > 0.99


def test_pv1_low_memory_eval_policy_is_frozen_and_nonmutating():
    original = {
        "num_workers": 6,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "series_cache_mb_per_worker": 256,
        "b7_eval_batch_size": 2,
        "device": "auto",
    }
    before = copy.deepcopy(original)
    safe = low_memory_eval_config(original)
    assert original == before
    assert safe["device"] == "auto"
    assert safe["b7_eval_batch_size"] == PV1_EVAL_BATCH_SIZE == 1
    assert safe["num_workers"] == PV1_EVAL_NUM_WORKERS == 1
    assert safe["prefetch_factor"] == PV1_EVAL_PREFETCH_FACTOR == 1
    assert safe["persistent_workers"] is PV1_EVAL_PERSISTENT_WORKERS is False
    assert safe["series_cache_mb_per_worker"] == PV1_EVAL_SERIES_CACHE_MB == 0
