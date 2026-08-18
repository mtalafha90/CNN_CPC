import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee import phase9_v2_supervision as p9v2


def _arrays(uids, active_cells):
    y = np.full((len(uids), len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros_like(y)
    for uid, target_index, positive in active_cells:
        i = uids.index(uid)
        y[i, target_index] = 0.85 if positive else 0.05
        w[i, target_index] = 0.50 if positive else 1.00
    return y, w


def test_phase9_v2_removes_same_holdout_but_retains_zero_weight_training_studies(monkeypatch):
    all_uids = ["holdout", "active", "rescued"]
    control_y, control_w = _arrays(
        all_uids,
        [("holdout", 0, True), ("active", 1, False)],
    )
    candidate_y, candidate_w = _arrays(
        all_uids,
        [("holdout", 0, True), ("active", 1, False), ("rescued", 2, True)],
    )
    hold_y = control_y[[0]].copy()
    hold_w = control_w[[0]].copy()

    def fake_arm(train_df, *, arm, b6_root, phase8_root):
        y, w = (control_y, control_w) if arm == "control" else (candidate_y, candidate_w)
        summary = {
            "active_studies": int((w.sum(axis=1) > 0).sum()),
            "usable_cells": int((w > 0).sum()),
            "positive_cells": int(((w > 0) & (y > 0.5)).sum()),
            "negative_cells": int(((w > 0) & (y < 0.5)).sum()),
        }
        return all_uids, y.copy(), w.copy(), summary, {"arm": arm}

    monkeypatch.setattr(p9v2, "load_phase9_arm_supervision", fake_arm)
    monkeypatch.setattr(
        p9v2,
        "load_phase9_v2_holdout",
        lambda *args, **kwargs: {
            "uids": ["holdout"],
            "targets": hold_y,
            "weights": hold_w,
            "pv2_split_sha256": "split",
            "pv2_validation_uid_sha256": "uids",
            "parent_pv1_split_sha256": "parent",
            "supervision_audit": {"usable_cells": 1, "positive_cells": 1, "negative_cells": 0},
            "limitation": "historically exposed weak-label surface",
        },
    )
    monkeypatch.setattr(p9v2, "PHASE9_V2_HOLDOUT_STUDIES", 1)
    monkeypatch.setattr(p9v2, "PHASE9_V2_TRAIN_STUDIES", 2)

    train = pd.DataFrame({"StudyInstanceUID": ["unused"]})
    cuids, _, cw, csum, _, _ = p9v2.prepare_phase9_v2_arm_supervision(
        train,
        arm="control",
        b6_root="b6",
        phase8_root="phase8",
        parent_pv1_manifest_path="pv1.json",
        pv2_manifest_path="pv2.json",
    )
    auids, _, aw, asum, _, _ = p9v2.prepare_phase9_v2_arm_supervision(
        train,
        arm="candidate",
        b6_root="b6",
        phase8_root="phase8",
        parent_pv1_manifest_path="pv1.json",
        pv2_manifest_path="pv2.json",
    )

    assert cuids == auids == ["active", "rescued"]
    assert csum["held_out_pv2_studies"] == asum["held_out_pv2_studies"] == 1
    assert csum["training_uid_sha256"] == asum["training_uid_sha256"]
    assert int((cw.sum(axis=1) == 0).sum()) == 1
    assert int((aw.sum(axis=1) == 0).sum()) == 0
    assert csum["pv2_holdout_original_b6_labels_unchanged_in_arm"] is True
    assert asum["pv2_holdout_original_b6_labels_unchanged_in_arm"] is True


def test_phase9_v2_rejects_candidate_change_on_holdout(monkeypatch):
    all_uids = ["holdout", "train"]
    b6_y, b6_w = _arrays(all_uids, [("holdout", 0, True), ("train", 1, False)])
    candidate_y = b6_y.copy()
    candidate_w = b6_w.copy()
    candidate_y[0, 0] = 0.05
    candidate_w[0, 0] = 1.0

    monkeypatch.setattr(
        p9v2,
        "load_phase9_arm_supervision",
        lambda *args, **kwargs: (
            all_uids,
            candidate_y,
            candidate_w,
            {"active_studies": 2, "usable_cells": 2, "positive_cells": 0, "negative_cells": 2},
            {"arm": "candidate"},
        ),
    )
    monkeypatch.setattr(
        p9v2,
        "load_phase9_v2_holdout",
        lambda *args, **kwargs: {
            "uids": ["holdout"],
            "targets": b6_y[[0]],
            "weights": b6_w[[0]],
            "pv2_split_sha256": "split",
            "pv2_validation_uid_sha256": "uids",
            "parent_pv1_split_sha256": "parent",
            "supervision_audit": {"usable_cells": 1, "positive_cells": 1, "negative_cells": 0},
            "limitation": "historically exposed weak-label surface",
        },
    )
    monkeypatch.setattr(p9v2, "PHASE9_V2_HOLDOUT_STUDIES", 1)
    monkeypatch.setattr(p9v2, "PHASE9_V2_TRAIN_STUDIES", 1)

    with pytest.raises(RuntimeError, match="holdout targets differ from frozen original B6"):
        p9v2.prepare_phase9_v2_arm_supervision(
            pd.DataFrame({"StudyInstanceUID": ["unused"]}),
            arm="candidate",
            b6_root="b6",
            phase8_root="phase8",
            parent_pv1_manifest_path="pv1.json",
            pv2_manifest_path="pv2.json",
        )


def test_phase9_v2_auc_hard_truth_masks_unsupervised_cells():
    from rsna_knee.phase9_v2_auc_addendum import hard_truth_from_b6

    y = np.full((2, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros_like(y)
    y[0, 0], w[0, 0] = 0.85, 0.50
    y[1, 0], w[1, 0] = 0.05, 1.00
    truth = hard_truth_from_b6(y, w)

    assert truth[0, 0] == 1.0
    assert truth[1, 0] == 0.0
    assert np.isnan(truth[:, 1:]).all()


def test_phase9_v2_auc_bootstrap_is_paired_and_strict_all_targets():
    from rsna_knee.phase9_v2_auc_addendum import paired_bootstrap_macro_auc_difference

    n = 40
    truth = np.empty((n, len(TARGETS)), dtype=np.float64)
    for j in range(len(TARGETS)):
        truth[:, j] = ((np.arange(n) + j) % 2).astype(np.float64)
    candidate = 0.1 + 0.8 * truth
    control = 0.9 - 0.8 * truth

    result = paired_bootstrap_macro_auc_difference(
        truth,
        control,
        candidate,
        n_bootstrap=300,
        seed=7,
    )

    assert result["control_macro_auc"] == pytest.approx(0.0)
    assert result["candidate_macro_auc"] == pytest.approx(1.0)
    assert result["point_difference"] == pytest.approx(1.0)
    assert result["ci_lower"] == pytest.approx(1.0)
    assert result["ci_upper"] == pytest.approx(1.0)
    assert result["probability_candidate_better"] == pytest.approx(1.0)
    assert result["strict_all_12_targets_per_replicate"] is True
    assert result["n_valid_replicates"] > 0


def test_phase9_v2_auc_requires_all_targets_defined_on_full_holdout():
    from rsna_knee.phase9_v2_auc_addendum import paired_bootstrap_macro_auc_difference

    n = 20
    truth = np.tile((np.arange(n) % 2)[:, None], (1, len(TARGETS))).astype(np.float64)
    truth[:, 0] = 1.0
    pred = np.tile(np.linspace(0.0, 1.0, n)[:, None], (1, len(TARGETS)))

    with pytest.raises(RuntimeError, match="does not define AUC for all 12 targets"):
        paired_bootstrap_macro_auc_difference(truth, pred, pred, n_bootstrap=20, seed=3)
