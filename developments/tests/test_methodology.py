from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.calibration import fit_calibration
from rsna_knee.constants import TARGETS
from rsna_knee.cotrain import (
    assign_crossfit_folds,
    choose_fold_candidate_root,
    consensus_arrays,
    load_fold_image_teacher,
)
from rsna_knee.dataset import DatasetConfig, KneeStudyDataset
from rsna_knee.report_labels import STATE_POSITIVE, STATE_UNMENTIONED
from rsna_knee.sampling import TwoPoolBatchSampler


def test_unmentioned_cells_receive_zero_direct_supervision_by_default():
    n = 30
    states = np.full((n, len(TARGETS)), STATE_UNMENTIONED, dtype=object)
    gold = np.zeros((n, len(TARGETS)), dtype=float)
    calibration = fit_calibration(states, gold)
    confidence = calibration.confidence(states)
    assert np.all(confidence == 0.0)


def test_frequent_but_uninformative_state_has_low_reliability():
    n = 40
    states = np.full((n, len(TARGETS)), STATE_POSITIVE, dtype=object)
    gold = np.zeros((n, len(TARGETS)), dtype=float)
    gold[::2] = 1.0
    calibration = fit_calibration(states, gold)
    confidence = calibration.confidence(states)
    assert float(confidence.max()) < 0.05


def test_crossfit_keeps_identical_reports_in_same_fold():
    df = pd.DataFrame(
        {"Report": ["same report", "same report", "different"], "StudyInstanceUID": ["a", "b", "c"]}
    )
    folds = assign_crossfit_folds(df, 3)
    assert folds.iloc[0] == folds.iloc[1]


def test_consensus_strengthens_agreement_and_downweights_conflict():
    report = np.array([[0.9, 0.9], [0.9, 0.1]], dtype=np.float32)
    report_conf = np.full_like(report, 0.4)
    image = np.array([[0.95, 0.05], [0.05, 0.05]], dtype=np.float32)
    _, confidence = consensus_arrays(report, report_conf, image)
    assert confidence[0, 0] == np.float32(0.9)
    assert confidence[0, 1] <= np.float32(0.05)
    assert confidence[1, 0] <= np.float32(0.05)
    assert confidence[1, 1] == np.float32(0.9)


def test_confident_crossfit_image_can_supervise_report_silence_modestly():
    report = np.array([[0.50, 0.50]], dtype=np.float32)
    report_conf = np.zeros_like(report)
    image = np.array([[0.98, 0.02]], dtype=np.float32)
    probability, confidence = consensus_arrays(report, report_conf, image)
    assert probability[0, 0] > 0.80 and probability[0, 1] < 0.20
    assert confidence[0, 0] == pytest.approx(0.20)
    assert confidence[0, 1] == pytest.approx(0.20)


def _prediction_frame(uids, value=0.7):
    data = {"StudyInstanceUID": list(uids)}
    for target in TARGETS:
        data[target] = value
    return pd.DataFrame(data)


def _write_stage1_candidate(root, fold, uids, inner_score, outer_score=0.5, inner_fold=1):
    folder = root / f"fold{fold}"
    folder.mkdir(parents=True)
    _prediction_frame(uids).to_csv(folder / "weak_oof.csv", index=False)
    (folder / "best.pt").write_bytes(b"checkpoint")
    (folder / "selection.json").write_text(
        json.dumps(
            {
                "stage": "stage1",
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "inner_macro_auc": inner_score,
                "outer_macro_auc": outer_score,
                "validation_tta_offsets": [-1, 0, 1],
            }
        )
    )
    return folder


def test_stage1_candidate_choice_uses_inner_score_not_outer_oof(tmp_path):
    a = tmp_path / "random"
    b = tmp_path / "ssl"
    _write_stage1_candidate(a, 0, ["w0"], inner_score=0.70, outer_score=0.95)
    _write_stage1_candidate(b, 0, ["w0"], inner_score=0.80, outer_score=0.10)
    selected, metadata = choose_fold_candidate_root([a, b], 0)
    assert selected == b
    assert metadata["criterion"] == "inner_macro_auc_only"
    assert metadata["inner_macro_auc"] == pytest.approx(0.80)


def test_stage2_loads_only_same_outer_fold_weak_teacher(tmp_path):
    df = pd.DataFrame({"StudyInstanceUID": ["gold0", "w0", "w1"], "crossfit_fold": [0, 0, 1]})
    gold = np.array([True, False, False])
    _write_stage1_candidate(tmp_path, 0, ["w0"], inner_score=0.7)
    image = load_fold_image_teacher(tmp_path, 0, df, gold)
    assert np.isfinite(image[1]).all()
    assert np.isnan(image[0]).all() and np.isnan(image[2]).all()


def test_stage2_rejects_teacher_prediction_from_wrong_fold(tmp_path):
    df = pd.DataFrame({"StudyInstanceUID": ["gold0", "w0", "w1"], "crossfit_fold": [0, 0, 1]})
    gold = np.array([True, False, False])
    _write_stage1_candidate(tmp_path, 0, ["w0", "w1"], inner_score=0.7)
    with pytest.raises(ValueError, match="unsafe studies"):
        load_fold_image_teacher(tmp_path, 0, df, gold)


def test_stage2_rejects_incomplete_fold_teacher(tmp_path):
    df = pd.DataFrame({"StudyInstanceUID": ["w0", "w0b"], "crossfit_fold": [0, 0]})
    gold = np.array([False, False])
    _write_stage1_candidate(tmp_path, 0, ["w0"], inner_score=0.7)
    with pytest.raises(ValueError, match="missing 1 expected"):
        load_fold_image_teacher(tmp_path, 0, df, gold)


def test_two_pool_single_gpu_batch_has_expected_trusted_quota():
    trusted = np.zeros(40, dtype=bool)
    trusted[:12] = True
    sampler = TwoPoolBatchSampler(trusted, batch_size=8, trusted_fraction=0.25, seed=7)
    batch = next(iter(sampler))
    assert len(batch) == 8
    assert sum(trusted[i] for i in batch) == 2


def test_two_pool_sampler_is_epoch_deterministic():
    trusted = np.array([True] * 5 + [False] * 15)
    sampler = TwoPoolBatchSampler(trusted, batch_size=4, trusted_fraction=0.25, seed=3)
    first = list(iter(sampler))
    sampler.set_epoch(0)
    assert first == list(iter(sampler))
    sampler.set_epoch(1)
    assert first != list(iter(sampler))


def test_two_pool_sampler_honors_explicit_deadline():
    trusted = np.array([True] * 5 + [False] * 15)
    sampler = TwoPoolBatchSampler(
        trusted,
        batch_size=4,
        trusted_fraction=0.25,
        deadline_monotonic=time.monotonic() - 1.0,
    )
    assert list(iter(sampler)) == []


def test_training_view_is_reproducible_given_torch_seed():
    config = DatasetConfig(
        data_root=".", n_slices=4, image_size=16, train_gap_choices=(1, 2),
        center_jitter=2, noise_std=0.01, slice_dropout=0.0,
        rotation_deg=2.0, translate_frac=0.01, scale_jitter=0.02,
        gamma_jitter=0.02, bias_field_strength=0.02, series_cache_mb=0,
    )
    raw = np.arange(20 * 12 * 12, dtype=np.float32).reshape(20, 12, 12)
    ds1 = KneeStudyDataset([], {}, config, train=True)
    ds2 = KneeStudyDataset([], {}, config, train=True)
    torch.manual_seed(123)
    first = ds1._training_view(raw)
    torch.manual_seed(123)
    second = ds2._training_view(raw)
    assert torch.equal(first, second)
