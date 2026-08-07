from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.calibration import fit_calibration
from rsna_knee.constants import TARGETS
from rsna_knee.cotrain import assign_crossfit_folds, consensus_arrays
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
    gold[::2] = 1.0  # positive state carries no information beyond 50% prior
    calibration = fit_calibration(states, gold)
    confidence = calibration.confidence(states)
    assert float(confidence.max()) < 0.05


def test_crossfit_keeps_identical_reports_in_same_fold():
    df = pd.DataFrame({
        "Report": ["same report", "same report", "different"],
        "StudyInstanceUID": ["a", "b", "c"],
    })
    folds = assign_crossfit_folds(df, 3)
    assert folds.iloc[0] == folds.iloc[1]


def test_consensus_strengthens_agreement_and_downweights_conflict():
    report = np.array([[0.9, 0.9], [0.9, 0.1]], dtype=np.float32)
    report_conf = np.full_like(report, 0.4)
    image = np.array([[0.95, 0.05], [0.05, 0.05]], dtype=np.float32)
    _, confidence = consensus_arrays(report, report_conf, image)
    assert confidence[0, 0] == np.float32(0.9)  # positive agreement
    assert confidence[0, 1] <= np.float32(0.05)  # direct disagreement
    assert confidence[1, 0] <= np.float32(0.05)
    assert confidence[1, 1] == np.float32(0.9)  # negative agreement


def test_two_pool_ddp_ranks_receive_disjoint_shards():
    trusted = np.zeros(40, dtype=bool)
    trusted[:12] = True
    rank0 = TwoPoolBatchSampler(trusted, batch_size=4, trusted_fraction=0.25, seed=7, rank=0, world_size=2)
    rank1 = TwoPoolBatchSampler(trusted, batch_size=4, trusted_fraction=0.25, seed=7, rank=1, world_size=2)
    batch0 = next(iter(rank0))
    batch1 = next(iter(rank1))
    assert len(batch0) == len(batch1) == 4
    assert set(batch0).isdisjoint(batch1)
    global_batch = batch0 + batch1
    assert sum(trusted[i] for i in global_batch) == 2


def test_two_pool_sampler_is_epoch_deterministic():
    trusted = np.array([True] * 5 + [False] * 15)
    sampler = TwoPoolBatchSampler(trusted, batch_size=4, trusted_fraction=0.25, seed=3)
    first = list(iter(sampler))
    sampler.set_epoch(0)
    assert first == list(iter(sampler))
    sampler.set_epoch(1)
    assert first != list(iter(sampler))
