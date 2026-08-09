from __future__ import annotations

import numpy as np

from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee.frozen_features import _candidate_grid
from rsna_knee.frozen_features_crossval import (
    _two_way_cv_predictions,
    select_target_candidate_crossval,
)


def test_two_way_cv_predictions_cover_both_non_outer_folds():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 20))
    y = np.array([0, 1, 0, 1] * 3, dtype=float)
    fold_ids = np.repeat([0, 1, 2], 4)

    indices, pred = _two_way_cv_predictions(
        x,
        y,
        fold_ids,
        1,
        2,
        n_components=3,
        c_value=0.1,
        seed=2026,
    )
    assert indices.tolist() == list(range(4, 12))
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()


def test_two_way_cv_never_uses_outer_fold():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(15, 12))
    y = np.array([0, 1, 0, 1, 0] * 3, dtype=float)
    folds = np.repeat([0, 1, 2], 5)
    indices, _ = _two_way_cv_predictions(
        x, y, folds, 1, 2, n_components=4, c_value=1.0, seed=1
    )
    assert not np.isin(indices, np.flatnonzero(folds == 0)).any()


def test_target_selector_returns_valid_policy_and_all_non_outer_studies():
    rng = np.random.default_rng(9)
    n = 18
    features = rng.normal(size=(n, len(DUAL_STREAMS), 12))
    present = np.ones((n, len(DUAL_STREAMS)), dtype=float)
    y = np.zeros((n, len(TARGETS)), dtype=float)
    for j in range(len(TARGETS)):
        y[:, j] = np.asarray(([0, 1, 0, 1, 1, 0] * 3)[:n], dtype=float)
    folds = np.repeat([0, 1, 2], 6)
    candidates = list(_candidate_grid([2, 4], [0.1], ["all", "prior"]))

    policy = select_target_candidate_crossval(
        features,
        present,
        y,
        folds,
        outer_fold=0,
        target_index=0,
        candidates=candidates,
        seed=2026,
    )
    assert policy["feature_mode"] in {"all", "prior"}
    assert policy["pca_components"] in {2, 4}
    assert policy["C"] == 0.1
    assert policy["cv_folds"] == [1, 2]
    assert policy["cv_studies"] == 12
    assert np.isfinite(policy["cv_auc"])
