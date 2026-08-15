from __future__ import annotations

import numpy as np

from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee.frozen_features_shared import select_shared_candidate


def test_shared_candidate_returns_one_policy_and_all_target_predictions():
    rng = np.random.default_rng(123)
    n = 18
    # Small synthetic pooled cache: [study, stream, pooled feature].
    features = rng.normal(size=(n, len(DUAL_STREAMS), 12))
    present = np.ones((n, len(DUAL_STREAMS)), dtype=float)
    present[::5, 1] = 0.0

    y = np.zeros((n, len(TARGETS)), dtype=float)
    for j in range(len(TARGETS)):
        # Every six-study block contains both classes for every target.
        y[:, j] = (np.arange(n) + j) % 2

    selection_train = np.zeros(n, dtype=bool)
    selection_train[:6] = True
    inner = np.zeros(n, dtype=bool)
    inner[6:12] = True

    candidates = [
        ("all", 2, 0.1),
        ("prior", 2, 0.1),
    ]
    policy, prediction = select_shared_candidate(
        features,
        present,
        y,
        selection_train,
        inner,
        candidates=candidates,
        seed=2026,
    )

    assert policy["feature_mode"] in {"all", "prior"}
    assert policy["pca_components"] == 2
    assert policy["C"] == 0.1
    assert np.isfinite(policy["inner_macro_auc"])
    assert set(policy["inner_per_target_auc"]) == set(TARGETS)
    assert prediction.shape == (6, len(TARGETS))
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()


def test_shared_candidate_does_not_return_target_specific_hyperparameters():
    rng = np.random.default_rng(321)
    n = 18
    features = rng.normal(size=(n, len(DUAL_STREAMS), 8))
    present = np.ones((n, len(DUAL_STREAMS)), dtype=float)
    y = np.column_stack([((np.arange(n) + j) % 2) for j in range(len(TARGETS))]).astype(float)

    selection_train = np.arange(n) < 6
    inner = (np.arange(n) >= 6) & (np.arange(n) < 12)
    policy, _ = select_shared_candidate(
        features,
        present,
        y,
        selection_train,
        inner,
        candidates=[("all", 2, 0.1)],
        seed=7,
    )

    assert "targets" not in policy
    assert policy["feature_mode"] == "all"
    assert policy["pca_components"] == 2
    assert policy["C"] == 0.1
