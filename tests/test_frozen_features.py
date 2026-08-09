from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee.frozen_features import (
    TARGET_STREAM_SUBSETS,
    _candidate_grid,
    _fit_predict,
    target_design_matrix,
)


def test_target_stream_subsets_are_valid_and_nonempty():
    assert set(TARGET_STREAM_SUBSETS) == set(TARGETS)
    valid = set(DUAL_STREAMS)
    for target, streams in TARGET_STREAM_SUBSETS.items():
        assert streams, target
        assert len(set(streams)) == len(streams)
        assert set(streams).issubset(valid)


def test_target_design_matrix_all_and_prior_shapes():
    n, d = 5, 9
    features = np.arange(n * len(DUAL_STREAMS) * d, dtype=float).reshape(n, len(DUAL_STREAMS), d)
    present = np.ones((n, len(DUAL_STREAMS)), dtype=float)
    present[0, 0] = 0

    all_x = target_design_matrix(features, present, "ACL", "all")
    prior_x = target_design_matrix(features, present, "ACL", "prior")

    assert all_x.shape == (n, len(DUAL_STREAMS) * d + len(DUAL_STREAMS))
    assert prior_x.shape == (n, 2 * d + 2)
    assert prior_x[0, -2] == 0
    assert prior_x[0, -1] == 1


def test_fit_predict_is_finite_and_bounded():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(20, 40))
    y_train = np.array([0, 1] * 10, dtype=float)
    x_eval = rng.normal(size=(6, 40))

    p = _fit_predict(
        x_train,
        y_train,
        x_eval,
        n_components=8,
        c_value=0.1,
        seed=2026,
    )
    assert p.shape == (6,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_fit_predict_constant_target_falls_back_safely():
    x_train = np.arange(40, dtype=float).reshape(10, 4)
    y_train = np.ones(10, dtype=float)
    x_eval = np.zeros((3, 4), dtype=float)
    p = _fit_predict(
        x_train,
        y_train,
        x_eval,
        n_components=4,
        c_value=1.0,
        seed=1,
    )
    np.testing.assert_allclose(p, 1.0)


def test_candidate_grid_is_deterministic_and_validates():
    grid = list(_candidate_grid([4, 8], [0.1, 1.0], ["all", "prior"]))
    assert grid == [
        ("all", 4, 0.1),
        ("all", 4, 1.0),
        ("all", 8, 0.1),
        ("all", 8, 1.0),
        ("prior", 4, 0.1),
        ("prior", 4, 1.0),
        ("prior", 8, 0.1),
        ("prior", 8, 1.0),
    ]
