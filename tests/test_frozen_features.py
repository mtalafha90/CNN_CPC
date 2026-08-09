from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee import frozen_features as b4
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


def test_nested_classical_oof_writes_complete_outer_predictions(tmp_path, monkeypatch):
    n = 12
    uids = np.asarray([f"study-{i:02d}" for i in range(n)], dtype=str)
    fold_ids = np.asarray([0, 1, 2] * 4, dtype=int)

    rows = {"StudyInstanceUID": uids, "Report": [f"report {i}" for i in range(n)]}
    # Every fold contains both classes for every target.
    for j, target in enumerate(TARGETS):
        rows[target] = np.asarray([(i // 3 + j) % 2 for i in range(n)], dtype=int)
    train = pd.DataFrame(rows)

    data_root = tmp_path / "data"
    data_root.mkdir()
    train.to_csv(data_root / "train.csv", index=False)

    rng = np.random.default_rng(9)
    features = rng.normal(size=(n, len(DUAL_STREAMS), 12)).astype(np.float32)
    present = np.ones((n, len(DUAL_STREAMS)), dtype=np.float32)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        study_uids=uids,
        features=features,
        present=present,
        stream_names=np.asarray(DUAL_STREAMS, dtype=str),
        pool_names=np.asarray(["mean", "std", "max"], dtype=str),
    )

    monkeypatch.setattr(
        b4,
        "make_balanced_gold_folds",
        lambda df, n_splits, seed: pd.Series(fold_ids, index=df.index, dtype=int),
    )

    out_root = tmp_path / "b4"
    payload = b4.nested_classical_oof(
        {"data_root": str(data_root), "seed": 2026, "n_folds": 3, "n_bootstrap": 20},
        feature_path=feature_path,
        out_root=out_root,
        pca_components=[2],
        c_values=[0.1],
        feature_modes=["all"],
        n_bootstrap=20,
    )

    combined = pd.read_csv(out_root / "oof.csv")
    assert len(combined) == n
    assert combined["StudyInstanceUID"].nunique() == n
    assert np.isfinite(combined[TARGETS].to_numpy(float)).all()
    assert set(payload["folds"]) == {"0", "1", "2"}
    for fold in range(3):
        fold_oof = pd.read_csv(out_root / f"fold{fold}" / "oof.csv")
        assert len(fold_oof) == int((fold_ids == fold).sum())
