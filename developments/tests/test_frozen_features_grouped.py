from __future__ import annotations

import numpy as np

from rsna_knee.constants import DUAL_STREAMS, TARGETS
from rsna_knee.frozen_features_grouped import (
    PATHOLOGY_GROUPS,
    _predict_group,
    _validate_groups,
    select_group_candidate,
)


def _synthetic_features(seed: int = 9):
    rng = np.random.default_rng(seed)
    n = 24
    d = 12
    features = rng.normal(size=(n, len(DUAL_STREAMS), d))
    present = np.ones((n, len(DUAL_STREAMS)), dtype=float)
    y = np.zeros((n, len(TARGETS)), dtype=float)
    for j in range(len(TARGETS)):
        y[:, j] = (np.arange(n) + j) % 2
    return features, present, y


def test_pathology_groups_partition_targets_exactly_once():
    _validate_groups()
    flattened = [target for targets in PATHOLOGY_GROUPS.values() for target in targets]
    assert set(flattened) == set(TARGETS)
    assert len(flattened) == len(TARGETS)
    assert len(set(flattened)) == len(TARGETS)


def test_pathology_groups_are_the_predeclared_four_groups():
    assert list(PATHOLOGY_GROUPS) == [
        "ligament_meniscus",
        "osteoarthritis",
        "fluid_inflammatory",
        "osseous_injury",
    ]
    assert PATHOLOGY_GROUPS["ligament_meniscus"] == (
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"
    )
    assert PATHOLOGY_GROUPS["osteoarthritis"] == (
        "Medial OA", "Lateral OA", "PF OA"
    )
    assert PATHOLOGY_GROUPS["fluid_inflammatory"] == (
        "Effusion", "Synovitis", "Baker's"
    )
    assert PATHOLOGY_GROUPS["osseous_injury"] == ("Contusion", "Fracture")


def test_predict_group_is_finite_and_has_group_width():
    features, present, y = _synthetic_features()
    train = np.arange(len(y)) < 16
    evaluate = ~train
    targets = PATHOLOGY_GROUPS["osteoarthritis"]

    pred = _predict_group(
        features,
        present,
        y,
        train,
        evaluate,
        targets,
        mode="all",
        n_components=4,
        c_value=0.1,
        seed=2026,
    )
    assert pred.shape == (int(evaluate.sum()), len(targets))
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()


def test_select_group_candidate_with_single_candidate_is_deterministic():
    features, present, y = _synthetic_features()
    selection_train = np.zeros(len(y), dtype=bool)
    selection_train[:12] = True
    inner = np.zeros(len(y), dtype=bool)
    inner[12:20] = True
    targets = PATHOLOGY_GROUPS["osseous_injury"]
    candidates = [("prior", 4, 0.1)]

    first = select_group_candidate(
        features,
        present,
        y,
        selection_train,
        inner,
        targets,
        candidates=candidates,
        seed=2026,
    )
    second = select_group_candidate(
        features,
        present,
        y,
        selection_train,
        inner,
        targets,
        candidates=candidates,
        seed=2026,
    )

    assert first == second
    assert first["feature_mode"] == "prior"
    assert first["pca_components"] == 4
    assert first["C"] == 0.1
    assert first["candidate_index"] == 0
    assert np.isfinite(first["inner_group_macro_auc"])
    assert set(first["inner_per_target_auc"]) == set(targets)
