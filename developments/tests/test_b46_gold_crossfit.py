from __future__ import annotations

import json

import numpy as np

from rsna_knee.b46_gold_crossfit import (
    B46_EXPERIMENT,
    B46_FOLD_SALT,
    B46_GOLD_STUDIES,
    B46_N_FOLDS,
    B46_VERSION,
    assign_gold_folds,
    heldout_uids,
    load_gold_fold_manifest,
    training_gold_uids,
)
from rsna_knee.constants import TARGETS


def _synthetic_labels() -> tuple[list[str], np.ndarray]:
    uids = [f"uid-{i:03d}" for i in range(B46_GOLD_STUDIES)]
    y = np.zeros((B46_GOLD_STUDIES, len(TARGETS)), dtype=np.int64)
    # Deterministic, nontrivial prevalences with both classes for every target.
    for i in range(B46_GOLD_STUDIES):
        for j in range(len(TARGETS)):
            y[i, j] = int(((i * (j + 3) + j * 7) % (j + 5)) < (2 + (j % 3)))
    assert ((y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))).all()
    return uids, y


def test_b46_fold_assignment_is_deterministic_complete_and_fixed_size():
    uids, y = _synthetic_labels()
    a = assign_gold_folds(uids, y)
    b = assign_gold_folds(uids, y)
    assert np.array_equal(a, b)
    assert sorted(np.bincount(a, minlength=B46_N_FOLDS).tolist()) == [11, 11, 12, 12, 12]
    assert set(a.tolist()) == set(range(B46_N_FOLDS))


def test_b46_manifest_loader_and_fold_partition(tmp_path):
    uids, y = _synthetic_labels()
    folds = assign_gold_folds(uids, y)
    rows = []
    for uid, fold, labels in zip(uids, folds, y):
        row = {"StudyInstanceUID": uid, "fold": int(fold)}
        row.update({target: int(value) for target, value in zip(TARGETS, labels)})
        rows.append(row)
    payload = {
        "experiment": B46_EXPERIMENT,
        "version": B46_VERSION,
        "n_gold_studies": B46_GOLD_STUDIES,
        "n_folds": B46_N_FOLDS,
        "fold_salt": B46_FOLD_SALT,
        "rows": rows,
    }
    path = tmp_path / "gold_folds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_gold_fold_manifest(path)

    all_held = []
    for fold in range(B46_N_FOLDS):
        held = heldout_uids(loaded, fold)
        train = training_gold_uids(loaded, fold)
        assert not set(held).intersection(train)
        assert len(held) + len(train) == B46_GOLD_STUDIES
        all_held.extend(held)
    assert sorted(all_held) == sorted(uids)
    assert len(set(all_held)) == B46_GOLD_STUDIES


def test_b46_assignment_changes_when_salt_changes():
    uids, y = _synthetic_labels()
    a = assign_gold_folds(uids, y, salt=B46_FOLD_SALT)
    b = assign_gold_folds(uids, y, salt=B46_FOLD_SALT + "-different")
    # Salt is part of the frozen tie-breaking contract; on a nontrivial surface
    # a different salt should not silently reproduce the entire assignment.
    assert not np.array_equal(a, b)


def test_b46_training_and_eval_modules_import():
    # Importing these catches syntax/import drift without constructing a model.
    from rsna_knee import b46_gold_crossfit_eval as evaluation
    from rsna_knee import b46_gold_crossfit_training as training

    assert callable(training.train_b46_fold)
    assert callable(evaluation.evaluate_b46_crossfit)
