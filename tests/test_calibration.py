"""Tests for fold-safe teacher calibration."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.calibration import (
    TeacherCalibration,
    calibration_split_mask,
    fit_calibration,
)
from rsna_knee.constants import TARGETS
from rsna_knee.report_labels import (
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
    predict_target,
    state_dataframe,
)


def _states(n: int, state: str) -> np.ndarray:
    return np.full((n, len(TARGETS)), state, dtype=object)


def test_rule_states_are_exposed():
    assert predict_target("Complete tear of the anterior cruciate ligament.", "ACL").state == STATE_POSITIVE
    assert predict_target("No tear of the anterior cruciate ligament.", "ACL").state == STATE_NEGATED
    assert predict_target("The anterior cruciate ligament is shown.", "ACL").state == STATE_UNCERTAIN
    assert predict_target("Unremarkable study.", "ACL").state == STATE_UNMENTIONED


def test_state_dataframe_shape():
    import pandas as pd

    df = pd.DataFrame({"Report": ["ACL tear", "no acl tear", ""]})
    states = state_dataframe(df)

    assert states.shape == (3, len(TARGETS))
    assert states[0, TARGETS.index("ACL")] == STATE_POSITIVE
    assert states[1, TARGETS.index("ACL")] == STATE_NEGATED


def test_positive_state_calibrates_above_negative_state():
    """The whole point: learn what each state is actually worth."""
    n = 40
    states = np.concatenate([_states(n, STATE_POSITIVE), _states(n, STATE_NEGATED)])
    gold = np.concatenate([np.ones((n, len(TARGETS))), np.zeros((n, len(TARGETS)))])

    calibration = fit_calibration(states, gold)

    for target in TARGETS:
        assert calibration.probability(target, STATE_POSITIVE) > 0.8
        assert calibration.probability(target, STATE_NEGATED) < 0.2


def test_estimates_are_smoothed_towards_the_prior():
    """A state seen twice must not be trusted as a hard 0 or 1.

    The prior has to be genuinely low for shrinkage to be visible, so the
    positives are a rare state among mostly negative studies — as they are in
    the real gold set.
    """
    states = np.concatenate([_states(2, STATE_POSITIVE), _states(40, STATE_UNMENTIONED)])
    gold = np.concatenate([np.ones((2, len(TARGETS))), np.zeros((40, len(TARGETS)))])

    calibration = fit_calibration(states, gold, alpha=5.0)

    probability = calibration.probability(TARGETS[0], STATE_POSITIVE)
    assert probability < 1.0
    # Two observations against alpha=5: the low prior should still dominate.
    assert probability < 0.8


def test_more_evidence_moves_the_estimate_further_from_the_prior():
    few = fit_calibration(_states(3, STATE_POSITIVE), np.ones((3, len(TARGETS))))
    many = fit_calibration(_states(200, STATE_POSITIVE), np.ones((200, len(TARGETS))))

    assert many.probability(TARGETS[0], STATE_POSITIVE) > few.probability(TARGETS[0], STATE_POSITIVE)


def test_unannotated_cells_are_ignored_not_counted_as_negative():
    """NaN means unknown. Counting it as 0 would bias every estimate downwards."""
    states = _states(10, STATE_POSITIVE)
    gold = np.full((10, len(TARGETS)), np.nan)
    gold[:5] = 1.0  # only half are annotated, all positive

    calibration = fit_calibration(states, gold)

    assert calibration.counts[(TARGETS[0], STATE_POSITIVE)] == 5
    assert calibration.probability(TARGETS[0], STATE_POSITIVE) > 0.8


def test_unseen_state_falls_back_to_the_prior():
    states = _states(20, STATE_POSITIVE)
    gold = np.zeros((20, len(TARGETS)))
    gold[:4] = 1.0

    calibration = fit_calibration(states, gold)

    # STATE_UNCERTAIN never appeared, so it must return the prior (0.2).
    assert calibration.probability(TARGETS[0], STATE_UNCERTAIN) == pytest.approx(0.2, abs=0.01)


def test_apply_maps_states_to_probabilities():
    train_states = np.concatenate([_states(30, STATE_POSITIVE), _states(30, STATE_NEGATED)])
    gold = np.concatenate([np.ones((30, len(TARGETS))), np.zeros((30, len(TARGETS)))])
    calibration = fit_calibration(train_states, gold)

    new_states = np.array([[STATE_POSITIVE] * len(TARGETS), [STATE_NEGATED] * len(TARGETS)], dtype=object)
    probabilities = calibration.apply(new_states)

    assert probabilities.shape == (2, len(TARGETS))
    assert (probabilities[0] > probabilities[1]).all()


def test_confidence_reflects_evidence_and_damps_silence():
    states = np.concatenate([_states(100, STATE_POSITIVE), _states(100, STATE_UNMENTIONED)])
    gold = np.concatenate([np.ones((100, len(TARGETS))), np.zeros((100, len(TARGETS)))])
    calibration = fit_calibration(states, gold)

    confidence = calibration.confidence(states)

    # Well-evidenced positives outrank unmentioned cells, which are damped
    # because radiologists routinely omit incidental findings.
    assert confidence[0, 0] > confidence[-1, 0]


def test_calibration_split_excludes_the_validation_fold():
    """The leak this module exists to prevent."""
    gold_present = np.array([True, True, True, False])
    folds = np.array([0, 1, 2, 1])

    mask = calibration_split_mask(gold_present, folds, validation_fold=1)

    assert mask.tolist() == [True, False, True, False]


def test_calibration_is_not_influenced_by_the_validation_fold():
    """Changing only validation-fold labels must not change the calibration."""
    n = 30
    states = _states(n, STATE_POSITIVE)
    folds = np.array([i % 3 for i in range(n)])
    gold_a = np.ones((n, len(TARGETS)))
    gold_b = gold_a.copy()
    gold_b[folds == 1] = 0.0  # flip only the validation fold

    mask = calibration_split_mask(np.ones(n, dtype=bool), folds, validation_fold=1)
    calibration_a = fit_calibration(states[mask], gold_a[mask])
    calibration_b = fit_calibration(states[mask], gold_b[mask])

    assert calibration_a.probability(TARGETS[0], STATE_POSITIVE) == pytest.approx(
        calibration_b.probability(TARGETS[0], STATE_POSITIVE)
    )


def test_round_trip_serialisation():
    calibration = fit_calibration(_states(20, STATE_POSITIVE), np.ones((20, len(TARGETS))))

    restored = TeacherCalibration.from_dict(calibration.to_dict())

    assert restored.probability(TARGETS[0], STATE_POSITIVE) == pytest.approx(
        calibration.probability(TARGETS[0], STATE_POSITIVE)
    )
    assert restored.counts == calibration.counts


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        fit_calibration(_states(5, STATE_POSITIVE), np.ones((4, len(TARGETS))))
