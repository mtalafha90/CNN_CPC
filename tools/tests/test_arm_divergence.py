"""Measuring how far two matched arms' predictions actually diverge.

The point of the tool is the discordant-pair fraction: ROC AUC is a function of
pair ordering alone, so that number bounds how far two arms' AUCs can possibly
differ. Everything else is context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.arm_divergence import (
    align,
    compare,
    describe,
    discordant_fraction,
    load_predictions,
)


def _frame(values, uids=None, targets=("ACL", "MCL")):
    values = np.asarray(values, dtype=float)
    uids = uids or [f"study-{i}" for i in range(len(values))]
    frame = pd.DataFrame(values, columns=list(targets))
    frame.insert(0, "StudyInstanceUID", uids)
    return frame.set_index("StudyInstanceUID").sort_index()


# --- the quantity that actually bounds an AUC difference ------------------


def test_identical_rankings_have_no_discordant_pairs():
    order = np.array([0.1, 0.4, 0.2, 0.9])
    assert discordant_fraction(order, order) == 0.0
    # A different scale is still the same ranking, so still no discordance.
    assert discordant_fraction(order, order * 3.0 + 5.0) == 0.0


def test_a_reversed_ranking_makes_every_pair_discordant():
    order = np.array([1.0, 2.0, 3.0, 4.0])
    assert discordant_fraction(order, -order) == 1.0


def test_one_swapped_pair_is_counted_once():
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([2.0, 1.0, 3.0])  # the first two change places
    assert discordant_fraction(left, right) == pytest.approx(1 / 3)


def test_ties_cannot_flip_a_ranking_so_they_are_not_discordant():
    left = np.array([1.0, 1.0, 2.0])
    right = np.array([2.0, 1.0, 1.0])
    # Only the (0,2) pair genuinely reverses; the tied pairs cannot.
    assert discordant_fraction(left, right) == pytest.approx(1 / 3)


def test_fewer_than_two_studies_has_no_pairs():
    assert discordant_fraction(np.array([0.5]), np.array([0.9])) == 0.0


# --- the comparison --------------------------------------------------------


def test_two_arms_that_agree_exactly_report_nothing_moved():
    frame = _frame([[0.1, 0.9], [0.5, 0.2], [0.7, 0.4]])
    result = compare(frame, frame.copy())
    assert result["studies"] == 3
    assert result["mean_abs_delta"] == 0.0
    assert result["mean_discordant_pair_fraction"] == 0.0
    assert result["min_spearman"] == pytest.approx(1.0)


def test_a_tiny_shift_that_does_not_reorder_anything_is_reported_as_such():
    """The case the B48/B49 gate analysis predicts."""
    control = _frame([[0.10, 0.90], [0.50, 0.20], [0.70, 0.40], [0.30, 0.60]])
    candidate = control + 1e-6  # every study moves, none changes place
    result = compare(control, candidate)
    assert result["mean_abs_delta"] == pytest.approx(1e-6)
    assert result["mean_discordant_pair_fraction"] == 0.0
    assert result["min_spearman"] == pytest.approx(1.0)


def test_a_genuine_reordering_shows_up():
    control = _frame([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    candidate = _frame([[0.4, 0.1], [0.3, 0.2], [0.2, 0.3], [0.1, 0.4]])
    result = compare(control, candidate)
    per_target = result["per_target"].set_index("target")
    assert per_target.loc["ACL", "discordant_pair_fraction"] == 1.0
    assert per_target.loc["MCL", "discordant_pair_fraction"] == 0.0
    assert per_target.loc["ACL", "spearman"] == pytest.approx(-1.0)


def test_studies_are_matched_by_uid_not_by_row_order():
    """Row order differing between files must not fabricate a difference."""
    control = _frame([[0.1, 0.9], [0.5, 0.2]], uids=["b", "a"])
    candidate = _frame([[0.5, 0.2], [0.1, 0.9]], uids=["a", "b"])
    result = compare(control, candidate)
    assert result["mean_abs_delta"] == 0.0


def test_only_shared_studies_and_targets_are_compared():
    control = _frame([[0.1, 0.9], [0.5, 0.2]], uids=["a", "b"])
    candidate = _frame([[0.1, 0.9], [0.5, 0.2]], uids=["a", "c"])
    result = compare(control, candidate)
    assert result["studies"] == 1


def test_no_shared_studies_is_an_error_not_a_silent_zero():
    control = _frame([[0.1, 0.9]], uids=["a"])
    candidate = _frame([[0.1, 0.9]], uids=["z"])
    with pytest.raises(ValueError, match="share no studies"):
        compare(control, candidate)


def test_no_shared_targets_is_an_error():
    control = _frame([[0.1, 0.9]], uids=["a"], targets=("ACL", "MCL"))
    candidate = _frame([[0.1, 0.9]], uids=["a"], targets=("Effusion", "Fracture"))
    with pytest.raises(ValueError, match="share no target"):
        align(control, candidate)


# --- reading files ---------------------------------------------------------


def test_a_prediction_csv_round_trips(tmp_path):
    path = tmp_path / "arm.csv"
    _frame([[0.1, 0.9], [0.5, 0.2]]).reset_index().to_csv(path, index=False)
    frame = load_predictions(path)
    assert list(frame.columns) == ["ACL", "MCL"]
    assert frame.index.name == "StudyInstanceUID"


def test_a_csv_without_the_uid_column_is_refused(tmp_path):
    path = tmp_path / "arm.csv"
    pd.DataFrame({"ACL": [0.1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no StudyInstanceUID"):
        load_predictions(path)


def test_duplicate_studies_are_refused(tmp_path):
    path = tmp_path / "arm.csv"
    pd.DataFrame({"StudyInstanceUID": ["a", "a"], "ACL": [0.1, 0.2]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="duplicate study rows"):
        load_predictions(path)


def test_the_report_states_no_verdict():
    """A threshold here would be quoted back as evidence."""
    frame = _frame([[0.1, 0.9], [0.5, 0.2]])
    text = describe(compare(frame, frame.copy())).lower()
    assert "discordant" in text
    for verdict in ("fail", "pass", "too similar", "no effect", "conclusion"):
        assert verdict not in text


# --- the ceiling an AUC difference cannot exceed --------------------------


def test_no_reordering_means_no_possible_auc_difference():
    from tools.arm_divergence import auc_difference_ceiling

    assert auc_difference_ceiling(0.0) == 0.0


def test_the_ceiling_is_the_discordant_fraction():
    """An AUC moves only on pairs the two arms order differently."""
    from tools.arm_divergence import auc_difference_ceiling

    assert auc_difference_ceiling(0.002382) == pytest.approx(0.002382)
    assert auc_difference_ceiling(1.0) == pytest.approx(1.0)


def test_the_ceiling_appears_in_the_report_with_its_assumption_stated():
    frame_a = _frame([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    frame_b = _frame([[0.4, 0.1], [0.3, 0.2], [0.2, 0.3], [0.1, 0.4]])
    text = describe(compare(frame_a, frame_b))
    assert "max |dAUC|" in text
    assert "assuming" in text, "a bound without its assumption invites misuse"
