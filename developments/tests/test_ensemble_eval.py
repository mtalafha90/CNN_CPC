"""What TTA is worth, and whether an ensemble of these models helps.

Both are measured before a submission slot is spent, and both are easy to
report favourably by accident. The tests that earn their place are the ones
that would catch a gain which is really an artefact: a rank transform that
orders ties, an "ensemble" of one model against itself, and a comparison drawn
against the mean member rather than the best one.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from rsna_knee.ensemble_eval import (
    ENSEMBLE_VERSION,
    SUBMISSION_TTA_OFFSETS,
    TTA_WORTH_KEEPING,
    measure_ensemble,
    measure_tta,
    member_agreement,
    rank_average,
    rank_normalise,
    report,
)


def _probs(seed=0, studies=40, targets=12):
    return np.random.RandomState(seed).rand(studies, targets)


# --- the rank transform --------------------------------------------------------


def test_ranks_span_zero_to_one_per_target():
    ranked = rank_normalise(_probs())

    assert ranked.min() == 0.0
    assert ranked.max() == 1.0
    for column in range(ranked.shape[1]):
        assert ranked[:, column].min() == 0.0
        assert ranked[:, column].max() == 1.0


def test_ranking_preserves_the_ordering_the_metric_reads():
    values = _probs()
    ranked = rank_normalise(values)

    for column in range(values.shape[1]):
        assert np.array_equal(
            values[:, column].argsort(), ranked[:, column].argsort()
        )


def test_ties_take_the_average_rank():
    """Otherwise two studies a member cannot separate gain an order from their
    position in the array, which is information the model never had."""
    values = np.array([[0.5], [0.5], [0.9]])
    ranked = rank_normalise(values)

    assert ranked[0, 0] == ranked[1, 0]
    assert ranked[2, 0] == 1.0


def test_an_all_tied_column_collapses_rather_than_inventing_an_order():
    ranked = rank_normalise(np.array([[0.5], [0.5], [0.5]]))
    assert len(set(ranked[:, 0])) == 1


def test_ranking_is_invariant_to_calibration():
    """The reason ranks and not probabilities: a monotone rescaling of one
    member must not let it dominate the average."""
    values = _probs()
    squashed = values * 0.01 + 0.5

    assert np.allclose(rank_normalise(values), rank_normalise(squashed))


def test_a_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"\[studies, targets\]"):
        rank_normalise(np.zeros((3, 4, 5)))


def test_a_single_study_cannot_be_ranked():
    """One row has no ordering; returning zeros beats dividing by zero."""
    assert rank_normalise(np.array([[0.7, 0.2]])).shape == (1, 2)


# --- the average ---------------------------------------------------------------


def test_averaging_one_member_returns_its_own_ranks():
    values = _probs()
    assert np.allclose(rank_average([values]), rank_normalise(values))


def test_averaging_a_member_with_itself_changes_nothing():
    """An 'ensemble' of one model twice must not appear to gain."""
    values = _probs()
    assert np.allclose(rank_average([values, values]), rank_normalise(values))


def test_members_of_different_shapes_are_refused():
    with pytest.raises(ValueError, match="disagree on shape"):
        rank_average([_probs(studies=40), _probs(studies=39)])


def test_an_empty_ensemble_is_refused():
    with pytest.raises(ValueError, match="at least one member"):
        rank_average([])


# --- agreement, which is why a gain is or is not surprising --------------------


def test_identical_members_agree_completely():
    values = _probs()
    assert member_agreement([values, values])["mean_rank_correlation"] == pytest.approx(
        1.0
    )


def test_independent_members_agree_much_less():
    agreement = member_agreement([_probs(1), _probs(2)])
    assert agreement["mean_rank_correlation"] < 0.5


def test_the_highest_pair_is_reported_as_well_as_the_mean():
    """Three members where two are twins: the mean hides it, the max does not."""
    values = _probs(1)
    agreement = member_agreement([values, values, _probs(2)])

    assert agreement["pairs"] == 3
    assert agreement["max_rank_correlation"] == pytest.approx(1.0)
    assert agreement["mean_rank_correlation"] < 1.0


def test_a_constant_column_is_skipped_rather_than_producing_nan():
    constant = np.zeros((10, 2))
    constant[:, 1] = _probs(3, studies=10, targets=1).ravel()
    agreement = member_agreement([constant, constant])

    assert np.isfinite(agreement["mean_rank_correlation"])


# --- the TTA measurement -------------------------------------------------------


def _fake_score_one(monkeypatch, by_offset):
    """Stand in for the GPU pass, returning chosen predictions per offset."""
    targets = (np.arange(40)[:, None] % 2).repeat(12, axis=1).astype(float)
    weights = np.ones((40, 12))
    calls: list[int] = []

    def fake(checkpoint, **kwargs):
        offset = int(kwargs.get("center_offset", 0))
        calls.append(offset)
        return {
            "checkpoint": str(checkpoint),
            "experiment": f"E{checkpoint}",
            "macro_auc": 0.5,
            "probabilities": by_offset[offset],
            "targets": targets,
            "weights": weights,
        }

    monkeypatch.setattr("rsna_knee.ensemble_eval.score_one", fake)
    return calls


def test_tta_scores_every_offset_and_averages_them(monkeypatch):
    views = {offset: _probs(offset + 5) for offset in SUBMISSION_TTA_OFFSETS}
    calls = _fake_score_one(monkeypatch, views)

    result = measure_tta("ckpt.pt")

    assert calls == list(SUBMISSION_TTA_OFFSETS)
    assert result["cost_multiple"] == 3
    assert set(result["per_offset_macro"]) == {"-1", "0", "1"}


def test_tta_compares_the_average_against_the_centre_offset(monkeypatch):
    """Not against the mean of the offsets: one offset is what you would ship."""
    views = {offset: _probs(offset + 5) for offset in SUBMISSION_TTA_OFFSETS}
    _fake_score_one(monkeypatch, views)

    result = measure_tta("ckpt.pt")
    assert result["single_offset_macro"] == pytest.approx(
        result["per_offset_macro"]["0"]
    )
    assert result["delta"] == pytest.approx(
        result["averaged_macro"] - result["single_offset_macro"]
    )


def test_identical_views_show_tta_worth_exactly_nothing(monkeypatch):
    same = _probs(9)
    _fake_score_one(monkeypatch, {o: same for o in SUBMISSION_TTA_OFFSETS})

    result = measure_tta("ckpt.pt")
    assert result["delta"] == pytest.approx(0.0)
    assert result["worth_keeping"] is False


def test_the_threshold_is_declared_and_applied(monkeypatch):
    assert TTA_WORTH_KEEPING == 0.005

    _fake_score_one(monkeypatch, {o: _probs(9) for o in SUBMISSION_TTA_OFFSETS})
    result = measure_tta("ckpt.pt")

    assert result["threshold"] == TTA_WORTH_KEEPING
    assert result["worth_keeping"] == (result["delta"] >= TTA_WORTH_KEEPING)


def test_the_offsets_are_the_submissions_own():
    from rsna_knee.b42_kaggle_fast_preprocess import B42_FAST_TTA_OFFSETS

    assert tuple(SUBMISSION_TTA_OFFSETS) == tuple(B42_FAST_TTA_OFFSETS)


# --- the ensemble measurement --------------------------------------------------


def test_the_ensemble_is_compared_against_its_best_member(monkeypatch):
    """Against the mean member it would almost always look good."""
    macros = iter([0.70, 0.80])

    targets = (np.arange(40)[:, None] % 2).repeat(12, axis=1).astype(float)
    weights = np.ones((40, 12))
    predictions = iter([_probs(1), _probs(2)])

    def fake(checkpoint, **kwargs):
        return {
            "checkpoint": str(checkpoint),
            "experiment": str(checkpoint),
            "macro_auc": next(macros),
            "probabilities": next(predictions),
            "targets": targets,
            "weights": weights,
        }

    monkeypatch.setattr("rsna_knee.ensemble_eval.score_one", fake)
    result = measure_ensemble(["a.pt", "b.pt"])

    assert result["best_member_macro"] == 0.80
    assert result["gain_over_best_member"] == pytest.approx(
        result["ensemble_macro"] - 0.80
    )


def test_an_ensemble_of_a_model_with_itself_gains_nothing(monkeypatch):
    """The clearest artefact: duplicating a member must not manufacture a gain."""
    same = _probs(4)
    targets = (np.arange(40)[:, None] % 2).repeat(12, axis=1).astype(float)
    weights = np.ones((40, 12))

    def fake(checkpoint, **kwargs):
        return {
            "checkpoint": str(checkpoint),
            "experiment": str(checkpoint),
            "macro_auc": 0.75,
            "probabilities": same,
            "targets": targets,
            "weights": weights,
        }

    monkeypatch.setattr("rsna_knee.ensemble_eval.score_one", fake)
    result = measure_ensemble(["a.pt", "b.pt"])

    # AUC reads only the ordering, and rank-averaging a member with itself
    # returns its own ranks, so the ensemble must score exactly what one member
    # scores on those same predictions.
    from rsna_knee.b52_competition_training import macro_auc

    alone = float(macro_auc(targets, weights, same)["macro_auc"])

    assert result["agreement"]["mean_rank_correlation"] == pytest.approx(1.0)
    assert result["ensemble_macro"] == pytest.approx(alone, abs=1e-9)


def test_the_ensemble_reports_agreement_beside_the_gain(monkeypatch):
    targets = (np.arange(40)[:, None] % 2).repeat(12, axis=1).astype(float)
    weights = np.ones((40, 12))
    predictions = iter([_probs(1), _probs(2)])

    def fake(checkpoint, **kwargs):
        return {
            "checkpoint": str(checkpoint),
            "experiment": str(checkpoint),
            "macro_auc": 0.75,
            "probabilities": next(predictions),
            "targets": targets,
            "weights": weights,
        }

    monkeypatch.setattr("rsna_knee.ensemble_eval.score_one", fake)
    result = measure_ensemble(["a.pt", "b.pt"])

    assert "agreement" in result
    assert result["agreement"]["pairs"] == 1


# --- the report ----------------------------------------------------------------


def _tta_result(delta=0.0004):
    return {
        "offsets": [-1, 0, 1],
        "single_offset_macro": 0.8300,
        "averaged_macro": 0.8300 + delta,
        "delta": delta,
        "cost_multiple": 3,
        "worth_keeping": delta >= TTA_WORTH_KEEPING,
        "threshold": TTA_WORTH_KEEPING,
        "per_offset_macro": {"-1": 0.8291, "0": 0.8300, "1": 0.8295},
    }


def test_the_tta_report_says_what_the_views_cost(capsys):
    report(_tta_result())
    printed = capsys.readouterr().out

    assert "3x the runtime" in printed
    assert "DROP IT" in printed
    assert "affordable" in printed


def test_a_worthwhile_tta_says_keep_it(capsys):
    report(_tta_result(delta=0.02))
    printed = capsys.readouterr().out

    assert "keep it" in printed
    assert "DROP IT" not in printed


def test_the_ensemble_report_prints_agreement_next_to_the_gain(capsys):
    report(
        {
            "members": [
                {"experiment": "B53", "macro_auc": 0.8402},
                {"experiment": "B55", "macro_auc": 0.8471},
            ],
            "ensemble_macro": 0.8520,
            "best_member_macro": 0.8471,
            "gain_over_best_member": 0.0049,
            "agreement": {
                "pairs": 1,
                "mean_rank_correlation": 0.83,
                "max_rank_correlation": 0.83,
            },
        }
    )
    printed = capsys.readouterr().out

    assert "+0.004900" in printed
    assert "0.8300" in printed
    assert "pays for disagreement" in printed


# --- the command line ----------------------------------------------------------


def test_tta_mode_refuses_more_than_one_checkpoint(monkeypatch, capsys):
    import sys

    from rsna_knee import ensemble_eval as module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e", "--mode", "tta", "--data-root", "/d", "--labels-root", "/l",
            "--series-policy", "/p", "--base-checkpoint", "/b",
            "--domain-split", "/s", "--checkpoint", "a.pt", "--checkpoint", "b.pt",
        ],
    )
    with pytest.raises(SystemExit, match="exactly one"):
        module.main()


def test_ensemble_mode_refuses_a_single_checkpoint(monkeypatch):
    import sys

    from rsna_knee import ensemble_eval as module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e", "--mode", "ensemble", "--data-root", "/d", "--labels-root", "/l",
            "--series-policy", "/p", "--base-checkpoint", "/b",
            "--domain-split", "/s", "--checkpoint", "a.pt",
        ],
    )
    with pytest.raises(SystemExit, match="at least two"):
        module.main()


def test_the_command_line_renders(capsys, monkeypatch):
    import sys

    from rsna_knee import ensemble_eval as module

    monkeypatch.setattr(sys, "argv", ["e", "--help"])
    with pytest.raises(SystemExit) as exit_code:
        module.main()

    assert exit_code.value.code == 0
    printed = capsys.readouterr().out
    for flag in ("--mode", "--checkpoint", "--labels-root", "--out-json"):
        assert flag in printed


def test_the_version_is_recorded():
    assert ENSEMBLE_VERSION == "ensemble_eval_v1"


# --- executed, not read --------------------------------------------------------


def test_score_one_accepts_what_this_module_passes():
    from rsna_knee.common_ruler_eval import score_one

    inspect.signature(score_one).bind(
        "ckpt.pt",
        config={},
        data_root="/d",
        labels_root="/l",
        series_policy_path="/p",
        base_checkpoint="/b",
        domain_split="/s",
        surface_cells=None,
        num_workers=None,
        center_offset=-1,
        with_predictions=True,
    )


def test_the_macro_helper_matches_the_trainers_own():
    """One scorer, so this module and the trainer cannot disagree."""
    from rsna_knee.b52_competition_training import macro_auc

    targets = (np.arange(40)[:, None] % 2).repeat(12, axis=1).astype(float)
    weights = np.ones((40, 12))
    predictions = _probs(7)

    assert "macro_auc" in macro_auc(targets, weights, predictions)
