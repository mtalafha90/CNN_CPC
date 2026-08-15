"""B24 governance: the properties that make the comparison worth running."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b24_protocol import (
    B24_FIXED_EPOCHS,
    GOLD_PROMOTION_MIN_PROBABILITY,
    MODE_CANDIDATE,
    MODE_CONTROL,
    cross_labeller_verdict,
    gold_promotion_decision,
    mode_identity,
    require_b24_contract,
    require_passed_labeller_gate,
)


def _config(**overrides):
    base = {
        "b7_epochs": B24_FIXED_EPOCHS,
        "b7_image_size": 224,
        "b7_batch_size": 2,
        "b7_n_slices": 16,
        "b7_transformer_layers": 2,
        "b7_transformer_heads": 8,
        "b7_pathology_layers": 1,
        "b12_1_series_pool_heads": 8,
        "seed": 2026,
        "b7_head_lr": 1e-4,
        "b7_encoder_lr": 0.0,
        "b17_encoder_frozen": True,
        "b18_expert_selection": False,
        "b24_surface": "matched",
    }
    base.update(overrides)
    return base


def test_the_b20_recipe_is_accepted_unchanged():
    require_b24_contract(_config())


@pytest.mark.parametrize(
    "override",
    [
        {"b7_epochs": 5},              # a different stopping point
        {"b7_head_lr": 3e-4},          # a different optimiser
        {"b7_image_size": 384},        # a different input
        {"b7_encoder_lr": 1e-5},       # an unfrozen encoder
        {"b17_encoder_frozen": False},
        {"b18_expert_selection": True},  # reintroduces selection optimism
        {"b24_surface": "full"},       # changes the batch count between arms
    ],
)
def test_any_recipe_drift_is_refused(override):
    # B24's whole claim is that only the labels changed, so drift is fatal.
    with pytest.raises(ValueError):
        require_b24_contract(_config(**override))


def test_the_two_arms_are_distinctly_identified():
    control_variant, control_name = mode_identity(MODE_CONTROL)
    candidate_variant, candidate_name = mode_identity(MODE_CANDIDATE)
    assert control_variant != candidate_variant
    assert "B6" in control_name and "B23" in candidate_name
    with pytest.raises(ValueError):
        mode_identity("something_else")


def test_training_is_refused_until_the_labeller_gate_has_passed(tmp_path):
    failing = tmp_path / "weak_holdout.json"
    failing.write_text(
        json.dumps({"labeller_gate": {"passed": False, "reasons": ["coverage too low"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        require_passed_labeller_gate(failing)


def test_training_is_refused_when_no_gate_exists_at_all(tmp_path):
    missing = tmp_path / "weak_holdout.json"
    missing.write_text(json.dumps({"surface": "b23_llm_holdout_v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no labeller gate"):
        require_passed_labeller_gate(missing)


def test_a_passed_gate_is_returned_for_the_record(tmp_path):
    passing = tmp_path / "weak_holdout.json"
    passing.write_text(
        json.dumps({"labeller_gate": {"passed": True, "reasons": [], "coverage": 0.9}}),
        encoding="utf-8",
    )
    gate = require_passed_labeller_gate(passing)
    assert gate["passed"] is True


# --- The cross-labeller reading ----------------------------------------------
# Each weak surface favours the arm trained by its own labeller, so only the
# pattern across both surfaces carries information.


def test_winning_only_on_its_own_surface_is_uninformative():
    verdict = cross_labeller_verdict(
        candidate_on_b23=0.80, control_on_b23=0.70,
        candidate_on_weak_v2=0.70, control_on_weak_v2=0.80,
    )
    assert verdict["strength"] == "uninformative"
    assert verdict["candidate_won_own_surface"] is True
    assert verdict["candidate_won_control_surface"] is False


def test_winning_on_the_controls_own_surface_is_the_strong_result():
    verdict = cross_labeller_verdict(
        candidate_on_b23=0.80, control_on_b23=0.70,
        candidate_on_weak_v2=0.78, control_on_weak_v2=0.72,
    )
    assert verdict["strength"] == "strong"
    assert "reproduces the B6 teacher better" in verdict["reading"]


def test_losing_on_its_own_surface_is_adverse():
    verdict = cross_labeller_verdict(
        candidate_on_b23=0.65, control_on_b23=0.75,
        candidate_on_weak_v2=0.60, control_on_weak_v2=0.80,
    )
    assert verdict["strength"] == "adverse"


def test_every_verdict_records_that_weak_surfaces_are_not_expert_truth():
    verdict = cross_labeller_verdict(
        candidate_on_b23=0.8, control_on_b23=0.7,
        candidate_on_weak_v2=0.8, control_on_weak_v2=0.7,
    )
    assert "not expert truth" in verdict["note"]


# --- The gold rule ------------------------------------------------------------
# B22 measured a 0.0439 within-run swing on this surface, so a bare point-
# estimate win must not promote anything.


def test_a_clear_win_promotes():
    assert gold_promotion_decision(paired_median=0.03, probability_candidate_better=0.97)["promoted"]


@pytest.mark.parametrize(
    "median,probability",
    [
        (0.002, 0.55),   # a small win well inside the noise
        (0.03, 0.80),    # a decent median but not confident enough
        (-0.01, 0.99),   # confident in the wrong direction
        (0.0, 0.95),     # exactly zero median
    ],
)
def test_anything_short_of_the_predeclared_rule_does_not_promote(median, probability):
    decision = gold_promotion_decision(
        paired_median=median, probability_candidate_better=probability
    )
    assert decision["promoted"] is False


def test_the_probability_threshold_is_the_predeclared_one():
    assert GOLD_PROMOTION_MIN_PROBABILITY == 0.95
    decision = gold_promotion_decision(paired_median=0.01, probability_candidate_better=0.95)
    assert decision["promoted"] is True
    assert decision["required_probability"] == 0.95


def test_the_decision_records_that_gold_is_not_independent_validation():
    decision = gold_promotion_decision(paired_median=0.01, probability_candidate_better=0.96)
    assert "not" in decision["interpretation"] and "independent" in decision["interpretation"]


def test_the_gold_acceptance_refuses_to_run_a_second_time(tmp_path):
    """One look means one look.

    A second run -- or a re-run after seeing the result and adjusting anything --
    would destroy the only property that makes the acceptance meaningful.
    """
    from rsna_knee.b24_accept import accept_b24

    out = tmp_path / "gold_acceptance"
    out.mkdir(parents=True)
    (out / "acceptance.json").write_text(json.dumps({"already": "consumed"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly one gold look"):
        accept_b24(
            {"data_root": str(tmp_path)},
            b20_checkpoint=tmp_path / "b20.pt",
            b24_checkpoint=tmp_path / "b24.pt",
            out_root=out,
        )
