"""B50's evaluator refuses a verdict its measurement could not support.

B48 and B49 were both recorded as `no_support` against a +0.010 threshold their
measurements could not reach -- their two arms ordered only 0.0015 and 0.0024 of
study pairs differently, which caps how far any AUC could move. That is a fact
about the endpoint, not about the mechanisms they tested, and filing it as
`no_support` misattributed it.

So the ceiling is computed first here, and `no_support` is reserved for a
measurement that could actually have passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.b50_adapted_hierarchy_eval import (
    B50_SEEN_TOLERANCE,
    B50_SUPPORT_DELTA,
    B50_SUPPORT_MIN_TARGETS,
    B50_SUPPORT_PROBABILITY,
    B50_SURFACES,
    decide,
    discordant_pair_fraction,
)
from pathlib import Path

from rsna_knee.constants import TARGETS


# --- the ceiling ----------------------------------------------------------


def test_identical_arms_can_move_no_auc_at_all():
    values = np.random.default_rng(0).random((20, len(TARGETS)))
    assert discordant_pair_fraction(values, values.copy()) == 0.0


def test_a_rescaled_prediction_is_the_same_ranking():
    values = np.random.default_rng(1).random((20, len(TARGETS)))
    assert discordant_pair_fraction(values, values * 3.0 + 0.5) == pytest.approx(0.0)


def test_a_reversed_ranking_moves_every_pair():
    values = np.random.default_rng(2).random((20, len(TARGETS)))
    assert discordant_pair_fraction(values, -values) == pytest.approx(1.0)


def test_the_ceiling_is_averaged_over_targets():
    rng = np.random.default_rng(3)
    control = rng.random((30, len(TARGETS)))
    candidate = control.copy()
    candidate[:, 0] = -control[:, 0]  # one target fully reversed, eleven identical
    assert discordant_pair_fraction(control, candidate) == pytest.approx(1 / 12, abs=1e-9)


# --- the decision ---------------------------------------------------------


def _primary(
    *,
    delta: float,
    ceiling: float,
    ci_lower: float,
    probability: float,
    improved: int,
    loto_positive: bool = True,
) -> dict:
    return {
        "delta": delta,
        "discordant_pair_fraction": ceiling,
        "max_possible_abs_delta": ceiling,
        "targets_improved_count": improved,
        "paired_bootstrap": {
            "ci_lower": ci_lower,
            "probability_candidate_better": probability,
        },
        "leave_one_target_out_candidate_minus_control": {
            name: (0.01 if loto_positive else -0.01) for name in TARGETS
        },
    }


def _seen(delta: float = 0.0) -> dict:
    return {"delta": delta}


def test_a_measurement_that_could_not_reach_the_threshold_is_named_as_such():
    """The B48 and B49 case, which must never be filed as no_support again."""
    for ceiling in (0.0015, 0.0024):
        verdict = decide(
            _primary(
                delta=0.0005,
                ceiling=ceiling,
                ci_lower=0.0003,
                probability=1.0,
                improved=10,
            ),
            _seen(),
        )
        assert verdict["outcome"] == "endpoint_underpowered"
        assert "not about the adapted hierarchy" in verdict["reason"]


def test_the_underpowered_check_runs_before_everything_else():
    """Even a result that would otherwise pass every clause."""
    verdict = decide(
        _primary(delta=0.05, ceiling=0.001, ci_lower=0.04, probability=1.0, improved=12),
        _seen(),
    )
    assert verdict["outcome"] == "endpoint_underpowered"


def test_a_clean_positive_is_supported():
    verdict = decide(
        _primary(delta=0.02, ceiling=0.20, ci_lower=0.01, probability=0.99, improved=9),
        _seen(0.0),
    )
    assert verdict["outcome"] == "supported"
    assert all(verdict["checks"].values())


def test_a_genuine_negative_is_no_support():
    verdict = decide(
        _primary(
            delta=-0.01, ceiling=0.20, ci_lower=-0.03, probability=0.05, improved=3
        ),
        _seen(),
    )
    assert verdict["outcome"] == "no_support"


def test_a_positive_that_misses_one_clause_is_inconclusive_not_supported():
    verdict = decide(
        _primary(
            # Above the delta bar and the interval clears zero, but only six
            # targets improve, so the effect may rest on too few findings.
            delta=0.012,
            ceiling=0.20,
            ci_lower=0.002,
            probability=0.97,
            improved=B50_SUPPORT_MIN_TARGETS - 1,
        ),
        _seen(),
    )
    assert verdict["outcome"] == "inconclusive"
    assert verdict["checks"]["targets_improved_at_least"] is False


def test_a_gain_carried_by_one_target_is_caught():
    """The failure that misled this project twice before."""
    verdict = decide(
        _primary(
            delta=0.02,
            ceiling=0.20,
            ci_lower=0.01,
            probability=0.99,
            improved=9,
            loto_positive=False,
        ),
        _seen(),
    )
    assert verdict["outcome"] != "supported"
    assert verdict["checks"]["every_leave_one_target_out_positive"] is False


def test_a_candidate_that_collapses_on_seen_scanners_is_not_supported():
    verdict = decide(
        _primary(delta=0.02, ceiling=0.20, ci_lower=0.01, probability=0.99, improved=9),
        _seen(B50_SEEN_TOLERANCE - 0.01),
    )
    assert verdict["outcome"] != "supported"
    assert verdict["checks"]["seen_scanner_delta_within_tolerance"] is False


def test_the_thresholds_are_the_ones_the_protocol_froze():
    assert B50_SUPPORT_DELTA == 0.010
    assert B50_SUPPORT_PROBABILITY == 0.95
    assert B50_SUPPORT_MIN_TARGETS == 7
    assert B50_SEEN_TOLERANCE == -0.005


def test_all_three_prediction_paths_are_scored():
    """B49 discarded its base predictions and could not separate them later."""
    assert B50_SURFACES == ("combined", "base", "local")


def test_the_primary_surface_is_the_one_a_submission_would_use():
    from rsna_knee.b50_adapted_hierarchy_eval import B50_PRIMARY_SPLIT

    assert B50_PRIMARY_SPLIT == "validation_unseen_scanners"
    assert B50_SURFACES[0] == "combined"


# --- the expert audit -----------------------------------------------------


def test_the_expert_audit_is_a_third_surface_that_selects_nothing():
    """It answers whether the gain is against reports or against knees."""
    import inspect

    from rsna_knee.b50_adapted_hierarchy_eval import (
        B50_COMPARATOR_SPLIT,
        B50_EXPERT58_SPLIT,
        B50_PRIMARY_SPLIT,
        decide,
        evaluate_b50_pair,
    )

    assert B50_EXPERT58_SPLIT == "expert58_audit"
    assert B50_EXPERT58_SPLIT not in (B50_PRIMARY_SPLIT, B50_COMPARATOR_SPLIT)

    # `decide` takes only the primary and the seen comparator. The expert
    # surface is reported beside the verdict and cannot enter it.
    parameters = list(inspect.signature(decide).parameters)
    assert parameters == ["primary", "seen"]

    source = inspect.getsource(evaluate_b50_pair)
    verdict_call = source.split("verdict = decide(")[1].split(")")[0]
    assert B50_EXPERT58_SPLIT not in verdict_call


def test_every_gold_cell_is_scored():
    """Unlike the weak surface, no expert cell is blank."""
    import pandas as pd

    from rsna_knee.b50_adapted_hierarchy_eval import (
        B50_EXPERT58_STUDIES,
        expert58_surface,
    )
    import rsna_knee.b50_adapted_hierarchy_eval as module

    frame = pd.DataFrame(
        {"StudyInstanceUID": [f"gold-{i}" for i in range(B50_EXPERT58_STUDIES)]}
    )
    for name in TARGETS:
        frame[name] = 1.0

    module.__dict__.setdefault("_orig", None)
    import rsna_knee.data as data_module

    original_load, original_mask = data_module.load_train_csv, data_module.gold_mask
    data_module.load_train_csv = lambda path: frame
    data_module.gold_mask = lambda f: pd.Series(True, index=f.index)
    try:
        uids, targets, weights = expert58_surface(Path("."), {})
    finally:
        data_module.load_train_csv, data_module.gold_mask = original_load, original_mask

    assert len(uids) == B50_EXPERT58_STUDIES
    assert targets.shape == (B50_EXPERT58_STUDIES, len(TARGETS))
    assert (weights > 0).all(), "every expert call is a real label"


def test_an_incomplete_expert_surface_is_refused():
    import pandas as pd

    from rsna_knee.b50_adapted_hierarchy_eval import expert58_surface
    import rsna_knee.data as data_module

    frame = pd.DataFrame({"StudyInstanceUID": ["gold-0", "gold-1"]})
    for name in TARGETS:
        frame[name] = 1.0

    original_load, original_mask = data_module.load_train_csv, data_module.gold_mask
    data_module.load_train_csv = lambda path: frame
    data_module.gold_mask = lambda f: pd.Series(True, index=f.index)
    try:
        with pytest.raises(ValueError, match="complete 58-study expert surface"):
            expert58_surface(Path("."), {})
    finally:
        data_module.load_train_csv, data_module.gold_mask = original_load, original_mask
