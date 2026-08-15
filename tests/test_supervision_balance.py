"""The prospective class-balance rule that justifies a targeted fill."""
from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.supervision_balance import (
    DEFAULT_IMBALANCE_THRESHOLD,
    balance_table,
    flag_unlearnable,
    format_balance,
)


def _surface(counts):
    """Build targets/weights with a declared positive/negative count per target."""
    rows = max(p + n for p, n in counts.values()) + 1
    y = np.full((rows, len(TARGETS)), 0.5)
    w = np.zeros((rows, len(TARGETS)))
    for j, target in enumerate(TARGETS):
        pos, neg = counts[target]
        y[:pos, j], w[:pos, j] = 0.85, 0.5
        y[pos : pos + neg, j], w[pos : pos + neg, j] = 0.05, 1.0
    return y, w


def _uniform(pos, neg):
    return {t: (pos, neg) for t in TARGETS}


def test_counts_only_include_cells_that_carry_supervision():
    y, w = _surface(_uniform(30, 20))
    table = balance_table(y, w).set_index("target")
    assert table.loc[TARGETS[0], "positive"] == 30
    assert table.loc[TARGETS[0], "negative"] == 20
    assert table.loc[TARGETS[0], "usable_cells"] == 50
    assert table.loc[TARGETS[0], "majority_share"] == pytest.approx(0.6)


def test_the_measured_synovitis_failure_is_flagged():
    # B6 supplied 322 positive and 13 negative Synovitis cells: 96.1% one class.
    counts = _uniform(200, 200)
    counts["Synovitis"] = (322, 13)
    flagged = flag_unlearnable(balance_table(*_surface(counts))).set_index("target")
    assert flagged.loc["Synovitis", "majority_share"] == pytest.approx(322 / 335, abs=1e-4)
    assert flagged.loc["Synovitis", "needs_fill"]
    assert flagged.loc["Synovitis", "minority_class"] == "negative"
    # A balanced target must not be swept up with it.
    assert not flagged.loc["ACL", "needs_fill"]


def test_a_target_with_too_few_minority_cells_is_flagged_even_when_the_share_passes():
    counts = _uniform(200, 200)
    counts["Fracture"] = (100, 12)  # 89.3% share passes, 12 negatives do not
    flagged = flag_unlearnable(balance_table(*_surface(counts))).set_index("target")
    assert flagged.loc["Fracture", "majority_share"] < DEFAULT_IMBALANCE_THRESHOLD
    assert flagged.loc["Fracture", "fails_minority_count"]
    assert flagged.loc["Fracture", "needs_fill"]


def test_the_rule_applies_uniformly_rather_than_naming_targets():
    # Whichever target is broken gets flagged; nothing is hard-coded.
    for broken in ("ACL", "Effusion", "Baker's"):
        counts = _uniform(200, 200)
        counts[broken] = (400, 5)
        flagged = flag_unlearnable(balance_table(*_surface(counts))).set_index("target")
        assert flagged.loc[broken, "needs_fill"]
        assert int(flagged["needs_fill"].sum()) == 1


def test_a_healthy_surface_flags_nothing():
    flagged = flag_unlearnable(balance_table(*_surface(_uniform(150, 150))))
    assert not flagged["needs_fill"].any()


def test_the_report_states_that_no_outcome_was_consulted():
    y, w = _surface(_uniform(150, 150))
    payload = {
        "labeller": "b6", "threshold": 0.9, "min_minority_cells": 30,
        "targets_needing_fill": [], "n_targets_needing_fill": 0,
        "table": flag_unlearnable(balance_table(y, w)).to_dict("records"),
    }
    text = format_balance(payload)
    assert "computed from training labels alone" in text
    assert "not" in text and "outcome-driven" in text


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        balance_table(np.zeros((4, len(TARGETS))), np.zeros((4, 3)))
