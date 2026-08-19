"""The parser audit must count agreement honestly, including the empty cases."""

from __future__ import annotations

import pandas as pd
import pytest

from tools.label_audit import _audit_target


def _frame(states, truths, confidences=None):
    if confidences is None:
        confidences = [0.9 if s in ("positive", "negated") else 0.0 for s in states]
    return pd.DataFrame({
        "T__truth": [float(t) for t in truths],
        "T__state": states,
        "T__confidence": confidences,
    })


def test_a_perfect_parser_scores_one():
    result = _audit_target(_frame(["positive", "negated"], [1, 0]), "T")
    assert result["yes_accuracy"] == 1.0
    assert result["no_accuracy"] == 1.0


def test_a_reversed_parser_scores_zero():
    result = _audit_target(_frame(["positive", "negated"], [0, 1]), "T")
    assert result["yes_accuracy"] == 0.0
    assert result["no_accuracy"] == 0.0


def test_yes_and_no_are_scored_separately():
    """The whole point: one can be reliable while the other is not."""
    result = _audit_target(
        _frame(["positive", "positive", "negated", "negated"], [1, 1, 0, 1]), "T"
    )
    assert result["yes_accuracy"] == 1.0
    assert result["no_accuracy"] == 0.5


def test_low_confidence_cells_are_not_counted():
    """Training ignores them, so the audit must ignore them too."""
    result = _audit_target(
        _frame(["positive", "negated"], [0, 1], confidences=[0.1, 0.1]), "T"
    )
    assert result["said_yes"] == 0
    assert result["said_no"] == 0
    assert result["said_nothing"] == 2


def test_silence_hiding_a_positive_is_counted():
    result = _audit_target(
        _frame(["unmentioned", "unmentioned"], [1, 0]), "T"
    )
    assert result["said_nothing"] == 2
    assert result["said_nothing_but_positive"] == 1


def test_a_target_the_parser_never_speaks_about_does_not_crash():
    result = _audit_target(_frame(["unmentioned"], [1]), "T")
    assert result["yes_accuracy"] is None
    assert result["no_accuracy"] is None


def test_the_expert_positive_rate_is_reported():
    result = _audit_target(_frame(["positive"] * 4, [1, 1, 0, 0]), "T")
    assert result["expert_positive_rate"] == pytest.approx(0.5)
