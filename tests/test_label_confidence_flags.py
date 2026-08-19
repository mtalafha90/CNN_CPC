"""The label targets must be settable, and settable only to sane values.

`tools.label_audit` measured what the frozen report parser is actually worth:
when it says a finding is present the expert agrees 69% of the time, and when
it says absent, 96%. Training meanwhile aims a "yes" at 0.85, a number nobody
had checked against anything. Testing that gap needs the target to be a
command-line choice rather than a value buried in a config file, which is what
these cover.
"""

from __future__ import annotations

import pytest

from model._implementation import LABEL_CONFIDENCE_KEYS, set_label_confidence


def test_a_measured_target_replaces_the_shipped_one():
    updated = set_label_confidence({"seed": 2026}, positive_target=0.70)
    assert updated[LABEL_CONFIDENCE_KEYS["positive_target"]] == pytest.approx(0.70)
    assert updated["seed"] == 2026, "unrelated settings must survive"


def test_the_original_config_is_left_alone():
    """A run that changes its caller's config is a run that taints the next one."""
    original = {"b7_positive_target": 0.85}
    set_label_confidence(original, positive_target=0.70)
    assert original["b7_positive_target"] == pytest.approx(0.85)


def test_targets_not_given_are_not_touched():
    updated = set_label_confidence({"b7_positive_target": 0.85}, positive_target=None)
    assert updated["b7_positive_target"] == pytest.approx(0.85)
    assert "b7_negative_target" not in updated


def test_both_targets_can_be_set_together():
    updated = set_label_confidence({}, positive_target=0.70, negative_target=0.04)
    assert updated["b7_positive_target"] == pytest.approx(0.70)
    assert updated["b7_negative_target"] == pytest.approx(0.04)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.2, 1.5, 85])
def test_a_target_outside_zero_to_one_is_refused(value):
    """0.85 and 85 are an easy slip, and one of them trains towards nonsense."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        set_label_confidence({}, positive_target=value)


def test_an_unknown_target_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="uncertain_target"):
        set_label_confidence({}, uncertain_target=0.5)


def test_the_flags_reach_the_command_line():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--positive-target" in result.stdout
    assert "--negative-target" in result.stdout
    assert "69%" in result.stdout, "the flag should carry the measurement behind it"
