"""The probe must refuse a bad setup before it spends time encoding MRI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tools import encoder_probe


def test_both_encoders_are_offered():
    assert encoder_probe.ENCODERS == ("report-aligned", "dinov3")


def test_report_aligned_without_a_checkpoint_is_refused_immediately():
    """Encoding takes many minutes; a missing path must fail in a second."""
    result = subprocess.run(
        [sys.executable, "-m", "tools.encoder_probe", "--encoder", "report-aligned",
         "--data-root", "/nonexistent", "--latin-script-labels", "a",
         "--all-script-labels", "b"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires --encoder-checkpoint" in result.stderr


def test_the_reading_guide_names_the_decision_it_supports():
    import inspect

    source = inspect.getsource(encoder_probe.main)
    assert "0.50" in source and "will not help" in source


@pytest.mark.parametrize("bad", ["latin", "everything"])
def test_an_unknown_supervision_surface_is_refused(bad):
    from model._implementation import report_label_supervision

    with pytest.raises(ValueError, match="surface must be one of"):
        report_label_supervision(None, surface=bad, latin_root="a", all_root="b")
