"""Turning a finished B57 arm into a submission.

A submission is the most expensive place in this project to find a bug: it
costs a slot and a day, and returns one number that cannot distinguish a bad
model from bad plumbing. B41 already paid that once.

So the tests here are weighted towards the guards -- the endpoint refusals and
the reproduction check -- rather than towards the happy path, which the
leaderboard will exercise anyway.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b57_models import ARMS
from rsna_knee.b57_protocol import VERSION
from rsna_knee.b57_submission import (
    REPRODUCTION_TOLERANCE,
    generate,
    load_endpoint,
    recorded_macro_auc,
    submission_frame,
    verify,
)
from rsna_knee.constants import SUBMISSION_COLUMNS, TARGETS


def _payload(**overrides):
    payload = {
        "version": VERSION,
        "arm": ARMS[1],
        "selection": "fixed_final_epoch",
        "completed_epochs": 12,
        "model_config": {"epochs": 12},
        "history": [{"validation": {"macro_auc": 0.774759}}],
    }
    payload.update(overrides)
    return payload


def _save(tmp_path, payload):
    import torch

    path = tmp_path / "final.pt"
    torch.save(payload, path)
    return path


# --- the endpoint is the endpoint ----------------------------------------------


def test_a_finished_arm_loads(tmp_path):
    assert load_endpoint(_save(tmp_path, _payload()))["arm"] == ARMS[1]


def test_a_payload_from_another_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a b57_v1 checkpoint"):
        load_endpoint(_save(tmp_path, _payload(version="b57_v2")))


def test_a_selection_rule_other_than_the_declared_endpoint_is_refused(tmp_path):
    """A best-epoch payload was chosen after seeing its own curve."""
    with pytest.raises(ValueError, match="chosen after seeing its own curve"):
        load_endpoint(_save(tmp_path, _payload(selection="best_validation_epoch")))


def test_an_arm_that_stopped_early_is_refused(tmp_path):
    """The DINOv2 candidate peaked at epoch 5 of 12; that is not the endpoint."""
    stopped = _payload(completed_epochs=5)
    with pytest.raises(ValueError, match="stopped at 5 of 12"):
        load_endpoint(_save(tmp_path, stopped))


def test_an_unknown_arm_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown B57 arm"):
        load_endpoint(_save(tmp_path, _payload(arm="something_else")))


def test_the_recorded_score_comes_from_the_last_epoch_not_the_best():
    """The peak is not what B57 claims, and must not be what this reports."""
    payload = _payload(history=[
        {"validation": {"macro_auc": 0.794906}},   # the peak, at epoch 5
        {"validation": {"macro_auc": 0.774759}},   # the endpoint
    ])
    assert recorded_macro_auc(payload) == 0.774759


def test_a_payload_with_no_history_cannot_be_verified():
    with pytest.raises(ValueError, match="no history to verify against"):
        recorded_macro_auc(_payload(history=[]))


# --- the reproduction check, which is the point --------------------------------


class _Model:
    """Returns whatever probabilities the test wants, through the real check."""

    def __init__(self, probabilities):
        self.probabilities = probabilities


def _verify_with(monkeypatch, reproduced, recorded=0.774759):
    """Run the real `verify` with its two data calls replaced."""
    from rsna_knee import b57_submission as module
    from rsna_knee.constants import TARGETS

    rows = 40
    generator = np.random.default_rng(0)
    target = (generator.random((rows, len(TARGETS))) > 0.5).astype(np.float64)
    weight = np.ones_like(target)

    monkeypatch.setattr(module, "validation_surface", lambda run_root, payload: object())
    monkeypatch.setattr(module, "make_loader", lambda *a, **k: object())
    monkeypatch.setattr(
        module, "predict",
        lambda model, loader, runtime: {
            "uids": np.arange(rows).astype(str),
            "prediction": generator.random((rows, len(TARGETS))),
            "target": target, "weight": weight,
        },
    )
    # Replace only the scoring, so the comparison logic itself stays real.
    monkeypatch.setattr(
        "rsna_knee.b52_competition_training.macro_auc",
        lambda t, w, p: {"macro_auc": reproduced, "per_target_auc": {}, "targets_defined": 12},
    )
    return module.verify(".", _payload(history=[{"validation": {"macro_auc": recorded}}]),
                         object(), object())


def test_an_inference_path_that_agrees_passes(monkeypatch):
    result = _verify_with(monkeypatch, reproduced=0.774759)

    assert result["agrees"] is True
    assert abs(result["delta"]) <= REPRODUCTION_TOLERANCE


def test_floating_point_drift_within_tolerance_passes(monkeypatch):
    """Autocast reorders sums; that is not a divergence."""
    result = _verify_with(monkeypatch, reproduced=0.774759 + 2e-4)
    assert result["agrees"] is True


def test_an_inference_path_that_disagrees_refuses_to_submit(monkeypatch):
    """The whole reason this function exists."""
    with pytest.raises(ValueError, match="Do not submit"):
        _verify_with(monkeypatch, reproduced=0.71)


def test_the_refusal_names_both_numbers(monkeypatch):
    """So the reader can see which way the path diverged."""
    with pytest.raises(ValueError) as problem:
        _verify_with(monkeypatch, reproduced=0.71)

    message = str(problem.value)
    assert "0.710000" in message and "0.774759" in message


def test_a_small_but_real_divergence_is_still_caught(monkeypatch):
    """A tolerance loose enough to pass a genuine bug would be worse than none."""
    with pytest.raises(ValueError, match="Do not submit"):
        _verify_with(monkeypatch, reproduced=0.774759 + 0.002)


# --- the file that gets uploaded ------------------------------------------------


def test_the_submission_has_the_competition_columns_in_order():
    frame = submission_frame(["a", "b"], np.full((2, len(TARGETS)), 0.5))
    assert list(frame.columns) == SUBMISSION_COLUMNS
    assert frame.columns[0] == "StudyInstanceUID"
    assert len(frame) == 2


def test_the_row_order_follows_the_study_list():
    probabilities = np.zeros((2, len(TARGETS)))
    probabilities[1, 0] = 1.0
    frame = submission_frame(["first", "second"], probabilities)

    assert list(frame["StudyInstanceUID"]) == ["first", "second"]
    assert frame.loc[1, TARGETS[0]] == 1.0


def test_a_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="expected"):
        submission_frame(["a", "b"], np.zeros((3, len(TARGETS))))


def test_a_nonfinite_probability_is_refused():
    values = np.full((1, len(TARGETS)), 0.5)
    values[0, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        submission_frame(["a"], values)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_values_outside_zero_to_one_are_refused(bad):
    values = np.full((1, len(TARGETS)), 0.5)
    values[0, 0] = bad
    with pytest.raises(ValueError, match="must be probabilities"):
        submission_frame(["a"], values)


def test_the_frame_survives_a_round_trip_to_csv(tmp_path):
    """It is written and then read by the grader, so it must reload identically."""
    frame = submission_frame(["a", "b"], np.full((2, len(TARGETS)), 0.25))
    path = tmp_path / "submission.csv"
    frame.to_csv(path, index=False)

    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == SUBMISSION_COLUMNS
    assert reloaded[TARGETS].to_numpy().tolist() == frame[TARGETS].to_numpy().tolist()


# --- executed against the real signatures ---------------------------------------


def test_every_call_this_module_makes_binds():
    """A wrong keyword here fails only once the model is on the card."""
    from rsna_knee.b57_protocol import load_protocol
    from rsna_knee.b57_training import make_dataset, make_loader, predict, runtime_for

    inspect.signature(load_protocol).bind(".")
    inspect.signature(make_dataset).bind(".", {}, "validation")
    inspect.signature(make_loader).bind(object(), object(), {})
    inspect.signature(predict).bind(object(), object(), object())
    inspect.signature(runtime_for).bind("auto", 0)


def test_the_model_is_rebuilt_without_fetching_public_weights():
    """An offline notebook has no public_init directory to read."""
    from rsna_knee.b57_models import build_model

    assert inspect.signature(build_model).parameters["public_root"].default is None

    from rsna_knee import b57_submission as module
    assert "public_root=None" in inspect.getsource(module.rebuild_model)


def test_no_test_time_augmentation_is_introduced():
    """The recorded number came from a single centre offset."""
    from rsna_knee import b57_submission as module

    source = inspect.getsource(module.test_surface)
    assert "center_offsets=(0,)" in source
    assert "tta_center_offsets = ()" in source


def test_generate_defaults_to_the_candidate_arm():
    assert inspect.signature(generate).parameters["arm"].default == ARMS[1]


def test_verification_is_on_by_default():
    """Skipping it must be a decision, not an oversight."""
    assert inspect.signature(generate).parameters["skip_verify"].default is False


def test_the_command_line_renders(capsys, monkeypatch):
    import sys

    from rsna_knee import b57_submission as module

    monkeypatch.setattr(sys, "argv", ["b57-submit", "--help"])
    with pytest.raises(SystemExit) as exit_code:
        module.main()

    assert exit_code.value.code == 0
    printed = capsys.readouterr().out
    for flag in ("--run-root", "--arm", "--data-root", "--out-path", "--skip-verify"):
        assert flag in printed
