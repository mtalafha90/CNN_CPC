"""Letting the trainers use DataLoader workers, safely.

The measured prize is a third off every epoch. The measured hazard is the
file-descriptor exhaustion that made `num_workers: 0` the config default in the
first place, so the tests that matter are the ones about the sharing strategy:
it must be in force whenever workers are, and a run must refuse to start rather
than discover otherwise in its eighth hour.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from rsna_knee.loader_throughput import (
    MEASURED_HOURS_WITH_WORKERS,
    MEASURED_HOURS_WITHOUT_WORKERS,
    SHARING_STRATEGY,
    add_worker_argument,
    apply_worker_override,
    measured_speedup,
    use_file_system_sharing,
)


@pytest.fixture(autouse=True)
def _restore_sharing_strategy():
    """These tests change global process state, so put it back."""
    before = torch.multiprocessing.get_sharing_strategy()
    yield
    torch.multiprocessing.set_sharing_strategy(before)


# --- the measurement it exists for --------------------------------------------


def test_the_measured_numbers_are_b53s():
    assert MEASURED_HOURS_WITHOUT_WORKERS == 4.5
    assert 3.0 <= MEASURED_HOURS_WITH_WORKERS <= 3.3


def test_the_speedup_is_about_a_third():
    assert 0.25 < measured_speedup() < 0.35


# --- leaving existing runs alone ----------------------------------------------


def test_none_does_not_touch_the_config():
    """Every run made before this flag existed must stay byte-identical."""
    settings = {"num_workers": 0, "other": "kept"}
    state = apply_worker_override(settings, None)

    assert settings == {"num_workers": 0, "other": "kept"}
    assert state["num_workers"] == 0
    assert state["source"] == "config"


def test_none_reports_whatever_the_config_said():
    assert apply_worker_override({"num_workers": 4}, None)["num_workers"] == 4


def test_a_config_with_no_workers_key_reads_as_zero():
    assert apply_worker_override({}, None)["num_workers"] == 0


def test_a_null_workers_value_reads_as_zero():
    """`num_workers: null` is valid YAML for 'decide for me'; the audit needs a number."""
    assert apply_worker_override({"num_workers": None}, None)["num_workers"] == 0


# --- the override -------------------------------------------------------------


def test_an_explicit_value_reaches_the_settings():
    settings = {"num_workers": 0}
    state = apply_worker_override(settings, 6)

    assert settings["num_workers"] == 6
    assert state["num_workers"] == 6
    assert state["source"] == "command line"


def test_an_explicit_zero_is_honoured_rather_than_ignored():
    """The way a run reproduces the old behaviour on purpose, not by default."""
    settings = {"num_workers": 6}
    state = apply_worker_override(settings, 0)

    assert settings["num_workers"] == 0
    assert state["source"] == "command line"


def test_a_negative_count_is_refused():
    with pytest.raises(ValueError, match=">= 0"):
        apply_worker_override({}, -1)


# --- the sharing strategy, which is the whole safety story --------------------


def test_workers_switch_the_strategy_to_file_system():
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    state = apply_worker_override({}, 6)

    assert state["sharing_strategy"] == SHARING_STRATEGY
    assert torch.multiprocessing.get_sharing_strategy() == SHARING_STRATEGY


def test_zero_workers_leaves_the_strategy_alone():
    """Nothing is shared between processes, so nothing needs changing."""
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    state = apply_worker_override({}, 0)

    assert state["sharing_strategy"] == "file_descriptor"


def test_the_strategy_is_set_before_the_state_is_returned():
    """The caller records what is in force, so it must already be in force."""
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    state = apply_worker_override({}, 2)

    assert torch.multiprocessing.get_sharing_strategy() == state["sharing_strategy"]


def test_workers_are_refused_when_the_strategy_will_not_take(monkeypatch):
    """A platform without file_system ignores the request rather than raising.

    Silently keeping file_descriptor is exactly the run that dies part-way
    through, so this refuses at the start instead.
    """
    monkeypatch.setattr(
        torch.multiprocessing, "get_all_sharing_strategies", lambda: {"file_descriptor"}
    )
    monkeypatch.setattr(
        torch.multiprocessing, "get_sharing_strategy", lambda: "file_descriptor"
    )

    with pytest.raises(RuntimeError, match="exhausts the file-descriptor table"):
        apply_worker_override({}, 6)


def test_the_refusal_names_the_way_out():
    source = inspect.getsource(apply_worker_override)
    assert "--num-workers 0" in source


def test_setting_the_strategy_returns_what_is_actually_in_force():
    assert use_file_system_sharing() == torch.multiprocessing.get_sharing_strategy()


# --- the flag -----------------------------------------------------------------


def test_the_flag_defaults_to_none_so_the_config_wins():
    import argparse

    parser = argparse.ArgumentParser()
    add_worker_argument(parser)

    assert parser.parse_args([]).num_workers is None
    assert parser.parse_args(["--num-workers", "6"]).num_workers == 6


def test_the_help_text_carries_the_measurement():
    import argparse

    parser = argparse.ArgumentParser()
    add_worker_argument(parser)
    # argparse rewraps help text, so a phrase can be split across lines.
    help_text = " ".join(parser.format_help().split())

    assert "6 workers" in help_text
    assert "deterministic" in help_text


def test_the_help_text_contains_no_percent_sign():
    """argparse puts help text through `%` formatting.

    A literal `%` there raises TypeError the moment anyone runs --help, which
    is a broken command line rather than a broken run -- and it is the kind of
    thing no training test would ever reach.
    """
    import argparse

    parser = argparse.ArgumentParser()
    add_worker_argument(parser)

    parser.format_help()  # must not raise
    assert "%" not in parser._actions[-1].help


@pytest.mark.parametrize(
    "module", ["b52_competition_training", "b53_augmented_training"]
)
def test_the_trainers_command_line_still_renders(module, capsys, monkeypatch):
    """The whole parser, not just this one flag.

    `--help` exits 0 after printing, so SystemExit is the success path here.
    monkeypatch restores sys.argv even when the assertion below fails.
    """
    import importlib
    import sys

    trainer = importlib.import_module(f"rsna_knee.{module}")
    monkeypatch.setattr(sys, "argv", [module, "--help"])

    with pytest.raises(SystemExit) as exit_code:
        trainer.main()

    assert exit_code.value.code == 0
    assert "--num-workers" in capsys.readouterr().out


# --- the trainers actually use it ---------------------------------------------


@pytest.mark.parametrize(
    "module",
    ["b52_competition_training", "b53_augmented_training"],
)
def test_the_trainer_overrides_before_it_resolves_the_runtime(module):
    """resolve_runtime reads num_workers out of the settings, so order decides.

    Read out of the syntax tree, not searched for as text: the line that calls
    apply_worker_override carries a comment naming resolve_runtime, and a
    string search finds the comment first.
    """
    import ast
    import importlib

    trainer = importlib.import_module(f"rsna_knee.{module}")
    function = getattr(trainer, f"train_{module.split('_')[0]}")
    tree = ast.parse(inspect.getsource(function).lstrip())

    called_at = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_at.setdefault(node.func.id, node.lineno)

    assert "apply_worker_override" in called_at, "the trainer never calls it"
    assert "resolve_runtime" in called_at
    assert called_at["apply_worker_override"] < called_at["resolve_runtime"]


@pytest.mark.parametrize(
    "module,entry",
    [
        ("b52_competition_training", "train_b52"),
        ("b53_augmented_training", "train_b53"),
    ],
)
def test_the_trainer_defaults_to_none(module, entry):
    """So a run that does not ask for workers is unchanged."""
    import importlib

    trainer = importlib.import_module(f"rsna_knee.{module}")
    parameters = inspect.signature(getattr(trainer, entry)).parameters

    assert parameters["num_workers"].default is None


@pytest.mark.parametrize(
    "module", ["b52_competition_training", "b53_augmented_training"]
)
def test_the_trainer_records_what_it_used(module):
    """A boolean nobody measured is what let B52 train on identical pixels."""
    import importlib

    source = inspect.getsource(importlib.import_module(f"rsna_knee.{module}"))
    assert '"loader": loader_state' in source
