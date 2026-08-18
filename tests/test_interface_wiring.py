"""The working-model interface must stay wired to the preserved implementation.

A facade that cannot import is worse than no facade: it looks clean and fails
only when someone tries to run it.
"""

from __future__ import annotations

import importlib

import pytest

from model._implementation import WORKING_ARCHITECTURE, network_spec, read_config
from model.architecture import TARGETS
from model.bootstrap import (
    developments_source,
    ensure_developments_source,
    repository_root,
)

COMMANDS = (
    "model.architecture",
    "training.train",
    "validation.validate",
    "testing.test",
)
LIBRARIES = (
    "model.preprocessing",
    "data.dataset",
)
ENTRY_POINTS = COMMANDS + LIBRARIES


def test_the_bootstrap_locates_the_preserved_implementation():
    assert developments_source() == repository_root() / "developments" / "src"
    assert ensure_developments_source().is_dir()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_entry_point_imports(name):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", COMMANDS)
def test_every_command_exposes_main(name):
    module = importlib.import_module(name)
    assert callable(getattr(module, "main", None))


def test_the_interface_agrees_with_the_implementation_on_the_targets():
    ensure_developments_source()
    from rsna_knee.constants import TARGETS as IMPLEMENTATION_TARGETS

    assert list(TARGETS) == list(IMPLEMENTATION_TARGETS)


def test_the_configured_model_is_the_declared_working_architecture():
    spec = network_spec(read_config("config/current_model.yaml"), normalize_input=True)
    assert spec["architecture"] == WORKING_ARCHITECTURE


def test_a_missing_implementation_fails_with_an_actionable_message(monkeypatch, tmp_path):
    import model.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "developments_source", lambda: tmp_path / "absent")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pip install -e"):
        bootstrap.ensure_developments_source()
