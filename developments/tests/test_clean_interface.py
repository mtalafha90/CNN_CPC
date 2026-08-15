"""The clean top-level interface must stay wired to the preserved implementation."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.architecture import CURRENT_MODEL, TARGETS  # noqa: E402
from model.bootstrap import developments_source, ensure_developments_source, repository_root  # noqa: E402


def test_the_bootstrap_locates_the_preserved_implementation():
    assert repository_root() == ROOT
    assert developments_source() == ROOT / "developments" / "src"
    assert ensure_developments_source().is_dir()


def test_every_clean_entry_point_imports():
    # A facade that cannot import is worse than no facade: it looks clean and
    # fails only when someone tries to run it.
    for name in ("training.train", "validation.validate", "testing.test", "data.dataset"):
        assert importlib.import_module(name) is not None


def test_the_clean_layer_agrees_with_the_implementation_on_the_targets():
    ensure_developments_source()
    from rsna_knee.constants import TARGETS as IMPL_TARGETS

    assert list(TARGETS) == list(IMPL_TARGETS)


def test_the_declared_model_is_still_b20():
    # The restructure was organizational; it must not have promoted anything.
    assert CURRENT_MODEL["name"] == "B20_crop_only_joint_focus"
    assert CURRENT_MODEL["canonical_epoch"] == 2
    assert CURRENT_MODEL["status"] == "ACTIVE WORKING MODEL"


def test_a_missing_implementation_fails_with_an_actionable_message(monkeypatch, tmp_path):
    import model.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "developments_source", lambda: tmp_path / "absent")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pip install -e"):
        bootstrap.ensure_developments_source()
