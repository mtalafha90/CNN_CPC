"""Make the preserved development implementation importable from the clean interface."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def developments_source() -> Path:
    return repository_root() / "developments" / "src"


def ensure_developments_source() -> Path:
    """Add the preserved `developments/src` implementation to `sys.path`.

    The current model is intentionally exposed through a small clean interface,
    while the full experiment lineage remains preserved verbatim under
    `developments/` for reproducibility.
    """
    source = developments_source()
    if source.is_dir():
        value = str(source)
        if value not in sys.path:
            sys.path.insert(0, value)
        return source

    # A non-editable `pip install .` copies only the clean interface packages
    # into site-packages, so the sibling `developments/` tree is not there.
    # That is still fine if the implementation is importable some other way;
    # only fail when it genuinely cannot be found.
    if importlib.util.find_spec("rsna_knee") is not None:
        return source

    raise RuntimeError(
        f"preserved implementation not found at {source} and 'rsna_knee' is not "
        "importable. Run from a source checkout, or install in editable mode "
        "with `pip install -e .` so the developments/ tree stays in place."
    )
