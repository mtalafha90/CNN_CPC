"""Make the preserved development implementation importable from the clean interface."""

from __future__ import annotations

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
    if not source.is_dir():
        raise RuntimeError(f"preserved implementation is missing: {source}")
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)
    return source
