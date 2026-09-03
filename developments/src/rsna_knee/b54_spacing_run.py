"""B54: carry the slice spacing all the way from the DICOM to the metadata sum.

`spacing_conditioning` supplies the component. This supplies the plumbing —
getting a number measured on disk into the one line of the model that adds up
what it knows about a series:

```python
metadata = plane + fluid + fat          # what it is
metadata = plane + fluid + fat + spacing  # and how much knee it holds
```

## Why this keys off the embeddings rather than a class name

That sum appears in at least six places in the lineage, in two roles. The base
holds one (`b12_variable_series`, added to slice features) and the sparse MIL
head holds another (`b36_sparse_mil`, added to tokens, zero-initialised).
`b35`, `b37`, `b38` and `b45` each reach into one of those.

Naming a class to subclass would pick one of them and silently miss the rest.
So `spacing_metadata` takes any module exposing `plane_embedding`,
`fluid_embedding` and `fat_embedding` — which all six do, with identical index
semantics — and reproduces their sum exactly, adding spacing only where the
conditioning has been installed. With nothing installed it is the frozen
expression, and a test asserts that equality rather than assuming it.

## Where the number comes from

Training reads `series_geometry.csv`, written once by `slice_geometry_scan`.
Any series the table misses falls through to three DICOM headers, which is
also what a submission set does, having no table. Both paths go through
`spacing_conditioning.resolve_spacing`.

A series whose spacing cannot be measured, and every padded slot in a batch,
carries `nan` and contributes exactly zero. It is never given a stand-in value.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .b12_variable_series import VariableSeriesKneeDataset, collate_variable_series
from .spacing_conditioning import (
    CONDITIONING_VERSION,
    MISSING_SPACING,
    SpacingConditioning,
    resolve_spacing,
    spacing_lookup,
    triplet_depth_mm,
)

B54_VERSION = "b54_spacing_conditioned_native_v1"
B54_RUN_ROOT = "runs/084_Experiment_B54_spacing_conditioned"

SPACING_KEY = "slice_spacing_mm"
EMBEDDING_NAMES = ("plane_embedding", "fluid_embedding", "fat_embedding")


# --- getting the number onto the series records -------------------------------


def attach_spacing(
    records: dict[str, list[dict]],
    *,
    series_geometry_csv: str | Path | None = None,
    data_root: str | Path | None = None,
    split: str = "train",
    read_headers_when_missing: bool = True,
) -> dict:
    """Add `slice_spacing_mm` to every series record, in place, and report.

    The report is the thing to read before training: if the table missed a
    third of the corpus, the conditioning is silent on a third of the corpus
    and the run is not testing what it claims to.
    """
    from .dicom import find_series_dir

    lookup = (
        spacing_lookup(series_geometry_csv) if series_geometry_csv is not None else None
    )
    stats = {
        "series": 0,
        "from_table": 0,
        "from_headers": 0,
        "unresolved": 0,
    }
    for study, entries in records.items():
        for record in entries:
            stats["series"] += 1
            series_uid = str(record["series_uid"])
            value = MISSING_SPACING
            if lookup is not None:
                value = resolve_spacing(study, series_uid, lookup=lookup)
                if np.isfinite(value):
                    stats["from_table"] += 1
            if not np.isfinite(value) and read_headers_when_missing and data_root:
                directory = find_series_dir(data_root, split, study, series_uid)
                if directory is not None:
                    value = resolve_spacing(study, series_uid, series_dir=directory)
                    if np.isfinite(value):
                        stats["from_headers"] += 1
            if not np.isfinite(value):
                stats["unresolved"] += 1
            record[SPACING_KEY] = float(value)

    stats["resolved_fraction"] = float(
        (stats["series"] - stats["unresolved"]) / max(stats["series"], 1)
    )
    return stats


def spacing_summary(records: dict[str, list[dict]], *, gap: int = 1) -> dict:
    """What the run will actually see, so it can be checked against the scan."""
    values = np.asarray(
        [
            record.get(SPACING_KEY, MISSING_SPACING)
            for entries in records.values()
            for record in entries
        ],
        dtype=float,
    )
    known = values[np.isfinite(values) & (values > 0)]
    if not known.size:
        return {"series": int(values.size), "with_a_spacing": 0}
    depths = triplet_depth_mm(known, gap=gap)
    quantiles = [5, 25, 50, 75, 95]
    return {
        "series": int(values.size),
        "with_a_spacing": int(known.size),
        "triplet_gap": int(gap),
        "slice_spacing_mm": {
            f"p{q}": float(np.percentile(known, q)) for q in quantiles
        },
        "triplet_depth_mm": {
            f"p{q}": float(np.percentile(depths, q)) for q in quantiles
        },
        "triplet_depth_ratio_p95_p05": float(
            np.percentile(depths, 95) / max(np.percentile(depths, 5), 1e-9)
        ),
    }


# --- the dataset --------------------------------------------------------------


class B54SpacingDataset(VariableSeriesKneeDataset):
    """`VariableSeriesKneeDataset` plus one tensor: the spacing of each series.

    Everything else in the item is untouched, so a model that ignores
    `series_spacing` behaves exactly as it did before.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        records = self.series_records[self.study_uids[idx]]
        item["series_spacing"] = torch.tensor(
            [float(record.get(SPACING_KEY, MISSING_SPACING)) for record in records],
            dtype=torch.float32,
        )
        return item


def collate_b54(batch: list[dict]) -> dict:
    """`collate_variable_series`, padding the spacing with `nan`.

    `nan` rather than zero so a padded slot is unmistakably "no series" rather
    than "a series of zero thickness". Both contribute nothing, but only one of
    them says so.
    """
    out = collate_variable_series(batch)
    if not batch or "series_spacing" not in batch[0]:
        return out
    max_k = max(int(item["series_spacing"].shape[0]) for item in batch)
    spacing = torch.full((len(batch), max_k), math.nan, dtype=torch.float32)
    for index, item in enumerate(batch):
        values = item["series_spacing"]
        spacing[index, : values.shape[0]] = values
    out["series_spacing"] = spacing
    return out


# --- the model side -----------------------------------------------------------


def install_spacing_conditioning(
    module: torch.nn.Module, *, enabled: bool = True
) -> SpacingConditioning:
    """Attach the conditioning to any module that already sums plane/fluid/fat.

    The width is taken from `plane_embedding`, so it cannot disagree with the
    features it will be added to.
    """
    missing = [name for name in EMBEDDING_NAMES if not hasattr(module, name)]
    if missing:
        raise ValueError(
            f"{type(module).__name__} has no {', '.join(missing)}; it is not one "
            "of the modules that sums series metadata, so spacing has nothing "
            "to be added to"
        )
    conditioning = SpacingConditioning(
        int(module.plane_embedding.embedding_dim), enabled=enabled
    )
    module.spacing_conditioning = conditioning
    return conditioning


def spacing_metadata(
    module: torch.nn.Module,
    series_meta: torch.Tensor,
    series_spacing: torch.Tensor | None = None,
) -> torch.Tensor:
    """`plane + fluid + fat`, plus spacing where the conditioning is installed.

    With no conditioning installed, or no spacing supplied, this is exactly the
    frozen expression. That equality is a test, not a claim.
    """
    plane = module.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
    fluid = module.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
    fat = module.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
    metadata = plane + fluid + fat

    conditioning = getattr(module, "spacing_conditioning", None)
    if conditioning is None or series_spacing is None:
        return metadata
    return metadata + conditioning(series_spacing.to(metadata.device))


def set_spacing_enabled(model: torch.nn.Module, enabled: bool) -> int:
    """Switch every installed conditioning on or off; returns how many moved.

    This is the free ablation: evaluate one trained checkpoint twice, once with
    `True` and once with `False`, and the difference is the spacing effect.
    """
    moved = 0
    for child in model.modules():
        if isinstance(child, SpacingConditioning):
            child.set_enabled(enabled)
            moved += 1
    return moved


def b54_state(model: torch.nn.Module | None = None) -> dict:
    """The audit payload: what this run declares about itself."""
    state = {
        "experiment": "B54",
        "version": B54_VERSION,
        "conditioning_version": CONDITIONING_VERSION,
        "conditioning_is_zero_initialised": True,
        "ablation": "set_spacing_enabled(model, False) on the trained checkpoint",
        "spacing_sources": ["series_geometry.csv", "dicom_headers"],
        "unresolved_spacing_contributes": 0.0,
    }
    if model is not None:
        conditionings = [
            child for child in model.modules() if isinstance(child, SpacingConditioning)
        ]
        state["conditioning_sites"] = len(conditionings)
        state["conditioning_enabled"] = [bool(c.enabled) for c in conditionings]
        state["conditioning_is_still_zero"] = [
            bool(torch.all(c.projection.weight == 0)) for c in conditionings
        ]
    return state


def preflight(
    records: dict[str, list[dict]],
    *,
    model: torch.nn.Module | None = None,
    minimum_resolved_fraction: float = 0.95,
) -> dict:
    """Refuse to start a long run that is not testing what it says it is.

    Two ways this run can be silently pointless: the spacing failed to resolve
    for most series, or the conditioning was never installed. Both are cheap to
    check now and expensive to discover afterwards.
    """
    summary = spacing_summary(records)
    total = max(int(summary["series"]), 1)
    resolved = float(summary.get("with_a_spacing", 0)) / total
    problems: list[str] = []
    if resolved < float(minimum_resolved_fraction):
        problems.append(
            f"only {resolved:.1%} of series have a measurable spacing, below the "
            f"{minimum_resolved_fraction:.0%} this run requires"
        )
    if model is not None:
        sites = sum(
            1 for child in model.modules() if isinstance(child, SpacingConditioning)
        )
        if sites == 0:
            problems.append(
                "no SpacingConditioning is installed, so the run would be a "
                "plain re-run under a new name"
            )
    return {
        "passed": not problems,
        "problems": problems,
        "resolved_fraction": resolved,
        "spacing_summary": summary,
        **b54_state(model),
    }
