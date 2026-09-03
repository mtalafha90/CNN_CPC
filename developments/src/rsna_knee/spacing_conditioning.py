"""Tell the model how thick a series is, which it has never been told.

`b12_1_hierarchical` gives every series three facts, as embeddings added to its
slice features: `plane_embedding`, `fluid_embedding`, `fat_embedding`. Plane,
fluid sensitivity, fat suppression — **what kind** of sequence it is.

B35 adds a fourth, `_position_basis`: where a slice sits inside its own stack,
as a fraction from 0 to 1. That is a *relative* coordinate. It says "40% of the
way through" and never says how far through that is in millimetres.

Nothing anywhere says how much knee an input holds. Measured across the corpus
(`slice_geometry_scan`, 24,371 series) that is not a small omission:

```text
slice spacing         p05 0.80 mm   p50 3.30 mm   max 8.33 mm
2.5D triplet depth    p05 1.59 mm   p50 6.60 mm   max 16.66 mm
```

`b7_triplet_gap` counts slices, not millimetres, so the depth of the three
channels is `2 x gap x spacing`. In 69.1% of studies the thickest triplet is at
least twice the thinnest, and the model fuses them into one prediction with no
way to tell which is which.

## Why this is not B10 again

`b12_use_physical_scale` is `false`, and B10 is the reason. But B10's own first
line calls itself "label-free **in-plane** physical-scale normalization":
PixelSpacing and field of view, left to right. Through-plane geometry has never
been normalised, and has never been offered to the model either.

## The shape of the fix

Deliberately the smallest thing that addresses the measurement:

```text
continuous          a basis function of the log spacing, not a lookup table
                    over bins whose edges nobody can justify
zero-initialised    at step 0 the model is numerically identical to the
                    baseline, so this cannot be blamed for a bad start
switchable          `set_enabled(False)` zeroes the contribution, so ONE
                    trained checkpoint yields its own ablation for free
```

Both properties are the house pattern, not an invention: B35 zero-initialises
every acquisition and position embedding for exactly this reason, and B47
replaced a positional lookup table with a continuous basis for exactly the
other one.

## Where the spacing comes from

Training reads it from the `series_geometry.csv` that `slice_geometry_scan`
already wrote — no DICOM headers opened. Submission cannot, since there is no
precomputed table for the test set, so it re-reads three headers per series.
Both paths go through `resolve_spacing`, so they cannot drift apart. A series
with no measurable spacing contributes exactly nothing, like a padded one.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CONDITIONING_VERSION = "spacing_conditioning_v1"

# Mirrors the width of B35's `_position_basis`, and its feature order.
SPACING_BASIS = 8

# The log-spacing window mapped onto [0, 1]. It brackets the measured corpus
# (0.80 mm to 8.33 mm) with room on both sides, so no real series is clipped.
SPACING_MIN_MM = 0.4
SPACING_MAX_MM = 12.8


def normalised_log_spacing(spacing_mm: torch.Tensor) -> torch.Tensor:
    """Map a slice spacing in mm onto [0, 1], logarithmically.

    Logarithmically because the difference that matters between 0.6 mm and
    1.2 mm is the same kind of difference as between 3 mm and 6 mm: a doubling
    of how much anatomy one channel step covers. A linear scale would spend
    almost all of its range on the thick half and crush the thin volumes into a
    corner.

    Non-finite or non-positive values map to 0 and should be masked out by the
    caller; they are not a spacing.
    """
    spacing = spacing_mm.float()
    valid = torch.isfinite(spacing) & (spacing > 0)
    safe = torch.where(valid, spacing, torch.full_like(spacing, SPACING_MIN_MM))
    span = math.log2(SPACING_MAX_MM / SPACING_MIN_MM)
    z = torch.log2(safe / SPACING_MIN_MM) / span
    return torch.where(valid, z.clamp(0.0, 1.0), torch.zeros_like(z))


def spacing_basis(spacing_mm: torch.Tensor) -> torch.Tensor:
    """Deterministic 8-D continuous description of a series' slice spacing.

    The same feature order as B35's through-plane position basis: the raw
    coordinate, its square, and three sine/cosine octaves.
    """
    z = normalised_log_spacing(spacing_mm)
    return torch.stack(
        [
            z,
            z.square(),
            torch.sin(math.pi * z),
            torch.cos(math.pi * z),
            torch.sin(2.0 * math.pi * z),
            torch.cos(2.0 * math.pi * z),
            torch.sin(4.0 * math.pi * z),
            torch.cos(4.0 * math.pi * z),
        ],
        dim=-1,
    )


def triplet_depth_mm(spacing_mm, gap: int = 1) -> np.ndarray:
    """What the three channels physically span. Stated once, used everywhere."""
    return 2.0 * int(gap) * np.asarray(spacing_mm, dtype=float)


class SpacingConditioning(nn.Module):
    """A zero-initialised projection of the spacing basis into feature space.

    Add its output to series features exactly where `plane + fluid + fat`
    already is. At initialisation it contributes zero, so a run that switches
    this on starts from the same numbers as one that does not.
    """

    def __init__(self, d_model: int, *, enabled: bool = True):
        super().__init__()
        if int(d_model) < 1:
            raise ValueError("d_model must be positive")
        self.d_model = int(d_model)
        self.projection = nn.Linear(SPACING_BASIS, self.d_model, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.enabled = bool(enabled)

    def set_enabled(self, enabled: bool) -> "SpacingConditioning":
        """Switch the contribution off to ablate a trained checkpoint."""
        self.enabled = bool(enabled)
        return self

    def forward(self, spacing_mm: torch.Tensor) -> torch.Tensor:
        """`spacing_mm` is [...]; the result is [..., d_model]."""
        basis = spacing_basis(spacing_mm)
        if not self.enabled:
            return torch.zeros(
                (*spacing_mm.shape, self.d_model),
                device=spacing_mm.device,
                dtype=self.projection.weight.dtype,
            )
        usable = (torch.isfinite(spacing_mm) & (spacing_mm > 0)).to(
            self.projection.weight.dtype
        )
        projected = self.projection(basis.to(self.projection.weight.dtype))
        return projected * usable.unsqueeze(-1)


# --- where the number comes from ---------------------------------------------

MISSING_SPACING = float("nan")


def spacing_lookup(series_geometry_csv: str | Path) -> dict[tuple[str, str], float]:
    """Study/series to slice spacing, from the table the scan already wrote."""
    frame = pd.read_csv(series_geometry_csv)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "slice_spacing_mm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{series_geometry_csv} is not a slice_geometry_scan table; missing "
            f"{', '.join(sorted(missing))}"
        )
    values = pd.to_numeric(frame["slice_spacing_mm"], errors="coerce")
    return {
        (str(study), str(series)): float(value)
        for study, series, value in zip(
            frame["StudyInstanceUID"], frame["SeriesInstanceUID"], values
        )
    }


def spacing_from_headers(series_dir: str | Path) -> float:
    """Read the spacing straight from a series, for a set with no table.

    Three headers, no pixel data. Trivial next to decoding every frame, which
    inference does anyway.
    """
    from .slice_geometry_scan import geometry_from_headers

    value = geometry_from_headers(series_dir).get("slice_spacing_mm", MISSING_SPACING)
    return float(value) if value is not None else MISSING_SPACING


def resolve_spacing(
    study_uid: str,
    series_uid: str,
    *,
    lookup: dict[tuple[str, str], float] | None = None,
    series_dir: str | Path | None = None,
) -> float:
    """The one way spacing is obtained, so training and submission agree.

    The precomputed table wins when it holds a usable value. Otherwise the
    headers are read, if a directory was given. Otherwise the series has no
    spacing and the conditioning contributes nothing for it.
    """
    if lookup is not None:
        value = lookup.get((str(study_uid), str(series_uid)), MISSING_SPACING)
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    if series_dir is not None:
        value = spacing_from_headers(series_dir)
        if np.isfinite(value) and value > 0:
            return float(value)
    return MISSING_SPACING
