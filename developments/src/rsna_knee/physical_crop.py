"""Crop a fixed number of millimetres, and point every knee the same way.

## Why the fractional crop is wrong

B37 through B54 crop 90% of whatever the native image happens to be. The
acquired field of view on this corpus has a **median of 160 mm and a range of
70 to 320 mm**, so "90% of native" means 63 mm of knee on one scanner and
288 mm on another. The model is shown a different amount of anatomy per site
and has no way to know which.

A fixed physical crop shows every study the same anatomy. Competitors measured
this bundle — physical-mm crop, laterality normalisation, per-series
normalisation — at **+0.030 on the leaderboard**, the largest single controlled
gain published for this competition.

`CROP_MM = 130` is their figure, not one derived here. It is chosen so the crop
is smaller than the acquired FOV for most series: a 160 mm crop would exceed the
image in roughly 60% of them and quietly do nothing at all, which is the same
failure mode as the fractional crop wearing different clothes.

## Laterality, and the mistake worth avoiding

Left and right knees are mirror images, so a model must either see both
orientations of every finding or be shown them in one canonical orientation.
The `Laterality` tag is missing on roughly half of studies and its missingness
varies by manufacturer, so it has to be derivable from geometry as well.

**The naive fix — mirror every right knee horizontally — is wrong for sagittal
series, which are the largest plane in this corpus.** A sagittal image's
in-plane axes are anterior-posterior and superior-inferior; there is no
left-right axis in the picture at all. Mirroring one would flip anterior and
posterior, teaching the model that the patella is at the back. What differs
between a left and a right sagittal acquisition is **which end of the slice
stack is medial**, so the correct normalisation there is to reverse the slice
order.

```text
coronal, axial    left-right lies in the image  ->  mirror the width axis
sagittal          left-right lies across slices ->  reverse the slice order
unknown plane     do nothing, and count it
```

Getting this wrong would be worse than not normalising, because it would be
wrong consistently.
"""
from __future__ import annotations

import numpy as np

PHYSICAL_CROP_VERSION = "physical_crop_v1"

#: Millimetres of knee to retain. The competitors' measured value.
CROP_MM = 130.0

#: The orientation everything is turned into. Either works; it must not change
#: between training and inference, so it is a constant rather than an argument.
CANONICAL_SIDE = "L"

#: Plausible in-plane spacing. Outside this the header is not believable and the
#: physical crop is refused rather than applied to a wrong number.
PLAUSIBLE_SPACING_MM = (0.05, 3.0)

SAGITTAL = "sagittal"
CORONAL = "coronal"
AXIAL = "axial"


def crop_pixels(pixel_spacing_mm: float, crop_mm: float = CROP_MM) -> int | None:
    """How many pixels span `crop_mm`, or None when the spacing is unusable."""
    spacing = float(pixel_spacing_mm)
    low, high = PLAUSIBLE_SPACING_MM
    if not np.isfinite(spacing) or not low <= spacing <= high:
        return None
    return max(2, int(round(float(crop_mm) / spacing)))


def crop_to_millimetres(
    triplets: np.ndarray,
    pixel_spacing_mm: float,
    *,
    crop_mm: float = CROP_MM,
    fallback_fraction: float = 0.90,
) -> tuple[np.ndarray, dict]:
    """Centre-crop `[S, C, H, W]` to a fixed physical size.

    Returns the crop and an audit of what actually happened, because three
    different things can, and a run that cannot tell them apart cannot say what
    it trained on:

    ```text
    physical      the spacing was usable and the crop fitted
    clipped       the crop was larger than the image, so less was removed
    fallback      no usable spacing; the old fractional crop was applied
    ```
    """
    x = np.asarray(triplets)
    if x.ndim != 4:
        raise ValueError(f"expected [S, C, H, W], got {x.shape}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if h < 2 or w < 2:
        raise ValueError("native spatial dimensions must be >= 2")

    wanted = crop_pixels(pixel_spacing_mm, crop_mm)
    if wanted is None:
        crop_h = max(2, min(h, int(round(h * float(fallback_fraction)))))
        crop_w = max(2, min(w, int(round(w * float(fallback_fraction)))))
        mode = "fallback"
    else:
        crop_h, crop_w = min(h, wanted), min(w, wanted)
        mode = "clipped" if (wanted > h or wanted > w) else "physical"

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = x[..., top : top + crop_h, left : left + crop_w]

    return cropped, {
        "mode": mode,
        "native_hw": [h, w],
        "cropped_hw": [int(crop_h), int(crop_w)],
        "wanted_pixels": wanted,
        "pixel_spacing_mm": float(pixel_spacing_mm),
        "retained_mm": (
            [float(crop_h * pixel_spacing_mm), float(crop_w * pixel_spacing_mm)]
            if wanted is not None
            else None
        ),
    }


def laterality_from_positions(image_positions) -> str | None:
    """Which knee, from the patient coordinate system.

    DICOM's patient frame puts **+x toward the patient's left**, so a left knee
    sits at positive x and a right knee at negative x. The median across slices
    is used rather than one slice, so a sagittal stack that crosses the midline
    is decided by where most of it lies.

    Returns None when the positions are missing or straddle zero closely enough
    that the sign is not informative -- guessing there would be worse than
    leaving the study alone.
    """
    if image_positions is None:
        return None
    positions = np.asarray(image_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[-1] != 3 or not len(positions):
        return None
    x = positions[:, 0]
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    median = float(np.median(x))
    # A knee sits well off the midline. Near zero the sign means nothing.
    if abs(median) < 10.0:
        return None
    return "L" if median > 0 else "R"


def resolve_laterality(tag_value=None, image_positions=None) -> dict:
    """Prefer the header's own answer, fall back to geometry, record which.

    Both are reported when both exist, so a run can measure how often they
    agree instead of assuming the derivation is sound.
    """
    tag = None
    if tag_value is not None:
        text = str(tag_value).strip().upper()
        if text.startswith("L"):
            tag = "L"
        elif text.startswith("R"):
            tag = "R"

    derived = laterality_from_positions(image_positions)
    resolved = tag or derived
    return {
        "laterality": resolved,
        "source": "tag" if tag else ("positions" if derived else None),
        "tag": tag,
        "derived": derived,
        "agree": None if (tag is None or derived is None) else bool(tag == derived),
    }


def normalise_laterality(
    triplets: np.ndarray,
    laterality: str | None,
    plane: str | None,
    *,
    canonical: str = CANONICAL_SIDE,
) -> tuple[np.ndarray, dict]:
    """Turn a knee into the canonical orientation, correctly for its plane.

    `[S, C, H, W]`, where S indexes slices through the stack. See the module
    docstring for why sagittal reverses the slice order rather than mirroring
    the image: a sagittal picture has no left-right axis to mirror, and doing
    it anyway would swap anterior and posterior on every such series.
    """
    x = np.asarray(triplets)
    if x.ndim != 4:
        raise ValueError(f"expected [S, C, H, W], got {x.shape}")

    if laterality is None:
        return x, {"applied": "none", "reason": "laterality unknown"}
    if str(laterality).upper() == str(canonical).upper():
        return x, {"applied": "none", "reason": "already canonical"}

    kind = str(plane or "").strip().lower()
    if kind == SAGITTAL:
        # Medial-lateral runs across the stack, not across the picture.
        return np.ascontiguousarray(x[::-1]), {"applied": "slice_order_reversed"}
    if kind in (CORONAL, AXIAL):
        return np.ascontiguousarray(x[..., ::-1]), {"applied": "width_mirrored"}
    return x, {"applied": "none", "reason": f"unknown plane {plane!r}"}


def summarise(records) -> dict:
    """Fold per-series audits into something a run can print and store."""
    rows = list(records)
    modes: dict[str, int] = {}
    applied: dict[str, int] = {}
    agreements = [r["agree"] for r in rows if r.get("agree") is not None]
    for row in rows:
        modes[row.get("mode", "unknown")] = modes.get(row.get("mode", "unknown"), 0) + 1
        key = row.get("applied", "none")
        applied[key] = applied.get(key, 0) + 1
    return {
        "version": PHYSICAL_CROP_VERSION,
        "series": len(rows),
        "crop_mm": CROP_MM,
        "canonical_side": CANONICAL_SIDE,
        "crop_mode": dict(sorted(modes.items())),
        "laterality_applied": dict(sorted(applied.items())),
        "laterality_tag_vs_derived_agreement": (
            float(np.mean(agreements)) if agreements else None
        ),
        "laterality_comparable_series": len(agreements),
    }


__all__ = [
    "CANONICAL_SIDE",
    "CROP_MM",
    "PHYSICAL_CROP_VERSION",
    "crop_pixels",
    "crop_to_millimetres",
    "laterality_from_positions",
    "normalise_laterality",
    "resolve_laterality",
    "summarise",
]
