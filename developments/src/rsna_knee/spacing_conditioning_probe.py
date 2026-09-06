"""Did the spacing term learn nothing, or learn something that cannot matter?

The Expert-58 ablation returned `+0.000152`. Two very different failures produce
that same number and the ablation cannot tell them apart:

```text
the model genuinely does not benefit from knowing the slice spacing
the term never grew large enough to influence anything
```

The backstop during training only proved the weight left exactly zero. Leaving
zero and reaching a magnitude capable of moving a transformer are not the same
claim. This reads the trained weight out of the checkpoint and measures which
happened. No GPU, no images, no forward pass.

## The number that matters is the spread, not the size

A term that adds the *same* vector to every series is a bias. The study
hierarchy can absorb a constant into `target_bias` and nothing downstream can
tell it apart from a shift in the mean — so a large constant contribution would
still do nothing at all.

What can carry information is how far the contribution **moves** as the spacing
changes:

```text
||c(0.8 mm) - c(5.0 mm)||     the corpus p05 to p95, the real range
```

That is the quantity compared against `plane + fluid + fat`, which is the sum it
was added to and whose scale the trained model demonstrably does respond to.

## How to read it

```text
spread / metadata norm        what it means
> 0.10                        a real term; the null is about the idea
0.01 to 0.10                  small but present; arguably underpowered
< 0.01                        the term is a rounding error on its own sum
```

Those bands are stated here, before the number is read, and they are a reading
aid rather than a threshold anything hangs on. The honest use of this module is
to distinguish "tested and refuted" from "not really tested", not to license a
second attempt at a higher learning rate — which would be fitting to the
endpoint and is refused elsewhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .spacing_conditioning import SPACING_BASIS, spacing_basis

PROBE_VERSION = "spacing_conditioning_probe_v1"

CONDITIONING_KEY = "spacing_conditioning.projection.weight"
METADATA_KEYS = ("plane_embedding.weight", "fluid_embedding.weight", "fat_embedding.weight")

#: The corpus range, from `slice_geometry_scan`: p05 0.80 mm, p50 3.30, p95 5.00.
CORPUS_SPACINGS_MM = (0.80, 1.50, 2.28, 3.30, 3.83, 5.00, 8.33)
CORPUS_P05_MM = 0.80
CORPUS_P95_MM = 5.00

#: Reading aids, declared before the number is read.
SPREAD_REAL = 0.10
SPREAD_PRESENT = 0.01


def load_conditioning(checkpoint: str | Path) -> tuple[torch.Tensor, dict, float]:
    """The trained projection weight, the base state, and the scale applied.

    The scale is not in the state dict -- it describes the host model rather
    than the weights -- so it comes from the run's audit. A checkpoint written
    before the scale existed gets 1.0, which is what it was trained with.
    """
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    base_state = payload.get("base_state")
    if not isinstance(base_state, dict):
        raise ValueError(f"{checkpoint} has no base_state to read")
    if CONDITIONING_KEY not in base_state:
        raise ValueError(
            f"{checkpoint} carries no {CONDITIONING_KEY}; it was trained without "
            "the spacing conditioning and there is nothing to probe"
        )
    weight = base_state[CONDITIONING_KEY].detach().float()
    if weight.ndim != 2 or int(weight.shape[1]) != SPACING_BASIS:
        raise ValueError(
            f"unexpected conditioning shape {tuple(weight.shape)}; expected "
            f"(d_model, {SPACING_BASIS})"
        )
    scale = float(payload.get("spacing", {}).get("conditioning_scale", 1.0))
    return weight, base_state, scale


def contributions(weight: torch.Tensor, spacings) -> torch.Tensor:
    """The vector the conditioning adds, for each spacing in millimetres."""
    basis = spacing_basis(torch.as_tensor(list(spacings), dtype=torch.float32))
    return basis @ weight.T


def metadata_scale(base_state: dict) -> dict:
    """How big the `plane + fluid + fat` sum typically is.

    Measured as the mean row norm of each embedding, summed. It is the scale the
    trained model demonstrably responds to, which makes it the right yardstick
    for a term added to the very same sum.
    """
    parts = {}
    for key in METADATA_KEYS:
        if key not in base_state:
            raise ValueError(f"base_state is missing {key}")
        rows = base_state[key].detach().float()
        # Row 0 is `padding_idx`, permanently zero, so it would drag the mean.
        live = rows[1:] if rows.shape[0] > 1 else rows
        parts[key] = float(live.norm(dim=-1).mean())
    total = float(sum(parts.values()))
    return {"per_embedding": parts, "typical_metadata_norm": total}


def probe(
    checkpoint: str | Path,
    *,
    spacings=CORPUS_SPACINGS_MM,
    out_json: str | Path | None = None,
) -> dict:
    """Measure whether the learned term is big enough to matter."""
    weight, base_state, applied_scale = load_conditioning(checkpoint)
    weight = weight * applied_scale
    scale = metadata_scale(base_state)
    typical = scale["typical_metadata_norm"]

    vectors = contributions(weight, spacings)
    norms = vectors.norm(dim=-1)

    thin, thick = contributions(weight, (CORPUS_P05_MM, CORPUS_P95_MM))
    spread = float((thin - thick).norm())

    return _finish(
        {
            "version": PROBE_VERSION,
            "checkpoint": str(Path(checkpoint).resolve()),
            "conditioning_scale": applied_scale,
            "weight_norm_raw": float(weight.norm() / applied_scale),
            "weight_norm": float(weight.norm()),
            "weight_is_exactly_zero": bool(torch.all(weight == 0)),
            "weight_max_abs": float(weight.abs().max()),
            "d_model": int(weight.shape[0]),
            "contribution_norm_by_spacing_mm": {
                f"{mm:g}": float(value) for mm, value in zip(spacings, norms)
            },
            "spread_p05_to_p95": spread,
            "typical_metadata_norm": typical,
            "spread_over_metadata": float(spread / typical) if typical else float("nan"),
            "mean_contribution_norm": float(norms.mean()),
            "mean_over_metadata": (
                float(float(norms.mean()) / typical) if typical else float("nan")
            ),
            **scale,
        },
        out_json,
    )


def _verdict(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unmeasurable"
    if ratio >= SPREAD_REAL:
        return "a real term: the null is about the idea, not the size"
    if ratio >= SPREAD_PRESENT:
        return "small but present: arguably underpowered rather than refuted"
    return "a rounding error on the sum it joins: this was not really tested"


def _finish(result: dict, out_json: str | Path | None) -> dict:
    result["verdict"] = _verdict(result["spread_over_metadata"])
    result["bands"] = {"real": SPREAD_REAL, "present": SPREAD_PRESENT}
    if out_json is not None:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _report(result: dict) -> None:
    print()
    print(f"  conditioning weight norm      {result['weight_norm']:.6f}")
    print(f"  largest single value          {result['weight_max_abs']:.6f}")
    print(f"  still exactly zero            {result['weight_is_exactly_zero']}")
    print()
    print("  what it adds, by slice spacing")
    for mm, value in result["contribution_norm_by_spacing_mm"].items():
        print(f"    {mm:>6} mm                 {value:.6f}")
    print()
    print("  the number that matters is the spread, not the size:")
    print("  a constant added to every series is a bias the hierarchy absorbs.")
    print()
    print(
        f"    spread, {CORPUS_P05_MM} to {CORPUS_P95_MM} mm       "
        f"{result['spread_p05_to_p95']:.6f}"
    )
    print(f"    typical plane+fluid+fat       {result['typical_metadata_norm']:.6f}")
    print(f"    ratio                         {result['spread_over_metadata']:.4f}")
    print()
    print(f"  {result['verdict']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        "Measure whether the trained spacing term is big enough to matter"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    _report(probe(args.checkpoint, out_json=args.out_json))


if __name__ == "__main__":
    main()
