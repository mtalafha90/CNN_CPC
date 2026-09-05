"""B54's model: B42, plus the one thing it was never told.

`b54_spacing_run` gets the measured slice spacing as far as the batch.
This is the last step — into the sum the study hierarchy computes over each
series:

```python
metadata = plane + fluid + fat          # what kind of sequence it is
                          + spacing     # and how much knee it holds
```

## It copies almost nothing

The obvious way to add a term to that sum is to override
`_base_logits_from_global` and reproduce its forty lines with one changed.
That duplicates frozen logic — the pooling, the padding masks, the empty-study
guard, the cross-attention — and any later correction to B37 would then stop
reaching B54.

It is not necessary. The parent computes

```python
x = global_feature[:, :, :B35_BASE_SLICES]
x = (x + base.slice_position[None, None, :, :] + metadata[:, :, None, :]) * mask
```

so a per-series vector added to `global_feature` **before** the parent runs
lands in exactly that sum, on every slice, and is masked identically. The
override is three lines and delegates the rest to `super()`.
`condition_global_feature` holds the arithmetic and is tested against a direct
reproduction of the parent's expression, so the equivalence is checked rather
than argued.

## Only the base is conditioned

The sparse MIL head sums the same three embeddings over its spatial tokens
(`b36_sparse_mil._tokens`) and could take the term too. It deliberately does
not, for now.

The base is where series are fused into one study prediction, which is exactly
where "these two series hold different amounts of knee" has to be known, and
it is the primary path — the head enters through a learned gate as a residual.
Conditioning one site is also the smaller change, and `install_spacing_
conditioning` can add the head later without touching anything here.
`b54_state` reports `conditioning_sites`, so any run records which choice it
made rather than leaving it to be inferred.

## What it is at initialisation

Numerically identical to B42. The conditioning is zero-initialised, and with no
conditioning installed, or no spacing supplied, `condition_global_feature`
returns its input unchanged.
"""
from __future__ import annotations

import torch

from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectSparseMILResidual,
)
from .b37_highres_sparse_mil import B37Forward
from .b54_spacing_run import B54_VERSION, b54_state
from .spacing_conditioning import SpacingConditioning


def condition_global_feature(
    module: torch.nn.Module,
    global_feature: torch.Tensor,
    series_spacing: torch.Tensor | None,
) -> torch.Tensor:
    """Add the per-series spacing term to every slice of `global_feature`.

    `module` is the one carrying the conditioning — the study base, whose
    `plane_embedding` set the width. Returns the input unchanged when there is
    no conditioning installed or no spacing to use, so the frozen path is
    reached by doing nothing rather than by a special case.
    """
    conditioning = getattr(module, "spacing_conditioning", None)
    if conditioning is None or series_spacing is None:
        return global_feature
    contribution = conditioning(series_spacing.to(global_feature.device))
    # [B, K, D] -> [B, K, 1, D]: one vector per series, on every slice of it,
    # which is exactly how `metadata[:, :, None, :]` enters the parent's sum.
    return global_feature + contribution[:, :, None, :].to(global_feature.dtype)


class B54SpacingConditionedMIL(B42ConstantAreaAspectSparseMILResidual):
    """B42, told how thick each series is.

    Construct it exactly as B42, load the pretrained checkpoint, and only then
    call `install_spacing_conditioning(model.base)`. Installing first adds a
    state-dict key the checkpoint does not have and a strict load will raise.
    """

    def _base_logits_from_global(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        series_spacing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super()._base_logits_from_global(
            condition_global_feature(self.base, global_feature, series_spacing),
            present,
            series_meta,
        )

    def forward(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        series_spacing: torch.Tensor | None = None,
    ) -> B37Forward:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)
        if series_spacing is not None and series_spacing.ndim == 1:
            series_spacing = series_spacing.unsqueeze(0)

        global_feature, spatial = self._encode_ragged_study(volumes, present)
        base_logits = self._base_logits_from_global(
            global_feature, present, series_meta, series_spacing
        )
        # The head is deliberately unconditioned; see the module docstring.
        local_logits, top_indices, top_values = self.head(
            spatial, present, series_meta, slice_position
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * local_logits.float()
        return B37Forward(
            logits=logits,
            base_logits=base_logits,
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values,
        )

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": B54_VERSION,
                "conditioned_on": "slice_spacing_mm",
                "conditioning_site": "study_base_only",
                "head_conditioned": False,
                **b54_state(self),
            }
        )
        return state


def spacing_from_batch(batch: list[dict] | dict) -> torch.Tensor | None:
    """Pull `series_spacing` out of whatever the collate produced.

    `collate_b42` keeps studies ragged as a list of item dicts, one per study;
    `collate_b54` pads them into one tensor. This accepts either, and returns
    None when the spacing is absent so an unconditioned run needs no branch of
    its own at the call site.
    """
    if isinstance(batch, dict):
        return batch.get("series_spacing")
    if not batch or "series_spacing" not in batch[0]:
        return None
    values = [item["series_spacing"] for item in batch]
    width = max(int(v.shape[0]) for v in values)
    out = torch.full((len(values), width), float("nan"), dtype=torch.float32)
    for index, value in enumerate(values):
        out[index, : value.shape[0]] = value
    return out


def requires_spacing(model: torch.nn.Module) -> bool:
    """Whether this model would actually use a spacing if given one."""
    return any(isinstance(m, SpacingConditioning) for m in model.modules())
