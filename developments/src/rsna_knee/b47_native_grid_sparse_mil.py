"""B47 — score evidence where the encoder actually produced it.

B37 raised the input to 448 pixels so that focal findings would have detail to
be found in. ConvNeXt-tiny has an output stride of 32, so a 448x448 slice leaves
a 14x14 feature map: 196 places the head could look. `_encode_chunk` then
average-pools that to a fixed 6x6 before any evidence is scored, so 36 places
survive. Roughly five-sixths of the localisation the resolution was bought for
is discarded between the encoder and the head that was built to use it.

At the B42 reference area each surviving cell averages about a sixth of the
cropped field of view in each direction -- on the order of 24 mm of knee. For a
meniscal tear or a subtle cruciate signal change that is a very coarse notion of
"where the evidence is", and it is a plausible reason the sparse residual has
only ever been worth about six thousandths.

There is a second fault that only appears in B42. B42 preserves the native
aspect ratio, so a series can arrive as 10x20 cells rather than 14x14. Pooling a
rectangle to a fixed *square* grid produces bins that are anisotropic and whose
physical extent differs from series to series -- and `region_embedding` is a
table indexed by grid position, so row 7 means a different piece of anatomy
depending on how the scan happened to be shaped. B37's square input did not have
this problem; B42 introduced it silently.

B47 fixes both, because they cannot be separated: keeping the native cells means
their number varies, and a fixed 36-row lookup table cannot describe a varying
number of cells.

    evidence grid   aspect-preserving, sized by a cell BUDGET rather than a
                    fixed side, so cells stay square-ish and a study's series
                    are described on comparable scales
    region identity a continuous function of the cell's normalised centre,
                    replacing the position-indexed lookup table, so it means the
                    same thing whatever the grid turned out to be
    ragged regions  series with different cell counts are padded to the study's
                    maximum and the padding is masked out before the top-k, the
                    same way absent series already are

Both changes are one capability -- *where evidence is scored* -- but the second
is machinery forced by the first, not an independent bet. So the arm that
isolates them is built in: running B47 at the control budget of 36 cells
reproduces B37's effective resolution while using the new continuous region
encoding, which separates "more places to look" from "a different way of saying
where". That control is the point of the design, and the retrospective's rule
about changing one high-level capability at a time is why it exists.

The region projection is zero-initialised, exactly as `region_embedding` is, so
at step zero the head asks the pretrained local representation the same question
B36's head does.

Nothing here modifies B36, B37 or B42. B47 subclasses them, so a B46 run in
flight is untouched.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .b35_target_spatial_residual import B35_BASE_SLICES, B35_POSITION_BASIS, _position_basis
from .b36_sparse_mil import B36SparseMILHead
from .b37_highres_sparse_mil import (
    B37_GRID_SIZE,
    B37_IMAGE_SIZE,
    B37_LOCAL_AUX_WEIGHT,
    B37_TEMPERATURE,
    B37_TOP_K,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42_REFERENCE_AREA,
    B42ConstantAreaAspectSparseMILResidual,
    require_b42_contract,
)
from .constants import N_TARGETS

B47_VERSION = "b47_native_grid_sparse_mil_v1"
B47_EXPERIMENT = "B47_NATIVE_GRID_SPARSE_MIL"
B47_RUN_ROOT = "runs/080_Experiment_B47_native_grid_sparse_mil"

# ConvNeXt-tiny's output stride. A square 448 slice yields 14 x 14 = 196 cells.
B47_ENCODER_STRIDE = 32
B47_SQUARE_REFERENCE_CELLS = (B37_IMAGE_SIZE // B47_ENCODER_STRIDE) ** 2  # 196

# The budget is 240, not 196, and the difference is not a safety margin -- it is
# measured. B42 holds the *anatomical* area near 448^2 and then reflection-pads
# up to stride alignment, so a rectangular series carries padding that a square
# one does not. Sweeping every source shape from 120 to 1300 pixels through
# `constant_area_shape` gives a cell count between 196 and 240, the maximum
# arising at extreme aspect ratios (a 144x1120 source aligns to 192x1280, i.e.
# 6 x 40 = 240 cells). A budget of 196 would therefore pool almost every
# non-square series and defeat the experiment; 240 lets the native arm pass the
# encoder's own grid through untouched for every geometry B42 can produce.
B47_NATIVE_REGION_BUDGET = 240

# The matched control: B37/B42's 6x6, reached through B47's own machinery.
B47_CONTROL_REGION_BUDGET = B37_GRID_SIZE * B37_GRID_SIZE  # 36

B47_REGION_BASIS = 12

B47_ARMS = ("control", "native")
B47_ARM_BUDGETS = {
    "control": B47_CONTROL_REGION_BUDGET,
    "native": B47_NATIVE_REGION_BUDGET,
}


def grid_for_budget(height: int, width: int, budget: int) -> tuple[int, int]:
    """Choose an aspect-preserving grid holding at most `budget` cells.

    A feature map already inside the budget is kept exactly as the encoder
    produced it -- that is the whole point, and it is why the native arm does no
    pooling at the reference geometry. A larger map is reduced by a single
    isotropic factor so the cells stay square-ish rather than being squashed
    onto a fixed square grid.
    """
    height, width = int(height), int(width)
    budget = int(budget)
    if height < 1 or width < 1:
        raise ValueError("B47 feature map must have positive extent")
    if budget < 1:
        raise ValueError("B47 region budget must be at least 1")
    if height * width <= budget:
        return height, width

    scale = math.sqrt(budget / float(height * width))
    grid_h = max(1, min(height, int(round(height * scale))))
    grid_w = max(1, min(width, int(round(width * scale))))
    # Rounding can overshoot the budget; shrink the longer side until it fits.
    while grid_h * grid_w > budget:
        if grid_h >= grid_w and grid_h > 1:
            grid_h -= 1
        elif grid_w > 1:
            grid_w -= 1
        else:
            break
    return grid_h, grid_w


def region_basis(grid_h: int, grid_w: int, *, device=None, dtype=None) -> torch.Tensor:
    """A continuous description of each cell's normalised centre.

    This replaces a table indexed by grid position. A lookup table can only
    describe the exact grid it was sized for, and its row 7 means whatever
    happened to land there; a function of the centre coordinate means the same
    thing at 6x6, at 14x14 and at 10x20, which is what lets one trained head see
    series of different shapes.

    Mirrors the through-plane basis in B35: raw coordinate, square, and three
    sine/cosine octaves, for each of the two in-plane axes.
    """
    grid_h, grid_w = int(grid_h), int(grid_w)
    ys = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / grid_h
    xs = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / grid_w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    features = []
    for axis in (grid_y.reshape(-1), grid_x.reshape(-1)):
        features.extend(
            [
                axis,
                axis.square(),
                torch.sin(math.pi * axis),
                torch.cos(math.pi * axis),
                torch.sin(2.0 * math.pi * axis),
                torch.cos(2.0 * math.pi * axis),
            ]
        )
    basis = torch.stack(features, dim=-1)
    if basis.shape[-1] != B47_REGION_BASIS:
        raise RuntimeError("B47 region basis width changed")
    return basis if dtype is None else basis.to(dtype=dtype)


class B47NativeGridSparseMILHead(B36SparseMILHead):
    """B36's evidence head with a resolution-independent region encoding.

    Everything that decides the answer is inherited unchanged: the same
    parameter-free layer norm, the same evidence classifiers, the same top-k
    log-mean-exp, the same zero-init gate. Only how a cell says where it is
    changes, plus the ability to mask individual cells rather than whole series.
    """

    def __init__(
        self,
        dim: int,
        *,
        region_budget: int = B47_NATIVE_REGION_BUDGET,
        top_k: int = B37_TOP_K,
        temperature: float = B37_TEMPERATURE,
        token_dropout: float = 0.0,
    ) -> None:
        # B36's constructor wants a square grid; give it the budget's side only
        # so the inherited parameters exist, then discard its region table.
        side = max(1, int(round(math.sqrt(float(region_budget)))))
        super().__init__(
            dim,
            grid_size=side,
            top_k=int(top_k),
            temperature=float(temperature),
            token_dropout=float(token_dropout),
        )
        self.region_budget = int(region_budget)
        if self.region_budget < self.top_k:
            raise ValueError("B47 region budget must be at least top_k")
        # The position-indexed table cannot describe a varying number of cells.
        del self.region_embedding
        self.region_projection = nn.Linear(B47_REGION_BASIS, self.dim, bias=False)
        nn.init.zeros_(self.region_projection.weight)

    def region_features(self, grid_h: int, grid_w: int, *, device, dtype) -> torch.Tensor:
        basis = region_basis(grid_h, grid_w, device=device, dtype=torch.float32)
        return self.region_projection(basis).to(dtype=dtype)

    def _tokens(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        *,
        region_valid: torch.Tensor | None = None,
        region_shapes: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial.ndim != 5:
            raise ValueError("B47 spatial features must be [B,K,S,R,D]")
        b, k, s, r, d = spatial.shape
        if d != self.dim:
            raise ValueError("B47 spatial feature width does not match the head")
        if r > self.region_budget:
            raise ValueError(
                f"B47 received {r} regions, above the frozen budget {self.region_budget}"
            )
        if present.shape != (b, k):
            raise ValueError("B47 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B47 series metadata shape mismatch")
        if slice_position.shape != (b, k, s):
            raise ValueError("B47 slice-position shape mismatch")
        if region_shapes is not None and len(region_shapes) != k:
            raise ValueError("B47 needs one grid shape per series")

        x = F.layer_norm(spatial.float(), (d,)).to(dtype=spatial.dtype)
        pos = self.position_projection(_position_basis(slice_position).to(x.device)).to(
            dtype=x.dtype
        )

        # Each series describes its own cells, because after aspect-preserving
        # pooling two series need not share a grid.
        if region_shapes is None:
            region_shapes = [(int(math.isqrt(r)), int(math.isqrt(r)))] * k
        region = x.new_zeros((k, r, d))
        for index, (grid_h, grid_w) in enumerate(region_shapes):
            cells = int(grid_h) * int(grid_w)
            if cells > r:
                raise ValueError("B47 grid shape exceeds the padded region count")
            region[index, :cells] = self.region_features(
                int(grid_h), int(grid_w), device=x.device, dtype=x.dtype
            )

        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = (plane + fluid + fat).to(dtype=x.dtype)

        x = x + pos[:, :, :, None, :] + region[None, :, None, :, :] + metadata[:, :, None, None, :]
        x = self.token_dropout(x)
        tokens = x.reshape(b, k * s * r, d)

        invalid = (
            (present <= 0)[:, :, None, None].expand(b, k, s, r).reshape(b, k * s * r)
        )
        if region_valid is not None:
            # Padding cells are not evidence. Without this a short series would
            # contribute zero-filled tokens to the top-k pool.
            if region_valid.shape != (k, r):
                raise ValueError("B47 region validity mask shape mismatch")
            padded = (
                (~region_valid.bool())[None, :, None, :].expand(b, k, s, r).reshape(b, k * s * r)
            )
            invalid = invalid | padded

        if invalid.all(dim=1).any():
            raise RuntimeError("B47 received a study with no readable MRI series")
        if int((~invalid).sum(dim=1).min().item()) < self.top_k:
            raise RuntimeError("B47 has fewer valid local tokens than top_k")
        return tokens, invalid

    def forward(
        self,
        spatial: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        *,
        region_valid: torch.Tensor | None = None,
        region_shapes: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, invalid = self._tokens(
            spatial,
            present,
            series_meta,
            slice_position,
            region_valid=region_valid,
            region_shapes=region_shapes,
        )
        # Kept in fp32 through scoring and selection. The top-k picks the best
        # eight of tens of thousands of tightly clustered scores, and bfloat16
        # carries about two or three significant digits, which orders instances
        # inside that band arbitrarily and differently on different hardware.
        score = torch.einsum(
            "bnd,td->btn",
            tokens.float(),
            self.evidence_weight.float(),
        ) + self.evidence_bias.float()[None, :, None]
        score = score.masked_fill(invalid[:, None, :], float("-inf"))

        top_values, top_indices = torch.topk(
            score, k=self.top_k, dim=-1, largest=True, sorted=True
        )
        tau = float(self.temperature)
        local_logits = tau * (
            torch.logsumexp(top_values / tau, dim=-1) - math.log(float(self.top_k))
        )
        return local_logits.to(dtype=spatial.dtype), top_indices, top_values


class B47NativeGridSparseMILResidual(B42ConstantAreaAspectSparseMILResidual):
    """B42's encoder and hierarchy, scoring evidence on the encoder's own grid."""

    def __init__(self, *args, region_budget: int = B47_NATIVE_REGION_BUDGET, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.region_budget = int(region_budget)
        self.head = B47NativeGridSparseMILHead(
            int(self.head.dim),
            region_budget=self.region_budget,
            top_k=int(self.head.top_k),
            temperature=float(self.head.temperature),
        )

    def _encode_chunk(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool only as far as the budget requires, preserving aspect ratio."""
        encoder = self.base.encoder
        normalized = encoder._normalize(chunk)
        fmap = encoder.features(normalized)
        global_feature = encoder.pre_classifier(encoder.avgpool(fmap)).reshape(
            chunk.shape[0], int(encoder.out_dim)
        )
        grid_h, grid_w = grid_for_budget(
            int(fmap.shape[-2]), int(fmap.shape[-1]), self.region_budget
        )
        if (grid_h, grid_w) != (int(fmap.shape[-2]), int(fmap.shape[-1])):
            fmap = F.adaptive_avg_pool2d(fmap, (grid_h, grid_w))
        normalized_grid = encoder.pre_classifier[0](fmap)
        spatial = normalized_grid.permute(0, 2, 3, 1).reshape(
            chunk.shape[0], grid_h * grid_w, int(encoder.out_dim)
        )
        return global_feature, spatial

    def _encode_ragged_study(
        self,
        volumes: list[torch.Tensor],
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """As B42, but series may differ in cell count, so pad and record it."""
        if not isinstance(volumes, list) or not volumes:
            raise ValueError("B47 expects a non-empty list of series tensors")
        present_flat = present[0] if present.ndim == 2 else present
        if len(volumes) != int(present_flat.numel()):
            raise ValueError("B47 volumes/present series count mismatch")

        global_rows: list[torch.Tensor | None] = []
        spatial_rows: list[torch.Tensor | None] = []
        shapes: list[tuple[int, int] | None] = []
        for series_tensor, flag in zip(volumes, present_flat):
            if float(flag.detach().item()) <= 0:
                global_rows.append(None)
                spatial_rows.append(None)
                shapes.append(None)
                continue
            base_global, base_spatial = self._encode_rect_group(
                series_tensor[:B35_BASE_SLICES]
            )
            extra_global, extra_spatial = self._encode_rect_group(
                series_tensor[B35_BASE_SLICES:]
            )
            global_rows.append(torch.cat((base_global, extra_global), dim=0))
            spatial_rows.append(torch.cat((base_spatial, extra_spatial), dim=0))
            shapes.append(
                grid_for_budget(
                    int(series_tensor.shape[-2]) // B47_ENCODER_STRIDE,
                    int(series_tensor.shape[-1]) // B47_ENCODER_STRIDE,
                    self.region_budget,
                )
            )

        readable = [row for row in spatial_rows if row is not None]
        if not readable:
            raise RuntimeError("B47 study has no readable MRI series")
        max_regions = max(int(row.shape[1]) for row in readable)
        template_global = next(row for row in global_rows if row is not None)
        template_shape = next(shape for shape in shapes if shape is not None)

        padded_spatial, valid_rows = [], []
        for index, row in enumerate(spatial_rows):
            if row is None:
                global_rows[index] = torch.zeros_like(template_global)
                row = readable[0].new_zeros(
                    (readable[0].shape[0], max_regions, readable[0].shape[2])
                )
                shapes[index] = template_shape
                valid = torch.zeros(max_regions, dtype=torch.bool, device=row.device)
            else:
                cells = int(row.shape[1])
                valid = torch.zeros(max_regions, dtype=torch.bool, device=row.device)
                valid[:cells] = True
                if cells < max_regions:
                    row = F.pad(row, (0, 0, 0, max_regions - cells))
            padded_spatial.append(row)
            valid_rows.append(valid)

        global_feature = torch.stack(
            [row for row in global_rows if row is not None], dim=0
        ).unsqueeze(0)
        spatial = torch.stack(padded_spatial, dim=0).unsqueeze(0)
        region_valid = torch.stack(valid_rows, dim=0)
        return global_feature, spatial, region_valid, [s for s in shapes if s is not None]

    def forward(self, volumes, present, series_meta, slice_position):
        from .b37_highres_sparse_mil import B37Forward

        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)

        global_feature, spatial, region_valid, shapes = self._encode_ragged_study(
            volumes, present
        )
        base_logits = self._base_logits_from_global(global_feature, present, series_meta)
        local_logits, top_indices, top_values = self.head(
            spatial,
            present,
            series_meta,
            slice_position,
            region_valid=region_valid,
            region_shapes=shapes,
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
                "version": B47_VERSION,
                "experiment": B47_EXPERIMENT,
                "region_budget": self.region_budget,
                "region_identity": "continuous normalised cell centre",
                "evidence_precision": "fp32 through scoring and top-k",
            }
        )
        return state


def require_b47_contract(config: dict) -> dict:
    """Freeze everything B42 froze, and the one thing B47 is allowed to move."""
    crop_policy = require_b42_contract(config)

    arm = str(config.get("b47_arm", "native"))
    if arm not in B47_ARMS:
        raise ValueError(f"B47 arm must be one of {B47_ARMS}; got {arm!r}")

    budget = int(config.get("b47_region_budget", B47_ARM_BUDGETS[arm]))
    if budget != B47_ARM_BUDGETS[arm]:
        raise ValueError(
            f"B47 arm {arm!r} freezes b47_region_budget={B47_ARM_BUDGETS[arm]}; got {budget}"
        )
    if budget < B37_TOP_K:
        raise ValueError("B47 region budget must be at least top_k")

    for key, frozen in (
        ("b37_top_k", B37_TOP_K),
        ("b37_temperature", B37_TEMPERATURE),
        ("b37_local_aux_weight", B37_LOCAL_AUX_WEIGHT),
    ):
        value = config.get(key, frozen)
        if not np.isclose(float(value), float(frozen), atol=1e-12, rtol=0):
            raise ValueError(f"B47 inherits frozen {key}={frozen}; got {value}")

    return {
        "crop_policy": crop_policy,
        "arm": arm,
        "region_budget": budget,
        "reference_pixel_area": B42_REFERENCE_AREA,
    }


def b47_state(arm: str = "native") -> dict:
    if arm not in B47_ARMS:
        raise ValueError(f"B47 arm must be one of {B47_ARMS}")
    return {
        "version": B47_VERSION,
        "experiment": B47_EXPERIMENT,
        "arm": arm,
        "region_budget": B47_ARM_BUDGETS[arm],
        "encoder_stride": B47_ENCODER_STRIDE,
        "square_reference_cells": B47_SQUARE_REFERENCE_CELLS,
        "measured_cell_range": [196, B47_NATIVE_REGION_BUDGET],
        "b37_b42_cells": B47_CONTROL_REGION_BUDGET,
        "grid_policy": "aspect preserving, budgeted by cell count",
        "region_identity": "continuous normalised cell centre, zero initialised",
        "ragged_regions": "padded to the study maximum and masked before top-k",
        "n_targets": N_TARGETS,
    }


__all__ = [
    "B47_ARMS",
    "B47_ARM_BUDGETS",
    "B47_CONTROL_REGION_BUDGET",
    "B47_ENCODER_STRIDE",
    "B47_EXPERIMENT",
    "B47_NATIVE_REGION_BUDGET",
    "B47_SQUARE_REFERENCE_CELLS",
    "B47_REGION_BASIS",
    "B47_RUN_ROOT",
    "B47_VERSION",
    "B47NativeGridSparseMILHead",
    "B47NativeGridSparseMILResidual",
    "b47_state",
    "grid_for_budget",
    "region_basis",
    "require_b47_contract",
]
