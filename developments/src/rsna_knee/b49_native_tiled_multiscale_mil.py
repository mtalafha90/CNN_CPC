"""B49 — full-FOV native-tiled, globally conditioned sparse MIL.

B48 deliberately held the B42 input representation fixed while asking whether
the B34 cross-series pathology query can improve sparse local evidence.  B49
answers a different, prospective representation question without changing B48:

* the B34 hierarchy receives one *separate* full-FOV, aspect-preserving,
  constant-area context view for its historical 16 slice centres;
* the local sparse branch receives all 32 deterministic 2.5-D centres at their
  native in-plane sampling, split into overlapping 640x640 tiles;
* tile overlap supplies edge context, but an ownership mask assigns every
  native location to exactly one tile before top-k MIL, preventing duplicated
  evidence from being counted twice;
* local scores are pooled online.  Only the top-k values and identities are
  retained, so a large study never needs all tile feature maps in memory.

The local branch never resizes or centre-crops a source image.  Reflection
padding is permitted only outside its native field of view to make a 640-pixel
tile.  The global context branch is intentionally downsampled and is labelled
as such in every checkpoint/audit; it is not the local evidence representation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .b35_target_spatial_residual import (
    B35_BASE_SLICES,
    B35_DENSE_SLICES,
    _position_basis,
    b35_centers,
)
from .b36_sparse_mil import B36SparseMILHead
from .b37_highres_sparse_mil import (
    B37_ENCODER_CHUNK_SIZE,
    B37_ENCODER_LR_SCALE,
    B37_ENCODER_TRAINABLE_STAGES,
    B37_GRID_SIZE,
    B37_IMAGE_SIZE,
    B37_LOCAL_AUX_WEIGHT,
    B37_TEMPERATURE,
    B37_TOP_K,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42_EFFECTIVE_BATCH,
    B42_REFERENCE_AREA,
    B42_REFERENCE_SIDE,
    B42ConstantAreaAspectSparseMILResidual,
    resize_triplets_constant_area,
)
from .constants import N_TARGETS
from .dicom import _normalise_volume, find_series_dir, read_dicom_series
from .b12_variable_series import VariableSeriesKneeDataset


B49_VERSION = "b49_full_fov_native_tiled_multiscale_global_conditioned_sparse_mil_v1"
B49_EXPERIMENT = "B49_full_FOV_native_tiled_multiscale_global_conditioned_sparse_MIL"
B49_NUMBERED_CONTAINER = "runs/082_Experiment_B49_native_tiled_multiscale_mil"
B49_RUN_ROOT = f"{B49_NUMBERED_CONTAINER}/b49_native_tiled_multiscale_mil"

# The local contract is deliberately pixel-native.  640 covers the second most
# common 640x640 matrix in one tile; 128 pixels of overlap preserve context at
# tile borders.  A source shorter than 640 is reflection-padded outside its FOV
# only.  The ownership mask below prevents that overlap from duplicating MIL
# evidence.
B49_TILE_SIZE = 640
B49_TILE_OVERLAP = 128
B49_TILE_STRIDE = B49_TILE_SIZE - B49_TILE_OVERLAP
B49_TILE_ENCODER_CHUNK_SIZE = 2
B49_TILE_PADDING = "reflect_outside_native_fov_only"
B49_LOCAL_PREPROCESSING = "full_fov_native_tiled_no_inplane_resize_no_center_crop"
B49_GLOBAL_CONTEXT_PREPROCESSING = "full_fov_constant_area_aspect_downsample_for_B34_context_only"
B49_GLOBAL_CONTEXT_REFERENCE_AREA = B42_REFERENCE_AREA
B49_GLOBAL_CONTEXT_ALIGNMENT = 32
B49_COORDINATE_BASIS = 12
B49_EVIDENCE_PRECISION = "fp32_scoring_and_topk_for_large_native_token_pool"

B49_CONTEXT_DIM = 96
B49_CONTEXT_EPS = 1e-6
B49_CONTEXT_METRIC = "cosine_low_rank_query_token_compatibility"
B49_CONTEXT_GATE_INIT = "zero_tanh_targetwise"
B49_CONTEXT_QUERY_GRADIENT = "detached_before_local_head"
B49_FIXED_EPOCHS = 2
B49_SUPERVISION = "report_only_weak_no_official_gold"
B49_VALIDATION_SURFACE = "frozen_scanner_grouped_domain_split_v1"

B49_STATIC_PRIOR_CONTROL = "static_prior_control"
B49_POST_CROSS_ATTENTION_CANDIDATE = "post_cross_attention_candidate"
B49_ARMS = (B49_STATIC_PRIOR_CONTROL, B49_POST_CROSS_ATTENTION_CANDIDATE)
B49_ARM_CONTEXT_SOURCE = {
    B49_STATIC_PRIOR_CONTROL: "pathology_prior_before_series_cross_attention",
    B49_POST_CROSS_ATTENTION_CANDIDATE: "post_series_cross_attention_query",
}


@dataclass(frozen=True)
class NativeTile:
    """One local native tile plus its non-overlapping evidence ownership area."""

    index: int
    y0: int
    x0: int
    y_owner_lo: float
    y_owner_hi: float
    x_owner_lo: float
    x_owner_hi: float


def native_tile_starts(
    length: int,
    *,
    tile_size: int = B49_TILE_SIZE,
    stride: int = B49_TILE_STRIDE,
) -> tuple[int, ...]:
    """Return centred/full-coverage tile origins for one native image axis.

    For a large source the number of intervals is the minimum needed to keep
    adjacent origins no farther apart than ``stride``.  Origins are then spread
    evenly over the available span, avoiding an unnecessarily tiny final step.
    For a smaller source one tile is centred and extends outside the source;
    that extension is later reflection-padded and never treated as native
    evidence.
    """
    length, tile_size, stride = int(length), int(tile_size), int(stride)
    if length < 1 or tile_size < 2 or stride < 1 or stride > tile_size:
        raise ValueError("B49 tile dimensions/stride are invalid")
    if length <= tile_size:
        return (-(tile_size - length) // 2,)
    span = length - tile_size
    intervals = int(math.ceil(span / float(stride)))
    starts = tuple(int(round(span * index / intervals)) for index in range(intervals + 1))
    if starts[0] != 0 or starts[-1] != span or len(set(starts)) != len(starts):
        raise RuntimeError("B49 failed to construct a valid full-FOV tile grid")
    return starts


def _axis_ownership(starts: Sequence[int], *, length: int, tile_size: int) -> tuple[tuple[float, float], ...]:
    """Partition one source axis at the midpoint of adjacent tile overlaps."""
    if not starts:
        raise ValueError("B49 tile ownership needs at least one start")
    bounds: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        lo = 0.0 if index == 0 else 0.5 * (starts[index - 1] + tile_size + start)
        hi = float(length) if index + 1 == len(starts) else 0.5 * (start + tile_size + starts[index + 1])
        if not (0.0 <= lo < hi <= float(length)):
            raise RuntimeError("B49 tile ownership is not a valid native partition")
        bounds.append((float(lo), float(hi)))
    if not np.isclose(bounds[0][0], 0.0) or not np.isclose(bounds[-1][1], float(length)):
        raise RuntimeError("B49 ownership does not span the complete native axis")
    for left, right in zip(bounds[:-1], bounds[1:]):
        if not np.isclose(left[1], right[0], atol=1e-8, rtol=0):
            raise RuntimeError("B49 tile ownership has a gap or overlap")
    return tuple(bounds)


def native_tile_layout(
    height: int,
    width: int,
    *,
    tile_size: int = B49_TILE_SIZE,
    stride: int = B49_TILE_STRIDE,
) -> tuple[NativeTile, ...]:
    """Return all full-FOV tiles and their deterministic ownership rectangles."""
    height, width = int(height), int(width)
    ys = native_tile_starts(height, tile_size=tile_size, stride=stride)
    xs = native_tile_starts(width, tile_size=tile_size, stride=stride)
    y_owner = _axis_ownership(ys, length=height, tile_size=int(tile_size))
    x_owner = _axis_ownership(xs, length=width, tile_size=int(tile_size))
    layout: list[NativeTile] = []
    for y_index, y0 in enumerate(ys):
        for x_index, x0 in enumerate(xs):
            layout.append(
                NativeTile(
                    index=len(layout),
                    y0=int(y0),
                    x0=int(x0),
                    y_owner_lo=y_owner[y_index][0],
                    y_owner_hi=y_owner[y_index][1],
                    x_owner_lo=x_owner[x_index][0],
                    x_owner_hi=x_owner[x_index][1],
                )
            )
    return tuple(layout)


def verify_native_tile_coverage(
    height: int,
    width: int,
    layout: Sequence[NativeTile],
    *,
    tile_size: int = B49_TILE_SIZE,
) -> None:
    """Prove that all native pixel centres have exactly one evidence owner."""
    height, width = int(height), int(width)
    if height < 1 or width < 1 or not layout:
        raise ValueError("B49 coverage verification needs a non-empty source/layout")
    # This check is deliberately exact at source-pixel centres, not a summary of
    # tile rectangles.  It catches both a missing edge pixel and duplicate owner.
    yy = np.arange(height, dtype=np.float64) + 0.5
    xx = np.arange(width, dtype=np.float64) + 0.5
    owner = np.zeros((height, width), dtype=np.uint8)
    for tile in layout:
        inside_native = (
            (yy[:, None] >= tile.y0)
            & (yy[:, None] < tile.y0 + int(tile_size))
            & (xx[None, :] >= tile.x0)
            & (xx[None, :] < tile.x0 + int(tile_size))
        )
        owned = (
            (yy[:, None] >= tile.y_owner_lo)
            & (yy[:, None] < tile.y_owner_hi)
            & (xx[None, :] >= tile.x_owner_lo)
            & (xx[None, :] < tile.x_owner_hi)
        )
        owner += (inside_native & owned).astype(np.uint8)
    if not np.all(owner == 1):
        raise RuntimeError("B49 tiles do not give each native source pixel exactly one owner")


def _tile_padding(
    height: int,
    width: int,
    layout: Sequence[NativeTile],
    *,
    tile_size: int,
) -> tuple[int, int, int, int]:
    """Return the shared native-volume reflection padding required by a layout."""
    top = max(0, -min(tile.y0 for tile in layout))
    left = max(0, -min(tile.x0 for tile in layout))
    bottom = max(0, max(tile.y0 + tile_size for tile in layout) - int(height))
    right = max(0, max(tile.x0 + tile_size for tile in layout) - int(width))
    return int(top), int(bottom), int(left), int(right)


def tile_feature_coordinates(
    tile: NativeTile,
    *,
    native_height: int,
    native_width: int,
    feature_height: int,
    feature_width: int,
    tile_size: int = B49_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a tile feature lattice to full-FOV coordinates and ownership validity.

    The coordinate convention uses the centre of an equal-area input cell.  It
    is recorded because ConvNeXt's true receptive field is broader than one
    cell; overlap gives border context while ownership makes the evidence pool
    one-to-one in native image space.
    """
    native_height, native_width = int(native_height), int(native_width)
    feature_height, feature_width = int(feature_height), int(feature_width)
    if min(native_height, native_width, feature_height, feature_width) < 1:
        raise ValueError("B49 feature-coordinate dimensions must be positive")
    y = float(tile.y0) + (torch.arange(feature_height, dtype=torch.float32) + 0.5) * (
        float(tile_size) / float(feature_height)
    )
    x = float(tile.x0) + (torch.arange(feature_width, dtype=torch.float32) + 0.5) * (
        float(tile_size) / float(feature_width)
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    valid = (
        (yy >= 0.0)
        & (yy < float(native_height))
        & (xx >= 0.0)
        & (xx < float(native_width))
        & (yy >= float(tile.y_owner_lo))
        & (yy < float(tile.y_owner_hi))
        & (xx >= float(tile.x_owner_lo))
        & (xx < float(tile.x_owner_hi))
    ).reshape(-1)
    coordinate = torch.stack(
        (
            (yy / float(native_height)).clamp(0.0, 1.0),
            (xx / float(native_width)).clamp(0.0, 1.0),
        ),
        dim=-1,
    ).reshape(-1, 2)
    return coordinate, valid


def coordinate_basis(coordinates: torch.Tensor) -> torch.Tensor:
    """Continuous 12-D full-FOV coordinate basis, matching B47's semantics."""
    if coordinates.ndim != 2 or int(coordinates.shape[-1]) != 2:
        raise ValueError("B49 coordinates must be [R,2]")
    features = []
    for axis in (coordinates[:, 0].float(), coordinates[:, 1].float()):
        features.extend(
            (
                axis,
                axis.square(),
                torch.sin(math.pi * axis),
                torch.cos(math.pi * axis),
                torch.sin(2.0 * math.pi * axis),
                torch.cos(2.0 * math.pi * axis),
            )
        )
    result = torch.stack(features, dim=-1)
    if int(result.shape[-1]) != B49_COORDINATE_BASIS:
        raise RuntimeError("B49 coordinate-basis width changed")
    return result


def _triplet_indices(centres: np.ndarray, *, n_frames: int, gap: int) -> np.ndarray:
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    return np.clip(
        np.asarray(centres, dtype=np.int64)[:, None] + offsets[None, :],
        0,
        int(n_frames) - 1,
    )


def full_fov_context_from_normalized(
    normalized: np.ndarray,
    centres: np.ndarray,
    *,
    gap: int,
) -> torch.Tensor:
    """Build the only B49 downsampled input: a full-FOV B34 context view."""
    x = np.asarray(normalized, dtype=np.float32)
    if x.ndim != 3 or len(x) < 1:
        raise ValueError("B49 context preprocessing requires [S,H,W]")
    if len(centres) < B35_BASE_SLICES:
        raise ValueError("B49 needs the historical first 16 slice centres")
    index = _triplet_indices(centres[:B35_BASE_SLICES], n_frames=len(x), gap=int(gap))
    triplets = x[index].astype(np.float32, copy=False)
    # This is a named, separate global context representation.  B49 local tiles
    # never call interpolate or this helper.
    return resize_triplets_constant_area(
        triplets,
        reference_area=B49_GLOBAL_CONTEXT_REFERENCE_AREA,
        alignment=B49_GLOBAL_CONTEXT_ALIGNMENT,
    )


def _padded_native_volume(
    normalized: np.ndarray,
    layout: Sequence[NativeTile],
    *,
    tile_size: int,
) -> tuple[np.ndarray, int, int]:
    """Reflection-pad outside the native FOV only, once per streamed series."""
    x = np.asarray(normalized, dtype=np.float32)
    if x.ndim != 3 or min(x.shape[1:]) < 2:
        raise ValueError("B49 reflection-tiled local branch needs a 2-D native image")
    top, bottom, left, right = _tile_padding(
        int(x.shape[1]), int(x.shape[2]), layout, tile_size=int(tile_size)
    )
    if not any((top, bottom, left, right)):
        return x, 0, 0
    padded = np.pad(
        x,
        ((0, 0), (top, bottom), (left, right)),
        mode="reflect",
    )
    return np.ascontiguousarray(padded), int(top), int(left)


def native_tile_triplet_chunks(
    normalized: np.ndarray,
    centres: np.ndarray,
    layout: Sequence[NativeTile],
    *,
    gap: int,
    chunk_size: int = B49_TILE_ENCODER_CHUNK_SIZE,
    tile_size: int = B49_TILE_SIZE,
) -> Iterator[tuple[NativeTile, np.ndarray, torch.Tensor]]:
    """Yield one small ``[C,3,640,640]`` native tile batch at a time.

    The generator intentionally holds only one source volume and one tile chunk
    at a time.  It does not make a resized full image or a giant tile tensor.
    """
    x = np.asarray(normalized, dtype=np.float32)
    centres = np.asarray(centres, dtype=np.int64).reshape(-1)
    if x.ndim != 3 or len(centres) != B35_DENSE_SLICES:
        raise ValueError("B49 local tile generator requires 32 centres from one [S,H,W] volume")
    if int(chunk_size) < 1:
        raise ValueError("B49 tile encoder chunk size must be positive")
    padded, y_shift, x_shift = _padded_native_volume(x, layout, tile_size=int(tile_size))
    index = _triplet_indices(centres, n_frames=len(x), gap=int(gap))
    for tile in layout:
        y0, x0 = int(tile.y0) + y_shift, int(tile.x0) + x_shift
        for begin in range(0, len(centres), int(chunk_size)):
            chosen = np.arange(begin, min(begin + int(chunk_size), len(centres)), dtype=np.int64)
            patch = padded[index[chosen], :, y0 : y0 + int(tile_size), x0 : x0 + int(tile_size)]
            expected = (len(chosen), 3, int(tile_size), int(tile_size))
            if tuple(patch.shape) != expected:
                raise RuntimeError(f"B49 tiled crop shape changed: {tuple(patch.shape)} != {expected}")
            yield tile, chosen, torch.from_numpy(np.ascontiguousarray(patch))


class B49NativeTiledFullFOVDataset(VariableSeriesKneeDataset):
    """Read only B49's low-resolution context tensors during data loading.

    Native local tiles are not materialised here.  Their source path, full-FOV
    geometry and deterministic slice centres are retained as a tiny descriptor;
    the model streams them after its global query is available.  That avoids
    storing hundreds of 640x640 tensors per high-resolution series in RAM.
    """

    def __init__(
        self,
        study_uids,
        series_records,
        config,
        *,
        center_offsets: tuple[int, ...] = (0,),
        targets=None,
        weights=None,
    ) -> None:
        super().__init__(
            study_uids,
            series_records,
            config,
            targets=targets,
            weights=weights,
            train=False,
        )
        self.center_offsets = tuple(int(value) for value in center_offsets)
        if not self.center_offsets:
            raise ValueError("B49 needs at least one deterministic centre offset")
        if int(config.image_size) != B37_IMAGE_SIZE:
            raise ValueError("B49 freezes the global context reference side at 448")
        if int(config.triplet_gap) != 1:
            raise ValueError("B49 freezes the 2.5-D triplet gap at 1")
        if int(config.series_cache_mb) != 0:
            raise ValueError("B49 disables dataset raw-volume caching to bound host memory")

    @staticmethod
    def _zero_context() -> torch.Tensor:
        return torch.zeros(
            B35_BASE_SLICES,
            3,
            B42_REFERENCE_SIDE,
            B42_REFERENCE_SIDE,
            dtype=torch.float32,
        )

    def _read_one_series(self, uid: str, record: dict) -> tuple[list[dict], list[torch.Tensor], bool, dict]:
        series_uid = str(record["series_uid"])
        directory = find_series_dir(self.config.data_root, self.config.split, uid, series_uid)
        if directory is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing B49 series {uid}/{series_uid}")
            return [], [], False, {"series_uid": series_uid, "present": False, "reason": "missing"}
        try:
            raw = self._read_volume(directory, str(record["plane"]).lower())
            normalized = _normalise_volume(raw)
            height, width = int(normalized.shape[1]), int(normalized.shape[2])
            layout = native_tile_layout(height, width)
            sources, contexts = [], []
            for offset in self.center_offsets:
                centres, position = b35_centers(
                    len(normalized),
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                )
                contexts.append(
                    full_fov_context_from_normalized(
                        normalized,
                        centres,
                        gap=int(self.config.triplet_gap),
                    )
                )
                sources.append(
                    {
                        "path": str(Path(directory).resolve()),
                        "study_uid": str(uid),
                        "series_uid": series_uid,
                        "native_height": height,
                        "native_width": width,
                        "centres": [int(value) for value in centres.tolist()],
                        "slice_positions": [float(value) for value in position.tolist()],
                        "tile_count": len(layout),
                    }
                )
            geometry = {
                "series_uid": series_uid,
                "present": True,
                "native_height": height,
                "native_width": width,
                "tile_count": len(layout),
                "tile_size": B49_TILE_SIZE,
                "tile_overlap": B49_TILE_OVERLAP,
                "tile_layout": [asdict(tile) for tile in layout],
            }
            return sources, contexts, True, geometry
        except Exception as exc:
            if self.config.strict_dicom:
                raise
            return [], [], False, {
                "series_uid": series_uid,
                "present": False,
                "reason": f"unreadable:{type(exc).__name__}",
            }

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        per_record: list[tuple[list[dict], list[torch.Tensor], bool, dict]] = [
            self._read_one_series(uid, record) for record in records
        ]
        present = torch.tensor([float(row[2]) for row in per_record], dtype=torch.float32)
        meta = torch.tensor(
            [
                [record["plane_id"], record["fluid_id"], record["fat_id"]]
                for record in records
            ],
            dtype=torch.long,
        )
        views: list[dict] = []
        for view_index, _offset in enumerate(self.center_offsets):
            context_volumes, local_sources, positions = [], [], []
            for sources, contexts, readable, _geometry in per_record:
                if readable:
                    context_volumes.append(contexts[view_index])
                    local_sources.append(sources[view_index])
                    positions.append(torch.tensor(sources[view_index]["slice_positions"], dtype=torch.float32))
                else:
                    context_volumes.append(self._zero_context())
                    local_sources.append(None)
                    positions.append(torch.zeros(B35_DENSE_SLICES, dtype=torch.float32))
            views.append(
                {
                    "context_volumes": context_volumes,
                    "local_sources": local_sources,
                    "slice_position": torch.stack(positions),
                }
            )
        item = {
            "study_uid": str(uid),
            "views": views,
            "present": present,
            "series_meta": meta,
            "geometry": [row[3] for row in per_record],
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], dtype=np.float32))
        return item


def collate_b49(items: list[dict]) -> list[dict]:
    """Keep ragged B49 studies and streaming source descriptors unpadded."""
    return list(items)


@dataclass(frozen=True)
class B49HeadPool:
    values: torch.Tensor | None
    ids: torch.Tensor | None


class B49NativeTiledMILHead(B36SparseMILHead):
    """Continuous full-FOV local evidence scorer with B48-style query residual."""

    def __init__(
        self,
        dim: int = 768,
        *,
        top_k: int = B37_TOP_K,
        temperature: float = B37_TEMPERATURE,
        context_dim: int = B49_CONTEXT_DIM,
    ) -> None:
        # The B36 constructor supplies the inherited evidence classifier,
        # through-plane/metadata embeddings and zero-start sparse residual.
        super().__init__(
            dim=int(dim),
            grid_size=B37_GRID_SIZE,
            top_k=int(top_k),
            temperature=float(temperature),
        )
        del self.region_embedding
        self.region_projection = nn.Linear(B49_COORDINATE_BASIS, self.dim, bias=False)
        nn.init.zeros_(self.region_projection.weight)
        self.context_dim = int(context_dim)
        if self.context_dim < 1:
            raise ValueError("B49 context dimension must be positive")
        self.context_query = nn.Linear(self.dim, self.context_dim, bias=False)
        self.context_key = nn.Linear(self.dim, self.context_dim, bias=False)
        self.context_gate = nn.Parameter(torch.zeros(N_TARGETS, dtype=torch.float32))
        nn.init.xavier_uniform_(self.context_query.weight)
        nn.init.xavier_uniform_(self.context_key.weight)

    def effective_context_gate(self) -> torch.Tensor:
        return torch.tanh(self.context_gate)

    def empty_pool(self) -> B49HeadPool:
        return B49HeadPool(values=None, ids=None)

    def _context_residual(
        self,
        tokens: torch.Tensor,
        global_query: torch.Tensor,
    ) -> torch.Tensor:
        if global_query.shape != (1, N_TARGETS, self.dim):
            raise ValueError("B49 global query must be [1,12,D]")
        query = F.layer_norm(global_query.detach().float(), (self.dim,))
        key = F.layer_norm(tokens.float(), (self.dim,))
        q = F.normalize(self.context_query(query), p=2.0, dim=-1, eps=B49_CONTEXT_EPS)
        k = F.normalize(self.context_key(key), p=2.0, dim=-1, eps=B49_CONTEXT_EPS)
        cosine = torch.einsum("btr,nr->tn", q, k)
        return self.effective_context_gate().float()[:, None] * cosine

    def score_tile_features(
        self,
        fmap: torch.Tensor,
        *,
        slice_positions: torch.Tensor,
        series_meta: torch.Tensor,
        coordinates: torch.Tensor,
        coordinate_valid: torch.Tensor,
        global_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return inherited/static scores, conditioned scores, and valid mask.

        ``fmap`` contains a few slice centres from one tile.  The output keeps
        its full map resolution; no adaptive spatial pooling occurs here.
        """
        if fmap.ndim != 4 or int(fmap.shape[1]) != self.dim:
            raise ValueError("B49 tile feature map must be [C,D,H,W]")
        count, _dim, fh, fw = fmap.shape
        regions = int(fh) * int(fw)
        if slice_positions.shape != (count,):
            raise ValueError("B49 tile slice positions do not match feature batch")
        if series_meta.shape != (3,):
            raise ValueError("B49 series metadata must be [plane,fluid,fat]")
        if coordinates.shape != (regions, 2) or coordinate_valid.shape != (regions,):
            raise ValueError("B49 tile coordinate lattice does not match feature map")

        tokens = fmap.permute(0, 2, 3, 1).reshape(count * regions, self.dim)
        tokens = F.layer_norm(tokens.float(), (self.dim,)).to(dtype=fmap.dtype)
        through_plane = self.position_projection(
            _position_basis(slice_positions.to(device=fmap.device))
        ).to(dtype=tokens.dtype)
        location = self.region_projection(
            coordinate_basis(coordinates.to(device=fmap.device))
        ).to(dtype=tokens.dtype)
        meta = series_meta.to(device=fmap.device, dtype=torch.long)
        acquisition = (
            self.plane_embedding(meta[0].clamp(0, 3))
            + self.fluid_embedding(meta[1].clamp(0, 2))
            + self.fat_embedding(meta[2].clamp(0, 2))
        ).to(dtype=tokens.dtype)
        tokens = tokens + through_plane[:, None, :].expand(-1, regions, -1).reshape_as(tokens)
        tokens = tokens + location[None, :, :].expand(count, -1, -1).reshape_as(tokens)
        tokens = self.token_dropout(tokens + acquisition[None, :])
        valid = coordinate_valid.to(device=fmap.device, dtype=torch.bool).repeat(count)
        if not bool(valid.any()):
            raise RuntimeError("B49 tile has no native evidence-owned feature cells")
        # Native tiling may supply thousands of competing local cells.  Keep
        # scoring/selection in fp32 so bfloat16 rounding cannot decide which
        # apparently tied cells enter the sparse evidence pool.
        static = torch.einsum("nd,td->tn", tokens.float(), self.evidence_weight.float())
        static = static + self.evidence_bias.float()[:, None]
        context = self._context_residual(tokens, global_query)
        conditioned = static + context
        static = static.masked_fill(~valid[None, :], float("-inf"))
        conditioned = conditioned.masked_fill(~valid[None, :], float("-inf"))
        return static, conditioned, valid

    def update_pool(
        self,
        pool: B49HeadPool,
        values: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> B49HeadPool:
        """Merge one streamed tile chunk into an exact global top-k pool."""
        if values.ndim != 2 or int(values.shape[0]) != N_TARGETS:
            raise ValueError("B49 pooled values must be [12,N]")
        if token_ids.shape != (int(values.shape[1]),):
            raise ValueError("B49 token IDs do not match pooled values")
        ids = token_ids.to(device=values.device, dtype=torch.long)[None, :].expand(N_TARGETS, -1)
        if pool.values is not None:
            values = torch.cat((pool.values, values), dim=-1)
            ids = torch.cat((pool.ids, ids), dim=-1)
        keep = min(self.top_k, int(values.shape[-1]))
        top_values, index = torch.topk(values, k=keep, dim=-1, largest=True, sorted=True)
        top_ids = torch.gather(ids, dim=-1, index=index)
        return B49HeadPool(values=top_values, ids=top_ids)

    def pooled_logits(self, pool: B49HeadPool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pool.values is None or pool.ids is None or int(pool.values.shape[-1]) != self.top_k:
            raise RuntimeError("B49 has fewer valid native local tokens than top-k")
        tau = float(self.temperature)
        logits = tau * (torch.logsumexp(pool.values.float() / tau, dim=-1) - math.log(float(self.top_k)))
        return logits[None, :], pool.ids[None, :, :], pool.values.float()[None, :, :]

    def state(self) -> dict:
        inherited = super().state()
        raw = self.context_gate.detach().float().cpu()
        inherited.update(
            {
                "version": B49_VERSION,
                "regions_per_slice": "variable_native_tile_feature_lattice",
                "region_identity": "continuous_full_fov_native_feature_cell_center",
                "region_basis": B49_COORDINATE_BASIS,
                "evidence_precision": B49_EVIDENCE_PRECISION,
                "context_metric": B49_CONTEXT_METRIC,
                "context_dim": self.context_dim,
                "context_eps": B49_CONTEXT_EPS,
                "context_gate_init": B49_CONTEXT_GATE_INIT,
                "context_gate_raw": [float(value) for value in raw.tolist()],
                "context_gate_effective": [float(value) for value in torch.tanh(raw).tolist()],
            }
        )
        return inherited


@dataclass(frozen=True)
class B49Forward:
    logits: torch.Tensor
    base_logits: torch.Tensor
    local_logits: torch.Tensor
    top_indices: torch.Tensor
    top_values: torch.Tensor
    context_query: torch.Tensor
    context_abs_mean: torch.Tensor
    topk_overlap_with_static: torch.Tensor | None
    native_tile_count: int
    native_valid_token_count: int


class B49NativeTiledMultiscaleMILResidual(B42ConstantAreaAspectSparseMILResidual):
    """Frozen B34 hierarchy plus streamed full-FOV native sparse evidence."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        encoder_trainable_stages: int = B37_ENCODER_TRAINABLE_STAGES,
        encoder_chunk_size: int = B37_ENCODER_CHUNK_SIZE,
        tile_encoder_chunk_size: int = B49_TILE_ENCODER_CHUNK_SIZE,
        arm: str = B49_POST_CROSS_ATTENTION_CANDIDATE,
        context_dim: int = B49_CONTEXT_DIM,
    ) -> None:
        if arm not in B49_ARMS:
            raise ValueError(f"B49 arm must be one of {B49_ARMS}; got {arm!r}")
        super().__init__(
            base_model,
            grid_size=B37_GRID_SIZE,
            top_k=B37_TOP_K,
            temperature=B37_TEMPERATURE,
            encoder_trainable_stages=int(encoder_trainable_stages),
            encoder_chunk_size=int(encoder_chunk_size),
        )
        self.arm = str(arm)
        self.context_source = B49_ARM_CONTEXT_SOURCE[self.arm]
        self.tile_encoder_chunk_size = int(tile_encoder_chunk_size)
        if self.tile_encoder_chunk_size < 1:
            raise ValueError("B49 tile encoder chunk size must be positive")
        # Replacing only the local head leaves B34 and the encoder-tail freeze
        # contract created by the parent untouched.
        with torch.random.fork_rng(devices=[]):
            self.head = B49NativeTiledMILHead(
                int(self.base.encoder.out_dim),
                top_k=B37_TOP_K,
                temperature=B37_TEMPERATURE,
                context_dim=int(context_dim),
            )

    @torch.no_grad()
    def _global_query_states(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the frozen B34 static/post-attention pathology states exactly."""
        base = self.base
        x = global_feature[:, :, :B35_BASE_SLICES]
        plane = base.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = base.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = base.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(x.dtype)
        x = (x + base.slice_position[None, None, :, :] + metadata[:, :, None, :]) * mask
        tokens = base._pool_real_series_b31(x, present)
        padding = present <= 0
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        memory = base.context(tokens, src_key_padding_mask=safe_padding)
        memory = memory.masked_fill(padding[:, :, None], 0.0)
        queries = base.pathology_tokens[None, :, :].expand(memory.shape[0], -1, -1)
        prior = base.pathology_context(queries)
        attended, _ = base.cross_attention(
            prior,
            memory,
            memory,
            key_padding_mask=safe_padding,
            need_weights=False,
        )
        static = base.dropout(base.query_norm(prior))
        post = base.dropout(base.query_norm(prior + attended))
        reconstructed = (post * base.target_weight[None, :, :]).sum(dim=-1) + base.target_bias
        reconstructed = torch.where(empty[:, None], base.target_bias[None, :], reconstructed)
        return static.detach(), post.detach(), reconstructed.detach()

    def _select_context_query(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> torch.Tensor:
        static, post, _ = self._global_query_states(global_feature, present, series_meta)
        return static if self.arm == B49_STATIC_PRIOR_CONTROL else post

    def _encode_context_study(
        self,
        context_volumes: list[torch.Tensor],
        present: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one ragged study's 16 full-FOV global context triplets."""
        present_flat = present[0] if present.ndim == 2 else present
        if not isinstance(context_volumes, list) or len(context_volumes) != int(present_flat.numel()):
            raise ValueError("B49 context volume/present series count mismatch")
        rows: list[torch.Tensor | None] = []
        template = None
        for volume, flag in zip(context_volumes, present_flat):
            if volume.ndim != 4 or tuple(volume.shape[1:2]) != (3,) or int(volume.shape[0]) != B35_BASE_SLICES:
                raise ValueError("B49 global context series must be [16,3,H,W]")
            if float(flag.detach().item()) <= 0:
                rows.append(None)
                continue
            global_feature, _discarded_spatial = self._encode_rect_group(volume)
            rows.append(global_feature)
            if template is None:
                template = global_feature
        if template is None:
            raise RuntimeError("B49 study has no readable global context series")
        result = [row if row is not None else torch.zeros_like(template) for row in rows]
        return torch.stack(result, dim=0).unsqueeze(0)

    def _encode_native_tile_fmap(self, chunk: torch.Tensor) -> torch.Tensor:
        if tuple(chunk.shape[1:]) != (3, B49_TILE_SIZE, B49_TILE_SIZE):
            raise ValueError("B49 native local encoder needs [C,3,640,640] tiles")
        encoder = self.base.encoder
        normalized = encoder._normalize(chunk)
        fmap = encoder.features(normalized)
        return encoder.pre_classifier[0](fmap)

    @staticmethod
    def _source_normalized(source: dict) -> np.ndarray:
        raw = source.get("raw_volume")
        if raw is None:
            raw = read_dicom_series(source["path"])
        normalized = _normalise_volume(np.asarray(raw, dtype=np.float32))
        declared = (int(source["native_height"]), int(source["native_width"]))
        if tuple(normalized.shape[1:]) != declared:
            raise RuntimeError(
                "B49 local DICOM geometry changed after global-context loading: "
                f"{tuple(normalized.shape[1:])} != {declared}"
            )
        return normalized

    def _stream_local_scores(
        self,
        local_sources: list[dict | None],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        global_query: torch.Tensor,
        *,
        audit_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
        """Encode/score native tiles sequentially and retain exact global top-k only."""
        present_flat = present[0] if present.ndim == 2 else present
        meta_flat = series_meta[0] if series_meta.ndim == 3 else series_meta
        position_flat = slice_position[0] if slice_position.ndim == 3 else slice_position
        if len(local_sources) != int(present_flat.numel()):
            raise ValueError("B49 local source/present series count mismatch")
        dynamic_pool = self.head.empty_pool()
        static_pool = self.head.empty_pool() if audit_context else None
        context_abs_sum = torch.zeros(N_TARGETS, dtype=torch.float32, device=global_query.device)
        valid_count = 0
        tile_count = 0
        for series_index, (source, flag) in enumerate(zip(local_sources, present_flat)):
            if float(flag.detach().item()) <= 0:
                continue
            if source is None:
                raise RuntimeError("B49 marks a local series present but has no source descriptor")
            normalized = self._source_normalized(source)
            height, width = int(normalized.shape[1]), int(normalized.shape[2])
            layout = native_tile_layout(height, width)
            centres = np.asarray(source["centres"], dtype=np.int64)
            declared_position = torch.tensor(source["slice_positions"], dtype=torch.float32)
            if centres.shape != (B35_DENSE_SLICES,) or declared_position.shape != (B35_DENSE_SLICES,):
                raise RuntimeError("B49 source descriptor does not contain 32 dense centres")
            # Catch a future dataset/model disagreement before it silently changes
            # through-plane position semantics.
            if not torch.allclose(
                declared_position,
                position_flat[series_index].detach().float().cpu(),
                atol=1e-6,
                rtol=0,
            ):
                raise RuntimeError("B49 source and batch slice positions disagree")
            for tile, chosen, cpu_tile in native_tile_triplet_chunks(
                normalized,
                centres,
                layout,
                gap=1,
                chunk_size=self.tile_encoder_chunk_size,
            ):
                device_tile = cpu_tile.to(global_query.device, non_blocking=True)
                use_checkpoint = bool(self.training and self.encoder_trainable_stages > 0)
                if use_checkpoint:
                    fmap = checkpoint(
                        self._encode_native_tile_fmap,
                        device_tile,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    fmap = self._encode_native_tile_fmap(device_tile)
                coordinate, coordinate_valid = tile_feature_coordinates(
                    tile,
                    native_height=height,
                    native_width=width,
                    feature_height=int(fmap.shape[-2]),
                    feature_width=int(fmap.shape[-1]),
                )
                chosen_tensor = torch.as_tensor(chosen, dtype=torch.long, device=fmap.device)
                static, conditioned, valid = self.head.score_tile_features(
                    fmap,
                    slice_positions=position_flat[series_index].index_select(0, chosen_tensor),
                    series_meta=meta_flat[series_index],
                    coordinates=coordinate,
                    coordinate_valid=coordinate_valid,
                    global_query=global_query,
                )
                regions = int(fmap.shape[-2]) * int(fmap.shape[-1])
                token_ids = (
                    int(series_index) * 1_000_000_000_000
                    + chosen_tensor[:, None] * 100_000_000
                    + int(tile.index) * 10_000
                    + torch.arange(regions, dtype=torch.long, device=fmap.device)[None, :]
                ).reshape(-1)
                dynamic_pool = self.head.update_pool(dynamic_pool, conditioned, token_ids)
                if static_pool is not None:
                    static_pool = self.head.update_pool(static_pool, static, token_ids)
                # `conditioned - static` is safe only on valid entries; invalid
                # entries contain -inf by design and do not enter this audit.
                residual = (conditioned[:, valid] - static[:, valid]).float()
                context_abs_sum += residual.abs().sum(dim=-1)
                valid_count += int(valid.sum().item())
                tile_count += 1 if int(chosen[0]) == 0 else 0
                del device_tile, fmap, static, conditioned, valid, token_ids, chosen_tensor
            del normalized
        local_logits, top_indices, top_values = self.head.pooled_logits(dynamic_pool)
        if valid_count < self.head.top_k:
            raise RuntimeError("B49 has fewer native evidence cells than sparse top-k")
        context_abs_mean = (context_abs_sum / float(valid_count))[None, :]
        overlap = None
        if static_pool is not None:
            _static_logits, static_ids, _static_values = self.head.pooled_logits(static_pool)
            overlap = (
                (top_indices[..., :, None] == static_ids[..., None, :])
                .any(dim=-1)
                .float()
                .mean(dim=-1)
            )
        return (
            local_logits,
            top_indices,
            top_values,
            context_abs_mean,
            overlap,
            int(tile_count),
            int(valid_count),
        )

    def forward(
        self,
        context_volumes: list[torch.Tensor],
        local_sources: list[dict | None],
        present: torch.Tensor,
        series_meta: torch.Tensor,
        slice_position: torch.Tensor,
        *,
        audit_context: bool = False,
    ) -> B49Forward:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        if slice_position.ndim == 2:
            slice_position = slice_position.unsqueeze(0)
        global_feature = self._encode_context_study(context_volumes, present)
        base_logits = self._base_logits_from_global(global_feature, present, series_meta)
        context_query = self._select_context_query(global_feature, present, series_meta)
        (
            local_logits,
            top_indices,
            top_values,
            context_abs_mean,
            overlap,
            tile_count,
            valid_tokens,
        ) = self._stream_local_scores(
            local_sources,
            present,
            series_meta,
            slice_position,
            context_query,
            audit_context=bool(audit_context),
        )
        gate = self.head.effective_gate().to(dtype=local_logits.dtype)
        logits = base_logits.float() + gate[None, :] * local_logits.float()
        return B49Forward(
            logits=logits,
            base_logits=base_logits,
            local_logits=local_logits,
            top_indices=top_indices,
            top_values=top_values,
            context_query=context_query,
            context_abs_mean=context_abs_mean,
            topk_overlap_with_static=overlap,
            native_tile_count=tile_count,
            native_valid_token_count=valid_tokens,
        )

    @torch.no_grad()
    def context_reconstruction_error(
        self,
        global_feature: torch.Tensor,
        present: torch.Tensor,
        series_meta: torch.Tensor,
    ) -> float:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        if series_meta.ndim == 2:
            series_meta = series_meta.unsqueeze(0)
        expected = self._base_logits_from_global(global_feature, present, series_meta)
        _static, _post, reconstructed = self._global_query_states(global_feature, present, series_meta)
        return float((expected.float() - reconstructed.float()).abs().max().item())

    def state(self) -> dict:
        state = super().state()
        state.update(
            {
                "version": B49_VERSION,
                "experiment": B49_EXPERIMENT,
                "arm": self.arm,
                "context_source": self.context_source,
                "tile_size": B49_TILE_SIZE,
                "tile_overlap": B49_TILE_OVERLAP,
                "tile_stride": B49_TILE_STRIDE,
                "tile_encoder_chunk_size": self.tile_encoder_chunk_size,
                "local_preprocessing": B49_LOCAL_PREPROCESSING,
                "global_context_preprocessing": B49_GLOBAL_CONTEXT_PREPROCESSING,
            }
        )
        return state


def b49_preprocessing_state() -> dict:
    return {
        "normalization": "full native volume p1-p99 before both branches",
        "local_branch": {
            "crop_fraction": 1.0,
            "centre_crop": False,
            "inplane_resize": False,
            "policy": B49_LOCAL_PREPROCESSING,
            "tile_size": B49_TILE_SIZE,
            "tile_overlap": B49_TILE_OVERLAP,
            "tile_stride": B49_TILE_STRIDE,
            "padding": B49_TILE_PADDING,
            "overlap_evidence_policy": "midpoint ownership; each native pixel centre has one owner",
            "evidence_precision": B49_EVIDENCE_PRECISION,
            "slice_centres": B35_DENSE_SLICES,
        },
        "global_context_branch": {
            "crop_fraction": 1.0,
            "centre_crop": False,
            "inplane_resize": True,
            "policy": B49_GLOBAL_CONTEXT_PREPROCESSING,
            "reference_pixel_area": B49_GLOBAL_CONTEXT_REFERENCE_AREA,
            "resize": "bilinear antialias=True align_corners=False",
            "slice_centres": B35_BASE_SLICES,
        },
    }


def require_b49_contract(config: dict, *, arm: str) -> dict:
    """Freeze B49's representation, matched mechanism and training controls."""
    if arm not in B49_ARMS:
        raise ValueError(f"B49 arm must be one of {B49_ARMS}; got {arm!r}")
    expected_int = {
        "b7_image_size": B37_IMAGE_SIZE,
        "b7_n_slices": B35_BASE_SLICES,
        "b7_triplet_gap": 1,
        "b37_grid_size": B37_GRID_SIZE,
        "b37_top_k": B37_TOP_K,
        "b37_encoder_trainable_stages": B37_ENCODER_TRAINABLE_STAGES,
        "b37_encoder_chunk_size": B37_ENCODER_CHUNK_SIZE,
        "b49_tile_size": B49_TILE_SIZE,
        "b49_tile_overlap": B49_TILE_OVERLAP,
        "b49_tile_stride": B49_TILE_STRIDE,
        "b49_tile_encoder_chunk_size": B49_TILE_ENCODER_CHUNK_SIZE,
        "b49_context_dim": B49_CONTEXT_DIM,
        "b49_fixed_epochs": B49_FIXED_EPOCHS,
        "b42_effective_batch": B42_EFFECTIVE_BATCH,
    }
    for key, frozen in expected_int.items():
        value = int(config.get(key, frozen))
        if value != frozen:
            raise ValueError(f"B49 freezes {key}={frozen}; got {value}")
    expected_float = {
        "b37_temperature": B37_TEMPERATURE,
        "b37_local_aux_weight": B37_LOCAL_AUX_WEIGHT,
        "b37_encoder_lr_scale": B37_ENCODER_LR_SCALE,
        "b49_local_crop_fraction": 1.0,
        "b49_global_context_crop_fraction": 1.0,
    }
    for key, frozen in expected_float.items():
        value = float(config.get(key, frozen))
        if not np.isclose(value, frozen, atol=1e-12, rtol=0):
            raise ValueError(f"B49 freezes {key}={frozen}; got {value}")
    expected_text = {
        "b49_local_preprocessing": B49_LOCAL_PREPROCESSING,
        "b49_global_context_preprocessing": B49_GLOBAL_CONTEXT_PREPROCESSING,
        "b49_tile_padding": B49_TILE_PADDING,
        "b49_evidence_precision": B49_EVIDENCE_PRECISION,
        "b49_context_metric": B49_CONTEXT_METRIC,
        "b49_context_gate_init": B49_CONTEXT_GATE_INIT,
        "b49_context_query_gradient": B49_CONTEXT_QUERY_GRADIENT,
        "b49_supervision": B49_SUPERVISION,
        "b49_validation_surface": B49_VALIDATION_SURFACE,
        "b49_checkpoint_selection": "none_fixed_epoch_2",
    }
    for key, frozen in expected_text.items():
        value = str(config.get(key, frozen))
        if value != frozen:
            raise ValueError(f"B49 freezes {key}={frozen!r}; got {value!r}")
    eps = float(config.get("b49_context_eps", B49_CONTEXT_EPS))
    if not np.isclose(eps, B49_CONTEXT_EPS, atol=1e-12, rtol=0):
        raise ValueError(f"B49 freezes b49_context_eps={B49_CONTEXT_EPS}; got {eps}")
    if int(config.get("series_cache_mb_per_worker", 0)) != 0:
        raise ValueError("B49 requires series_cache_mb_per_worker=0")
    return {
        "arm": arm,
        "context_source": B49_ARM_CONTEXT_SOURCE[arm],
        "context_dim": B49_CONTEXT_DIM,
        "tile_size": B49_TILE_SIZE,
        "tile_overlap": B49_TILE_OVERLAP,
        "tile_stride": B49_TILE_STRIDE,
        "evidence_precision": B49_EVIDENCE_PRECISION,
        "supervision": B49_SUPERVISION,
    }


def b49_state(arm: str) -> dict:
    if arm not in B49_ARMS:
        raise ValueError(f"B49 arm must be one of {B49_ARMS}; got {arm!r}")
    return {
        "version": B49_VERSION,
        "experiment": B49_EXPERIMENT,
        "arm": arm,
        "context_source": B49_ARM_CONTEXT_SOURCE[arm],
        "context_metric": B49_CONTEXT_METRIC,
        "context_dim": B49_CONTEXT_DIM,
        "context_eps": B49_CONTEXT_EPS,
        "context_gate_init": B49_CONTEXT_GATE_INIT,
        "context_query_gradient": B49_CONTEXT_QUERY_GRADIENT,
        "tile_size": B49_TILE_SIZE,
        "tile_overlap": B49_TILE_OVERLAP,
        "tile_stride": B49_TILE_STRIDE,
        "local_preprocessing": B49_LOCAL_PREPROCESSING,
        "global_context_preprocessing": B49_GLOBAL_CONTEXT_PREPROCESSING,
        "supervision": B49_SUPERVISION,
        "validation_surface": B49_VALIDATION_SURFACE,
    }


__all__ = [
    "B49_ARMS",
    "B49_ARM_CONTEXT_SOURCE",
    "B49_CONTEXT_DIM",
    "B49_EXPERIMENT",
    "B49_EVIDENCE_PRECISION",
    "B49_FIXED_EPOCHS",
    "B49_GLOBAL_CONTEXT_PREPROCESSING",
    "B49_LOCAL_PREPROCESSING",
    "B49_NUMBERED_CONTAINER",
    "B49_POST_CROSS_ATTENTION_CANDIDATE",
    "B49_RUN_ROOT",
    "B49_STATIC_PRIOR_CONTROL",
    "B49_SUPERVISION",
    "B49_TILE_ENCODER_CHUNK_SIZE",
    "B49_TILE_OVERLAP",
    "B49_TILE_SIZE",
    "B49_TILE_STRIDE",
    "B49_VALIDATION_SURFACE",
    "B49_VERSION",
    "B49Forward",
    "B49NativeTiledFullFOVDataset",
    "B49NativeTiledMILHead",
    "B49NativeTiledMultiscaleMILResidual",
    "NativeTile",
    "b49_preprocessing_state",
    "b49_state",
    "collate_b49",
    "coordinate_basis",
    "full_fov_context_from_normalized",
    "native_tile_layout",
    "native_tile_starts",
    "native_tile_triplet_chunks",
    "require_b49_contract",
    "tile_feature_coordinates",
    "verify_native_tile_coverage",
]
