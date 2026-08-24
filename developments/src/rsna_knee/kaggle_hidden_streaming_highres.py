"""Shared hidden-safe streaming execution helpers for B39/B41.

This module changes inference plumbing only.  It never changes a checkpoint,
model operation, crop, resize, TTA offset, sparse-MIL rule, sigmoid transform,
or probability aggregation.

The historical high-resolution datasets materialize every TTA view for every
series in a study before model execution.  Hidden studies can be much larger
than the three-study public sample, so that layout can transiently multiply host
RAM.  The helpers below instead:

1. decode and normalize every native series exactly once;
2. retain only the normalized native arrays for the current study;
3. construct one complete TTA view across all series;
4. run the frozen model for that view;
5. release the resized view before constructing the next one.

The per-view preprocessing functions are intentionally factored from the exact
existing B39/B41 fast helpers and are unit-tested for exact tensor equality.
"""
from __future__ import annotations

import ctypes
import gc
import resource
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .b35_target_spatial_residual import B35_DENSE_SLICES, b35_centers
from .b37_highres_sparse_mil import (
    B37_IMAGE_SIZE,
    _native_center_crop as _b37_native_center_crop,
)
from .b41_highres_aspect_sparse_mil import (
    B41_IMAGE_SIZE,
    B41_PAD_VALUE,
    _native_center_crop as _b41_native_center_crop,
    resize_triplets_aspect_preserving_pad,
)
from .dicom import _normalise_volume, find_series_dir


def process_rss_gib() -> tuple[float, float]:
    """Return current and peak process RSS in GiB without optional dependencies."""
    current_kib = 0.0
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                current_kib = float(line.split()[1])
                break
    except OSError:
        current_kib = 0.0
    peak_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return current_kib / (1024.0**2), peak_kib / (1024.0**2)


def trim_host_memory() -> None:
    """Release Python cycles and return free glibc arenas when supported."""
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def normalized_view_b39(
    normalized: np.ndarray,
    *,
    gap: int,
    center_offset: int,
    crop_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one exact B39/B37 448x448 view from an already-normalized volume."""
    gap = int(gap)
    if gap < 1:
        raise ValueError("B39 streaming 2.5D gap must be positive")
    centers, position = b35_centers(
        len(normalized),
        gap=gap,
        center_offset=int(center_offset),
    )
    triplet_offsets = np.asarray([-gap, 0, gap], dtype=np.int64)
    index = np.clip(
        centers[:, None] + triplet_offsets[None, :],
        0,
        len(normalized) - 1,
    )
    triplets = normalized[index].astype(np.float32, copy=False)
    cropped = _b37_native_center_crop(triplets, float(crop_fraction))
    tensor = torch.from_numpy(np.ascontiguousarray(cropped))
    image = F.interpolate(
        tensor,
        size=(B37_IMAGE_SIZE, B37_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return image, torch.from_numpy(position)


def normalized_view_b41(
    normalized: np.ndarray,
    *,
    gap: int,
    center_offset: int,
    crop_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one exact B41 aspect-preserving padded view from normalized native data."""
    gap = int(gap)
    if gap < 1:
        raise ValueError("B41 streaming 2.5D gap must be positive")
    centers, position = b35_centers(
        len(normalized),
        gap=gap,
        center_offset=int(center_offset),
    )
    triplet_offsets = np.asarray([-gap, 0, gap], dtype=np.int64)
    index = np.clip(
        centers[:, None] + triplet_offsets[None, :],
        0,
        len(normalized) - 1,
    )
    triplets = normalized[index].astype(np.float32, copy=False)
    cropped = _b41_native_center_crop(triplets, float(crop_fraction))
    image = resize_triplets_aspect_preserving_pad(
        cropped,
        image_size=B41_IMAGE_SIZE,
        pad_value=B41_PAD_VALUE,
    )
    return image, torch.from_numpy(position)


def load_normalized_study(
    reader,
    *,
    uid: str,
    records: list[dict],
) -> list[np.ndarray]:
    """Decode each eligible series once and retain only its normalized native array."""
    normalized: list[np.ndarray] = []
    for series_index, record in enumerate(records):
        series_uid = str(record["series_uid"])
        plane = str(record["plane"])
        path = find_series_dir(
            reader.config.data_root,
            reader.config.split,
            str(uid),
            series_uid,
        )
        if path is None:
            raise FileNotFoundError(
                f"hidden streaming missing series: study={uid} series={series_uid} "
                f"series_index={series_index}"
            )
        try:
            raw = reader._read_volume(path, plane.lower())
            norm = _normalise_volume(raw)
        except Exception as exc:
            raise RuntimeError(
                f"hidden streaming DICOM failure: study={uid} series={series_uid} "
                f"series_index={series_index} plane={plane}: {type(exc).__name__}: {exc}"
            ) from exc
        normalized.append(np.asarray(norm, dtype=np.float32))
        del raw, norm
    return normalized


def build_streamed_view(
    normalized_series: list[np.ndarray],
    records: list[dict],
    *,
    gap: int,
    center_offset: int,
    crop_fraction: float,
    preprocess_view: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize exactly one [K,32,3,448,448] study view and its metadata.

    The destination tensor is preallocated and filled one series at a time.  This
    avoids retaining K individual resized tensors and then allocating a second
    K-series copy through ``torch.stack``.
    """
    if len(normalized_series) != len(records) or not records:
        raise ValueError("streaming normalized-series/record surface mismatch")
    count = len(records)
    volume = torch.empty(
        count,
        B35_DENSE_SLICES,
        3,
        B37_IMAGE_SIZE,
        B37_IMAGE_SIZE,
        dtype=torch.float32,
    )
    position = torch.empty(count, B35_DENSE_SLICES, dtype=torch.float32)
    meta = torch.empty(count, 3, dtype=torch.long)
    for series_index, (normalized, record) in enumerate(zip(normalized_series, records)):
        image, series_position = preprocess_view(
            normalized,
            gap=int(gap),
            center_offset=int(center_offset),
            crop_fraction=float(crop_fraction),
        )
        expected = (B35_DENSE_SLICES, 3, B37_IMAGE_SIZE, B37_IMAGE_SIZE)
        if tuple(image.shape) != expected or tuple(series_position.shape) != (B35_DENSE_SLICES,):
            raise RuntimeError(
                f"streamed view shape changed for series_index={series_index}: "
                f"image={tuple(image.shape)} position={tuple(series_position.shape)}"
            )
        volume[series_index].copy_(image)
        position[series_index].copy_(series_position)
        meta[series_index] = torch.tensor(
            [
                int(record["plane_id"]),
                int(record["fluid_id"]),
                int(record["fat_id"]),
            ],
            dtype=torch.long,
        )
        del image, series_position
    present = torch.ones(count, dtype=torch.float32)
    return volume, position, present, meta


def infer_streaming_shard(
    *,
    endpoint_name: str,
    rank: int,
    indices: list[int],
    uids: list[str],
    variable_index: dict,
    model,
    reader,
    tta_offsets: tuple[int, ...],
    gap: int,
    crop_fraction: float,
    preprocess_view: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    global_started: float,
    runtime_hours: float,
    reserve_minutes: float,
    timing_safety_factor: float,
) -> tuple[list[tuple[int, str, np.ndarray]], dict]:
    """Infer one hidden shard with one materialized TTA view and no time-based abort."""
    device = torch.device(f"cuda:{int(rank)}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    rows: list[tuple[int, str, np.ndarray]] = []
    durations: list[float] = []
    max_rss = max_rss_peak = 0.0

    with torch.inference_mode():
        for local_position, index in enumerate(indices):
            uid = str(uids[index])
            records = variable_index.get(uid, [])
            if not records:
                raise RuntimeError(f"{endpoint_name} hidden row {index} has zero MRI series")
            rss, rss_peak = process_rss_gib()
            print(
                f"[{endpoint_name} hidden-safe gpu{rank}] start row={index} "
                f"shard={local_position + 1}/{len(indices)} uid={uid} "
                f"series={len(records)} rss={rss:.2f}GiB rss_peak={rss_peak:.2f}GiB",
                flush=True,
            )
            started = time.monotonic()
            normalized_series = load_normalized_study(reader, uid=uid, records=records)
            view_probabilities: list[torch.Tensor] = []
            for center_offset in tta_offsets:
                volume_cpu, position_cpu, present_cpu, meta_cpu = build_streamed_view(
                    normalized_series,
                    records,
                    gap=int(gap),
                    center_offset=int(center_offset),
                    crop_fraction=float(crop_fraction),
                    preprocess_view=preprocess_view,
                )
                volume = volume_cpu.unsqueeze(0).to(device, non_blocking=True)
                position = position_cpu.unsqueeze(0).to(device, non_blocking=True)
                present = present_cpu.unsqueeze(0).to(device, non_blocking=True)
                series_meta = meta_cpu.unsqueeze(0).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(volume, present, series_meta, position)
                view_probabilities.append(torch.sigmoid(output.logits.float()).cpu())
                del (
                    volume_cpu,
                    position_cpu,
                    present_cpu,
                    meta_cpu,
                    volume,
                    position,
                    present,
                    series_meta,
                    output,
                )

            probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
            if probability.shape[-1] <= 0 or not torch.isfinite(probability).all():
                raise RuntimeError(f"{endpoint_name} hidden row {index} produced invalid probabilities")
            rows.append((int(index), uid, probability.numpy()[0]))
            del normalized_series, view_probabilities, probability
            trim_host_memory()

            duration = time.monotonic() - started
            durations.append(duration)
            rss, rss_peak = process_rss_gib()
            max_rss = max(max_rss, rss)
            max_rss_peak = max(max_rss_peak, rss_peak)

            # Telemetry only.  The former B39/B41 code raised RuntimeError here,
            # which could fail a hidden rerun after an unusually slow early study.
            window = np.asarray(durations[-5:], dtype=np.float64)
            projected = float(
                window.mean()
                * (len(indices) - local_position - 1)
                * timing_safety_factor
            )
            elapsed = time.monotonic() - global_started
            available = (
                float(runtime_hours) * 3600.0
                - float(reserve_minutes) * 60.0
                - elapsed
            )
            print(
                f"[{endpoint_name} hidden-safe gpu{rank}] done row={index} "
                f"seconds={duration:.1f} projected_remaining={projected/60.0:.1f}min "
                f"available={available/60.0:.1f}min telemetry_only=True "
                f"rss={rss:.2f}GiB "
                f"cuda_alloc={torch.cuda.memory_allocated(device)/(1024**3):.2f}GiB",
                flush=True,
            )

    stats = {
        "rank": int(rank),
        "studies": int(len(indices)),
        "elapsed_seconds": float(sum(durations)),
        "mean_study_seconds": float(np.mean(durations)) if durations else 0.0,
        "max_study_seconds": float(np.max(durations)) if durations else 0.0,
        "max_observed_rss_gib": float(max_rss),
        "process_rss_peak_gib": float(max_rss_peak),
        "cuda_peak_allocated_gib": float(
            torch.cuda.max_memory_allocated(device) / (1024**3)
        ),
        "cuda_peak_reserved_gib": float(
            torch.cuda.max_memory_reserved(device) / (1024**3)
        ),
    }
    return rows, stats


__all__ = [
    "build_streamed_view",
    "infer_streaming_shard",
    "load_normalized_study",
    "normalized_view_b39",
    "normalized_view_b41",
    "process_rss_gib",
    "trim_host_memory",
]
