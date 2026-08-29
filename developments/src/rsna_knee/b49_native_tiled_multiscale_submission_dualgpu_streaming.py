"""Hidden-safe dual-T4 inference for the frozen B49 candidate endpoint.

This execution-only companion keeps B49's candidate checkpoint, full-FOV tile
geometry, three centre offsets, model operations and raw sigmoid averaging
fixed.  It shards complete studies across two GPUs, normalizes each native
series once, materializes one TTA context view at a time, and trims released
host arenas after each study while retaining CUDA allocator blocks.

The B49 model source is checksum-validated by the completed checkpoint, so it
is intentionally not edited.  Only after that validation this module installs
a process-local adapter that lets the unchanged local-tile model reuse the
already-normalized native array that produced its context tensor.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b17_submission import _validate_sample_submission, _validate_submission
from .b35_target_spatial_residual import b35_centers
from .b35_training import sha256_file
from .b37_highres_sparse_submission import (
    B37_SUBMISSION_MAX_HOURS,
    B37_SUBMISSION_MIN_RESERVE_MINUTES,
    _submission_budget,
)
from .b49_native_tiled_multiscale_mil import (
    B49_POST_CROSS_ATTENTION_CANDIDATE,
    B49NativeTiledFullFOVDataset,
    B49NativeTiledMultiscaleMILResidual,
    full_fov_context_from_normalized,
    native_tile_layout,
)
from .b49_native_tiled_multiscale_submission import (
    B49_SUBMISSION_EXPERIMENT,
    _verified_candidate,
    require_b49_candidate_submission_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .dicom import find_series_dir
from .kaggle_hidden_streaming_highres import (
    load_normalized_study,
    process_rss_gib,
    trim_host_memory,
)
from .runtime import autocast, resolve_runtime


B49_HIDDEN_SAFE_EXECUTION_VERSION = "b49_hidden_dual_t4_streaming_context_normonce_v1"
B49_HIDDEN_SAFE_GPU_COUNT = 2
B49_HIDDEN_SAFE_TIMING_SAFETY_FACTOR = 1.35
B49_PRELOADED_NORMALIZED_SOURCE_KEY = "_b49_hidden_safe_normalized_volume"

_ORIGINAL_B49_SOURCE_NORMALIZED = B49NativeTiledMultiscaleMILResidual._source_normalized


def preloaded_or_disk_b49_source_normalized(source: dict) -> np.ndarray:
    """Return a geometry-checked preloaded native array or the original result."""
    preloaded = source.get(B49_PRELOADED_NORMALIZED_SOURCE_KEY)
    if preloaded is None:
        return _ORIGINAL_B49_SOURCE_NORMALIZED(source)
    normalized = np.asarray(preloaded, dtype=np.float32)
    if normalized.ndim != 3 or len(normalized) < 1:
        raise ValueError("B49 hidden-safe preloaded source must be [S,H,W]")
    declared = (int(source["native_height"]), int(source["native_width"]))
    if tuple(normalized.shape[1:]) != declared:
        raise RuntimeError(
            "B49 hidden-safe preloaded local geometry changed: "
            f"{tuple(normalized.shape[1:])} != {declared}"
        )
    return normalized


def install_b49_preloaded_source_adapter() -> None:
    """Install the process-local adapter only if no other adapter is present."""
    active = B49NativeTiledMultiscaleMILResidual._source_normalized
    if active is preloaded_or_disk_b49_source_normalized:
        return
    if active is not _ORIGINAL_B49_SOURCE_NORMALIZED:
        raise RuntimeError("B49 hidden-safe source normalization adapter is unexpectedly replaced")
    B49NativeTiledMultiscaleMILResidual._source_normalized = staticmethod(
        preloaded_or_disk_b49_source_normalized
    )


def b49_streamed_study_metadata(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact TTA-invariant study inputs once."""
    if not records:
        raise ValueError("B49 hidden-safe study has no eligible MRI series")
    present = torch.ones(len(records), dtype=torch.float32)
    meta = torch.tensor(
        [
            [int(record["plane_id"]), int(record["fluid_id"]), int(record["fat_id"])]
            for record in records
        ],
        dtype=torch.long,
    )
    return present, meta


def b49_streamed_source_paths(reader, *, uid: str, records: list[dict]) -> list[str]:
    """Resolve the tiny source descriptors used by the unchanged local branch."""
    paths: list[str] = []
    for series_index, record in enumerate(records):
        series_uid = str(record["series_uid"])
        path = find_series_dir(reader.config.data_root, reader.config.split, str(uid), series_uid)
        if path is None:
            raise FileNotFoundError(
                f"B49 hidden-safe missing series: study={uid} series={series_uid} "
                f"series_index={series_index}"
            )
        paths.append(str(Path(path).resolve()))
    return paths


def build_b49_streamed_view(
    normalized_series: list[np.ndarray],
    records: list[dict],
    source_paths: list[str],
    *,
    study_uid: str,
    gap: int,
    center_offset: int,
) -> tuple[list[torch.Tensor], list[dict], torch.Tensor]:
    """Build one exact B49 context view plus local tile source descriptors."""
    if not records or len(normalized_series) != len(records) or len(source_paths) != len(records):
        raise ValueError("B49 hidden-safe normalized-series/record/path surface mismatch")
    if int(gap) != 1:
        raise ValueError("B49 hidden-safe inference freezes the 2.5-D triplet gap at 1")

    context_volumes: list[torch.Tensor] = []
    local_sources: list[dict] = []
    positions = torch.empty((len(records), 32), dtype=torch.float32)
    for series_index, (normalized, record, path) in enumerate(
        zip(normalized_series, records, source_paths)
    ):
        native = np.asarray(normalized, dtype=np.float32)
        if native.ndim != 3 or len(native) < 1:
            raise ValueError(f"B49 hidden-safe series {series_index} must be [S,H,W]")
        height, width = int(native.shape[1]), int(native.shape[2])
        centres, position = b35_centers(
            len(native), gap=int(gap), center_offset=int(center_offset)
        )
        context = full_fov_context_from_normalized(native, centres, gap=int(gap))
        if context.ndim != 4 or int(context.shape[0]) != 16 or int(context.shape[1]) != 3:
            raise RuntimeError(
                f"B49 hidden-safe context shape changed for series {series_index}: {tuple(context.shape)}"
            )
        context_volumes.append(context)
        positions[series_index].copy_(torch.from_numpy(position.astype(np.float32, copy=False)))
        local_sources.append(
            {
                "path": str(path),
                "study_uid": str(study_uid),
                "series_uid": str(record["series_uid"]),
                "native_height": height,
                "native_width": width,
                "centres": [int(value) for value in centres.tolist()],
                "slice_positions": [float(value) for value in position.tolist()],
                "tile_count": len(native_tile_layout(height, width)),
                B49_PRELOADED_NORMALIZED_SOURCE_KEY: native,
            }
        )
    return context_volumes, local_sources, positions


def _verify_replica_payloads(payload0: dict, payload1: dict, domain0: dict, domain1: dict) -> None:
    """Make a two-device execution mismatch fail before test scoring."""
    if domain0 != domain1:
        raise RuntimeError("B49 hidden-safe replicas disagree on the frozen domain split")
    for key in (
        "experiment",
        "version",
        "arm",
        "seed",
        "base_checkpoint_sha256",
        "encoder_sha256_final",
        "config_sha256",
        "source_sha256",
        "matched_pair_identity",
    ):
        if payload0.get(key) != payload1.get(key):
            raise RuntimeError(f"B49 hidden-safe replicas disagree on {key}")
    if str(payload0.get("arm")) != B49_POST_CROSS_ATTENTION_CANDIDATE:
        raise RuntimeError("B49 hidden-safe path loaded a non-candidate checkpoint")


def _infer_b49_hidden_safe_shard(
    *,
    rank: int,
    indices: list[int],
    uids: list[str],
    variable_index: dict,
    model,
    reader,
    offsets: tuple[int, ...],
    runtime,
    global_started: float,
    runtime_hours: float,
    reserve_minutes: float,
) -> tuple[list[tuple[int, str, np.ndarray]], dict]:
    """Infer one complete-study shard with bounded host/TTA lifetime."""
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
                raise RuntimeError(f"B49 hidden-safe row {index} has zero eligible MRI series")
            rss, rss_peak = process_rss_gib()
            print(
                f"[B49 hidden-safe gpu{rank}] start row={index} "
                f"shard={local_position + 1}/{len(indices)} uid={uid} "
                f"series={len(records)} rss={rss:.2f}GiB rss_peak={rss_peak:.2f}GiB",
                flush=True,
            )
            started = time.monotonic()
            normalized_series = load_normalized_study(reader, uid=uid, records=records)
            source_paths = b49_streamed_source_paths(reader, uid=uid, records=records)
            present_cpu, meta_cpu = b49_streamed_study_metadata(records)
            present = present_cpu.to(device, non_blocking=True)
            meta = meta_cpu.to(device, non_blocking=True)
            view_probabilities: list[torch.Tensor] = []

            for center_offset in offsets:
                context_cpu, local_sources, position_cpu = build_b49_streamed_view(
                    normalized_series,
                    records,
                    source_paths,
                    study_uid=uid,
                    gap=int(reader.config.triplet_gap),
                    center_offset=int(center_offset),
                )
                context = [volume.to(device, non_blocking=True) for volume in context_cpu]
                position = position_cpu.to(device, non_blocking=True)
                with autocast(runtime):
                    output = model(context, local_sources, present, meta, position)
                view_probabilities.append(torch.sigmoid(output.logits.float()).cpu())
                del context_cpu, local_sources, position_cpu, context, position, output

            probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
            if tuple(probability.shape) != (1, len(TARGETS)) or not torch.isfinite(probability).all():
                raise RuntimeError(f"B49 hidden-safe row {index} produced invalid probabilities")
            rows.append((int(index), uid, probability.numpy()[0]))
            del (
                normalized_series,
                source_paths,
                present_cpu,
                meta_cpu,
                present,
                meta,
                view_probabilities,
                probability,
            )
            # Keep CUDA allocations cached for the next study.  Trimming host
            # arenas addresses the hidden-run pressure without GPU reallocation.
            trim_host_memory()

            duration = time.monotonic() - started
            durations.append(duration)
            rss, rss_peak = process_rss_gib()
            max_rss = max(max_rss, rss)
            max_rss_peak = max(max_rss_peak, rss_peak)
            window = np.asarray(durations[-5:], dtype=np.float64)
            projected = float(
                window.mean()
                * (len(indices) - local_position - 1)
                * B49_HIDDEN_SAFE_TIMING_SAFETY_FACTOR
            )
            elapsed = time.monotonic() - global_started
            available = float(runtime_hours) * 3600.0 - float(reserve_minutes) * 60.0 - elapsed
            print(
                f"[B49 hidden-safe gpu{rank}] done row={index} seconds={duration:.1f} "
                f"projected_remaining={projected / 60.0:.1f}min "
                f"available={available / 60.0:.1f}min telemetry_only=True "
                f"rss={rss:.2f}GiB "
                f"cuda_alloc={torch.cuda.memory_allocated(device) / (1024**3):.2f}GiB",
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
        "cuda_peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "cuda_peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
    }
    return rows, stats


def generate_b49_candidate_submission_dual_gpu_streaming(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    candidate_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
    manifest_path: str | Path | None = None,
) -> Path:
    """Write the frozen B49 candidate submission using two T4-safe replicas."""
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)
    offsets = require_b49_candidate_submission_contract(settings)
    runtime = resolve_runtime(settings)
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < B49_HIDDEN_SAFE_GPU_COUNT:
        raise RuntimeError(
            f"B49 hidden-safe submission requires two CUDA devices; visible={visible}"
        )
    if runtime.num_workers != 0 or runtime.pin_memory:
        raise RuntimeError("B49 hidden-safe submission requires workers=0 and pin_memory=False")
    print(runtime.describe(), flush=True)
    print(
        "[B49 hidden-safe] using cuda:0 + cuda:1; one TTA context view at a time; "
        f"offsets={list(offsets)}; runtime guard=telemetry only",
        flush=True,
    )

    candidate_path = Path(candidate_checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    domain_path = Path(domain_split).resolve()
    if not candidate_path.is_file() or not base_path.is_file() or not domain_path.is_file():
        raise FileNotFoundError("B49 hidden-safe candidate/base checkpoint or domain split is missing")
    candidate_sha = sha256_file(candidate_path)

    # Load sequentially, and verify original trained-source fingerprints before
    # installing the process-local preloaded-array adapter.
    model0, payload0, domain0 = _verified_candidate(
        settings=settings,
        root=root,
        labels_root=labels_root,
        base_checkpoint=base_path,
        domain_split=domain_path,
        candidate_checkpoint=candidate_path,
        device=torch.device("cuda:0"),
    )
    model1, payload1, domain1 = _verified_candidate(
        settings=settings,
        root=root,
        labels_root=labels_root,
        base_checkpoint=base_path,
        domain_split=domain_path,
        candidate_checkpoint=candidate_path,
        device=torch.device("cuda:1"),
    )
    _verify_replica_payloads(payload0, payload1, domain0, domain1)
    model0.eval()
    model1.eval()
    install_b49_preloaded_source_adapter()

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("B49 hidden-safe test.csv has no studies")
    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    missing = [uid for uid, count in zip(uids, counts) if count == 0]
    if missing:
        raise ValueError(
            f"B49 hidden-safe found {len(missing)} test study/studies with no eligible MRI series"
        )

    dataset_config = make_b7_dataset_config(settings, root, train=False)
    dataset_config.split = "test"
    dataset_config.tta_center_offsets = ()
    readers = [
        B49NativeTiledFullFOVDataset(
            uids, variable_index, dataset_config, center_offsets=(0,)
        )
        for _ in range(B49_HIDDEN_SAFE_GPU_COUNT)
    ]
    shards = [
        list(range(rank, len(uids), B49_HIDDEN_SAFE_GPU_COUNT))
        for rank in range(B49_HIDDEN_SAFE_GPU_COUNT)
    ]
    budget = _submission_budget(
        settings,
        max_hours=B37_SUBMISSION_MAX_HOURS,
        min_reserve_minutes=B37_SUBMISSION_MIN_RESERVE_MINUTES,
    )
    global_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=B49_HIDDEN_SAFE_GPU_COUNT) as pool:
        futures = [
            pool.submit(
                _infer_b49_hidden_safe_shard,
                rank=rank,
                indices=shards[rank],
                uids=uids,
                variable_index=variable_index,
                model=(model0 if rank == 0 else model1),
                reader=readers[rank],
                offsets=offsets,
                runtime=runtime,
                global_started=global_started,
                runtime_hours=float(budget.max_hours),
                reserve_minutes=float(budget.reserve_minutes),
            )
            for rank in range(B49_HIDDEN_SAFE_GPU_COUNT)
        ]
        rows: list[tuple[int, str, np.ndarray]] = []
        worker_stats: list[dict] = []
        for future in futures:
            shard_rows, stats = future.result()
            rows.extend(shard_rows)
            worker_stats.append(stats)

    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(uids))):
        raise RuntimeError("B49 hidden-safe output row indices are incomplete or duplicated")
    if [row[1] for row in rows] != uids:
        raise RuntimeError("B49 hidden-safe changed test.csv StudyInstanceUID order")
    probabilities = np.stack([row[2] for row in rows], axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output = Path(out_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    elapsed_hours = (time.monotonic() - global_started) / 3600.0
    manifest = {
        "submission_role": "exploratory_candidate_only_posthoc_hidden_test_inference_not_promotion_evidence",
        "experiment": B49_SUBMISSION_EXPERIMENT,
        "execution_version": B49_HIDDEN_SAFE_EXECUTION_VERSION,
        "execution_only_change": True,
        "candidate_arm": B49_POST_CROSS_ATTENTION_CANDIDATE,
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": candidate_sha,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": str(payload0["base_checkpoint_sha256"]),
        "domain_split_sha256": domain0["sha256"],
        "domain_split_rows_sha256": domain0["rows_sha256"],
        "trained_source_sha256_verified_before_adapter": True,
        "trained_source_sha256": payload0["source_sha256"],
        "matched_pair_identity": payload0["matched_pair_identity"],
        "completed_epochs": int(payload0["completed_epochs"]),
        "gpu_count": B49_HIDDEN_SAFE_GPU_COUNT,
        "study_sharding": "test-row index modulo 2",
        "cross_study_batching": False,
        "tta_offsets": list(offsets),
        "prediction_policy": "raw_sigmoid_probabilities; no_thresholding; no_calibration; no_blending",
        "hidden_safe_execution": {
            "native_volume_normalizations_per_series": 1,
            "tta_context_materialization": "one complete study view at a time",
            "all_tta_context_views_materialized": False,
            "normalized_native_series_retained_for_current_study": True,
            "local_tiles": "frozen native 640px tiles in chunks of two",
            "cuda_cache_reused_between_studies": True,
            "host_trim_after_each_study": True,
            "runtime_projection": "telemetry_only_no_exception",
            "strict_dicom": True,
        },
        "worker_stats": worker_stats,
        "test_studies": len(uids),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(elapsed_hours),
        "runtime_budget_hours": float(budget.max_hours),
        "runtime_reserve_minutes": float(budget.reserve_minutes),
        "sample_submission_validation": sample_validation,
        "submission_path": str(output),
        "submission_sha256": sha256_file(output),
        "governance": (
            "Exact completed B49 post-cross-attention candidate. Execution changes only "
            "study-to-GPU sharding and temporary data lifetime: each series is normalized "
            "once, one frozen TTA context view is materialized at a time, and the model "
            "uses the identical normalized native array for its unchanged local tile stream. "
            "No training, checkpoint selection, crop, resize, tile, TTA, threshold, "
            "calibration, blending, or prediction rule changed."
        ),
    }
    manifest_output = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output, flush=True)
    print(manifest_output, flush=True)
    print(f"submission_sha256 {manifest['submission_sha256']}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Write a hidden-safe dual-T4 B49 candidate Kaggle submission")
    parser.add_argument("--config", default="config/b49_native_tiled_multiscale.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    generate_b49_candidate_submission_dual_gpu_streaming(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        candidate_checkpoint=args.candidate_checkpoint,
        out_path=args.out,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B49_HIDDEN_SAFE_EXECUTION_VERSION",
    "B49_PRELOADED_NORMALIZED_SOURCE_KEY",
    "b49_streamed_source_paths",
    "b49_streamed_study_metadata",
    "build_b49_streamed_view",
    "generate_b49_candidate_submission_dual_gpu_streaming",
    "install_b49_preloaded_source_adapter",
    "preloaded_or_disk_b49_source_normalized",
]
