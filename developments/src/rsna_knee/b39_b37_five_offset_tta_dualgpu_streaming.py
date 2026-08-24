"""Hidden-safe dual-T4 streaming inference for the frozen B39 endpoint.

B39 remains the exact B37 fixed-E2 checkpoint averaged over the prospectively
frozen five offsets [-2,-1,0,1,2].  This module changes execution only after both
B39 and B41 passed the visible sample notebook but Kaggle reported a hidden
``Notebook Threw Exception`` rerun.

The hidden-safe path normalizes each native series once, retains only normalized
native arrays for the current study, materializes one 448x448 TTA view at a time,
keeps CUDA allocator blocks cached, trims host allocations after each study, and
turns the previous proactive runtime exception into telemetry only.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .b12_variable_series import build_variable_series_index
from .b17_submission import _validate_sample_submission, _validate_submission
from .b35_training import sha256_file
from .b37_highres_sparse_mil import B37_VERSION, B37HighResSparseDataset
from .b37_highres_sparse_submission import _b37_test_dataset_config, _submission_budget
from .b37_highres_sparse_training import B37_EXPERIMENT
from .b39_b37_five_offset_tta import (
    B39_RUNTIME_BUDGET_HOURS,
    B39_RUNTIME_RESERVE_MINUTES,
    B39_SUBMISSION_EXPERIMENT,
    B39_TIMING_SAFETY_FACTOR,
    B39_TTA_OFFSETS,
    B39_VERSION,
    require_b39_five_offset_contract,
)
from . import b39_b37_five_offset_tta_dualgpu as _legacy
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .kaggle_hidden_streaming_highres import infer_streaming_shard, normalized_view_b39
from .runtime import resolve_runtime

B39_STREAMING_EXECUTION_VERSION = "b39_hidden_dual_t4_streaming_views_normonce_noabort_v5"
B39_STREAMING_GPU_COUNT = 2


def generate_b39_submission_dual_gpu_streaming(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Generate exact frozen B39 probabilities with hidden-safe view streaming."""
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)
    crop_policy = require_b39_five_offset_contract(settings)

    runtime = resolve_runtime(settings)
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < B39_STREAMING_GPU_COUNT:
        raise RuntimeError(
            f"B39 hidden-safe streaming requires two CUDA devices; visible={visible}"
        )
    if runtime.num_workers != 0 or runtime.pin_memory:
        raise RuntimeError("B39 hidden-safe streaming requires workers=0 and pin_memory=False")
    print(runtime.describe(), flush=True)
    print(
        f"[B39 hidden-safe] using cuda:0 + cuda:1; one TTA view at a time; "
        f"offsets={list(B39_TTA_OFFSETS)}; runtime guard=telemetry only",
        flush=True,
    )

    checkpoint_path = Path(checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    if not checkpoint_path.is_file() or not base_path.is_file():
        raise FileNotFoundError("B39 hidden-safe checkpoint/base checkpoint is missing")
    checkpoint_sha = sha256_file(checkpoint_path)
    base_sha = sha256_file(base_path)

    model0, payload0 = _legacy._load_replica(checkpoint_path, base_path, torch.device("cuda:0"))
    model1, payload1 = _legacy._load_replica(checkpoint_path, base_path, torch.device("cuda:1"))
    if payload0.get("encoder_sha256_final") != payload1.get("encoder_sha256_final"):
        raise RuntimeError("B39 hidden-safe replicas do not share the same encoder endpoint")
    if payload0.get("experiment") != B37_EXPERIMENT or payload0.get("version") != B37_VERSION:
        raise RuntimeError("B39 hidden-safe source checkpoint is not the frozen B37 endpoint")

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("B39 hidden-safe test.csv contains no studies")
    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    missing = [uid for uid, count in zip(uids, counts) if count == 0]
    if missing:
        raise ValueError(
            f"B39 hidden-safe found {len(missing)} test study/studies with zero eligible MRI series"
        )

    dataset_config = _b37_test_dataset_config(settings, root)
    readers = [
        B37HighResSparseDataset(
            uids,
            variable_index,
            dataset_config,
            crop_focus_policy=crop_policy,
            center_offsets=(0,),
        )
        for _ in range(B39_STREAMING_GPU_COUNT)
    ]
    shards = [list(range(rank, len(uids), B39_STREAMING_GPU_COUNT)) for rank in range(2)]
    budget = _submission_budget(
        settings,
        max_hours=B39_RUNTIME_BUDGET_HOURS,
        min_reserve_minutes=B39_RUNTIME_RESERVE_MINUTES,
    )
    global_started = time.monotonic()

    with ThreadPoolExecutor(max_workers=B39_STREAMING_GPU_COUNT) as pool:
        futures = [
            pool.submit(
                infer_streaming_shard,
                endpoint_name="B39",
                rank=rank,
                indices=shards[rank],
                uids=uids,
                variable_index=variable_index,
                model=(model0 if rank == 0 else model1),
                reader=readers[rank],
                tta_offsets=tuple(B39_TTA_OFFSETS),
                gap=int(dataset_config.triplet_gap),
                crop_fraction=float(crop_policy["crop_fraction"]),
                preprocess_view=normalized_view_b39,
                global_started=global_started,
                runtime_hours=float(budget.max_hours),
                reserve_minutes=float(budget.reserve_minutes),
                timing_safety_factor=float(B39_TIMING_SAFETY_FACTOR),
            )
            for rank in range(B39_STREAMING_GPU_COUNT)
        ]
        rows: list[tuple[int, str, np.ndarray]] = []
        worker_stats: list[dict] = []
        for future in futures:
            shard_rows, stats = future.result()
            rows.extend(shard_rows)
            worker_stats.append(stats)

    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(uids))):
        raise RuntimeError("B39 hidden-safe output row indices are incomplete or duplicated")
    if [row[1] for row in rows] != uids:
        raise RuntimeError("B39 hidden-safe changed StudyInstanceUID order")

    probabilities = np.stack([row[2] for row in rows], axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    elapsed_hours = (time.monotonic() - global_started) / 3600.0
    manifest = {
        "experiment": B39_SUBMISSION_EXPERIMENT,
        "version": B39_VERSION,
        "execution_version": B39_STREAMING_EXECUTION_VERSION,
        "execution_only_change": True,
        "gpu_count": 2,
        "study_sharding": "test-row index modulo 2",
        "cross_study_batching": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_sha,
        "checkpoint_base_sha256_verified": base_sha == str(payload0.get("base_checkpoint_sha256", "")),
        "fixed_endpoint": True,
        "completed_epochs": int(payload0.get("completed_epochs", -1)),
        "prediction": "frozen B37/B39 combined sparse-MIL logits; raw sigmoid probability",
        "thresholding_used": False,
        "blending_used": False,
        "tta_center_offsets": list(B39_TTA_OFFSETS),
        "tta_aggregation": "mean of per-view sigmoid probabilities",
        "sparse_mil": payload0.get("sparse_mil"),
        "hidden_safe_execution": {
            "native_volume_normalizations_per_series": 1,
            "tta_materialization": "one complete study view at a time",
            "all_tta_study_tensor_materialized": False,
            "normalized_native_series_retained_for_current_study": True,
            "cuda_cache_reused_between_studies": True,
            "host_trim_after_each_study": True,
            "runtime_projection": "telemetry_only_no_exception",
            "strict_dicom": True,
        },
        "worker_stats": worker_stats,
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(elapsed_hours),
        "runtime_budget_hours": float(budget.max_hours),
        "runtime_reserve_minutes": float(budget.reserve_minutes),
        "submission_sha256": sha256_file(output),
        "governance": (
            "Exact prospective B39 five-offset endpoint. Hidden-safe execution changes "
            "memory lifetime and runtime telemetry only: one TTA view is materialized "
            "at a time after one native normalization per series; frozen B37 checkpoint, "
            "90% native crop, direct 448 resize, offsets [-2,-1,0,1,2], sparse-MIL, "
            "sigmoid averaging, thresholds and blending are unchanged."
        ),
        **sample_validation,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output, flush=True)
    print(manifest_path, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return output


__all__ = [
    "B39_STREAMING_EXECUTION_VERSION",
    "generate_b39_submission_dual_gpu_streaming",
]
