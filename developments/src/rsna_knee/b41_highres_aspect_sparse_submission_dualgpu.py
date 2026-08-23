"""Dual-GPU hidden-test inference for the frozen B41 endpoint.

This module changes execution infrastructure only.  The B41 checkpoint,
preprocessing, three centre-offset views, sparse-MIL head and probability
averaging are unchanged.  Kaggle exposes two T4 GPUs for this competition, so
studies are deterministically sharded by test-row index across two identical
model replicas.  Each worker processes one complete study at a time on one GPU;
there is no cross-study batching and therefore no change to model semantics.
"""
from __future__ import annotations

import gc
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
from .b37_highres_sparse_eval import load_b37_checkpoint
from .b37_highres_sparse_submission import (
    B37_SUBMISSION_MAX_HOURS,
    _b37_test_dataset_config,
    _submission_budget,
    projected_remaining_seconds,
)
from .b41_highres_aspect_sparse_mil import (
    B41_EXPERIMENT,
    B41_VERSION,
    B41HighResAspectSparseDataset,
    b41_preprocessing_state,
)
from .b41_highres_aspect_sparse_submission import (
    B41_SUBMISSION_EXPERIMENT,
    B41_SUBMISSION_MIN_RESERVE_MINUTES,
    B41_SUBMISSION_TTA_OFFSETS,
    _require_b41_checkpoint_contract,
    generate_b41_submission,
    require_b41_submission_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .runtime import resolve_runtime

B41_DUALGPU_EXECUTION_VERSION = "b41_hidden_dual_t4_study_shards_v1"
B41_DUALGPU_COUNT = 2
B41_DUALGPU_TIMING_SAFETY_FACTOR = 1.35


def _release_worker_memory(device: torch.device) -> None:
    """Release one completed study without touching model parameters."""
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _load_replica(
    checkpoint_path: Path,
    base_path: Path,
    device: torch.device,
):
    """Load and verify one byte-identical B41 model replica on one GPU."""
    model, payload = load_b37_checkpoint(
        checkpoint_path,
        base_checkpoint=base_path,
        device=device,
        expected_version=B41_VERSION,
        expected_experiment=B41_EXPERIMENT,
        checkpoint_label="B41",
    )
    _require_b41_checkpoint_contract(payload)
    model.eval()
    return model, payload


def _infer_shard(
    *,
    rank: int,
    indices: list[int],
    model,
    dataset,
    global_started: float,
    max_hours: float,
    reserve_minutes: float,
) -> list[tuple[int, str, np.ndarray]]:
    """Run one deterministic test-row shard on exactly one CUDA device."""
    device = torch.device(f"cuda:{int(rank)}")
    torch.cuda.set_device(device)
    durations: list[float] = []
    rows: list[tuple[int, str, np.ndarray]] = []

    with torch.inference_mode():
        for local_position, index in enumerate(indices):
            if durations:
                remaining_local = len(indices) - local_position
                projected = projected_remaining_seconds(
                    durations,
                    remaining_studies=remaining_local,
                    safety_factor=B41_DUALGPU_TIMING_SAFETY_FACTOR,
                )
                elapsed = time.monotonic() - global_started
                available = (
                    float(max_hours) * 3600.0
                    - float(reserve_minutes) * 60.0
                    - elapsed
                )
                if projected > available:
                    raise RuntimeError(
                        f"B41 dual-GPU shard {rank} cannot finish inside the "
                        f"remaining runtime budget: projected={projected/60.0:.1f} "
                        f"min available={available/60.0:.1f} min"
                    )

            started = time.monotonic()
            item = dataset[index]
            volumes = item["volumes"]
            position = item["slice_position"]
            if volumes.ndim != 6 or int(volumes.shape[0]) != len(B41_SUBMISSION_TTA_OFFSETS):
                raise RuntimeError(
                    f"B41 dual-GPU TTA volume shape changed for row {index}: "
                    f"{tuple(volumes.shape)}"
                )
            if position.ndim != 3 or int(position.shape[0]) != len(B41_SUBMISSION_TTA_OFFSETS):
                raise RuntimeError("B41 dual-GPU slice-position TTA shape changed")

            present = item["present"].unsqueeze(0).to(device, non_blocking=True)
            series_meta = item["series_meta"].unsqueeze(0).to(device, non_blocking=True)
            view_probabilities = []
            for view in range(int(volumes.shape[0])):
                image = volumes[view].unsqueeze(0).to(device, non_blocking=True)
                pos = position[view].unsqueeze(0).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(image, present, series_meta, pos)
                view_probabilities.append(torch.sigmoid(output.logits.float()).cpu())
                del image, pos, output

            probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
            if not torch.isfinite(probability).all():
                raise RuntimeError(
                    f"B41 dual-GPU produced non-finite probabilities for row {index}"
                )
            uid = str(item["study_uid"])
            rows.append((int(index), uid, probability.numpy()[0]))

            del item, volumes, position, present, series_meta, view_probabilities, probability
            _release_worker_memory(device)
            durations.append(time.monotonic() - started)

            completed = local_position + 1
            if completed % 10 == 0 or completed == len(indices):
                remaining = projected_remaining_seconds(
                    durations,
                    remaining_studies=len(indices) - completed,
                    safety_factor=B41_DUALGPU_TIMING_SAFETY_FACTOR,
                )
                print(
                    f"[B41 dual submit gpu{rank}] {completed}/{len(indices)} "
                    f"shard elapsed={sum(durations)/60.0:.1f} min "
                    f"estimated_remaining={remaining/60.0:.1f} min",
                    flush=True,
                )
    return rows


def generate_b41_submission_dual_gpu(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Run the exact B41 endpoint using two independent Kaggle T4 study shards."""
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)
    crop_policy = require_b41_submission_contract(settings)

    runtime = resolve_runtime(settings)
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < B41_DUALGPU_COUNT:
        print(
            f"[B41 dual submit] only {visible} CUDA device(s) visible; "
            "falling back to the audited single-GPU path",
            flush=True,
        )
        output = generate_b41_submission(
            settings,
            data_root=root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=out_path,
        )
        if output is None:
            raise RuntimeError("B41 single-GPU fallback unexpectedly returned no output")
        return output

    if runtime.num_workers != 0 or runtime.pin_memory:
        raise RuntimeError("B41 dual-GPU submission still requires workers=0 and pin_memory=False")
    print(runtime.describe(), flush=True)
    print(
        f"[B41 dual submit] using cuda:0 + cuda:1; exact TTA offsets="
        f"{list(B41_SUBMISSION_TTA_OFFSETS)}",
        flush=True,
    )

    checkpoint_path = Path(checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B41 checkpoint is missing: {checkpoint_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"B41 base checkpoint is missing: {base_path}")

    # Load replicas sequentially so checkpoint deserialization cannot compete for host RAM.
    model0, payload0 = _load_replica(checkpoint_path, base_path, torch.device("cuda:0"))
    model1, payload1 = _load_replica(checkpoint_path, base_path, torch.device("cuda:1"))
    if payload0.get("encoder_sha256_final") != payload1.get("encoder_sha256_final"):
        raise RuntimeError("B41 dual-GPU replicas do not describe the same encoder endpoint")

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("test.csv contains no studies")
    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    missing = [uid for uid, count in zip(uids, counts) if count == 0]
    if missing:
        raise ValueError(
            f"B41 dual-GPU submission found {len(missing)} test study/studies "
            "with zero eligible MRI series"
        )

    dataset_config = _b37_test_dataset_config(settings, root)
    # Independent no-cache dataset instances avoid shared mutable state between workers.
    datasets = [
        B41HighResAspectSparseDataset(
            uids,
            variable_index,
            dataset_config,
            crop_focus_policy=crop_policy,
            center_offsets=B41_SUBMISSION_TTA_OFFSETS,
        )
        for _ in range(B41_DUALGPU_COUNT)
    ]

    shards = [
        list(range(rank, len(uids), B41_DUALGPU_COUNT))
        for rank in range(B41_DUALGPU_COUNT)
    ]
    budget = _submission_budget(
        settings,
        max_hours=B37_SUBMISSION_MAX_HOURS,
        min_reserve_minutes=B41_SUBMISSION_MIN_RESERVE_MINUTES,
    )
    global_started = time.monotonic()

    with ThreadPoolExecutor(max_workers=B41_DUALGPU_COUNT) as pool:
        futures = [
            pool.submit(
                _infer_shard,
                rank=rank,
                indices=shards[rank],
                model=(model0 if rank == 0 else model1),
                dataset=datasets[rank],
                global_started=global_started,
                max_hours=budget.max_hours,
                reserve_minutes=budget.reserve_minutes,
            )
            for rank in range(B41_DUALGPU_COUNT)
        ]
        rows = []
        for future in futures:
            rows.extend(future.result())

    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(uids))):
        raise RuntimeError("B41 dual-GPU output row indices are incomplete or duplicated")
    uid_rows = [row[1] for row in rows]
    if uid_rows != uids:
        raise RuntimeError("B41 dual-GPU submission changed StudyInstanceUID order")

    probabilities = np.stack([row[2] for row in rows], axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    base_checkpoint_sha256 = sha256_file(base_path)
    elapsed_hours = (time.monotonic() - global_started) / 3600.0
    manifest = {
        "experiment": B41_SUBMISSION_EXPERIMENT,
        "version": B41_VERSION,
        "execution_version": B41_DUALGPU_EXECUTION_VERSION,
        "execution_only_change": True,
        "gpu_count": B41_DUALGPU_COUNT,
        "study_sharding": "test-row index modulo 2",
        "cross_study_batching": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_base_sha256_verified": (
            base_checkpoint_sha256 == str(payload0.get("base_checkpoint_sha256", ""))
        ),
        "fixed_endpoint": True,
        "completed_epochs": int(payload0.get("completed_epochs", -1)),
        "prediction": "frozen B41 combined sparse-MIL logits; raw sigmoid probability",
        "thresholding_used": False,
        "blending_used": False,
        "preprocessing": b41_preprocessing_state(),
        "crop_policy": crop_policy,
        "sparse_mil": payload0.get("sparse_mil"),
        "tta_center_offsets": list(B41_SUBMISSION_TTA_OFFSETS),
        "tta_aggregation": "mean of per-view sigmoid probabilities",
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "workers_per_gpu": 0,
        "pin_memory": False,
        "strict_dicom": True,
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(elapsed_hours),
        "runtime_budget_hours": float(budget.max_hours),
        "runtime_reserve_minutes": float(budget.reserve_minutes),
        "submission_sha256": sha256_file(output_path),
        "governance": (
            "Exact frozen B41 fixed-E2 endpoint. Dual-GPU study sharding changes "
            "execution wall time only; preprocessing, checkpoint, TTA, probability "
            "aggregation, sparse-MIL, thresholds and blending are unchanged."
        ),
        **sample_validation,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output_path, flush=True)
    print(manifest_path, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return output_path


__all__ = [
    "B41_DUALGPU_EXECUTION_VERSION",
    "generate_b41_submission_dual_gpu",
]
