"""Dual-T4 Kaggle inference for the frozen B42 constant-area endpoint.

This launcher changes execution only. The fixed B42 checkpoint, constant-area
native-aspect geometry, ragged per-series encoding, three centre offsets
[-1, 0, +1], sparse-MIL head and probability averaging are unchanged.

Submission-only acceleration is limited to:
1. deterministic study sharding over two independent T4 model replicas;
2. computing the identical native-volume normalization once per series instead
   of three times, then executing each historical crop/resize/pad view separately;
3. retaining CUDA allocator blocks between studies instead of emptying the cache.

The trained encoder execution chunk remains exactly 4.
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
from .b37_highres_sparse_submission import (
    B37_SUBMISSION_MAX_HOURS,
    B37_SUBMISSION_MIN_RESERVE_MINUTES,
    _b37_test_dataset_config,
    _submission_budget,
    projected_remaining_seconds,
)
from .b42_constant_area_aspect_sparse_eval import load_b42_checkpoint
from .b42_constant_area_aspect_sparse_mil import (
    B42_EXPERIMENT,
    B42_VERSION,
    b42_preprocessing_state,
    require_b42_contract,
)
from .b42_kaggle_fast_preprocess import (
    B42_FAST_TTA_OFFSETS,
    B42KaggleNormalizeOnceDataset,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .runtime import resolve_runtime

B42_FROZEN_CHECKPOINT_SHA256 = (
    "399f0b04c818ce767af539e4f33226b6f5d6223a389814f508fd8f84c95afce3"
)
B42_SUBMISSION_EXPERIMENT = "B42_frozen_constant_area_native_aspect_hidden_test_inference"
B42_FAST_EXECUTION_VERSION = "b42_hidden_dual_t4_normonce_chunk4_cache_reuse_v1"
B42_FAST_ENCODER_CHUNK_SIZE = 4
B42_DUALGPU_COUNT = 2
B42_TIMING_SAFETY_FACTOR = 1.35
B42_SUBMISSION_TTA_OFFSETS = B42_FAST_TTA_OFFSETS


def _verify_checkpoint_identity(
    checkpoint_path: Path, expected_sha256: str = B42_FROZEN_CHECKPOINT_SHA256
) -> str:
    """Refuse any checkpoint but the one this run is declared to be using.

    The default pins B42's own frozen endpoint and is what every B42 submission
    uses. A sibling launcher may pin a different, deliberately chosen artefact
    (B51 does), which keeps the guarantee -- a hidden run cannot silently use a
    file nobody named -- while allowing a different endpoint to be submitted.
    """
    observed = sha256_file(checkpoint_path)
    if observed != expected_sha256:
        raise ValueError(
            "B42 hidden submission requires the declared checkpoint: "
            f"expected {expected_sha256}, got {observed}"
        )
    return observed


def _load_replica(
    checkpoint_path: Path,
    base_path: Path,
    device: torch.device,
):
    model, payload = load_b42_checkpoint(
        checkpoint_path,
        base_checkpoint=base_path,
        device=device,
    )
    if payload.get("experiment") != B42_EXPERIMENT or payload.get("version") != B42_VERSION:
        raise ValueError("B42 checkpoint identity changed")
    if int(payload.get("model_state", {}).get("encoder_chunk_size", -1)) != 4:
        raise ValueError("B42 hidden submission requires trained encoder chunk size 4")
    model.eval()
    return model, payload


def _b42_endpoint_manifest(payload: dict) -> dict:
    """What the manifest says about *which endpoint* produced the predictions.

    Split out from the manifest body because it is the only part that describes
    B42's own frozen run rather than this execution. A sibling launcher
    submitting a different endpoint supplies its own, so it cannot inherit
    claims that are untrue of it -- `fixed_endpoint` and B42's completed-epoch
    and training-population counts above all. Writing those unchanged into
    another endpoint's manifest would put a false provenance record beside a
    real submission, which is worse than having no manifest at all.
    """
    return {
        "experiment": B42_SUBMISSION_EXPERIMENT,
        "version": B42_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "training_studies": int(payload.get("training_studies", -1)),
        "training_series": int(payload.get("training_series", -1)),
        "training_supervision_cells": int(payload.get("training_supervision_cells", -1)),
        "prediction": "frozen B42 combined sparse-MIL logits; raw sigmoid probability",
        "governance": (
            "Exact frozen B42 fixed-E2 endpoint. Do not change checkpoint, reference "
            "area 448^2, 90% native crop, native-aspect constant-area resize, reflection "
            "stride padding, ragged encoding, offsets [-1,0,1], sparse-MIL, thresholds "
            "or blending after Expert-58."
        ),
    }


ON_UNREADABLE_RAISE = "raise"
ON_UNREADABLE_FALLBACK = "fallback"
ON_UNREADABLE_MODES = (ON_UNREADABLE_RAISE, ON_UNREADABLE_FALLBACK)

# A study nothing could be read from still needs a row, and every target gets the
# same number. ROC AUC only sees ordering, so one constant leaves those studies
# tied among themselves and contributes nothing either way -- which is the honest
# outcome for a study the model never saw.
DEFAULT_FALLBACK_PROBABILITY = 0.5


def _infer_one_study(index: int, dataset, model, device: torch.device):
    """One complete ragged study. Every shape assertion the frozen path makes."""
    item = dataset[index]
    volumes_all = item["volumes"]
    position_all = item["slice_position"]
    present_cpu = item["present"]

    if not isinstance(volumes_all, list) or not volumes_all:
        raise RuntimeError(f"B42 row {index} has no ragged series tensors")
    if position_all.ndim != 3 or int(position_all.shape[1]) != len(B42_SUBMISSION_TTA_OFFSETS):
        raise RuntimeError(
            f"B42 row {index} TTA position shape changed: {tuple(position_all.shape)}"
        )
    for series_tensor in volumes_all:
        if series_tensor.ndim != 5 or int(series_tensor.shape[0]) != len(B42_SUBMISSION_TTA_OFFSETS):
            raise RuntimeError(
                f"B42 row {index} TTA series shape changed: {tuple(series_tensor.shape)}"
            )

    present = present_cpu.to(device, non_blocking=True)
    series_meta = item["series_meta"].to(device, non_blocking=True)
    view_probabilities: list[torch.Tensor] = []

    for view in range(len(B42_SUBMISSION_TTA_OFFSETS)):
        view_volumes = [
            series_tensor[view].to(device, non_blocking=True)
            for series_tensor in volumes_all
        ]
        position = position_all[:, view].to(device, non_blocking=True)
        # device.type, not a literal: identical on the real path, where the device
        # is always CUDA, and it lets the shard be exercised on CPU by a test.
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = model(view_volumes, present, series_meta, position)
        view_probabilities.append(torch.sigmoid(output.logits.float()).cpu())
        del view_volumes, position, output

    probability = torch.stack(view_probabilities, dim=0).mean(dim=0)
    if probability.shape != (1, len(TARGETS)) or not torch.isfinite(probability).all():
        raise RuntimeError(f"B42 row {index} produced invalid probabilities")

    shapes = [
        (int(series_tensor.shape[-2]), int(series_tensor.shape[-1]))
        for series_tensor, flag in zip(volumes_all, present_cpu)
        if float(flag.item()) > 0
    ]
    uid = str(item["study_uid"])
    result = (uid, probability.numpy()[0], shapes)

    del (
        item, volumes_all, position_all, present_cpu, present,
        series_meta, view_probabilities, probability,
    )
    return result


def _infer_shard(
    *,
    rank: int,
    indices: list[int],
    model,
    dataset,
    global_started: float,
    max_hours: float,
    reserve_minutes: float,
    uids: list[str] | None = None,
    on_unreadable: str = ON_UNREADABLE_RAISE,
    fallback_probability: float = DEFAULT_FALLBACK_PROBABILITY,
    device: torch.device | None = None,
) -> tuple[list[tuple[int, str, np.ndarray, list[tuple[int, int]]]], list[dict]]:
    """Infer complete ragged studies on one fixed CUDA device.

    `on_unreadable="raise"` is the frozen behaviour: any study that cannot be
    read ends the whole run. `on_unreadable="fallback"` gives that one study a
    constant prediction and carries on, which is what a hidden run needs -- one
    unreadable study out of 1,300 should not destroy the other 1,299.
    """
    if on_unreadable not in ON_UNREADABLE_MODES:
        raise ValueError(f"on_unreadable must be one of {ON_UNREADABLE_MODES}")
    device = torch.device(f"cuda:{int(rank)}") if device is None else device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rows: list[tuple[int, str, np.ndarray, list[tuple[int, int]]]] = []
    failures: list[dict] = []
    durations: list[float] = []

    with torch.inference_mode():
        for local_position, index in enumerate(indices):
            if durations:
                remaining_local = len(indices) - local_position
                projected = projected_remaining_seconds(
                    durations,
                    remaining_studies=remaining_local,
                    safety_factor=B42_TIMING_SAFETY_FACTOR,
                )
                elapsed = time.monotonic() - global_started
                available = (
                    float(max_hours) * 3600.0
                    - float(reserve_minutes) * 60.0
                    - elapsed
                )
                if projected > available:
                    raise RuntimeError(
                        f"B42 dual-GPU shard {rank} cannot finish inside the runtime "
                        f"budget: projected={projected/60.0:.1f} min "
                        f"available={available/60.0:.1f} min"
                    )

            started = time.monotonic()
            if on_unreadable == ON_UNREADABLE_RAISE:
                uid, probability, shapes = _infer_one_study(index, dataset, model, device)
                rows.append((int(index), uid, probability, shapes))
            else:
                try:
                    uid, probability, shapes = _infer_one_study(index, dataset, model, device)
                except torch.OutOfMemoryError as first:
                    # Memory, not data. The cache is deliberately kept warm between
                    # studies, so an unusually large study can fail where it would
                    # have fitted from clean. Give it one clean attempt.
                    print(
                        f"[B42 dual submit gpu{rank}] row {index} out of memory, "
                        f"retrying once with an empty cache: {first}",
                        flush=True,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        uid, probability, shapes = _infer_one_study(index, dataset, model, device)
                    except Exception as second:  # noqa: BLE001
                        uid, probability, shapes = None, None, None
                        error = second
                except Exception as only:  # noqa: BLE001
                    uid, probability, shapes = None, None, None
                    error = only

                if probability is None:
                    uid = str(uids[index]) if uids is not None else f"row-{index}"
                    probability = np.full(
                        len(TARGETS), float(fallback_probability), dtype=np.float32
                    )
                    shapes = []
                    failures.append({
                        "index": int(index),
                        "study_uid": uid,
                        "error": f"{type(error).__name__}: {error}",
                    })
                    print(
                        f"[B42 dual submit gpu{rank}] row {index} uid={uid} unreadable, "
                        f"predicting {fallback_probability} for all targets: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                rows.append((int(index), uid, probability, shapes))

            # Do not call gc.collect() or torch.cuda.empty_cache() here. Normal
            # reference counting releases tensors while CUDA allocator blocks stay
            # available for the next study, changing allocation overhead only.
            durations.append(time.monotonic() - started)

            completed = local_position + 1
            if completed % 10 == 0 or completed == len(indices):
                remaining = projected_remaining_seconds(
                    durations,
                    remaining_studies=len(indices) - completed,
                    safety_factor=B42_TIMING_SAFETY_FACTOR,
                )
                print(
                    f"[B42 dual submit gpu{rank}] {completed}/{len(indices)} "
                    f"shard elapsed={sum(durations)/60.0:.1f} min "
                    f"estimated_remaining={remaining/60.0:.1f} min",
                    flush=True,
                )

    return rows, failures


def generate_b42_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
    expected_checkpoint_sha256: str = B42_FROZEN_CHECKPOINT_SHA256,
    load_replica=_load_replica,
    endpoint_manifest=_b42_endpoint_manifest,
    on_unreadable: str = ON_UNREADABLE_RAISE,
    fallback_probability: float = DEFAULT_FALLBACK_PROBABILITY,
) -> Path:
    """Generate the exact frozen B42 hidden-test submission on two Kaggle T4s.

    `load_replica` and `endpoint_manifest` exist so a sibling launcher can submit
    a different endpoint through this exact inference loop rather than copying
    it. Both default to B42's own, so a call that passes neither behaves exactly
    as it did before they existed. Everything they do not cover -- the geometry,
    the sharding, the TTA offsets, the aggregation, the runtime guard and the
    validation -- is shared code and cannot diverge between endpoints.
    """
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)
    crop_policy = require_b42_contract(settings)

    runtime = resolve_runtime(settings)
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible < B42_DUALGPU_COUNT:
        raise RuntimeError(
            f"B42 Kaggle hidden submission requires two CUDA devices; visible={visible}"
        )
    if runtime.num_workers != 0 or runtime.pin_memory:
        raise RuntimeError("B42 hidden submission requires workers=0 and pin_memory=False")
    print(runtime.describe(), flush=True)
    print(
        f"[B42 dual submit] using cuda:0 + cuda:1; exact TTA offsets="
        f"{list(B42_SUBMISSION_TTA_OFFSETS)}",
        flush=True,
    )

    checkpoint_path = Path(checkpoint).resolve()
    base_path = Path(base_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B42 checkpoint is missing: {checkpoint_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"B42 base checkpoint is missing: {base_path}")
    checkpoint_sha256 = _verify_checkpoint_identity(
        checkpoint_path, expected_checkpoint_sha256
    )

    # Load replicas sequentially so checkpoint deserialization does not compete
    # for host RAM. Both replicas are reconstructed and fingerprint-verified.
    model0, payload0 = load_replica(checkpoint_path, base_path, torch.device("cuda:0"))
    model1, payload1 = load_replica(checkpoint_path, base_path, torch.device("cuda:1"))
    if payload0.get("encoder_sha256_final") != payload1.get("encoder_sha256_final"):
        raise RuntimeError("B42 dual-GPU replicas do not share the same encoder endpoint")

    test = load_test_csv(root / settings.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("test.csv contains no studies")
    series = load_series_csv(root / settings.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    missing = [uid for uid, count in zip(uids, counts) if count == 0]
    if missing and on_unreadable == ON_UNREADABLE_RAISE:
        raise ValueError(
            f"B42 submission found {len(missing)} test study/studies with zero MRI series"
        )
    if missing:
        print(
            f"[B42 dual submit] {len(missing)} study/studies have no series with a "
            f"recognised anatomical plane; they will be predicted at "
            f"{fallback_probability}",
            flush=True,
        )

    dataset_config = _b37_test_dataset_config(settings, root)
    if on_unreadable == ON_UNREADABLE_FALLBACK:
        # A series that cannot be found or decoded becomes a zero volume with
        # present=0, which the study aggregation already masks out. Under the
        # frozen strict setting the same series ends the run instead, and one bad
        # series in roughly 7,000 is a near certainty on a hidden set that three
        # clean example studies cannot reveal.
        dataset_config.strict_dicom = False
    datasets = [
        B42KaggleNormalizeOnceDataset(
            uids,
            variable_index,
            dataset_config,
            crop_focus_policy=crop_policy,
            center_offsets=B42_SUBMISSION_TTA_OFFSETS,
        )
        for _ in range(B42_DUALGPU_COUNT)
    ]

    shards = [
        list(range(rank, len(uids), B42_DUALGPU_COUNT))
        for rank in range(B42_DUALGPU_COUNT)
    ]
    budget = _submission_budget(
        settings,
        max_hours=B37_SUBMISSION_MAX_HOURS,
        min_reserve_minutes=B37_SUBMISSION_MIN_RESERVE_MINUTES,
    )
    global_started = time.monotonic()

    with ThreadPoolExecutor(max_workers=B42_DUALGPU_COUNT) as pool:
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
                uids=uids,
                on_unreadable=on_unreadable,
                fallback_probability=fallback_probability,
            )
            for rank in range(B42_DUALGPU_COUNT)
        ]
        rows = []
        failures: list[dict] = []
        for future in futures:
            shard_rows, shard_failures = future.result()
            rows.extend(shard_rows)
            failures.extend(shard_failures)
    failures.sort(key=lambda record: record["index"])

    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(uids))):
        raise RuntimeError("B42 dual-GPU output indices are incomplete or duplicated")
    uid_rows = [row[1] for row in rows]
    if uid_rows != uids:
        raise RuntimeError("B42 dual-GPU submission changed StudyInstanceUID order")

    probabilities = np.stack([row[2] for row in rows], axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    shape_rows = [shape for row in rows for shape in row[3]]
    if not shape_rows:
        raise RuntimeError("B42 submission observed no readable MRI tensor shapes")
    heights = np.asarray([shape[0] for shape in shape_rows], dtype=np.int64)
    widths = np.asarray([shape[1] for shape in shape_rows], dtype=np.int64)
    tensor_pixels = heights * widths
    feature_cells = (heights // 32) * (widths // 32)
    geometry_summary = {
        "n_series": int(len(shape_rows)),
        "rectangular_series": int(np.sum(heights != widths)),
        "square_series": int(np.sum(heights == widths)),
        "height_min": int(heights.min()),
        "height_median": float(np.median(heights)),
        "height_max": int(heights.max()),
        "width_min": int(widths.min()),
        "width_median": float(np.median(widths)),
        "width_max": int(widths.max()),
        "tensor_pixels_median": float(np.median(tensor_pixels)),
        "feature_cells_median": float(np.median(feature_cells)),
    }

    base_checkpoint_sha256 = sha256_file(base_path)
    elapsed_hours = (time.monotonic() - global_started) / 3600.0
    manifest = {
        **endpoint_manifest(payload0),
        "execution_version": B42_FAST_EXECUTION_VERSION,
        "execution_only_change": True,
        "gpu_count": B42_DUALGPU_COUNT,
        "study_sharding": "test-row index modulo 2",
        "cross_study_batching": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_base_sha256_verified": (
            base_checkpoint_sha256 == str(payload0.get("base_checkpoint_sha256", ""))
        ),
        # Leakage hygiene, not endpoint identity: true of every endpoint this
        # path may submit, so it stays here rather than moving with the rest.
        "gold_studies_used_in_gradient": int(payload0.get("gold_studies_used_in_gradient", -1)),
        "gold_labels_used": bool(payload0.get("gold_labels_used", True)),
        # How many rows the model did not actually produce. A submission that had
        # to guess for some studies is still a submission, but the number of them
        # is the first thing anyone reading the score needs to know.
        "on_unreadable": on_unreadable,
        "fallback_probability": float(fallback_probability),
        "studies_predicted_from_fallback": len(failures),
        "studies_predicted_from_fallback_fraction": (
            float(len(failures)) / float(len(uids)) if uids else 0.0
        ),
        "studies_with_no_recognised_plane": len(missing),
        "fallback_studies": failures[:50],
        "fallback_studies_truncated": max(0, len(failures) - 50),
        "thresholding_used": False,
        "blending_used": False,
        "preprocessing": b42_preprocessing_state(),
        "crop_policy": crop_policy,
        "sparse_mil": payload0.get("sparse_mil"),
        "encoder_sha256_initial": payload0.get("encoder_sha256_initial"),
        "encoder_sha256_final": payload0.get("encoder_sha256_final"),
        "tta_center_offsets": list(B42_SUBMISSION_TTA_OFFSETS),
        "tta_aggregation": "mean of per-view sigmoid probabilities",
        "checkpoint_encoder_chunk_size": 4,
        "execution_encoder_chunk_size": 4,
        "historical_volume_normalizations_per_series": 3,
        "execution_volume_normalizations_per_series": 1,
        "tta_resize_calls_per_series": 3,
        "cuda_cache_reused_between_studies": True,
        "per_study_cuda_empty_cache": False,
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "workers_per_gpu": 0,
        "pin_memory": False,
        "strict_dicom": True,
        "geometry": geometry_summary,
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(elapsed_hours),
        "runtime_budget_hours": float(budget.max_hours),
        "runtime_reserve_minutes": float(budget.reserve_minutes),
        "timing_safety_factor": B42_TIMING_SAFETY_FACTOR,
        "submission_sha256": sha256_file(output_path),
        "runtime_acceleration_scope": (
            "Inference execution only: dual-T4 complete-study sharding, one exact "
            "full-native normalization per series reused for the same three offsets, "
            "and CUDA allocator cache reuse. Each B42 crop, constant-area isotropic "
            "resize and reflection stride-pad operation remains a separate historical "
            "view; encoder chunk remains 4."
        ),
        **sample_validation,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(output_path, flush=True)
    print(manifest_path, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    print(
        "[B42 fast submit] normalize_once=True execution_chunk=4 "
        "dual_gpu=True cuda_cache_reuse=True",
        flush=True,
    )
    if failures:
        print(
            f"[B42 fast submit] WARNING {len(failures)}/{len(uids)} studies "
            f"({100.0 * len(failures) / len(uids):.2f}%) were predicted at "
            f"{fallback_probability} because they could not be read. The score "
            "reflects that; see fallback_studies in the manifest.",
            flush=True,
        )
    return output_path


__all__ = [
    "B42_DUALGPU_COUNT",
    "DEFAULT_FALLBACK_PROBABILITY",
    "ON_UNREADABLE_FALLBACK",
    "ON_UNREADABLE_MODES",
    "ON_UNREADABLE_RAISE",
    "B42_FAST_ENCODER_CHUNK_SIZE",
    "B42_FAST_EXECUTION_VERSION",
    "B42_FROZEN_CHECKPOINT_SHA256",
    "B42_SUBMISSION_TTA_OFFSETS",
    "generate_b42_submission_dual_gpu_fast",
]
