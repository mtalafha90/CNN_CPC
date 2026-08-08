from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import RuntimeBudget
from .calibration import fit_calibration
from .constants import DUAL_STREAMS, TARGETS
from .data import (
    backfill_series_metadata,
    build_series_index,
    gold_mask,
    load_series_csv,
    load_train_csv,
    make_balanced_gold_folds,
)
from .dicom import find_series_dir, read_dicom_series
from .policy import validate_competition_config
from .report_labels import STATES, state_dataframe
from .runtime import default_workers


def _decode_one(task: tuple[str, str, str, str]) -> dict:
    data_root, study_uid, series_uid, split = task
    path = find_series_dir(data_root, split, study_uid, series_uid)
    if path is None:
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "found": False,
            "decoded": False,
            "candidate_files": 0,
            "file_decode_failures": 0,
            "file_decode_failure_rate": 1.0,
            "decoded_frames": 0,
            "error": "series directory not found",
        }
    try:
        _, stats = read_dicom_series(path, return_stats=True)
        candidates = int(stats["candidate_files"])
        failures = int(stats["decode_failures"])
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "found": True,
            "decoded": True,
            "candidate_files": candidates,
            "file_decode_failures": failures,
            "file_decode_failure_rate": float(failures / max(candidates, 1)),
            "decoded_frames": int(stats["decoded_frames"]),
            "error": "",
        }
    except Exception as exc:
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "found": True,
            "decoded": False,
            "candidate_files": 0,
            "file_decode_failures": 0,
            "file_decode_failure_rate": 1.0,
            "decoded_frames": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _confidence_histogram(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values, dtype=float)
    return {
        "zero": int((values == 0).sum()),
        "gt0_lt0.10": int(((values > 0) & (values < 0.10)).sum()),
        "0.10_to_lt0.35": int(((values >= 0.10) & (values < 0.35)).sum()),
        "0.35_to_lt0.60": int(((values >= 0.35) & (values < 0.60)).sum()),
        "ge0.60": int((values >= 0.60).sum()),
    }


def run_audit(
    config: dict,
    *,
    out_dir: str | Path,
    full_decode: bool = True,
    max_hours: float | None = None,
) -> dict:
    """Audit the full training surface before expensive model experiments."""
    validate_competition_config(config, purpose="train")
    budget = RuntimeBudget(
        max_hours=float(max_hours or config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    global_failure_limit = float(config.get("audit_max_global_file_decode_failure_rate", 0.02))
    series_failure_limit = float(config.get("audit_max_series_file_decode_failure_rate", 0.20))
    if not 0 <= global_failure_limit <= 1 or not 0 <= series_failure_limit <= 1:
        raise ValueError("audit decode-failure thresholds must be in [0,1]")

    root = Path(config["data_root"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")

    gold = gold_mask(train)
    states = state_dataframe(train)
    teacher = fit_calibration(
        states[gold.to_numpy()],
        train.loc[gold, TARGETS].to_numpy(dtype=np.float64),
        alpha=float(config.get("calibration_alpha", 5.0)),
    )
    confidence = teacher.confidence(
        states,
        unmentioned_weight=float(config.get("unmentioned_weight", 0.0)),
        uncertain_weight_cap=float(config.get("uncertain_weight_cap", 0.10)),
    )

    state_counts = {}
    confidence_counts = {}
    for j, target in enumerate(TARGETS):
        state_counts[target] = {state: int((states[:, j] == state).sum()) for state in STATES}
        confidence_counts[target] = _confidence_histogram(confidence[:, j])

    folds = make_balanced_gold_folds(
        train,
        int(config.get("n_folds", 3)),
        int(config.get("seed", 2026)),
    )
    fold_counts = {}
    for fold in range(int(config.get("n_folds", 3))):
        mask = gold & folds.eq(fold)
        per_target = {}
        for target in TARGETS:
            values = train.loc[mask, target]
            known = values.notna()
            positives = int(values[known].sum()) if known.any() else 0
            per_target[target] = {
                "known": int(known.sum()),
                "positive": positives,
                "negative": int(known.sum()) - positives,
            }
        fold_counts[str(fold)] = {"studies": int(mask.sum()), "targets": per_target}

    index = build_series_index(series, train["StudyInstanceUID"], mode="dual")
    stream_counts = {name: 0 for name in DUAL_STREAMS}
    selected_pairs: set[tuple[str, str]] = set()
    for uid in train["StudyInstanceUID"].astype(str):
        mapping = index.get(uid, {})
        for name in DUAL_STREAMS:
            series_uid = mapping.get(name)
            if series_uid:
                stream_counts[name] += 1
                selected_pairs.add((uid, str(series_uid)))
    stream_missing = {name: len(train) - stream_counts[name] for name in DUAL_STREAMS}

    decode_rows: list[dict] = []
    decode_complete = not full_decode
    if full_decode:
        tasks = [(str(root), uid, series_uid, "train") for uid, series_uid in sorted(selected_pairs)]
        requested_workers = config.get("audit_workers")
        if requested_workers is None:
            requested_workers = default_workers(config.get("num_workers"))
        workers = max(1, int(requested_workers))
        ctx = mp.get_context(str(config.get("multiprocessing_context", "spawn")))
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        try:
            for result in executor.map(_decode_one, tasks, chunksize=1):
                decode_rows.append(result)
                if len(decode_rows) % 250 == 0:
                    print(f"[audit] decoded {len(decode_rows)}/{len(tasks)} selected series")
                if not budget.can_start(60.0):
                    print("[audit] stopping decode audit before runtime reserve")
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        decode_complete = len(decode_rows) == len(tasks)
        pd.DataFrame(decode_rows).to_csv(out / "series_decode_audit.csv", index=False)

    candidate_files = sum(int(row.get("candidate_files", 0)) for row in decode_rows)
    file_failures = sum(int(row.get("file_decode_failures", 0)) for row in decode_rows)
    decoded_series = sum(bool(row.get("decoded")) for row in decode_rows)
    series_failed = len(decode_rows) - decoded_series
    series_with_partial_failures = sum(
        bool(row.get("decoded")) and int(row.get("file_decode_failures", 0)) > 0
        for row in decode_rows
    )
    series_over_limit = sum(
        float(row.get("file_decode_failure_rate", 0.0)) > series_failure_limit
        for row in decode_rows
    )
    global_failure_rate = float(file_failures / max(candidate_files, 1))

    payload = {
        "studies": int(len(train)),
        "gold_studies": int(gold.sum()),
        "non_gold_studies": int((~gold).sum()),
        "series_rows": int(len(series)),
        "metadata": metadata_stats,
        "teacher_state_counts": state_counts,
        "teacher_confidence_counts": confidence_counts,
        "gold_fold_counts": fold_counts,
        "selected_stream_counts": stream_counts,
        "missing_stream_counts": stream_missing,
        "unique_selected_series": int(len(selected_pairs)),
        "decode_audit": {
            "requested": bool(full_decode),
            "complete": bool(decode_complete),
            "series_checked": int(len(decode_rows)),
            "series_decoded": int(decoded_series),
            "series_failed": int(series_failed),
            "series_with_partial_file_failures": int(series_with_partial_failures),
            "series_over_file_failure_limit": int(series_over_limit),
            "series_file_failure_limit": float(series_failure_limit),
            "candidate_files": int(candidate_files),
            "file_decode_failures": int(file_failures),
            "file_decode_failure_rate": global_failure_rate,
            "global_file_failure_limit": float(global_failure_limit),
        },
        "runtime": budget.to_dict(),
    }
    (out / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if full_decode and not decode_complete:
        raise RuntimeError(
            "full decode audit did not finish inside the configured budget; "
            f"partial results are in {out / 'series_decode_audit.csv'}"
        )
    if full_decode and series_failed:
        raise RuntimeError(
            f"full audit found {series_failed} selected MRI series that could not be decoded; "
            f"inspect {out / 'series_decode_audit.csv'}"
        )
    if full_decode and series_over_limit:
        raise RuntimeError(
            f"full audit found {series_over_limit} selected series exceeding the "
            f"{series_failure_limit:.1%} per-series file failure limit"
        )
    if full_decode and global_failure_rate > global_failure_limit:
        raise RuntimeError(
            f"global DICOM file failure rate {global_failure_rate:.2%} exceeds "
            f"audit limit {global_failure_limit:.2%}"
        )
    return payload
