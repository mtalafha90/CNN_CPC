"""Evaluate matched B20/B31/B33 controls on prospective weak-validation v1.

Primary metric: macro across the 12 targets of B6-weighted soft-label BCE on the
untouched 20% study partition (lower is better). Secondary metric: macro AUC of
hard B6 positive/negated states for targets where both classes are present.

This is a fresh architecture-selection signal, not independent clinical/expert
validation. No expert labels are read by this module.

PV1 evaluation is intentionally memory-conservative. Models are loaded and
predicted one at a time, then deleted before the next checkpoint is loaded. The
evaluation DataLoader is also frozen to batch_size=1, one worker, one prefetched
batch, no persistent worker, and no per-worker raw-series cache. These changes
alter resource use only; the frozen [-1,0,1] TTA and prediction/metric semantics
remain unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, load_frozen_b6_export, make_b7_dataset_config, prepare_b7_supervision
from .b12_variable_series import audit_variable_series_surface, build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b13_training import B13_SERIES_SIGNATURE
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .prospective_weak_v1 import PV1_VALIDATION_STUDIES, validate_prospective_weak_v1_manifest
from .prospective_weak_v1_training import load_prospective_weak_v1_checkpoint
from .runtime import resolve_runtime

PV1_EVAL_VERSION = "1.1.0"
PV1_EVAL_BATCH_SIZE = 1
PV1_EVAL_NUM_WORKERS = 1
PV1_EVAL_PREFETCH_FACTOR = 1
PV1_EVAL_SERIES_CACHE_MB = 0
PV1_EVAL_PERSISTENT_WORKERS = False


def low_memory_eval_config(config: dict) -> dict:
    """Return the frozen PV1 evaluation resource policy without mutating config."""
    safe = dict(config)
    safe["num_workers"] = PV1_EVAL_NUM_WORKERS
    safe["persistent_workers"] = PV1_EVAL_PERSISTENT_WORKERS
    safe["prefetch_factor"] = PV1_EVAL_PREFETCH_FACTOR
    safe["series_cache_mb_per_worker"] = PV1_EVAL_SERIES_CACHE_MB
    safe["b7_eval_batch_size"] = PV1_EVAL_BATCH_SIZE
    return safe


def _release_model(model, payload) -> None:
    """Drop checkpoint/model references before loading the next matched control."""
    del model
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def macro_weighted_soft_bce(targets: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != w.shape or y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError("PV1 metric arrays must all have shape [N,12]")
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    cell = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    per_target = {}
    values = []
    for j, target in enumerate(TARGETS):
        mass = float(w[:, j].sum())
        if mass <= 0:
            per_target[target] = None
            continue
        value = float((cell[:, j] * w[:, j]).sum() / mass)
        per_target[target] = value
        values.append(value)
    if len(values) != len(TARGETS):
        raise ValueError("PV1 primary metric requires positive validation weight mass for all 12 targets")
    return {"macro_weighted_soft_bce": float(np.mean(values)), "per_target": per_target}


def weak_state_auc(targets: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(targets)
    w = np.asarray(weights)
    p = np.asarray(probabilities)
    per_target = {}
    values = []
    for j, target in enumerate(TARGETS):
        active = w[:, j] > 0
        hard = (y[active, j] > 0.5).astype(np.int64)
        pred = p[active, j]
        if hard.size == 0 or np.unique(hard).size < 2:
            per_target[target] = None
            continue
        auc = float(roc_auc_score(hard, pred))
        per_target[target] = auc
        values.append(auc)
    return {
        "macro_auc_defined_targets": None if not values else float(np.mean(values)),
        "n_defined_targets": int(len(values)),
        "per_target": per_target,
    }


def paired_bootstrap_loss_difference(
    targets: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    n = int(targets.shape[0])
    rng = np.random.default_rng(int(seed))
    diffs = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        try:
            ref = macro_weighted_soft_bce(targets[idx], weights[idx], reference[idx])["macro_weighted_soft_bce"]
            cand = macro_weighted_soft_bce(targets[idx], weights[idx], candidate[idx])["macro_weighted_soft_bce"]
        except ValueError:
            continue
        diffs.append(float(cand - ref))
    if not diffs:
        raise RuntimeError("PV1 paired bootstrap produced no valid replicates")
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "difference_definition": "candidate_macro_weighted_soft_bce - reference_macro_weighted_soft_bce",
        "lower_is_better": True,
        "median_difference": float(np.median(arr)),
        "ci_lower": float(np.quantile(arr, 0.025)),
        "ci_upper": float(np.quantile(arr, 0.975)),
        "probability_candidate_better": float(np.mean(arr < 0.0)),
        "n_valid_replicates": int(arr.size),
    }


def _subset_supervision(full_uids, targets, weights, subset_uids):
    row = {str(uid): i for i, uid in enumerate(full_uids)}
    idx = np.asarray([row[str(uid)] for uid in subset_uids], dtype=np.int64)
    return targets[idx], weights[idx]


def _write_single_model_predictions(
    out: Path,
    *,
    model_name: str,
    val_uids: list[str],
    prediction: np.ndarray,
    split_sha256: str,
    encoder_sha256: str,
) -> None:
    """Persist completed model predictions immediately for crash diagnostics."""
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", val_uids)
    frame.to_csv(out / f"{model_name}_predictions.csv", index=False)
    meta = {
        "evaluation_version": PV1_EVAL_VERSION,
        "model_name": model_name,
        "split_sha256": split_sha256,
        "encoder_sha256": encoder_sha256,
        "validation_studies": len(val_uids),
        "prediction_shape": list(prediction.shape),
        "status": "prediction pass completed; final PV1 comparison may still be pending",
    }
    (out / f"{model_name}_prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def evaluate_prospective_weak_v1(
    config: dict,
    *,
    split_manifest_path: str | Path,
    b6_root: str | Path,
    b20_checkpoint: str | Path,
    b31_checkpoint: str | Path,
    b33_checkpoint: str | Path,
    out_root: str | Path = "runs/prospective_weak_v1/eval",
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    eval_config = low_memory_eval_config(config)
    runtime = resolve_runtime(eval_config)
    print(runtime.describe())
    print(
        "[PV1 eval] low-memory policy: sequential checkpoints | "
        f"batch={PV1_EVAL_BATCH_SIZE} workers={PV1_EVAL_NUM_WORKERS} "
        f"prefetch={PV1_EVAL_PREFETCH_FACTOR} persistent={PV1_EVAL_PERSISTENT_WORKERS} "
        f"series_cache_mb={PV1_EVAL_SERIES_CACHE_MB}"
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    full_uids, full_targets, full_weights, _ = prepare_b7_supervision(train, b6_frame)
    full_uids = [str(x) for x in full_uids]
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    validate_prospective_weak_v1_manifest(split, full_uids)
    val_uids = [str(x) for x in split["validation_uids"]]
    if len(val_uids) != PV1_VALIDATION_STUDIES:
        raise RuntimeError("PV1 validation count changed")
    val_targets, val_weights = _subset_supervision(full_uids, full_targets, full_weights, val_uids)

    expected_audit = split["post_assignment_supervision_audit"]["validation"]
    if int((val_weights > 0).sum()) != int(expected_audit["usable_cells"]):
        raise RuntimeError("PV1 validation supervision no longer matches split manifest")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_series_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV1 full series surface no longer matches frozen B13 signature")
    val_index = build_variable_series_index(series, val_uids)
    val_series = int(sum(len(val_index.get(uid, [])) for uid in val_uids))
    if val_series <= 0 or any(len(val_index.get(uid, [])) == 0 for uid in val_uids):
        raise RuntimeError("PV1 validation subset contains study with no eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("PV1 freezes validation TTA at [-1,0,1]")
    ds = CropFocusedVariableSeriesKneeDataset(
        val_uids,
        val_index,
        make_b7_dataset_config(eval_config, root, train=False, tta_offsets=offsets),
        targets=val_targets,
        weights=val_weights,
        train=False,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=PV1_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 34_100_000),
    )

    checkpoints = {
        "b20": b20_checkpoint,
        "b31": b31_checkpoint,
        "b33": b33_checkpoint,
    }
    split_sha = str(split["split_sha256"])
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    encoder_shas: set[str] = set()

    # Load only ONE model/checkpoint payload at a time. Keeping three payloads
    # resident duplicated large state_dicts in CPU RAM and, with TTA batches and
    # worker prefetching, contributed to a confirmed systemd-oomd kill.
    for name in ("b20", "b31", "b33"):
        path = checkpoints[name]
        print(f"[PV1 eval] loading {name}: {path}")
        model, payload = load_prospective_weak_v1_checkpoint(
            path, expected_split_sha256=split_sha, device=runtime.device
        )
        if payload.get("model_name") != name:
            raise ValueError(f"PV1 checkpoint/model mismatch for {name}")
        if payload.get("crop_focus_policy") != crop_policy:
            raise ValueError(f"PV1 crop policy mismatch for {name}")
        encoder_sha = str(payload.get("encoder_sha256_initial", ""))
        if not encoder_sha:
            raise RuntimeError(f"PV1 {name} checkpoint has no encoder fingerprint")
        encoder_shas.add(encoder_sha)
        if len(encoder_shas) > 1:
            raise RuntimeError("PV1 matched controls do not share one encoder fingerprint")

        model.eval()
        print(f"[PV1 eval] predicting {name}")
        pred_uids, pred = predict_b12_1(model, loader, runtime)
        if [str(x) for x in pred_uids] != val_uids:
            raise RuntimeError(f"PV1 {name} prediction order changed")
        pred = np.asarray(pred, dtype=np.float32)
        predictions[name] = pred
        metrics[name] = {
            "primary": macro_weighted_soft_bce(val_targets, val_weights, pred),
            "secondary": weak_state_auc(val_targets, val_weights, pred),
        }
        _write_single_model_predictions(
            out,
            model_name=name,
            val_uids=val_uids,
            prediction=pred,
            split_sha256=split_sha,
            encoder_sha256=encoder_sha,
        )
        print(
            f"[PV1 eval] completed {name}: "
            f"weak-BCE={metrics[name]['primary']['macro_weighted_soft_bce']:.10f}; releasing model"
        )
        _release_model(model, payload)

    if len(encoder_shas) != 1:
        raise RuntimeError("PV1 evaluation did not certify one shared encoder fingerprint")

    seed = int(config.get("seed", 2026))
    paired = {
        "b31_minus_b20": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b20"], predictions["b31"],
            n_bootstrap=n_bootstrap, seed=seed + 34_201,
        ),
        "b33_minus_b20": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b20"], predictions["b33"],
            n_bootstrap=n_bootstrap, seed=seed + 34_202,
        ),
        "b33_minus_b31": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b31"], predictions["b33"],
            n_bootstrap=n_bootstrap, seed=seed + 34_203,
        ),
    }

    ranking = sorted(
        (name for name in metrics),
        key=lambda name: metrics[name]["primary"]["macro_weighted_soft_bce"],
    )
    result = {
        "evaluation_version": PV1_EVAL_VERSION,
        "surface": "prospective weak-validation v1; 20% untouched StudyInstanceUID partition",
        "split_sha256": split_sha,
        "training_studies_per_control": int(split["training_studies"]),
        "validation_studies": len(val_uids),
        "validation_series": val_series,
        "independent_expert_validation": False,
        "weak_label_validation": True,
        "expert_labels_read": False,
        "primary_selection_metric": "macro of per-target B6-weighted soft-label BCE; lower is better",
        "secondary_metric": "macro AUC over B6 positive/negated states where both classes are present",
        "tta_offsets": list(offsets),
        "memory_policy": {
            "sequential_model_loading": True,
            "models_resident_simultaneously": 1,
            "eval_batch_size": PV1_EVAL_BATCH_SIZE,
            "num_workers": PV1_EVAL_NUM_WORKERS,
            "prefetch_factor": PV1_EVAL_PREFETCH_FACTOR,
            "persistent_workers": PV1_EVAL_PERSISTENT_WORKERS,
            "series_cache_mb_per_worker": PV1_EVAL_SERIES_CACHE_MB,
            "prediction_semantics_changed": False,
        },
        "metrics": metrics,
        "paired_primary_loss_bootstrap": paired,
        "primary_metric_ranking_best_first": ranking,
        "encoder_sha256": next(iter(encoder_shas)),
        "metadata_repair": metadata_stats,
        "governance": (
            "This surface is frozen before B34 and may be used for future architecture selection. "
            "It does not replace independent expert or hidden competition evaluation. Do not alter the split "
            "after inspecting model outcomes. The v1.1 low-memory implementation changes resource management "
            "only; the frozen validation studies, TTA, predictions and metrics are unchanged."
        ),
    }

    frame = pd.DataFrame({"StudyInstanceUID": val_uids})
    for j, target in enumerate(TARGETS):
        frame[f"{target}__target"] = val_targets[:, j]
        frame[f"{target}__weight"] = val_weights[:, j]
        for name in ("b20", "b31", "b33"):
            frame[f"{target}__{name}"] = predictions[name][:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nPV1 PROSPECTIVE WEAK-VALIDATION RESULT")
    print("-" * 72)
    for name in ("b20", "b31", "b33"):
        primary = metrics[name]["primary"]["macro_weighted_soft_bce"]
        auc = metrics[name]["secondary"]["macro_auc_defined_targets"]
        n_auc = metrics[name]["secondary"]["n_defined_targets"]
        print(f"{name}: weak-BCE={primary:.10f} | weak-state macro AUC={auc} ({n_auc}/12 targets)")
    print("ranking:", " > ".join(ranking))
    print(out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate prospective weak-validation v1 matched controls")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--b20-checkpoint", required=True)
    ap.add_argument("--b31-checkpoint", required=True)
    ap.add_argument("--b33-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/prospective_weak_v1/eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_prospective_weak_v1(
        config,
        split_manifest_path=args.split_manifest,
        b6_root=args.b6_root,
        b20_checkpoint=args.b20_checkpoint,
        b31_checkpoint=args.b31_checkpoint,
        b33_checkpoint=args.b33_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
