"""Evaluate the frozen B29 architecture as a post-result PV1 mechanistic addendum.

The original PV1 B20/B31/B33 comparison is immutable and must remain the primary
prospective architecture-selection result.  This module is intentionally
separate because the decision to evaluate B29 on PV1 was made only after the
original PV1 result had been observed.

B29 itself predates PV1 and is not modified here.  The addendum uses the exact
same 624 validation studies, B6 targets/weights, frozen B16 encoder, B20 crop,
[-1,0,1] TTA, primary weighted-soft-BCE metric, secondary weak-state AUC, and
low-memory evaluation policy.  Existing B20/B31/B33 prediction files are read
and validated rather than re-predicted.  Only B29 is newly inferred.

Predeclared global mechanism comparisons:
  1. B29 - B20: does the frozen learned complementary summary help B20?
  2. B29 - B33: learned complementary query versus exact uniform mean.
  3. B31 - B29: incremental effect of B31 local-context scoring.

No target-wise model selection, target-specific follow-up, or blend is allowed
from this addendum.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, load_frozen_b6_export, make_b7_dataset_config, prepare_b7_supervision
from .b12_variable_series import audit_variable_series_surface, build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b13_training import B13_SERIES_SIGNATURE
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .prospective_weak_v1 import PV1_VALIDATION_STUDIES, validate_prospective_weak_v1_manifest
from .prospective_weak_v1_b29_training import (
    PV1_B29_ADDENDUM_ROLE,
    load_pv1_b29_addendum_checkpoint,
)
from .prospective_weak_v1_eval import (
    PV1_EVAL_BATCH_SIZE,
    PV1_EVAL_NUM_WORKERS,
    PV1_EVAL_PERSISTENT_WORKERS,
    PV1_EVAL_PREFETCH_FACTOR,
    PV1_EVAL_SERIES_CACHE_MB,
    PV1_EVAL_VERSION,
    low_memory_eval_config,
    macro_weighted_soft_bce,
    paired_bootstrap_loss_difference,
    weak_state_auc,
)
from .runtime import resolve_runtime

PV1_B29_ADDENDUM_EVAL_VERSION = "1.0.0"
PV1_B29_ADDENDUM_ANALYSIS = "post_pv1_frozen_b29_global_mechanism_decomposition"
PV1_B29_REFERENCE_MODELS = ("b20", "b31", "b33")
PV1_B29_ORIGINAL_RANKING = ("b31", "b33", "b20")
PV1_B29_COMPARISONS = (
    "b29_minus_b20",
    "b29_minus_b33",
    "b31_minus_b29",
)
PV1_B29_REFERENCE_METRIC_TOLERANCE = 1e-8


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _subset_supervision(full_uids, targets, weights, subset_uids):
    row = {str(uid): i for i, uid in enumerate(full_uids)}
    try:
        idx = np.asarray([row[str(uid)] for uid in subset_uids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"PV1 addendum split UID missing from B6 supervision: {exc}") from exc
    return targets[idx], weights[idx]


def _load_original_comparison(reference_eval_root: Path, split_sha256: str) -> dict:
    path = reference_eval_root / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"original PV1 comparison missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if str(result.get("split_sha256", "")) != str(split_sha256):
        raise ValueError("original PV1 comparison split fingerprint mismatch")
    if bool(result.get("expert_labels_read", True)):
        raise ValueError("original PV1 comparison unexpectedly read expert labels")
    if result.get("weak_label_validation") is not True:
        raise ValueError("original PV1 comparison is not marked weak-label validation")
    ranking = tuple(result.get("primary_metric_ranking_best_first", []))
    if ranking != PV1_B29_ORIGINAL_RANKING:
        raise ValueError(
            f"original PV1 ranking changed; expected {PV1_B29_ORIGINAL_RANKING}, got {ranking}"
        )
    return result


def _load_reference_predictions(
    reference_eval_root: Path,
    *,
    model_name: str,
    val_uids: list[str],
    split_sha256: str,
    encoder_sha256: str,
) -> np.ndarray:
    if model_name not in PV1_B29_REFERENCE_MODELS:
        raise ValueError(f"unknown PV1 addendum reference model {model_name!r}")

    meta_path = reference_eval_root / f"{model_name}_prediction_meta.json"
    pred_path = reference_eval_root / f"{model_name}_predictions.csv"
    if not meta_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"PV1 reference prediction artifacts missing for {model_name}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("evaluation_version", "")) != PV1_EVAL_VERSION:
        raise ValueError(f"PV1 {model_name} reference evaluation version mismatch")
    if str(meta.get("model_name", "")) != model_name:
        raise ValueError(f"PV1 {model_name} reference metadata model mismatch")
    if str(meta.get("split_sha256", "")) != str(split_sha256):
        raise ValueError(f"PV1 {model_name} reference split mismatch")
    if str(meta.get("encoder_sha256", "")) != str(encoder_sha256):
        raise ValueError(f"PV1 {model_name} reference encoder mismatch")
    if int(meta.get("validation_studies", -1)) != len(val_uids):
        raise ValueError(f"PV1 {model_name} reference validation count mismatch")

    frame = pd.read_csv(pred_path)
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"PV1 {model_name} reference prediction columns changed")
    got_uids = frame["StudyInstanceUID"].astype(str).tolist()
    if got_uids != val_uids:
        raise ValueError(f"PV1 {model_name} reference UID order changed")
    pred = frame[TARGETS].to_numpy(dtype=np.float32)
    if pred.shape != (len(val_uids), len(TARGETS)) or not np.isfinite(pred).all():
        raise ValueError(f"PV1 {model_name} reference predictions invalid")
    return pred


def _write_b29_predictions(
    out: Path,
    *,
    val_uids: list[str],
    prediction: np.ndarray,
    split_sha256: str,
    encoder_sha256: str,
) -> None:
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", val_uids)
    frame.to_csv(out / "b29_predictions.csv", index=False)
    meta = {
        "addendum_evaluation_version": PV1_B29_ADDENDUM_EVAL_VERSION,
        "base_pv1_evaluation_version": PV1_EVAL_VERSION,
        "analysis": PV1_B29_ADDENDUM_ANALYSIS,
        "analysis_role": PV1_B29_ADDENDUM_ROLE,
        "model_name": "b29",
        "split_sha256": split_sha256,
        "encoder_sha256": encoder_sha256,
        "validation_studies": len(val_uids),
        "prediction_shape": list(prediction.shape),
        "original_pv1_result_already_observed": True,
        "architecture_frozen_before_pv1": True,
    }
    (out / "b29_prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_paired_predictions(
    out: Path,
    *,
    val_uids: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    data: dict[str, object] = {"StudyInstanceUID": val_uids}
    for j, target in enumerate(TARGETS):
        data[f"{target}__target"] = targets[:, j]
        data[f"{target}__weight"] = weights[:, j]
        for model_name in ("b20", "b29", "b31", "b33"):
            data[f"{target}__{model_name}"] = predictions[model_name][:, j]
    pd.DataFrame(data).to_csv(out / "paired_predictions.csv", index=False)


def evaluate_pv1_b29_addendum(
    config: dict,
    *,
    split_manifest_path: str | Path,
    b6_root: str | Path,
    b29_checkpoint: str | Path,
    reference_eval_root: str | Path = "runs/prospective_weak_v1/eval",
    out_root: str | Path = "runs/prospective_weak_v1/b29_addendum/eval",
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    eval_config = low_memory_eval_config(config)
    runtime = resolve_runtime(eval_config)
    print(runtime.describe())
    print(
        "[PV1-B29 addendum] low-memory policy: one new model only | "
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
    split_sha = str(split["split_sha256"])
    val_uids = [str(x) for x in split["validation_uids"]]
    if len(val_uids) != PV1_VALIDATION_STUDIES:
        raise RuntimeError("PV1 B29 addendum validation count changed")
    val_targets, val_weights = _subset_supervision(full_uids, full_targets, full_weights, val_uids)

    expected_audit = split["post_assignment_supervision_audit"]["validation"]
    if int((val_weights > 0).sum()) != int(expected_audit["usable_cells"]):
        raise RuntimeError("PV1 B29 validation supervision no longer matches split manifest")

    reference_root = Path(reference_eval_root)
    original = _load_original_comparison(reference_root, split_sha)
    encoder_sha = str(original.get("encoder_sha256", ""))
    if not encoder_sha:
        raise RuntimeError("original PV1 comparison has no shared encoder fingerprint")

    predictions: dict[str, np.ndarray] = {}
    for name in PV1_B29_REFERENCE_MODELS:
        predictions[name] = _load_reference_predictions(
            reference_root,
            model_name=name,
            val_uids=val_uids,
            split_sha256=split_sha,
            encoder_sha256=encoder_sha,
        )

    # Recompute the original metrics from the persisted reference predictions and
    # require numerical agreement with the immutable original comparison.json.
    reference_metrics: dict[str, dict] = {}
    for name in PV1_B29_REFERENCE_MODELS:
        reference_metrics[name] = {
            "primary": macro_weighted_soft_bce(val_targets, val_weights, predictions[name]),
            "secondary": weak_state_auc(val_targets, val_weights, predictions[name]),
        }
        recomputed = reference_metrics[name]["primary"]["macro_weighted_soft_bce"]
        recorded = float(original["metrics"][name]["primary"]["macro_weighted_soft_bce"])
        if abs(recomputed - recorded) > PV1_B29_REFERENCE_METRIC_TOLERANCE:
            raise RuntimeError(
                f"PV1 {name} persisted predictions do not reproduce original primary metric: "
                f"{recomputed} vs {recorded}"
            )

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_series_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("PV1 B29 full series surface no longer matches frozen B13 signature")
    val_index = build_variable_series_index(series, val_uids)
    val_series = int(sum(len(val_index.get(uid, [])) for uid in val_uids))
    if val_series != int(original.get("validation_series", -1)):
        raise RuntimeError("PV1 B29 validation series count differs from original PV1 evaluation")
    if any(len(val_index.get(uid, [])) == 0 for uid in val_uids):
        raise RuntimeError("PV1 B29 validation subset contains study with no eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("PV1 B29 addendum freezes validation TTA at [-1,0,1]")

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

    print(f"[PV1-B29 addendum] loading B29: {b29_checkpoint}")
    model, payload = load_pv1_b29_addendum_checkpoint(
        b29_checkpoint,
        expected_split_sha256=split_sha,
        device=runtime.device,
    )
    if payload.get("crop_focus_policy") != crop_policy:
        raise ValueError("PV1 B29 crop policy mismatch")
    if str(payload.get("encoder_sha256_initial", "")) != encoder_sha:
        raise ValueError("PV1 B29 encoder does not match original PV1 controls")

    model.eval()
    print("[PV1-B29 addendum] predicting B29")
    pred_uids, b29_pred = predict_b12_1(model, loader, runtime)
    if [str(x) for x in pred_uids] != val_uids:
        raise RuntimeError("PV1 B29 prediction order changed")
    b29_pred = np.asarray(b29_pred, dtype=np.float32)
    if b29_pred.shape != (len(val_uids), len(TARGETS)) or not np.isfinite(b29_pred).all():
        raise RuntimeError("PV1 B29 predictions invalid")
    predictions["b29"] = b29_pred

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    _write_b29_predictions(
        out,
        val_uids=val_uids,
        prediction=b29_pred,
        split_sha256=split_sha,
        encoder_sha256=encoder_sha,
    )

    del pred_uids
    del model
    del payload
    _release_unused_memory()

    metrics = dict(reference_metrics)
    metrics["b29"] = {
        "primary": macro_weighted_soft_bce(val_targets, val_weights, predictions["b29"]),
        "secondary": weak_state_auc(val_targets, val_weights, predictions["b29"]),
    }

    seed = int(config.get("seed", 2026))
    paired = {
        "b29_minus_b20": paired_bootstrap_loss_difference(
            val_targets,
            val_weights,
            predictions["b20"],
            predictions["b29"],
            n_bootstrap=n_bootstrap,
            seed=seed + 34_301,
        ),
        "b29_minus_b33": paired_bootstrap_loss_difference(
            val_targets,
            val_weights,
            predictions["b33"],
            predictions["b29"],
            n_bootstrap=n_bootstrap,
            seed=seed + 34_302,
        ),
        "b31_minus_b29": paired_bootstrap_loss_difference(
            val_targets,
            val_weights,
            predictions["b29"],
            predictions["b31"],
            n_bootstrap=n_bootstrap,
            seed=seed + 34_303,
        ),
    }

    ranking = sorted(
        ("b20", "b29", "b31", "b33"),
        key=lambda name: metrics[name]["primary"]["macro_weighted_soft_bce"],
    )

    _write_paired_predictions(
        out,
        val_uids=val_uids,
        targets=val_targets,
        weights=val_weights,
        predictions=predictions,
    )

    result = {
        "addendum_evaluation_version": PV1_B29_ADDENDUM_EVAL_VERSION,
        "base_pv1_evaluation_version": PV1_EVAL_VERSION,
        "analysis": PV1_B29_ADDENDUM_ANALYSIS,
        "analysis_role": PV1_B29_ADDENDUM_ROLE,
        "surface": "same frozen PV1 624-study weak-label validation partition",
        "split_sha256": split_sha,
        "training_studies_b29": int(split["training_studies"]),
        "validation_studies": len(val_uids),
        "validation_series": val_series,
        "independent_expert_validation": False,
        "weak_label_validation": True,
        "expert_labels_read": False,
        "original_pv1_result_already_observed": True,
        "architecture_frozen_before_pv1": True,
        "prospective_primary_selection": False,
        "original_primary_ranking_best_first": list(PV1_B29_ORIGINAL_RANKING),
        "mechanistic_addendum_ranking_best_first": ranking,
        "primary_metric": "macro of per-target B6-weighted soft-label BCE; lower is better",
        "secondary_metric": "macro AUC over B6 positive/negated states where both classes are present",
        "predeclared_global_comparisons": list(PV1_B29_COMPARISONS),
        "tta_offsets": list(offsets),
        "memory_policy": {
            "new_models_loaded": 1,
            "reference_predictions_reused_from_original_pv1": True,
            "eval_batch_size": PV1_EVAL_BATCH_SIZE,
            "num_workers": PV1_EVAL_NUM_WORKERS,
            "prefetch_factor": PV1_EVAL_PREFETCH_FACTOR,
            "persistent_workers": PV1_EVAL_PERSISTENT_WORKERS,
            "series_cache_mb_per_worker": PV1_EVAL_SERIES_CACHE_MB,
            "prediction_semantics_changed": False,
        },
        "metrics": metrics,
        "paired_primary_loss_bootstrap": paired,
        "encoder_sha256": encoder_sha,
        "metadata_repair": metadata_stats,
        "governance": (
            "The original B20/B31/B33 PV1 result remains the prospective architecture-selection result. "
            "This B29 run was defined only after that result was observed, although B29 itself was frozen "
            "before PV1. Use this addendum only for global mechanism decomposition. Do not use target-level "
            "outcomes for target switching, blending, gate tuning, or a B29.1/B31.1 follow-up."
        ),
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[PV1-B29 addendum] complete")
    for name in ("b20", "b29", "b31", "b33"):
        print(
            f"  {name}: BCE={metrics[name]['primary']['macro_weighted_soft_bce']:.10f} "
            f"AUC={metrics[name]['secondary']['macro_auc_defined_targets']}"
        )
    print(f"  mechanistic ranking: {ranking}")
    print(out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate frozen B29 as the PV1 post-result mechanistic addendum")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--b29-checkpoint", required=True)
    ap.add_argument("--reference-eval-root", default="runs/prospective_weak_v1/eval")
    ap.add_argument("--out-root", default="runs/prospective_weak_v1/b29_addendum/eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_pv1_b29_addendum(
        config,
        split_manifest_path=args.split_manifest,
        b6_root=args.b6_root,
        b29_checkpoint=args.b29_checkpoint,
        reference_eval_root=args.reference_eval_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
