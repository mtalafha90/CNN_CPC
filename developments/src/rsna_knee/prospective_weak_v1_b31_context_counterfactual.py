"""PV1 post-result counterfactual audit of the trained B31 local-context branch.

This audit is defined after the original PV1 B20/B31/B33 comparison and after the
frozen-B29 mechanistic addendum. It is therefore a post-result mechanism audit,
not a new prospective model-selection experiment and not independent clinical
validation.

The already-trained PV1 B31 checkpoint is evaluated under one intervention only:
its trained depthwise local_context.weight is set to exact zero *at inference*.
Every other parameter is left unchanged. This isolates whether the learned local
context is needed by the final B31 function, as distinct from an optimization or
training-path effect caused by having the branch present during training.

Predeclared global comparisons on the same frozen 624-study PV1 surface:
  1. B31-context-zero - B31-normal  [primary counterfactual test]
  2. B31-context-zero - B29         [does ablation collapse toward B29?]
  3. B31-context-zero - B33         [relation to the uniform-mean comparator]

Difference is always candidate weighted-soft-BCE minus reference weighted-soft-
BCE, so negative favors the candidate. No target-wise switching, blending, gate
retuning, kernel retuning, B29.1, or B31.1 may be derived from this audit.
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
from .prospective_weak_v1_b29_eval import (
    PV1_B29_ADDENDUM_EVAL_VERSION,
    PV1_B29_ADDENDUM_ROLE,
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
from .prospective_weak_v1_training import load_prospective_weak_v1_checkpoint
from .runtime import resolve_runtime

PV1_B31_CONTEXT_COUNTERFACTUAL_VERSION = "1.0.0"
PV1_B31_CONTEXT_COUNTERFACTUAL_ANALYSIS = "post_pv1_trained_b31_context_zero_inference_counterfactual"
PV1_B31_CONTEXT_COUNTERFACTUAL_ROLE = "post_pv1_global_mechanism_counterfactual"
PV1_B31_CONTEXT_COUNTERFACTUAL_COMPARISONS = (
    "b31_context_zero_minus_b31",
    "b31_context_zero_minus_b29",
    "b31_context_zero_minus_b33",
)
PV1_B31_CONTEXT_REFERENCE_TOLERANCE = 1e-8


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
    idx = np.asarray([row[str(uid)] for uid in subset_uids], dtype=np.int64)
    return targets[idx], weights[idx]


def zero_b31_local_context_for_counterfactual(model) -> tuple[dict, dict]:
    """Zero only B31 local_context weights in-place and return before/after states."""
    if not hasattr(model, "local_context") or not hasattr(model, "local_context_state"):
        raise TypeError("counterfactual requires a B31 model with local_context")
    before = model.local_context_state()
    if int(before.get("parameter_count", -1)) != 2304:
        raise ValueError("B31 local-context parameter contract changed")
    if float(before.get("weight_l2", 0.0)) <= 0.0:
        raise ValueError("trained B31 local context is already zero; counterfactual is not informative")
    with torch.no_grad():
        model.local_context.weight.zero_()
    after = model.local_context_state()
    if float(after.get("weight_l2", -1.0)) != 0.0:
        raise RuntimeError("failed to zero B31 local context exactly")
    if int(torch.count_nonzero(model.local_context.weight).item()) != 0:
        raise RuntimeError("B31 counterfactual local-context tensor is not exactly zero")
    return before, after


def classify_primary_counterfactual(paired: dict) -> str:
    """Predeclared interpretation of context-zero minus normal B31 primary loss."""
    lo = float(paired["ci_lower"])
    hi = float(paired["ci_upper"])
    if lo > 0.0:
        return "trained_context_directly_improves_final_inference"
    if hi < 0.0:
        return "trained_context_is_harmful_at_inference_despite_training_path"
    return "direct_inference_effect_unresolved_optimization_path_remains_plausible"


def _load_original_pv1(reference_root: Path, split_sha: str) -> dict:
    path = reference_root / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if str(result.get("evaluation_version", "")) != PV1_EVAL_VERSION:
        raise ValueError("original PV1 evaluation version mismatch")
    if str(result.get("split_sha256", "")) != split_sha:
        raise ValueError("original PV1 split mismatch")
    if bool(result.get("expert_labels_read", True)) or result.get("weak_label_validation") is not True:
        raise ValueError("original PV1 governance mismatch")
    if tuple(result.get("primary_metric_ranking_best_first", [])) != ("b31", "b33", "b20"):
        raise ValueError("original PV1 ranking changed")
    return result


def _load_prediction(
    root: Path,
    *,
    model_name: str,
    val_uids: list[str],
    split_sha: str,
    encoder_sha: str,
    meta_kind: str,
) -> np.ndarray:
    pred_path = root / f"{model_name}_predictions.csv"
    meta_path = root / f"{model_name}_prediction_meta.json"
    if not pred_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing persisted predictions for {model_name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("model_name", "")) != model_name:
        raise ValueError(f"{model_name} metadata model mismatch")
    if str(meta.get("split_sha256", "")) != split_sha:
        raise ValueError(f"{model_name} prediction split mismatch")
    if str(meta.get("encoder_sha256", "")) != encoder_sha:
        raise ValueError(f"{model_name} prediction encoder mismatch")
    if int(meta.get("validation_studies", -1)) != len(val_uids):
        raise ValueError(f"{model_name} prediction validation count mismatch")
    if meta_kind == "pv1" and str(meta.get("evaluation_version", "")) != PV1_EVAL_VERSION:
        raise ValueError(f"{model_name} PV1 prediction version mismatch")
    if meta_kind == "b29_addendum":
        if str(meta.get("addendum_evaluation_version", "")) != PV1_B29_ADDENDUM_EVAL_VERSION:
            raise ValueError("B29 addendum prediction version mismatch")
        if str(meta.get("analysis_role", "")) != PV1_B29_ADDENDUM_ROLE:
            raise ValueError("B29 addendum analysis-role mismatch")

    frame = pd.read_csv(pred_path)
    expected = ["StudyInstanceUID", *TARGETS]
    if list(frame.columns) != expected:
        raise ValueError(f"{model_name} prediction columns changed")
    if frame["StudyInstanceUID"].astype(str).tolist() != val_uids:
        raise ValueError(f"{model_name} prediction UID order changed")
    pred = frame[TARGETS].to_numpy(dtype=np.float32)
    if pred.shape != (len(val_uids), len(TARGETS)) or not np.isfinite(pred).all():
        raise ValueError(f"{model_name} predictions invalid")
    return pred


def _load_b29_addendum(b29_root: Path, split_sha: str) -> dict:
    path = b29_root / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if str(result.get("addendum_evaluation_version", "")) != PV1_B29_ADDENDUM_EVAL_VERSION:
        raise ValueError("B29 addendum evaluation version mismatch")
    if str(result.get("analysis_role", "")) != PV1_B29_ADDENDUM_ROLE:
        raise ValueError("B29 addendum role mismatch")
    if str(result.get("split_sha256", "")) != split_sha:
        raise ValueError("B29 addendum split mismatch")
    if tuple(result.get("mechanistic_addendum_ranking_best_first", [])) != ("b31", "b33", "b29", "b20"):
        raise ValueError("B29 addendum ranking changed")
    return result


def _write_predictions(out: Path, val_uids: list[str], pred: np.ndarray, *, split_sha: str, encoder_sha: str) -> None:
    frame = pd.DataFrame(pred, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", val_uids)
    frame.to_csv(out / "b31_context_zero_predictions.csv", index=False)
    meta = {
        "counterfactual_version": PV1_B31_CONTEXT_COUNTERFACTUAL_VERSION,
        "analysis": PV1_B31_CONTEXT_COUNTERFACTUAL_ANALYSIS,
        "analysis_role": PV1_B31_CONTEXT_COUNTERFACTUAL_ROLE,
        "model_name": "b31_context_zero",
        "source_model": "b31",
        "intervention": "set trained local_context.weight to exact zero at inference only",
        "split_sha256": split_sha,
        "encoder_sha256": encoder_sha,
        "validation_studies": len(val_uids),
        "prediction_shape": list(pred.shape),
    }
    (out / "b31_context_zero_prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def evaluate_b31_context_counterfactual(
    config: dict,
    *,
    split_manifest_path: str | Path,
    b6_root: str | Path,
    b31_checkpoint: str | Path,
    reference_eval_root: str | Path = "runs/prospective_weak_v1/eval",
    b29_addendum_eval_root: str | Path = "runs/prospective_weak_v1/b29_addendum/eval",
    out_root: str | Path = "runs/prospective_weak_v1/b31_context_counterfactual",
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    eval_config = low_memory_eval_config(config)
    runtime = resolve_runtime(eval_config)
    print(runtime.describe())
    print(
        "[PV1 B31 context CF] inference-only intervention | "
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
        raise RuntimeError("B31 context counterfactual validation count changed")
    val_targets, val_weights = _subset_supervision(full_uids, full_targets, full_weights, val_uids)

    reference_root = Path(reference_eval_root)
    original = _load_original_pv1(reference_root, split_sha)
    encoder_sha = str(original.get("encoder_sha256", ""))
    if not encoder_sha:
        raise RuntimeError("original PV1 encoder SHA missing")
    b29_root = Path(b29_addendum_eval_root)
    b29_addendum = _load_b29_addendum(b29_root, split_sha)

    predictions = {
        "b31": _load_prediction(reference_root, model_name="b31", val_uids=val_uids, split_sha=split_sha, encoder_sha=encoder_sha, meta_kind="pv1"),
        "b33": _load_prediction(reference_root, model_name="b33", val_uids=val_uids, split_sha=split_sha, encoder_sha=encoder_sha, meta_kind="pv1"),
        "b29": _load_prediction(b29_root, model_name="b29", val_uids=val_uids, split_sha=split_sha, encoder_sha=encoder_sha, meta_kind="b29_addendum"),
    }

    reference_metrics = {}
    for name in ("b31", "b33"):
        reference_metrics[name] = {
            "primary": macro_weighted_soft_bce(val_targets, val_weights, predictions[name]),
            "secondary": weak_state_auc(val_targets, val_weights, predictions[name]),
        }
        got = reference_metrics[name]["primary"]["macro_weighted_soft_bce"]
        expected = float(original["metrics"][name]["primary"]["macro_weighted_soft_bce"])
        if abs(got - expected) > PV1_B31_CONTEXT_REFERENCE_TOLERANCE:
            raise RuntimeError(f"{name} persisted predictions no longer reproduce original PV1 metric")
    reference_metrics["b29"] = {
        "primary": macro_weighted_soft_bce(val_targets, val_weights, predictions["b29"]),
        "secondary": weak_state_auc(val_targets, val_weights, predictions["b29"]),
    }
    b29_got = reference_metrics["b29"]["primary"]["macro_weighted_soft_bce"]
    b29_expected = float(b29_addendum["metrics"]["b29"]["primary"]["macro_weighted_soft_bce"])
    if abs(b29_got - b29_expected) > PV1_B31_CONTEXT_REFERENCE_TOLERANCE:
        raise RuntimeError("B29 persisted predictions no longer reproduce addendum metric")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_series_summary, _ = audit_variable_series_surface(series, full_uids)
    if full_series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B31 context counterfactual full series surface changed")
    val_index = build_variable_series_index(series, val_uids)
    val_series = int(sum(len(val_index.get(uid, [])) for uid in val_uids))
    if val_series != int(original.get("validation_series", -1)):
        raise RuntimeError("B31 context counterfactual validation series count changed")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B31 context counterfactual freezes TTA at [-1,0,1]")
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

    print(f"[PV1 B31 context CF] loading trained B31: {b31_checkpoint}")
    model, payload = load_prospective_weak_v1_checkpoint(
        b31_checkpoint, expected_split_sha256=split_sha, device=runtime.device
    )
    if str(payload.get("model_name", "")) != "b31":
        raise ValueError("counterfactual checkpoint is not B31")
    if payload.get("crop_focus_policy") != crop_policy:
        raise ValueError("counterfactual B31 crop policy mismatch")
    if str(payload.get("encoder_sha256_initial", "")) != encoder_sha:
        raise ValueError("counterfactual B31 encoder does not match PV1 references")

    trained_context_state, zero_context_state = zero_b31_local_context_for_counterfactual(model)
    model.eval()
    print(
        "[PV1 B31 context CF] context zeroed at inference only | "
        f"trained_l2={trained_context_state['weight_l2']:.8f} -> 0"
    )
    pred_uids, zero_pred = predict_b12_1(model, loader, runtime)
    if [str(x) for x in pred_uids] != val_uids:
        raise RuntimeError("B31 context-zero prediction order changed")
    zero_pred = np.asarray(zero_pred, dtype=np.float32)
    if zero_pred.shape != (len(val_uids), len(TARGETS)) or not np.isfinite(zero_pred).all():
        raise RuntimeError("B31 context-zero predictions invalid")
    predictions["b31_context_zero"] = zero_pred

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    _write_predictions(out, val_uids, zero_pred, split_sha=split_sha, encoder_sha=encoder_sha)

    del pred_uids
    del model
    del payload
    _release_unused_memory()

    metrics = dict(reference_metrics)
    metrics["b31_context_zero"] = {
        "primary": macro_weighted_soft_bce(val_targets, val_weights, zero_pred),
        "secondary": weak_state_auc(val_targets, val_weights, zero_pred),
    }

    seed = int(config.get("seed", 2026))
    paired = {
        "b31_context_zero_minus_b31": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b31"], zero_pred,
            n_bootstrap=n_bootstrap, seed=seed + 34_401,
        ),
        "b31_context_zero_minus_b29": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b29"], zero_pred,
            n_bootstrap=n_bootstrap, seed=seed + 34_402,
        ),
        "b31_context_zero_minus_b33": paired_bootstrap_loss_difference(
            val_targets, val_weights, predictions["b33"], zero_pred,
            n_bootstrap=n_bootstrap, seed=seed + 34_403,
        ),
    }
    primary_interpretation = classify_primary_counterfactual(paired["b31_context_zero_minus_b31"])

    delta = zero_pred.astype(np.float64) - predictions["b31"].astype(np.float64)
    prediction_shift = {
        "mean_absolute_probability_change": float(np.mean(np.abs(delta))),
        "root_mean_square_probability_change": float(np.sqrt(np.mean(delta * delta))),
        "maximum_absolute_probability_change": float(np.max(np.abs(delta))),
    }

    data: dict[str, object] = {"StudyInstanceUID": val_uids}
    for j, target in enumerate(TARGETS):
        data[f"{target}__target"] = val_targets[:, j]
        data[f"{target}__weight"] = val_weights[:, j]
        for name in ("b29", "b31", "b33", "b31_context_zero"):
            data[f"{target}__{name}"] = predictions[name][:, j]
    pd.DataFrame(data).to_csv(out / "paired_predictions.csv", index=False)

    result = {
        "counterfactual_version": PV1_B31_CONTEXT_COUNTERFACTUAL_VERSION,
        "analysis": PV1_B31_CONTEXT_COUNTERFACTUAL_ANALYSIS,
        "analysis_role": PV1_B31_CONTEXT_COUNTERFACTUAL_ROLE,
        "surface": "same frozen PV1 624-study weak-label validation partition",
        "split_sha256": split_sha,
        "validation_studies": len(val_uids),
        "validation_series": val_series,
        "independent_expert_validation": False,
        "weak_label_validation": True,
        "expert_labels_read": False,
        "original_pv1_result_already_observed": True,
        "b29_addendum_result_already_observed": True,
        "prospective_primary_selection": False,
        "intervention": "same trained B31 checkpoint with local_context.weight set to exact zero at inference only",
        "all_other_checkpoint_parameters_unchanged": True,
        "primary_metric": "macro of per-target B6-weighted soft-label BCE; lower is better",
        "secondary_metric": "macro AUC over B6 positive/negated states where both classes are present",
        "predeclared_global_comparisons": list(PV1_B31_CONTEXT_COUNTERFACTUAL_COMPARISONS),
        "primary_counterfactual_comparison": "b31_context_zero_minus_b31",
        "primary_counterfactual_interpretation": primary_interpretation,
        "tta_offsets": list(offsets),
        "memory_policy": {
            "new_models_loaded": 1,
            "reference_predictions_reused": True,
            "eval_batch_size": PV1_EVAL_BATCH_SIZE,
            "num_workers": PV1_EVAL_NUM_WORKERS,
            "prefetch_factor": PV1_EVAL_PREFETCH_FACTOR,
            "persistent_workers": PV1_EVAL_PERSISTENT_WORKERS,
            "series_cache_mb_per_worker": PV1_EVAL_SERIES_CACHE_MB,
        },
        "trained_local_context_state_before_intervention": trained_context_state,
        "local_context_state_after_intervention": zero_context_state,
        "prediction_shift_vs_normal_b31": prediction_shift,
        "metrics": metrics,
        "paired_primary_loss_bootstrap": paired,
        "encoder_sha256": encoder_sha,
        "metadata_repair": metadata_stats,
        "governance": (
            "This is an inference-only post-result mechanism audit. The original PV1 selection remains B31 > B33 > B20, "
            "and the B29 addendum remains post-result mechanism decomposition. Use only the global predeclared comparisons. "
            "Do not create target-wise switches, blends, context masks, kernel retunes, B29.1, or B31.1 from these outcomes."
        ),
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[PV1 B31 context CF] complete")
    print(f"  B31 normal BCE       = {metrics['b31']['primary']['macro_weighted_soft_bce']:.10f}")
    print(f"  B31 context-zero BCE = {metrics['b31_context_zero']['primary']['macro_weighted_soft_bce']:.10f}")
    print(f"  interpretation       = {primary_interpretation}")
    print(out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("PV1 trained-B31 local-context inference counterfactual")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--b31-checkpoint", required=True)
    ap.add_argument("--reference-eval-root", default="runs/prospective_weak_v1/eval")
    ap.add_argument("--b29-addendum-eval-root", default="runs/prospective_weak_v1/b29_addendum/eval")
    ap.add_argument("--out-root", default="runs/prospective_weak_v1/b31_context_counterfactual")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_b31_context_counterfactual(
        config,
        split_manifest_path=args.split_manifest,
        b6_root=args.b6_root,
        b31_checkpoint=args.b31_checkpoint,
        reference_eval_root=args.reference_eval_root,
        b29_addendum_eval_root=args.b29_addendum_eval_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
