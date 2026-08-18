"""Evaluate Phase 9 v2 on the frozen 499-study PV2 weak-label holdout.

The holdout is removed from both Phase-9 v2 training arms before gradients.  Its
labels are always the original frozen B6 targets/weights, which Phase 8 leaves
unchanged.  Primary metric and paired bootstrap reuse the frozen PV1/PV2 weak-
label evaluation machinery: macro per-target B6-weighted soft-label BCE.

PV2 remains historically exposed and is not independent clinical validation.
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

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .phase9_matched_supervision_v2_training import load_phase9_v2_checkpoint
from .phase9_v2_supervision import (
    PHASE9_V2_HOLDOUT_SERIES,
    PHASE9_V2_HOLDOUT_STUDIES,
    PHASE9_V2_VERSION,
    load_phase9_v2_holdout,
)
from .prospective_weak_v1_eval import (
    PV1_EVAL_BATCH_SIZE,
    PV1_EVAL_NUM_WORKERS,
    PV1_EVAL_PERSISTENT_WORKERS,
    PV1_EVAL_PREFETCH_FACTOR,
    PV1_EVAL_SERIES_CACHE_MB,
    low_memory_eval_config,
    macro_weighted_soft_bce,
    paired_bootstrap_loss_difference,
    weak_state_auc,
)
from .runtime import resolve_runtime

PHASE9_V2_EVAL_VERSION = "2.0.0"


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _write_predictions(
    out: Path,
    arm: str,
    uids: list[str],
    prediction: np.ndarray,
    *,
    split_sha256: str,
    encoder_sha256: str,
) -> None:
    frame = pd.DataFrame(prediction, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame.to_csv(out / f"{arm}_pv2_predictions.csv", index=False)
    (out / f"{arm}_prediction_meta.json").write_text(
        json.dumps(
            {
                "evaluation_version": PHASE9_V2_EVAL_VERSION,
                "phase9_version": PHASE9_V2_VERSION,
                "arm": arm,
                "pv2_split_sha256": split_sha256,
                "encoder_sha256": encoder_sha256,
                "validation_studies": len(uids),
                "prediction_shape": list(prediction.shape),
                "weak_label_validation": True,
                "independent_clinical_validation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_phase9_v2(
    config: dict,
    *,
    b6_root: str | Path,
    parent_pv1_manifest_path: str | Path,
    pv2_manifest_path: str | Path,
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    out_root: str | Path = "runs/phase9_matched_supervision_v2/eval",
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    eval_config = low_memory_eval_config(config)
    runtime = resolve_runtime(eval_config)
    print(runtime.describe())
    print(
        "[Phase9v2 eval] frozen PV2 holdout | original B6 labels only | "
        f"batch={PV1_EVAL_BATCH_SIZE} workers={PV1_EVAL_NUM_WORKERS}"
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    holdout = load_phase9_v2_holdout(
        train,
        b6_root=b6_root,
        parent_pv1_manifest_path=parent_pv1_manifest_path,
        pv2_manifest_path=pv2_manifest_path,
    )
    uids = [str(x) for x in holdout["uids"]]
    targets = np.asarray(holdout["targets"], dtype=np.float32)
    weights = np.asarray(holdout["weights"], dtype=np.float32)
    if len(uids) != PHASE9_V2_HOLDOUT_STUDIES:
        raise RuntimeError("Phase 9 v2 evaluation holdout count changed")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise RuntimeError("Phase 9 v2 PV2 holdout contains a study with no eligible MRI series")
    if int(sum(counts)) != PHASE9_V2_HOLDOUT_SERIES:
        raise RuntimeError(
            f"Phase 9 v2 expected {PHASE9_V2_HOLDOUT_SERIES} PV2 holdout MRI series, got {sum(counts)}"
        )

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("Phase 9 v2 freezes TTA at [-1,0,1]")
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        index,
        make_b7_dataset_config(eval_config, root, train=False, tta_offsets=offsets),
        targets=targets,
        weights=weights,
        train=False,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=PV1_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 40_300_000),
    )

    split_sha = str(holdout["pv2_split_sha256"])
    checkpoints = {"control": control_checkpoint, "candidate": candidate_checkpoint}
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    encoder_shas: set[str] = set()
    matched_contract: list[dict] = []
    model_specs: list[dict] = []

    for arm in ("control", "candidate"):
        print(f"[Phase9v2 eval] loading {arm}: {checkpoints[arm]}")
        model, payload = load_phase9_v2_checkpoint(
            checkpoints[arm],
            expected_arm=arm,
            expected_pv2_split_sha256=split_sha,
            device=runtime.device,
        )
        if payload.get("crop_focus_policy") != crop_policy:
            raise ValueError(f"Phase 9 v2 {arm} crop policy mismatch")
        if int(payload.get("pv2_holdout_used_in_gradient", -1)) != 0:
            raise RuntimeError(f"Phase 9 v2 {arm} checkpoint consumed PV2 holdout")
        if str(payload.get("pv2_validation_uid_sha256", "")) != str(holdout["pv2_validation_uid_sha256"]):
            raise RuntimeError(f"Phase 9 v2 {arm} holdout UID fingerprint mismatch")

        encoder_sha = str(payload.get("encoder_sha256_initial", ""))
        encoder_shas.add(encoder_sha)
        if len(encoder_shas) > 1:
            raise RuntimeError("Phase 9 v2 arms do not share the exact same frozen B16 encoder")
        model_specs.append(payload.get("model_spec"))
        matched_contract.append(
            {
                "arm": arm,
                "training_uid_sha256": payload.get("training_uid_sha256"),
                "construction_seed": payload.get("construction_seed"),
                "loader_seed": payload.get("loader_seed"),
                "post_seed": payload.get("post_construction_training_seed"),
                "training_studies": payload.get("training_studies"),
                "training_series": payload.get("training_series"),
                "pv2_holdout_studies": payload.get("pv2_holdout_studies"),
                "completed_epochs": payload.get("completed_epochs"),
            }
        )

        model.eval()
        state = model.b34_state()
        if state.get("training_context_active") is not False:
            raise RuntimeError("Phase 9 v2 B34 evaluation did not deactivate training scaffold")
        if state.get("eval_context_exact_bypass") is not True:
            raise RuntimeError("Phase 9 v2 B34 exact eval bypass contract missing")
        if int(state.get("inference_context_parameters_used", -1)) != 0:
            raise RuntimeError("Phase 9 v2 B34 unexpectedly uses scaffold parameters at inference")

        pred_uids, pred = predict_b12_1(model, loader, runtime)
        if [str(x) for x in pred_uids] != uids:
            raise RuntimeError(f"Phase 9 v2 {arm} PV2 prediction order changed")
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != targets.shape or not np.isfinite(pred).all():
            raise RuntimeError(f"Phase 9 v2 {arm} predictions invalid")
        predictions[arm] = pred
        metrics[arm] = {
            "primary": macro_weighted_soft_bce(targets, weights, pred),
            "secondary": weak_state_auc(targets, weights, pred),
        }
        _write_predictions(
            out,
            arm,
            uids,
            pred,
            split_sha256=split_sha,
            encoder_sha256=encoder_sha,
        )
        print(
            f"[Phase9v2 eval] {arm} BCE="
            f"{metrics[arm]['primary']['macro_weighted_soft_bce']:.10f}"
        )
        del model, payload, pred_uids
        _release_unused_memory()

    if model_specs[0] != model_specs[1]:
        raise RuntimeError("Phase 9 v2 arms do not share the exact same B34 model specification")
    left, right = matched_contract
    for key in (
        "training_uid_sha256",
        "construction_seed",
        "loader_seed",
        "post_seed",
        "training_studies",
        "training_series",
        "pv2_holdout_studies",
        "completed_epochs",
    ):
        if left[key] != right[key]:
            raise RuntimeError(f"Phase 9 v2 matched contract differs between arms for {key}")

    paired = paired_bootstrap_loss_difference(
        targets,
        weights,
        predictions["control"],
        predictions["candidate"],
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 2026)) + 40_303,
    )

    data = {"StudyInstanceUID": uids}
    for j, target in enumerate(TARGETS):
        data[f"{target}__target"] = targets[:, j]
        data[f"{target}__weight"] = weights[:, j]
        data[f"{target}__control"] = predictions["control"][:, j]
        data[f"{target}__candidate"] = predictions["candidate"][:, j]
    pd.DataFrame(data).to_csv(out / "paired_pv2_predictions.csv", index=False)

    result = {
        "evaluation_version": PHASE9_V2_EVAL_VERSION,
        "phase9_version": PHASE9_V2_VERSION,
        "surface": "frozen 499-study PV2 validation holdout removed from both Phase-9 v2 gradients",
        "pv2_split_sha256": split_sha,
        "pv2_validation_uid_sha256": holdout["pv2_validation_uid_sha256"],
        "validation_studies": len(uids),
        "validation_series": int(sum(counts)),
        "weak_label_validation": True,
        "independent_clinical_validation": False,
        "historically_downstream_unseen": False,
        "phase9_v2_gradient_exposure": False,
        "expert_labels_read": False,
        "primary_metric": "macro of per-target original-B6-weighted soft-label BCE; lower is better",
        "secondary_metric": "macro ROC AUC over original B6 positive/negated states where both classes are present",
        "paired_difference_definition": "candidate macro weighted BCE minus control macro weighted BCE; negative favors Phase-8 supervision",
        "metrics": metrics,
        "paired_primary_loss_bootstrap": paired,
        "matched_training_contract": matched_contract,
        "encoder_sha256": next(iter(encoder_shas)),
        "model_spec_identical": True,
        "tta_offsets": list(offsets),
        "metadata_repair": metadata_stats,
        "pv2_limitation": holdout["limitation"],
        "governance": (
            "PV2 is a frozen weak-label Phase-9 v2 holdout, not independent clinical validation. "
            "It may test the global supervision-treatment effect because its original B6 cells were excluded from both Phase-9 v2 gradients and are unchanged by Phase 8. "
            "Do not perform target/script filtering, B34 retuning, or model promotion from this result alone; hidden competition or new external expert labels remain required."
        ),
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "control_BCE": metrics["control"]["primary"]["macro_weighted_soft_bce"],
                "candidate_BCE": metrics["candidate"]["primary"]["macro_weighted_soft_bce"],
                "candidate_minus_control": paired,
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate Phase-9 v2 on frozen PV2 holdout")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--parent-pv1-manifest", required=True)
    ap.add_argument("--pv2-manifest", required=True)
    ap.add_argument("--control-checkpoint", required=True)
    ap.add_argument("--candidate-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision_v2/eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_phase9_v2(
        config,
        b6_root=args.b6_root,
        parent_pv1_manifest_path=args.parent_pv1_manifest,
        pv2_manifest_path=args.pv2_manifest,
        control_checkpoint=args.control_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
