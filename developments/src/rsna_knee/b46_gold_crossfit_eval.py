"""Evaluate the five frozen B46 folds as one leakage-free 58-study OOF prediction."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b17_training import encoder_state_sha256
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b35_training import sha256_file
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b37_highres_sparse_mil import B37_EXPERT58_ROOT
from .b42_constant_area_aspect_sparse_mil import (
    B42_EXPERT58_ROOT,
    B42ConstantAreaAspectDataset,
    B42ConstantAreaAspectSparseMILResidual,
    collate_b42,
)
from .b46_gold_crossfit import (
    B46_EXPERIMENT,
    B46_FIXED_EPOCHS,
    B46_GOLD_CELL_WEIGHT,
    B46_N_FOLDS,
    B46_RUN_ROOT,
    B46_VERSION,
    heldout_uids,
    load_gold_fold_manifest,
    require_b46_contract,
)
from .b46_gold_crossfit_training import B46_CHECKPOINT_TEMPLATE
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime

B46_EVAL_LOADER_SEED_OFFSET = 55_300_000
B46_BOOTSTRAP_B42_SEED_OFFSET = 55_400_000
B46_BOOTSTRAP_B37_SEED_OFFSET = 55_500_000


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_predictions(path: Path, uids: list[str]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing historical Expert-58 predictions: {path}")
    frame = pd.read_csv(path)
    required = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != required:
        raise ValueError(f"prediction columns changed: {path}")
    if frame["StudyInstanceUID"].astype(str).tolist() != uids:
        raise RuntimeError(f"prediction UID order changed: {path}")
    arr = frame[TARGETS].to_numpy(np.float64)
    if arr.shape != (len(uids), len(TARGETS)) or not np.isfinite(arr).all():
        raise RuntimeError(f"invalid prediction matrix: {path}")
    return arr


def _summary(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, dict[str, float]]:
    macro, auc = macro_auc_from_arrays(truth, prediction)
    return float(macro), {target: float(value) for target, value in zip(TARGETS, auc)}


def _load_b46_model(
    checkpoint: Path,
    *,
    config: dict,
    base_checkpoint: Path,
    manifest_sha: str,
    fold: int,
    expected_heldout: list[str],
    device,
):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B46_EXPERIMENT or payload.get("version") != B46_VERSION:
        raise ValueError(f"{checkpoint} is not a frozen B46 checkpoint")
    if payload.get("fixed_endpoint") is not True:
        raise ValueError("B46 checkpoint is not fixed")
    if int(payload.get("completed_epochs", -1)) != B46_FIXED_EPOCHS:
        raise ValueError("B46 checkpoint did not complete fixed E2")
    if int(payload.get("fold", -1)) != int(fold):
        raise ValueError("B46 checkpoint fold identity changed")
    if str(payload.get("fold_manifest_sha256", "")) != manifest_sha:
        raise ValueError("B46 fold manifest fingerprint mismatch")
    if [str(x) for x in payload.get("heldout_gold_uids", [])] != [str(x) for x in expected_heldout]:
        raise RuntimeError("B46 checkpoint held-out UIDs differ from frozen manifest")
    training_gold = set(str(x) for x in payload.get("training_gold_uids", []))
    if training_gold.intersection(expected_heldout):
        raise RuntimeError("B46 held-out gold UID appears in checkpoint training list")
    if int(payload.get("heldout_gold_studies_used_in_gradient", -1)) != 0:
        raise RuntimeError("B46 checkpoint reports held-out gold leakage")
    if not bool(payload.get("gold_labels_used", False)):
        raise RuntimeError("B46 checkpoint did not use the declared gold anchor")
    if not np.isclose(float(payload.get("gold_cell_weight", -1)), B46_GOLD_CELL_WEIGHT):
        raise RuntimeError("B46 gold cell weight changed")
    if str(payload.get("target_balance_source", "")) != "weak_only_frozen":
        raise RuntimeError("B46 target-balance source changed")

    if sha256_file(base_checkpoint) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B46 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_checkpoint, expected_arm="llm_fill", device="cpu")
    model = B42ConstantAreaAspectSparseMILResidual(
        base,
        grid_size=int(config["b37_grid_size"]),
        top_k=int(config["b37_top_k"]),
        temperature=float(config["b37_temperature"]),
        encoder_trainable_stages=int(config["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b37_encoder_chunk_size"]),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    if encoder_state_sha256(model.base.encoder) != str(payload.get("encoder_sha256_final", "")):
        raise RuntimeError("B46 reconstructed encoder fingerprint changed")
    return model, payload


def _leave_one_target_out_deltas(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for j, target in enumerate(TARGETS):
        keep = [k for k in range(len(TARGETS)) if k != j]
        ref_macro, _ = macro_auc_from_arrays(truth[:, keep], reference[:, keep])
        cand_macro, _ = macro_auc_from_arrays(truth[:, keep], candidate[:, keep])
        result[target] = float(cand_macro - ref_macro)
    return result


@torch.no_grad()
def evaluate_b46_crossfit(
    config: dict,
    *,
    data_root: str | Path,
    base_checkpoint: str | Path,
    fold_manifest: str | Path,
    run_root: str | Path = B46_RUN_ROOT,
    b42_expert58_root: str | Path = B42_EXPERT58_ROOT,
    b37_expert58_root: str | Path = B37_EXPERT58_ROOT,
    out_root: str | Path | None = None,
    n_bootstrap: int = 5000,
) -> dict:
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b46_contract(settings)
    settings["b7_eval_batch_size"] = 1
    root = Path(settings["data_root"])
    base_path = Path(base_checkpoint).resolve()
    manifest_path = Path(fold_manifest).resolve()
    manifest = load_gold_fold_manifest(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B46 requires the complete 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    manifest_uids = [str(row["StudyInstanceUID"]) for row in manifest["rows"]]
    if set(manifest_uids) != set(uids):
        raise RuntimeError("B46 manifest does not cover exactly the official gold UIDs")

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    all_index = build_variable_series_index(series, uids)
    counts = [len(all_index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B46 Expert-58 MRI series surface changed")

    b42_prediction = _read_predictions(
        Path(b42_expert58_root) / "b42_combined_predictions.csv", uids
    )
    b37_prediction = _read_predictions(
        Path(b37_expert58_root) / "b37_combined_predictions.csv", uids
    )
    b42_macro, b42_auc = _summary(truth, b42_prediction)
    b37_macro, b37_auc = _summary(truth, b37_prediction)

    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    run_root = Path(run_root)
    prediction_by_uid: dict[str, np.ndarray] = {}
    global_by_uid: dict[str, np.ndarray] = {}
    fold_records = []

    for fold in range(B46_N_FOLDS):
        fold_uids = heldout_uids(manifest, fold)
        fold_index = {uid: all_index[uid] for uid in fold_uids}
        dcfg = make_b7_dataset_config(settings, root, train=False)
        dcfg.tta_center_offsets = ()
        dataset = B42ConstantAreaAspectDataset(
            fold_uids,
            fold_index,
            dcfg,
            crop_focus_policy=crop_policy,
            center_offsets=B37_EVAL_OFFSETS,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_b42,
            **runtime.loader_kwargs(
                seed=int(settings.get("seed", 2026)) + B46_EVAL_LOADER_SEED_OFFSET + fold
            ),
        )
        checkpoint = run_root / f"fold_{fold}" / B46_CHECKPOINT_TEMPLATE.format(fold=fold)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, payload = _load_b46_model(
            checkpoint,
            config=settings,
            base_checkpoint=base_path,
            manifest_sha=manifest_sha,
            fold=fold,
            expected_heldout=fold_uids,
            device=runtime.device,
        )
        scored = []
        for items in loader:
            if len(items) != 1:
                raise RuntimeError("B46 OOF evaluation requires one ragged study per batch")
            item = items[0]
            uid = str(item["study_uid"])
            scored.append(uid)
            present = item["present"].to(runtime.device, non_blocking=True)
            meta = item["series_meta"].to(runtime.device, non_blocking=True)
            position_all = item["slice_position"].to(runtime.device, non_blocking=True)
            combined_views = []
            global_views = []
            for view in range(len(B37_EVAL_OFFSETS)):
                volumes = [
                    series_tensor[view].to(runtime.device, non_blocking=True)
                    for series_tensor in item["volumes"]
                ]
                position = position_all[:, view]
                with autocast(runtime):
                    output = model(volumes, present, meta, position)
                combined_views.append(torch.sigmoid(output.logits.float()))
                global_views.append(torch.sigmoid(output.base_logits.float()))
                del volumes, position, output
            prediction_by_uid[uid] = torch.stack(combined_views).mean(dim=0).cpu().numpy()[0]
            global_by_uid[uid] = torch.stack(global_views).mean(dim=0).cpu().numpy()[0]
            del item, items, present, meta, position_all, combined_views, global_views
            _release()
        if scored != fold_uids:
            raise RuntimeError(f"B46 fold {fold} OOF UID order changed")
        fold_records.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "heldout_studies": len(fold_uids),
                "heldout_uids": fold_uids,
                "training_gold_studies": int(payload.get("gold_studies_used_in_gradient", -1)),
                "heldout_gold_studies_used_in_gradient": int(
                    payload.get("heldout_gold_studies_used_in_gradient", -1)
                ),
            }
        )
        del model, dataset, loader, payload
        _release()
        print(f"[B46 OOF] fold {fold} PASS ({len(fold_uids)} studies)", flush=True)

    if set(prediction_by_uid) != set(uids):
        missing = sorted(set(uids).difference(prediction_by_uid))
        extra = sorted(set(prediction_by_uid).difference(uids))
        raise RuntimeError(f"B46 OOF coverage mismatch missing={missing} extra={extra}")
    prediction = np.stack([prediction_by_uid[uid] for uid in uids], axis=0)
    global_prediction = np.stack([global_by_uid[uid] for uid in uids], axis=0)
    if not np.isfinite(prediction).all() or not np.isfinite(global_prediction).all():
        raise RuntimeError("B46 OOF prediction contains non-finite values")

    b46_macro, b46_auc = _summary(truth, prediction)
    b46_global_macro, b46_global_auc = _summary(truth, global_prediction)
    delta = float(b46_macro - b42_macro)
    paired_b42 = compare_runs(
        truth,
        b42_prediction,
        prediction,
        n_bootstrap=int(n_bootstrap),
        seed=int(settings.get("seed", 2026)) + B46_BOOTSTRAP_B42_SEED_OFFSET,
    )
    paired_b37 = compare_runs(
        truth,
        b37_prediction,
        prediction,
        n_bootstrap=int(n_bootstrap),
        seed=int(settings.get("seed", 2026)) + B46_BOOTSTRAP_B37_SEED_OFFSET,
    )
    loto = _leave_one_target_out_deltas(truth, b42_prediction, prediction)
    improved = sum(b46_auc[target] > b42_auc[target] for target in TARGETS)
    min_loto = float(min(loto.values()))

    # Frozen before any B46 OOF result is observed.
    strong_support = bool(
        delta >= 0.010
        and float(paired_b42["ci_lower"]) > 0.0
        and improved >= 8
        and min_loto > 0.0
    )
    directional_support = bool(
        not strong_support
        and delta >= 0.010
        and float(paired_b42["probability_b_better"]) >= 0.90
        and improved >= 7
        and min_loto > 0.0
    )
    if strong_support:
        verdict = "strong_support_label_mismatch_bottleneck"
    elif directional_support:
        verdict = "directional_support_label_mismatch_bottleneck"
    elif delta < 0.005 or float(paired_b42["probability_b_better"]) < 0.65:
        verdict = "no_support_for_gold_anchor_at_frozen_weight"
    else:
        verdict = "inconclusive"

    per_target = {
        target: {
            "b37_auc": b37_auc[target],
            "b42_auc": b42_auc[target],
            "b46_global_oof_auc": b46_global_auc[target],
            "b46_oof_auc": b46_auc[target],
            "b46_minus_b42": float(b46_auc[target] - b42_auc[target]),
        }
        for target in TARGETS
    }
    result = {
        "evaluation_role": (
            "prospectively cross-fitted 58-study official-gold OOF diagnostic; every row is "
            "predicted by a model whose gradients excluded that study. The fold/weight/architecture "
            "were frozen before OOF inspection. This is not a hidden competition test."
        ),
        "experiment": B46_EXPERIMENT,
        "version": B46_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "studies": len(uids),
        "series": int(sum(counts)),
        "gold_cell_weight": B46_GOLD_CELL_WEIGHT,
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "macro_auc": {
            "b37_combined": b37_macro,
            "b42_parent_combined": b42_macro,
            "b46_global_oof": b46_global_macro,
            "b46_oof": b46_macro,
            "b46_minus_b42": delta,
            "b46_minus_b37": float(b46_macro - b37_macro),
        },
        "paired_bootstrap": {
            "b46_minus_b42": paired_b42,
            "b46_minus_b37": paired_b37,
        },
        "coherence": {
            "targets_improved_vs_b42": int(improved),
            "targets_total": len(TARGETS),
            "leave_one_target_out_b46_minus_b42": loto,
            "minimum_leave_one_target_out_delta": min_loto,
        },
        "predeclared_decision_rule": {
            "strong_support": (
                "delta>=+0.010, paired 95% CI lower>0, >=8/12 targets improve, "
                "and every leave-one-target-out macro delta remains >0"
            ),
            "directional_support": (
                "delta>=+0.010, P(B46>B42)>=0.90, >=7/12 targets improve, "
                "and every leave-one-target-out macro delta remains >0"
            ),
            "no_support": "delta<+0.005 OR P(B46>B42)<0.65",
            "otherwise": "inconclusive",
        },
        "verdict": verdict,
        "per_target": per_target,
        "folds": fold_records,
        "metadata_repair": metadata_stats,
        "governance": (
            "Do not tune the 4.0 gold-cell weight, folds, B42 geometry, sparse settings, "
            "learning rates, epoch count, target subset, or target-wise mixtures from this OOF result."
        ),
    }

    output_root = Path(out_root) if out_root is not None else Path(run_root) / "oof"
    output_root.mkdir(parents=True, exist_ok=True)
    for name, arr in (
        ("b46_oof_predictions", prediction),
        ("b46_global_oof_predictions", global_prediction),
    ):
        frame = pd.DataFrame(arr, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(output_root / f"{name}.csv", index=False)
    (output_root / "crossfit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("B46 GOLD-ANCHORED CROSSFIT OOF: PASS", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate frozen B46 five-fold official-gold OOF")
    parser.add_argument("--config", default="config/b46_gold_anchored_crossfit.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--run-root", default=B46_RUN_ROOT)
    parser.add_argument("--b42-expert58-root", default=B42_EXPERT58_ROOT)
    parser.add_argument("--b37-expert58-root", default=B37_EXPERT58_ROOT)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b46_crossfit(
        config,
        data_root=args.data_root,
        base_checkpoint=args.base_checkpoint,
        fold_manifest=args.fold_manifest,
        run_root=args.run_root,
        b42_expert58_root=args.b42_expert58_root,
        b37_expert58_root=args.b37_expert58_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
