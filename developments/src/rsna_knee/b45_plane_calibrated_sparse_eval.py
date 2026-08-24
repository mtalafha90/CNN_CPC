"""Reused Expert-58 diagnostic for the frozen fixed-E2 B45 endpoint.

B45 is evaluated once after training is complete.  This evaluator is descriptive
only: Expert-58 is reused mechanistic evidence and is not a tuning or checkpoint
selection surface.  The fixed three offsets [-1, 0, +1] are scored and B45 is
compared with the already-recorded B42 parent and B37 hidden champion endpoint.
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
from .b12_variable_series import build_variable_series_index
from .b17_training import encoder_state_sha256
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b35_training import B35_EXPECTED_CELLS, B35_EXPECTED_SERIES, sha256_file
from .b37_highres_sparse_eval import (
    B37_BASE_TOLERANCE,
    B37_EVAL_OFFSETS,
    B37_EXPECTED_BASE_MACRO,
    FOCAL_SIX,
)
from .b37_highres_sparse_mil import B37_EXPERT58_ROOT
from .b42_constant_area_aspect_sparse_mil import (
    B42_EXPERT58_ROOT,
    B42ConstantAreaAspectDataset,
    collate_b42,
)
from .b45_plane_calibrated_sparse_mil import (
    B45_EXPERIMENT,
    B45_EXPERT58_ROOT,
    B45_VERSION,
    B45PlaneCalibratedSparseMILResidual,
    require_b45_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .phase9_supervision import REPORT_ONLY_STUDIES
from .runtime import autocast, resolve_runtime

B45_EVAL_LOADER_SEED_OFFSET = 54_300_000
B45_EVAL_BOOTSTRAP_B42_SEED_OFFSET = 54_400_000
B45_EVAL_BOOTSTRAP_B37_SEED_OFFSET = 54_500_000


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


def _focal(aucs: dict[str, float]) -> float:
    return float(np.mean([aucs[target] for target in FOCAL_SIX]))


def load_b45_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    device,
):
    """Reconstruct and verify the exact frozen B45 E2 model."""
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B45_EXPERIMENT:
        raise ValueError("checkpoint is not the fixed B45 experiment")
    if payload.get("version") != B45_VERSION:
        raise ValueError("checkpoint is not the fixed B45 version")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != 2:
        raise ValueError("B45 evaluation requires the completed fixed-E2 checkpoint")
    if str(payload.get("checkpoint_selection", "")) != "none; fixed epoch 2":
        raise ValueError("B45 checkpoint-selection contract changed")
    if int(payload.get("training_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B45 checkpoint training-study count changed")
    if int(payload.get("training_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B45 checkpoint training-series count changed")
    if int(payload.get("training_supervision_cells", -1)) != B35_EXPECTED_CELLS:
        raise ValueError("B45 checkpoint supervision surface changed")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B45 checkpoint unexpectedly used expert gradients")
    if bool(payload.get("gold_labels_used", True)):
        raise ValueError("B45 checkpoint unexpectedly used expert labels")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B45 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")

    sparse = payload.get("sparse_mil", {})
    finetune = payload.get("encoder_finetune", {})
    model_state = payload.get("model_state", {})
    routing = payload.get("plane_routing", {})
    model = B45PlaneCalibratedSparseMILResidual(
        base,
        grid_size=int(sparse.get("grid_size", 6)),
        top_k=int(sparse.get("top_k_per_available_plane", 8)),
        temperature=float(sparse.get("temperature", 1.0)),
        encoder_trainable_stages=int(finetune.get("encoder_trainable_stages", 1)),
        encoder_chunk_size=int(model_state.get("encoder_chunk_size", 4)),
        router_temperature=float(routing.get("router_temperature", 1.0)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()

    observed_encoder = encoder_state_sha256(model.base.encoder)
    expected_encoder = str(payload.get("encoder_sha256_final", ""))
    if observed_encoder != expected_encoder:
        raise RuntimeError("B45 reconstructed encoder fingerprint changed")
    if model.head.plane_embedding.weight.requires_grad:
        raise RuntimeError("B45 reconstructed forbidden trainable plane embedding")
    return model, payload


@torch.no_grad()
def evaluate_b45(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    b37_expert58_root: str | Path = B37_EXPERT58_ROOT,
    b42_expert58_root: str | Path = B42_EXPERT58_ROOT,
    out_root: str | Path = B45_EXPERT58_ROOT,
    n_bootstrap: int = 5000,
) -> dict:
    """Score frozen B45 once on reused Expert-58 and compare to B42/B37."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b45_contract(settings)
    settings["b7_eval_batch_size"] = 1
    root = Path(settings["data_root"])

    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B45 requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B45 Expert-58 MRI series surface changed")

    b37_root = Path(b37_expert58_root)
    b42_root = Path(b42_expert58_root)
    base_prediction = _read_predictions(b37_root / "base_224_predictions.csv", uids)
    b37_combined = _read_predictions(b37_root / "b37_combined_predictions.csv", uids)
    b42_combined = _read_predictions(b42_root / "b42_combined_predictions.csv", uids)

    base_macro, base_auc = _summary(truth, base_prediction)
    if abs(base_macro - B37_EXPECTED_BASE_MACRO) > B37_BASE_TOLERANCE:
        raise RuntimeError(
            f"historical base predictions changed: expected ~{B37_EXPECTED_BASE_MACRO:.10f}, "
            f"got {base_macro:.10f}"
        )
    b37_macro, b37_auc = _summary(truth, b37_combined)
    b42_macro, b42_auc = _summary(truth, b42_combined)

    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    dcfg = make_b7_dataset_config(settings, root, train=False)
    dcfg.tta_center_offsets = ()
    dataset = B42ConstantAreaAspectDataset(
        uids,
        index,
        dcfg,
        crop_focus_policy=crop_policy,
        center_offsets=B37_EVAL_OFFSETS,
        targets=truth.astype(np.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(
            seed=int(settings.get("seed", 2026)) + B45_EVAL_LOADER_SEED_OFFSET
        ),
    )

    model, payload = load_b45_checkpoint(
        checkpoint,
        base_checkpoint=base_checkpoint,
        device=runtime.device,
    )

    global_rows: list[np.ndarray] = []
    combined_rows: list[np.ndarray] = []
    scored_uids: list[str] = []
    effective_weight_sum = np.zeros((len(TARGETS), 3), dtype=np.float64)
    effective_weight_n = np.zeros((len(TARGETS), 3), dtype=np.float64)
    plane_available_counts = np.zeros(3, dtype=np.int64)

    for batch_index, items in enumerate(loader, start=1):
        if len(items) != 1:
            raise RuntimeError("B45 Expert-58 evaluation requires one ragged study per batch")
        item = items[0]
        uid = str(item["study_uid"])
        scored_uids.append(uid)

        present = item["present"].to(runtime.device, non_blocking=True)
        meta = item["series_meta"].to(runtime.device, non_blocking=True)
        position_all = item["slice_position"].to(runtime.device, non_blocking=True)
        if position_all.ndim != 3 or int(position_all.shape[1]) != len(B37_EVAL_OFFSETS):
            raise RuntimeError("B45 TTA slice-position shape changed")
        if not isinstance(item["volumes"], list) or not item["volumes"]:
            raise RuntimeError("B45 ragged volume list changed")

        global_views: list[torch.Tensor] = []
        combined_views: list[torch.Tensor] = []
        for view in range(len(B37_EVAL_OFFSETS)):
            volumes = [
                series_tensor[view].to(runtime.device, non_blocking=True)
                for series_tensor in item["volumes"]
            ]
            position = position_all[:, view]
            with autocast(runtime):
                output = model(volumes, present, meta, position)
            global_views.append(torch.sigmoid(output.base_logits.float()))
            combined_views.append(torch.sigmoid(output.logits.float()))

            weights = output.plane_weights[0].detach().float().cpu().numpy()
            available = output.plane_available[0].detach().cpu().numpy().astype(bool)
            plane_available_counts += available.astype(np.int64)
            for plane in range(3):
                if available[plane]:
                    effective_weight_sum[:, plane] += weights[:, plane]
                    effective_weight_n[:, plane] += 1.0
            del volumes, position, output

        global_rows.append(torch.stack(global_views).mean(dim=0).cpu().numpy())
        combined_rows.append(torch.stack(combined_views).mean(dim=0).cpu().numpy())
        del item, items, present, meta, position_all, global_views, combined_views
        _release()
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"[B45 Expert58] {batch_index}/{len(loader)}", flush=True)

    if scored_uids != uids:
        raise RuntimeError("B45 Expert-58 study order changed")
    global_prediction = np.concatenate(global_rows, axis=0)
    combined_prediction = np.concatenate(combined_rows, axis=0)
    if not np.isfinite(global_prediction).all() or not np.isfinite(combined_prediction).all():
        raise RuntimeError("B45 produced non-finite Expert-58 predictions")

    b45_global_macro, b45_global_auc = _summary(truth, global_prediction)
    b45_combined_macro, b45_combined_auc = _summary(truth, combined_prediction)

    effective_mean = np.divide(
        effective_weight_sum,
        effective_weight_n,
        out=np.zeros_like(effective_weight_sum),
        where=effective_weight_n > 0,
    )
    learned_router = payload.get("plane_routing", {}).get("router_weights_all_planes", [])

    per_target = {}
    for i, target in enumerate(TARGETS):
        per_target[target] = {
            "base_224_auc": base_auc[target],
            "b37_combined_auc": b37_auc[target],
            "b42_combined_auc": b42_auc[target],
            "b45_global_auc": b45_global_auc[target],
            "b45_combined_auc": b45_combined_auc[target],
            "b45_minus_b42": float(b45_combined_auc[target] - b42_auc[target]),
            "b45_minus_b37": float(b45_combined_auc[target] - b37_auc[target]),
            "learned_router_all_planes": learned_router[i] if len(learned_router) == len(TARGETS) else None,
            "mean_effective_router_on_expert58": effective_mean[i].tolist(),
        }

    result = {
        "evaluation_role": (
            "reused post-training Expert-58 descriptive diagnostic; not independent test evidence "
            "and not a B45 tuning or checkpoint-selection criterion"
        ),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint).resolve()),
        "fixed_endpoint": True,
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "studies": len(uids),
        "series": int(sum(counts)),
        "macro_auc": {
            "base_224": base_macro,
            "b37_combined": b37_macro,
            "b42_combined": b42_macro,
            "b45_global": b45_global_macro,
            "b45_combined": b45_combined_macro,
            "b45_minus_b42": float(b45_combined_macro - b42_macro),
            "b45_minus_b37": float(b45_combined_macro - b37_macro),
        },
        "focal_six_auc": {
            "base_224": _focal(base_auc),
            "b37_combined": _focal(b37_auc),
            "b42_combined": _focal(b42_auc),
            "b45_global": _focal(b45_global_auc),
            "b45_combined": _focal(b45_combined_auc),
        },
        "paired_bootstrap": {
            "b45_minus_b42": compare_runs(
                truth,
                b42_combined,
                combined_prediction,
                n_bootstrap=int(n_bootstrap),
                seed=int(settings.get("seed", 2026)) + B45_EVAL_BOOTSTRAP_B42_SEED_OFFSET,
            ),
            "b45_minus_b37": compare_runs(
                truth,
                b37_combined,
                combined_prediction,
                n_bootstrap=int(n_bootstrap),
                seed=int(settings.get("seed", 2026)) + B45_EVAL_BOOTSTRAP_B37_SEED_OFFSET,
            ),
        },
        "plane_router": {
            "learned_weights_all_planes": learned_router,
            "mean_effective_weights_expert58": effective_mean.tolist(),
            "available_plane_view_counts": plane_available_counts.tolist(),
        },
        "per_target": per_target,
        "metadata_repair": metadata_stats,
        "governance": (
            "B45 is a prospectively frozen E2 endpoint. Expert-58 results are descriptive reuse only; "
            "do not change plane routing, pooling, geometry, learning rates, epoch count, target subset, "
            "thresholds, or checkpoint based on this evaluation."
        ),
    }

    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, prediction in (
        ("b45_global", global_prediction),
        ("b45_combined", combined_prediction),
    ):
        frame = pd.DataFrame(prediction, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(output_root / f"{name}_predictions.csv", index=False)
    (output_root / "expert58.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("B45 EXPERT58 DIAGNOSTIC: PASS", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate frozen B45 plane-calibrated sparse MIL")
    parser.add_argument("--config", default="config/b45_plane_calibrated_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--b37-expert58-root", default=B37_EXPERT58_ROOT)
    parser.add_argument("--b42-expert58-root", default=B42_EXPERT58_ROOT)
    parser.add_argument("--out-root", default=B45_EXPERT58_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b45(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        b37_expert58_root=args.b37_expert58_root,
        b42_expert58_root=args.b42_expert58_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()


__all__ = ["evaluate_b45", "load_b45_checkpoint"]
