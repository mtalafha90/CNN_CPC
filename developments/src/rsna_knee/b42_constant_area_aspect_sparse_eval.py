"""Reused Expert-58 diagnostic for the fixed B42 constant-area endpoint.

B42 is evaluated once after its prospective fixed-E2 training endpoint exists.
The evaluator does not tune B42. It scores B42 with the frozen three offsets
[-1, 0, +1], compares against the already-recorded B37 and B41 Expert-58
predictions, and records paired B42-minus-B37/B41 bootstrap diagnostics.
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
from .b41_highres_aspect_sparse_mil import B41_EXPERT58_ROOT
from .b42_constant_area_aspect_sparse_mil import (
    B42_EXPERIMENT,
    B42_EXPERT58_ROOT,
    B42_REFERENCE_AREA,
    B42_STRIDE_ALIGNMENT,
    B42_VERSION,
    B42ConstantAreaAspectDataset,
    B42ConstantAreaAspectSparseMILResidual,
    b42_preprocessing_state,
    collate_b42,
    require_b42_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .phase9_supervision import REPORT_ONLY_STUDIES
from .runtime import autocast, resolve_runtime

B42_EVAL_LOADER_SEED_OFFSET = 52_300_000
B42_EVAL_BOOTSTRAP_B37_SEED_OFFSET = 52_400_000
B42_EVAL_BOOTSTRAP_B41_SEED_OFFSET = 52_500_000
B42_EVAL_BOOTSTRAP_BASE_SEED_OFFSET = 52_600_000


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_b42_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    device,
):
    """Reconstruct and verify the exact fixed-E2 B42 ragged model."""
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B42_EXPERIMENT:
        raise ValueError("checkpoint is not the fixed B42 experiment")
    if payload.get("version") != B42_VERSION:
        raise ValueError("checkpoint is not the fixed B42 version")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != 2:
        raise ValueError("B42 evaluation requires the completed fixed-E2 checkpoint")
    if int(payload.get("training_studies", -1)) != REPORT_ONLY_STUDIES:
        raise ValueError("B42 checkpoint training-study count changed")
    if int(payload.get("training_series", -1)) != B35_EXPECTED_SERIES:
        raise ValueError("B42 checkpoint training-series count changed")
    if int(payload.get("training_supervision_cells", -1)) != B35_EXPECTED_CELLS:
        raise ValueError("B42 checkpoint supervision surface changed")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B42 checkpoint unexpectedly used expert gradients")
    if bool(payload.get("gold_labels_used", True)):
        raise ValueError("B42 checkpoint unexpectedly used expert labels")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B42 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")

    sparse = payload.get("sparse_mil", {})
    finetune = payload.get("encoder_finetune", {})
    model_state = payload.get("model_state", {})
    model = B42ConstantAreaAspectSparseMILResidual(
        base,
        grid_size=int(sparse.get("grid_size", 6)),
        top_k=int(sparse.get("top_k", 8)),
        temperature=float(sparse.get("temperature", 1.0)),
        encoder_trainable_stages=int(finetune.get("encoder_trainable_stages", 1)),
        encoder_chunk_size=int(model_state.get("encoder_chunk_size", 4)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()

    observed_encoder = encoder_state_sha256(model.base.encoder)
    expected_encoder = str(payload.get("encoder_sha256_final", ""))
    if observed_encoder != expected_encoder:
        raise RuntimeError("B42 reconstructed encoder fingerprint changed")
    return model, payload


def _read_reference_predictions(path: Path, uids: list[str]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing historical Expert-58 predictions: {path}")
    frame = pd.read_csv(path)
    required = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != required:
        raise ValueError(f"prediction columns changed: {path}")
    observed_uids = frame["StudyInstanceUID"].astype(str).tolist()
    if observed_uids != uids:
        raise RuntimeError(f"prediction UID order changed: {path}")
    prediction = frame[TARGETS].to_numpy(np.float64)
    if prediction.shape != (len(uids), len(TARGETS)) or not np.isfinite(prediction).all():
        raise RuntimeError(f"invalid prediction matrix: {path}")
    return prediction


def _summary(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, dict[str, float]]:
    macro, auc = macro_auc_from_arrays(truth, prediction)
    return float(macro), {target: float(value) for target, value in zip(TARGETS, auc)}


def _focal(aucs: dict[str, float]) -> float:
    return float(np.mean([aucs[target] for target in FOCAL_SIX]))


@torch.no_grad()
def evaluate_b42(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    b37_expert58_root: str | Path = B37_EXPERT58_ROOT,
    b41_expert58_root: str | Path = B41_EXPERT58_ROOT,
    out_root: str | Path = B42_EXPERT58_ROOT,
    n_bootstrap: int = 5000,
    experiment_label: str = "b42",
) -> dict:
    """Score a B42-shaped endpoint against recorded B37/B41 Expert-58 outputs.

    `experiment_label` names what is being scored, in the written filenames and
    in the result keys. It defaults to "b42" so every existing B42 evaluation is
    unchanged. A sibling endpoint that reuses this exact inference recipe -- B51
    does, since B50's class alters only requires_grad -- passes its own label, so
    its results cannot be mistaken for B42's months later.
    """
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b42_contract(settings)
    settings["b7_eval_batch_size"] = 1
    root = Path(settings["data_root"])

    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B42 requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B42 Expert-58 MRI series surface changed")

    b37_root = Path(b37_expert58_root)
    b41_root = Path(b41_expert58_root)
    base_prediction = _read_reference_predictions(b37_root / "base_224_predictions.csv", uids)
    b37_global = _read_reference_predictions(b37_root / "b37_global_448_predictions.csv", uids)
    b37_combined = _read_reference_predictions(b37_root / "b37_combined_predictions.csv", uids)
    b41_global = _read_reference_predictions(b41_root / "b41_global_448_predictions.csv", uids)
    b41_combined = _read_reference_predictions(b41_root / "b41_combined_predictions.csv", uids)

    base_macro, base_auc = _summary(truth, base_prediction)
    if abs(base_macro - B37_EXPECTED_BASE_MACRO) > B37_BASE_TOLERANCE:
        raise RuntimeError(
            f"historical base predictions changed: expected ~{B37_EXPECTED_BASE_MACRO:.10f}, "
            f"got {base_macro:.10f}"
        )
    b37_global_macro, b37_global_auc = _summary(truth, b37_global)
    b37_combined_macro, b37_combined_auc = _summary(truth, b37_combined)
    b41_global_macro, b41_global_auc = _summary(truth, b41_global)
    b41_combined_macro, b41_combined_auc = _summary(truth, b41_combined)

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
            seed=int(settings.get("seed", 2026)) + B42_EVAL_LOADER_SEED_OFFSET
        ),
    )

    model, payload = load_b42_checkpoint(
        checkpoint,
        base_checkpoint=base_checkpoint,
        device=runtime.device,
    )

    global_rows: list[np.ndarray] = []
    combined_rows: list[np.ndarray] = []
    scored_uids: list[str] = []
    top1_sum = np.zeros(len(TARGETS), dtype=np.float64)
    topk_sum = np.zeros(len(TARGETS), dtype=np.float64)
    gap_sum = np.zeros(len(TARGETS), dtype=np.float64)
    selected_count = 0
    unique_sum = 0
    unique_n = 0
    geometry_rows: list[dict] = []

    for batch_index, items in enumerate(loader, start=1):
        if len(items) != 1:
            raise RuntimeError("B42 Expert-58 evaluation requires one ragged study per batch")
        item = items[0]
        uid = str(item["study_uid"])
        scored_uids.append(uid)

        present_cpu = item["present"]
        meta = item["series_meta"].to(runtime.device, non_blocking=True)
        present = present_cpu.to(runtime.device, non_blocking=True)
        position_all = item["slice_position"].to(runtime.device, non_blocking=True)
        if position_all.ndim != 3 or int(position_all.shape[1]) != len(B37_EVAL_OFFSETS):
            raise RuntimeError("B42 TTA slice-position shape changed")
        if not isinstance(item["volumes"], list) or not item["volumes"]:
            raise RuntimeError("B42 ragged volume list changed")

        for geom, flag in zip(item["geometry"], present_cpu):
            if float(flag.item()) <= 0:
                continue
            height = int(geom["height"])
            width = int(geom["width"])
            if height % B42_STRIDE_ALIGNMENT or width % B42_STRIDE_ALIGNMENT:
                raise RuntimeError("B42 evaluated tensor is not stride aligned")
            feature_h = height // B42_STRIDE_ALIGNMENT
            feature_w = width // B42_STRIDE_ALIGNMENT
            feature_cells = feature_h * feature_w
            geometry_rows.append(
                {
                    "StudyInstanceUID": uid,
                    "series_uid": str(geom["series_uid"]),
                    "height": height,
                    "width": width,
                    "tensor_pixels": int(height * width),
                    "tensor_area_vs_448sq": float((height * width) / B42_REFERENCE_AREA),
                    "feature_height": feature_h,
                    "feature_width": feature_w,
                    "feature_cells": feature_cells,
                    "feature_cells_vs_14x14": float(feature_cells / 196.0),
                    "rectangular": bool(height != width),
                }
            )

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

            values = output.top_values.float()
            top1_sum += values[:, :, 0].sum(dim=0).cpu().numpy()
            topk_sum += values.mean(dim=-1).sum(dim=0).cpu().numpy()
            gap_sum += (values[:, :, 0] - values[:, :, -1]).sum(dim=0).cpu().numpy()
            selected_count += int(values.shape[0])
            indices = output.top_indices.detach().cpu()
            for bi in range(indices.shape[0]):
                unique_sum += int(torch.unique(indices[bi]).numel())
                unique_n += 1
            del volumes, position, output

        global_rows.append(torch.stack(global_views).mean(dim=0).cpu().numpy())
        combined_rows.append(torch.stack(combined_views).mean(dim=0).cpu().numpy())
        del item, items, meta, present, present_cpu, position_all, global_views, combined_views
        _release()
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"[B42 Expert58] {batch_index}/{len(loader)}", flush=True)

    if scored_uids != uids:
        raise RuntimeError("B42 Expert-58 study order changed")
    global_prediction = np.concatenate(global_rows, axis=0)
    combined_prediction = np.concatenate(combined_rows, axis=0)
    if not np.isfinite(global_prediction).all() or not np.isfinite(combined_prediction).all():
        raise RuntimeError("B42 produced non-finite Expert-58 predictions")

    b42_global_macro, b42_global_auc = _summary(truth, global_prediction)
    b42_combined_macro, b42_combined_auc = _summary(truth, combined_prediction)

    focal = {
        "base_224": _focal(base_auc),
        "b37_global_448": _focal(b37_global_auc),
        "b37_combined": _focal(b37_combined_auc),
        "b41_global_448": _focal(b41_global_auc),
        "b41_combined": _focal(b41_combined_auc),
        f"{experiment_label}_global_rectangular": _focal(b42_global_auc),
        f"{experiment_label}_combined": _focal(b42_combined_auc),
    }

    per_target = {}
    for target in TARGETS:
        per_target[target] = {
            "base_224_auc": base_auc[target],
            "b37_global_448_auc": b37_global_auc[target],
            "b37_combined_auc": b37_combined_auc[target],
            "b41_global_448_auc": b41_global_auc[target],
            "b41_combined_auc": b41_combined_auc[target],
            "b42_global_rectangular_auc": b42_global_auc[target],
            "b42_combined_auc": b42_combined_auc[target],
            "b42_minus_b37_combined": float(b42_combined_auc[target] - b37_combined_auc[target]),
            "b42_minus_b41_combined": float(b42_combined_auc[target] - b41_combined_auc[target]),
            "b42_sparse_residual_increment": float(b42_combined_auc[target] - b42_global_auc[target]),
        }

    geometry = pd.DataFrame(geometry_rows)
    if len(geometry) != B18_EXPECTED_GOLD_SERIES:
        raise RuntimeError(
            f"B42 geometry audit expected {B18_EXPECTED_GOLD_SERIES} readable series; got {len(geometry)}"
        )
    geometry_summary = {
        "n_series": int(len(geometry)),
        "rectangular_series": int(geometry["rectangular"].sum()),
        "square_series": int((~geometry["rectangular"]).sum()),
        "height_min": int(geometry["height"].min()),
        "height_median": float(geometry["height"].median()),
        "height_max": int(geometry["height"].max()),
        "width_min": int(geometry["width"].min()),
        "width_median": float(geometry["width"].median()),
        "width_max": int(geometry["width"].max()),
        "tensor_pixels_median": float(geometry["tensor_pixels"].median()),
        "tensor_area_vs_448sq_median": float(geometry["tensor_area_vs_448sq"].median()),
        "feature_cells_min": int(geometry["feature_cells"].min()),
        "feature_cells_median": float(geometry["feature_cells"].median()),
        "feature_cells_max": int(geometry["feature_cells"].max()),
        "feature_cells_vs_14x14_median": float(geometry["feature_cells_vs_14x14"].median()),
    }

    result = {
        "evaluation_role": (
            f"reused Expert-58 development diagnostic for the {experiment_label} endpoint; "
            f"not independent test evidence and not a {experiment_label} tuning or promotion criterion"
        ),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint).resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "n_studies": len(uids),
        "n_series": int(sum(counts)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "macro_auc": {
            "base_224": base_macro,
            "b37_global_448": b37_global_macro,
            "b37_combined": b37_combined_macro,
            "b41_global_448": b41_global_macro,
            "b41_combined": b41_combined_macro,
            f"{experiment_label}_global_rectangular": b42_global_macro,
            f"{experiment_label}_combined": b42_combined_macro,
        },
        f"{experiment_label}_sparse_residual_macro_increment": float(b42_combined_macro - b42_global_macro),
        f"{experiment_label}_minus_b37_combined_macro": float(b42_combined_macro - b37_combined_macro),
        f"{experiment_label}_minus_b41_combined_macro": float(b42_combined_macro - b41_combined_macro),
        "focal_six": list(FOCAL_SIX),
        "focal_six_auc": focal,
        f"{experiment_label}_minus_b37_focal_six": float(focal[f"{experiment_label}_combined"] - focal["b37_combined"]),
        f"{experiment_label}_minus_b41_focal_six": float(focal[f"{experiment_label}_combined"] - focal["b41_combined"]),
        "per_target": per_target,
        f"paired_{experiment_label}_minus_b37_bootstrap": compare_runs(
            truth,
            b37_combined,
            combined_prediction,
            n_bootstrap=int(n_bootstrap),
            seed=int(settings.get("seed", 2026)) + B42_EVAL_BOOTSTRAP_B37_SEED_OFFSET,
        ),
        f"paired_{experiment_label}_minus_b41_bootstrap": compare_runs(
            truth,
            b41_combined,
            combined_prediction,
            n_bootstrap=int(n_bootstrap),
            seed=int(settings.get("seed", 2026)) + B42_EVAL_BOOTSTRAP_B41_SEED_OFFSET,
        ),
        f"paired_{experiment_label}_minus_base_bootstrap": compare_runs(
            truth,
            base_prediction,
            combined_prediction,
            n_bootstrap=int(n_bootstrap),
            seed=int(settings.get("seed", 2026)) + B42_EVAL_BOOTSTRAP_BASE_SEED_OFFSET,
        ),
        "mil": {
            "top_k": int(model.head.top_k),
            "grid_size": int(model.head.grid_size),
            "regions_per_slice": int(model.head.n_regions),
            "mean_unique_selected_locations_across_12_targets": float(unique_sum / max(unique_n, 1)),
            "max_possible_selected_locations_across_12_targets": int(len(TARGETS) * model.head.top_k),
            "per_target": {
                target: {
                    "mean_top1_evidence": float(top1_sum[i] / max(selected_count, 1)),
                    "mean_topk_evidence": float(topk_sum[i] / max(selected_count, 1)),
                    "mean_top1_minus_kth": float(gap_sum[i] / max(selected_count, 1)),
                }
                for i, target in enumerate(TARGETS)
            },
        },
        "geometry": geometry_summary,
        "preprocessing": b42_preprocessing_state(),
        "training_contract": {
            "completed_epochs": int(payload.get("completed_epochs", -1)),
            "training_studies": int(payload.get("training_studies", -1)),
            "training_series": int(payload.get("training_series", -1)),
            "training_supervision_cells": int(payload.get("training_supervision_cells", -1)),
            "encoder_sha256_initial": payload.get("encoder_sha256_initial"),
            "encoder_sha256_final": payload.get("encoder_sha256_final"),
        },
        "metadata_repair": metadata_stats,
        # Which weights produced these numbers. Without this the folder is just
        # a set of Expert-58 scores with nothing saying what was scored.
        "evaluated_endpoint": {
            "label": experiment_label,
            "experiment": payload.get("experiment"),
            "version": payload.get("version"),
            "converted_from": payload.get("converted_from"),
        },
        "governance": (
            "B42 remains fixed at reference area 448^2, native 90% crop, isotropic "
            "constant-area resize, reflection padding only to stride 32, ragged-series "
            "encoding, 32 centres, 6x6, top-k=8, B37 supervision and fixed epoch 2. "
            "Do not tune B42 from this reused Expert-58 diagnostic."
        ),
    }

    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, prediction in (
        (f"{experiment_label}_global_rectangular", global_prediction),
        (f"{experiment_label}_combined", combined_prediction),
    ):
        frame = pd.DataFrame(prediction, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(output_root / f"{name}_predictions.csv", index=False)
    geometry.to_csv(output_root / "geometry.csv", index=False)
    (output_root / "expert58.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate fixed B42 constant-area rectangular sparse MIL")
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--b37-expert58-root", default=B37_EXPERT58_ROOT)
    parser.add_argument("--b41-expert58-root", default=B41_EXPERT58_ROOT)
    parser.add_argument("--out-root", default=B42_EXPERT58_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    evaluate_b42(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        b37_expert58_root=args.b37_expert58_root,
        b41_expert58_root=args.b41_expert58_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
