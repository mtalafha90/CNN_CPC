"""Perturbation-based occlusion sensitivity for the active B20 working model.

No training is performed. Local patches are replaced with a blur of the same MRI
content. Positive delta_probability means removing that information lowers the
target probability. This is model-dependence evidence, not lesion segmentation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b18_visualize import (
    CAM_LAYERS,
    _choose_success,
    _gradcam_for_study,
    _load_gold_surface,
    _normalize_cam,
    _resolve_target,
    _view_probabilities,
)
from .b20_crop_focus import (
    CropFocusedVariableSeriesKneeDataset,
    load_b20_checkpoint,
    require_b20_contract,
)
from .b20_occlusion_utils import (
    accumulate_patch_map,
    blur_tokens,
    cached_probability,
    encode_modified_tokens,
    grid_starts,
    iou,
    metadata_vector,
    pearson,
    positive_percentile_mask,
    save_panel,
)
from .b20_visualize import _predict_gold
from .constants import TARGETS, TARGET_SLUGS
from .runtime import autocast, resolve_runtime

OCCLUSION_VARIANT = "b20_blur_occlusion_sensitivity_v1"


def _direct_probability(model, volumes, present, series_meta, runtime, target_idx):
    model.eval()
    with torch.no_grad(), autocast(runtime):
        logits = model(volumes, present, series_meta)
    return float(torch.sigmoid(logits.float())[0, target_idx].cpu())


def run_occlusion(
    config: dict,
    *,
    checkpoint: str | Path,
    uid: str,
    target: str,
    out_dir: str | Path,
    probability_threshold: float = 0.50,
    view_offset: int | None = None,
    series_index: int | None = None,
    slice_index: int | None = None,
    cam_layer: str = "28x28",
    patch_size: int = 28,
    stride: int = 14,
    blur_kernel: int = 15,
    scope: str = "slice",
) -> Path:
    if scope not in {"slice", "series"}:
        raise ValueError("scope must be 'slice' or 'series'")
    if (series_index is None) != (slice_index is None):
        raise ValueError("series_index and slice_index must be supplied together")
    if cam_layer not in CAM_LAYERS:
        raise ValueError(f"unknown CAM layer {cam_layer!r}")

    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, payload = load_b20_checkpoint(checkpoint, device=runtime.device)
    model.eval()
    if payload.get("crop_focus_policy") != crop_policy:
        raise ValueError("B20 occlusion config crop policy differs from checkpoint")

    root = Path(config["data_root"])
    gold, variable_index, metadata_stats = _load_gold_surface(config, root)
    prediction, offsets = _predict_gold(
        model, config, root, gold, variable_index, runtime, crop_policy
    )
    target_idx = _resolve_target(target)
    if target_idx is None:
        raise ValueError("B20 occlusion requires an explicit target")
    row_idx, selected_target = _choose_success(
        gold,
        prediction,
        target_idx=target_idx,
        uid=str(uid),
        probability_threshold=float(probability_threshold),
    )
    if int(selected_target) != int(target_idx):
        raise RuntimeError("target-selection mismatch")

    uid = str(gold.iloc[row_idx]["StudyInstanceUID"])
    truth = float(gold.iloc[row_idx][TARGETS[target_idx]])
    tta_probability = float(prediction[row_idx, target_idx])

    one_ds = CropFocusedVariableSeriesKneeDataset(
        [uid],
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=np.asarray(
            [[float(gold.iloc[row_idx][name]) for name in TARGETS]], dtype=np.float32
        ),
        train=False,
        crop_focus_policy=crop_policy,
    )
    item = one_ds[0]
    volumes_views = item["volumes"]
    present = item["present"].unsqueeze(0).to(runtime.device)
    series_meta = item["series_meta"].unsqueeze(0).to(runtime.device)

    model.eval()
    view_probs = _view_probabilities(
        model, volumes_views, present, series_meta, runtime, target_idx
    )
    if view_offset is None:
        view_idx = int(np.argmax(view_probs))
    else:
        if int(view_offset) not in offsets:
            raise ValueError(f"view_offset must be one of {list(offsets)}")
        view_idx = offsets.index(int(view_offset))
    explained_offset = int(offsets[view_idx])
    volumes = volumes_views[view_idx].unsqueeze(0).to(runtime.device)

    cams, importance, gradcam_probability, pairs = _gradcam_for_study(
        model,
        volumes,
        present,
        series_meta,
        target_idx=target_idx,
        runtime=runtime,
        cam_layer=cam_layer,
    )
    if series_index is None:
        selected_row = int(torch.argmax(importance).item())
    else:
        requested = (int(series_index), int(slice_index))
        if requested not in pairs:
            raise ValueError(f"requested series/slice pair {requested} is not active")
        selected_row = pairs.index(requested)
    series_idx, slice_idx = pairs[selected_row]
    record = variable_index[uid][series_idx]

    image = volumes[0, series_idx, slice_idx, 1].detach().float().cpu().numpy()
    cam_up = F.interpolate(
        cams[selected_row][None, None], size=image.shape, mode="bilinear", align_corners=False
    )[0, 0].detach().float().cpu().numpy()
    cam = _normalize_cam(cam_up)

    # Cache all unchanged frozen-encoder features once. Every perturbation below
    # re-encodes only the modified token(s), then runs the unchanged B20 head.
    model.eval()
    with torch.no_grad(), autocast(runtime):
        baseline_features = model._encode_slices(volumes, present, series_meta)
        cached_baseline = cached_probability(model, baseline_features, present, target_idx)
    direct_baseline = _direct_probability(
        model, volumes, present, series_meta, runtime, target_idx
    )
    consistency_delta = abs(direct_baseline - cached_baseline)
    if consistency_delta > 2e-4:
        raise RuntimeError(
            "cached-feature path does not reproduce direct B20 inference: "
            f"direct={direct_baseline:.8f}, cached={cached_baseline:.8f}, "
            f"delta={consistency_delta:.3g}"
        )

    h, w = image.shape
    y_starts = grid_starts(h, int(patch_size), int(stride))
    x_starts = grid_starts(w, int(patch_size), int(stride))

    if scope == "slice":
        source_tokens = volumes[0, series_idx, slice_idx : slice_idx + 1].detach()
        position_indices = [slice_idx]
    else:
        source_tokens = volumes[0, series_idx].detach()
        position_indices = list(range(int(source_tokens.shape[0])))
    blurred = blur_tokens(source_tokens, int(blur_kernel))
    metadata = metadata_vector(model, series_meta, series_idx)

    rows = []
    total_patches = len(y_starts) * len(x_starts)
    patch_counter = 0
    print(
        f"[B20 occlusion] UID={uid} target={TARGETS[target_idx]} "
        f"plane={record['plane']} series={series_idx} slice={slice_idx} "
        f"scope={scope} patches={total_patches}",
        flush=True,
    )

    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + int(patch_size), h)
                x1 = min(x0 + int(patch_size), w)
                modified = source_tokens.clone()
                modified[:, :, y0:y1, x0:x1] = blurred[:, :, y0:y1, x0:x1]

                with autocast(runtime):
                    encoded = encode_modified_tokens(model, modified)
                    pos_idx = torch.as_tensor(
                        position_indices, device=encoded.device, dtype=torch.long
                    )
                    modified_features = (
                        encoded + model.slice_position[pos_idx] + metadata[None]
                    )
                    candidate = baseline_features.clone()
                    if scope == "slice":
                        candidate[0, series_idx, slice_idx] = modified_features[0]
                    else:
                        candidate[0, series_idx, : modified_features.shape[0]] = modified_features
                    probability = cached_probability(
                        model, candidate, present, target_idx
                    )

                delta = float(cached_baseline - probability)
                rows.append(
                    {
                        "patch_index": patch_counter,
                        "x0": int(x0),
                        "x1": int(x1),
                        "y0": int(y0),
                        "y1": int(y1),
                        "center_x": float((x0 + x1 - 1) / 2),
                        "center_y": float((y0 + y1 - 1) / 2),
                        "occluded_probability": float(probability),
                        "delta_probability": delta,
                        "supportive": bool(delta > 0),
                    }
                )
                patch_counter += 1
                if patch_counter % 25 == 0 or patch_counter == total_patches:
                    print(
                        f"[B20 occlusion] {patch_counter}/{total_patches} patches",
                        flush=True,
                    )

    occ_map, coverage = accumulate_patch_map(h, w, rows)
    positive_map = np.maximum(occ_map, 0.0)
    cam_top20 = positive_percentile_mask(cam, 80)
    occ_top20 = positive_percentile_mask(positive_map, 80)
    top_rows = sorted(rows, key=lambda r: float(r["delta_probability"]), reverse=True)

    summary = {
        "variant": OCCLUSION_VARIANT,
        "model": "B20_crop_only_joint_focus",
        "checkpoint": str(Path(checkpoint)),
        "selected_epoch": int(payload.get("selected_epoch", -1)),
        "crop_focus_policy": crop_policy,
        "study_uid": uid,
        "target": TARGETS[target_idx],
        "target_slug": TARGET_SLUGS[target_idx],
        "expert_truth": truth,
        "frozen_three_view_tta_probability": tta_probability,
        "per_view_probabilities": {
            str(offset): float(prob) for offset, prob in zip(offsets, view_probs)
        },
        "explained_view_offset": explained_offset,
        "explained_view_probability_gradcam_forward": float(gradcam_probability),
        "direct_baseline_probability": float(direct_baseline),
        "cached_baseline_probability": float(cached_baseline),
        "cached_vs_direct_absolute_delta": float(consistency_delta),
        "selected_series_index": int(series_idx),
        "selected_series_uid": str(record["series_uid"]),
        "selected_plane": str(record["plane"]),
        "selected_slice_index": int(slice_idx),
        "cam_layer": cam_layer,
        "scope": scope,
        "patch_size": int(patch_size),
        "stride": int(stride),
        "blur_kernel": int(blur_kernel),
        "n_patches": int(len(rows)),
        "max_probability_drop": float(max(r["delta_probability"] for r in rows)),
        "max_probability_increase_after_occlusion": float(
            max(-float(r["delta_probability"]) for r in rows)
        ),
        "mean_absolute_patch_delta": float(
            np.mean([abs(float(r["delta_probability"])) for r in rows])
        ),
        "supportive_patch_fraction": float(
            np.mean([float(r["delta_probability"]) > 0 for r in rows])
        ),
        "cam_vs_positive_occlusion_pearson": pearson(cam, positive_map),
        "top20_cam_vs_top20_positive_occlusion_iou": iou(cam_top20, occ_top20),
        "top_five_supportive_patches": top_rows[:5],
        "metadata_repair": metadata_stats,
        "interpretation": (
            "Positive delta_probability means local blur lowered the target probability. "
            "This is perturbation-based model dependence, not lesion segmentation or independent validation."
        ),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{uid}_{TARGET_SLUGS[target_idx]}_{scope}_occlusion"
    pd.DataFrame(rows).sort_values(
        "delta_probability", ascending=False, ignore_index=True
    ).to_csv(out / f"{stem}_patches.csv", index=False)
    np.save(out / f"{stem}_map.npy", occ_map.astype(np.float32))
    np.save(out / f"{stem}_coverage.npy", coverage.astype(np.float32))
    (out / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    title = (
        f"B20 causal occlusion | {TARGETS[target_idx]} | truth={truth:.0f} | "
        f"baseline p={direct_baseline:.3f} | view {explained_offset:+d} | "
        f"{record['plane']} | series {series_idx} | slice {slice_idx} | scope={scope}"
    )
    panel_path = out / f"{stem}_panel.png"
    save_panel(
        panel_path, image=image, cam=cam, occlusion=occ_map, records=rows, title=title
    )
    print(json.dumps(summary, indent=2))
    print(panel_path)
    return panel_path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20-occlusion")
    parser.add_argument("--config", default="configs/b20_crop_focus.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default="runs/b20_crop_focus/b20_model.pt")
    parser.add_argument("--uid", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.50)
    parser.add_argument("--view-offset", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--series-index", type=int, default=None)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--cam-layer", choices=sorted(CAM_LAYERS), default="28x28")
    parser.add_argument("--patch-size", type=int, default=28)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--blur-kernel", type=int, default=15)
    parser.add_argument(
        "--scope",
        choices=["slice", "series"],
        default="slice",
        help="slice perturbs one 2.5D token; series applies the same patch to all sampled tokens in that MRI series",
    )
    parser.add_argument("--out-dir", default="runs/b20_crop_focus/occlusion")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    run_occlusion(
        config,
        checkpoint=args.checkpoint,
        uid=args.uid,
        target=args.target,
        out_dir=args.out_dir,
        probability_threshold=args.probability_threshold,
        view_offset=args.view_offset,
        series_index=args.series_index,
        slice_index=args.slice_index,
        cam_layer=args.cam_layer,
        patch_size=args.patch_size,
        stride=args.stride,
        blur_kernel=args.blur_kernel,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
