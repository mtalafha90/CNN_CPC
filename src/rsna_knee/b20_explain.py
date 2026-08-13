"""Robust B20 explanation view for diagnosing focal versus diffuse Grad-CAM evidence.

This is an interpretation diagnostic only. It does not retrain B20 and does not
pretend that Grad-CAM is lesion segmentation. The legacy fixed absolute CAM
threshold is shown alongside two less peak-sensitive summaries:

* percentile mask: pixels in the top CAM percentile;
* cumulative-mass mask: smallest high-CAM set containing a requested fraction
  of total positive CAM mass.

These views help distinguish a genuinely off-target saliency pattern from a
visualization artifact caused by normalizing to one extreme CAM maximum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from .b20_visualize import _predict_gold
from .constants import TARGETS, TARGET_SLUGS
from .runtime import resolve_runtime


def _mass_mask(cam: np.ndarray, fraction: float) -> np.ndarray:
    values = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=bool)
    flat = values.reshape(-1)
    order = np.argsort(-flat, kind="mergesort")
    cumulative = np.cumsum(flat[order]) / total
    stop = int(np.searchsorted(cumulative, fraction, side="left"))
    stop = min(stop, len(order) - 1)
    mask = np.zeros(flat.shape, dtype=bool)
    mask[order[: stop + 1]] = True
    return mask.reshape(values.shape)


def _cam_metrics(cam: np.ndarray) -> dict:
    values = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    h, w = values.shape
    total = float(values.sum())
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
    if total > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        com_y = float((values * yy).sum() / total)
        com_x = float((values * xx).sum() / total)
        p = values.reshape(-1) / total
        positive = p > 0
        entropy = float(-(p[positive] * np.log(p[positive])).sum() / np.log(p.size))
    else:
        com_y = com_x = entropy = float("nan")
    return {
        "cam_peak_y": int(peak_y),
        "cam_peak_x": int(peak_x),
        "cam_peak_y_norm": float(peak_y / max(h - 1, 1)),
        "cam_peak_x_norm": float(peak_x / max(w - 1, 1)),
        "cam_center_of_mass_y": com_y,
        "cam_center_of_mass_x": com_x,
        "cam_center_of_mass_y_norm": float(com_y / max(h - 1, 1)),
        "cam_center_of_mass_x_norm": float(com_x / max(w - 1, 1)),
        "cam_normalized_entropy": entropy,
        "cam_mean": float(values.mean()),
        "cam_std": float(values.std()),
        "cam_p50": float(np.percentile(values, 50)),
        "cam_p75": float(np.percentile(values, 75)),
        "cam_p80": float(np.percentile(values, 80)),
        "cam_p90": float(np.percentile(values, 90)),
        "cam_p95": float(np.percentile(values, 95)),
        "cam_p99": float(np.percentile(values, 99)),
    }


def _save_panel(
    out_dir: Path,
    stem: str,
    image: np.ndarray,
    cam: np.ndarray,
    absolute_mask: np.ndarray,
    percentile_mask: np.ndarray,
    mass_mask: np.ndarray,
    *,
    title: str,
    absolute_threshold: float,
    percentile: float,
    mass_fraction: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.asarray(image, dtype=np.float32), 0, 1)
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0, 1)

    plt.imsave(out_dir / f"{stem}_original.png", image, cmap="gray", vmin=0, vmax=1)
    plt.imsave(out_dir / f"{stem}_gradcam_continuous.png", cam, cmap="turbo", vmin=0, vmax=1)
    plt.imsave(out_dir / f"{stem}_mask_absolute.png", absolute_mask.astype(float), cmap="gray", vmin=0, vmax=1)
    plt.imsave(out_dir / f"{stem}_mask_percentile.png", percentile_mask.astype(float), cmap="gray", vmin=0, vmax=1)
    plt.imsave(out_dir / f"{stem}_mask_mass.png", mass_mask.astype(float), cmap="gray", vmin=0, vmax=1)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.7), constrained_layout=True)

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(cam, cmap="turbo", alpha=0.48, vmin=0, vmax=1)
    axes[1].set_title("Continuous Grad-CAM")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    levels = [np.percentile(cam, q) for q in (80, 90, 95)]
    levels = sorted(set(float(x) for x in levels if np.isfinite(x) and 0 < x < 1))
    if levels:
        axes[2].contour(cam, levels=levels, linewidths=1.2)
    axes[2].set_title("CAM contours\n80/90/95th pct")

    axes[3].imshow(absolute_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Absolute mask\nCAM >= {absolute_threshold:.2f}")

    axes[4].imshow(percentile_mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title(f"Top CAM region\n>= p{percentile:g}")

    axes[5].imshow(image, cmap="gray", vmin=0, vmax=1)
    if mass_mask.any() and (~mass_mask).any():
        axes[5].contour(mass_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.4)
    if percentile_mask.any() and (~percentile_mask).any():
        axes[5].contour(percentile_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
    axes[5].set_title(f"Evidence extent\nyellow={mass_fraction:.0%} mass; cyan=p{percentile:g}")

    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    path = out_dir / f"{stem}_explanation_panel.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20-explain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.50)
    parser.add_argument("--view-offset", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--series-index", type=int, default=None)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--cam-layer", choices=sorted(CAM_LAYERS), default="28x28")
    parser.add_argument("--absolute-threshold", type=float, default=0.65)
    parser.add_argument("--cam-percentile", type=float, default=80.0)
    parser.add_argument("--cam-mass", type=float, default=0.80)
    parser.add_argument("--out-dir", default="runs/b20_crop_focus/explanation")
    args = parser.parse_args()

    if (args.series_index is None) != (args.slice_index is None):
        raise ValueError("--series-index and --slice-index must be supplied together")
    if not 0 <= args.probability_threshold <= 1:
        raise ValueError("--probability-threshold must be in [0,1]")
    if not 0 < args.absolute_threshold < 1:
        raise ValueError("--absolute-threshold must be in (0,1)")
    if not 0 < args.cam_percentile < 100:
        raise ValueError("--cam-percentile must be in (0,100)")
    if not 0 < args.cam_mass < 1:
        raise ValueError("--cam-mass must be in (0,1)")

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    model, payload = load_b20_checkpoint(args.checkpoint, device=runtime.device)
    model.eval()
    if payload.get("crop_focus_policy") != crop_policy:
        raise ValueError("B20 explanation config crop policy differs from checkpoint")

    root = Path(config["data_root"])
    gold, variable_index, metadata_stats = _load_gold_surface(config, root)
    prediction, offsets = _predict_gold(model, config, root, gold, variable_index, runtime, crop_policy)
    requested_target = _resolve_target(args.target)
    row_idx, target_idx = _choose_success(
        gold,
        prediction,
        target_idx=requested_target,
        uid=args.uid,
        probability_threshold=args.probability_threshold,
    )

    uid = str(gold.iloc[row_idx]["StudyInstanceUID"])
    truth = float(gold.iloc[row_idx][TARGETS[target_idx]])
    tta_probability = float(prediction[row_idx, target_idx])

    one_ds = CropFocusedVariableSeriesKneeDataset(
        [uid],
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=np.asarray([[float(gold.iloc[row_idx][target]) for target in TARGETS]], dtype=np.float32),
        train=False,
        crop_focus_policy=crop_policy,
    )
    item = one_ds[0]
    volumes_views = item["volumes"]
    present = item["present"].unsqueeze(0).to(runtime.device)
    series_meta = item["series_meta"].unsqueeze(0).to(runtime.device)

    model.eval()
    view_probs = _view_probabilities(model, volumes_views, present, series_meta, runtime, target_idx)
    view_idx = int(np.argmax(view_probs)) if args.view_offset is None else offsets.index(int(args.view_offset))
    view_offset = int(offsets[view_idx])

    volumes = volumes_views[view_idx].unsqueeze(0).to(runtime.device)
    cams, importance, explained_probability, pairs = _gradcam_for_study(
        model,
        volumes,
        present,
        series_meta,
        target_idx=target_idx,
        runtime=runtime,
        cam_layer=args.cam_layer,
    )

    if args.series_index is None:
        selected_row = int(torch.argmax(importance).item())
    else:
        pair = (int(args.series_index), int(args.slice_index))
        if pair not in pairs:
            raise ValueError(f"requested series/slice pair {pair} is not active")
        selected_row = pairs.index(pair)

    series_idx, slice_idx = pairs[selected_row]
    record = variable_index[uid][series_idx]
    image = volumes[0, series_idx, slice_idx, 1].detach().float().cpu().numpy()
    cam_up = F.interpolate(
        cams[selected_row][None, None], size=image.shape, mode="bilinear", align_corners=False
    )[0, 0].detach().float().cpu().numpy()
    cam = _normalize_cam(cam_up)

    absolute_mask = cam >= float(args.absolute_threshold)
    percentile_value = float(np.percentile(cam, args.cam_percentile))
    percentile_mask = cam >= percentile_value
    mass_mask = _mass_mask(cam, float(args.cam_mass))
    metrics = _cam_metrics(cam)

    max_importance = float(importance.max().detach().cpu())
    rankings = []
    for row, ((series_number, slice_number), score) in enumerate(zip(pairs, importance.detach().float().cpu().numpy())):
        series_record = variable_index[uid][series_number]
        rankings.append({
            "cam_row": row,
            "series_index": int(series_number),
            "slice_index": int(slice_number),
            "series_uid": str(series_record["series_uid"]),
            "plane": str(series_record["plane"]),
            "cam_importance": float(score),
            "cam_importance_relative": float(score / max(max_importance, 1e-12)),
        })

    out_dir = Path(args.out_dir)
    stem = f"{uid}_{TARGET_SLUGS[target_idx]}"
    title = (
        f"B20 crop-only | {TARGETS[target_idx]} | truth={truth:.0f} | TTA p={tta_probability:.3f} | "
        f"view {view_offset:+d} p={explained_probability:.3f} | {record['plane']} | "
        f"series {series_idx} | slice {slice_idx}"
    )
    panel = _save_panel(
        out_dir,
        stem,
        image,
        cam,
        absolute_mask,
        percentile_mask,
        mass_mask,
        title=title,
        absolute_threshold=float(args.absolute_threshold),
        percentile=float(args.cam_percentile),
        mass_fraction=float(args.cam_mass),
    )

    pd.DataFrame(rankings).sort_values("cam_importance", ascending=False, ignore_index=True).to_csv(
        out_dir / f"{stem}_slice_rankings.csv", index=False
    )

    metadata = {
        "experiment": "B20_crop_only_explanation_v2",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_epoch": int(payload.get("selected_epoch", -1)),
        "study_uid": uid,
        "target": TARGETS[target_idx],
        "target_slug": TARGET_SLUGS[target_idx],
        "expert_truth": truth,
        "frozen_three_view_tta_probability": tta_probability,
        "per_view_probabilities": {str(o): float(p) for o, p in zip(offsets, view_probs)},
        "explained_view_offset": view_offset,
        "explained_view_probability": float(explained_probability),
        "cam_layer": args.cam_layer,
        "absolute_threshold": float(args.absolute_threshold),
        "absolute_mask_fraction": float(absolute_mask.mean()),
        "cam_percentile": float(args.cam_percentile),
        "cam_percentile_value": percentile_value,
        "percentile_mask_fraction": float(percentile_mask.mean()),
        "cam_mass_fraction_requested": float(args.cam_mass),
        "mass_mask_fraction": float(mass_mask.mean()),
        "selected_series_index": int(series_idx),
        "selected_series_uid": str(record["series_uid"]),
        "selected_plane": str(record["plane"]),
        "selected_slice_index": int(slice_idx),
        "metadata_repair": metadata_stats,
        "interpretation": (
            "Grad-CAM model evidence diagnostic, not lesion segmentation. Absolute-threshold masks are "
            "peak-sensitive; percentile and cumulative-mass summaries show broader evidence extent."
        ),
        **metrics,
    }
    metadata_path = out_dir / f"{stem}_explanation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({
        "panel": str(panel),
        "metadata": str(metadata_path),
        "study_uid": uid,
        "target": TARGETS[target_idx],
        "tta_probability": tta_probability,
        "explained_view_probability": float(explained_probability),
        "absolute_mask_fraction": float(absolute_mask.mean()),
        "percentile_mask_fraction": float(percentile_mask.mean()),
        "mass_mask_fraction": float(mass_mask.mean()),
        "cam_normalized_entropy": metrics["cam_normalized_entropy"],
        "cam_peak_xy_norm": [metrics["cam_peak_x_norm"], metrics["cam_peak_y_norm"]],
        "cam_center_of_mass_xy_norm": [metrics["cam_center_of_mass_x_norm"], metrics["cam_center_of_mass_y_norm"]],
    }, indent=2))


if __name__ == "__main__":
    main()
