"""Visualize one correctly detected B20 expert case with Grad-CAM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_1_gold_eval import predict_b12_1
from .b12_variable_series import collate_variable_series
from .b18_visualize import (
    CAM_LAYERS,
    _choose_success,
    _gradcam_for_study,
    _load_gold_surface,
    _normalize_cam,
    _resolve_target,
    _save_visuals,
    _view_probabilities,
)
from .b20_crop_focus import (
    CropFocusedVariableSeriesKneeDataset,
    load_b20_checkpoint,
    require_b20_contract,
)
from .constants import TARGETS, TARGET_SLUGS
from .runtime import resolve_runtime


def _predict_gold(model, config, root, gold, variable_index, runtime, crop_policy):
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B20 visualization requires frozen TTA offsets [-1,0,1]")
    uids = gold["StudyInstanceUID"].tolist()
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 28_100_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("B20 expert prediction order changed")
    return prediction, offsets


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20-visualize")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.50)
    parser.add_argument("--view-offset", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--series-index", type=int, default=None)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--cam-layer", choices=sorted(CAM_LAYERS), default="14x14")
    parser.add_argument("--cam-threshold", type=float, default=0.60)
    parser.add_argument("--out-dir", default="runs/b20_crop_focus/visualization")
    args = parser.parse_args()

    if (args.series_index is None) != (args.slice_index is None):
        raise ValueError("--series-index and --slice-index must be supplied together")
    if not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("--probability-threshold must be in [0,1]")
    if not 0.0 < args.cam_threshold < 1.0:
        raise ValueError("--cam-threshold must be in (0,1)")

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, payload = load_b20_checkpoint(args.checkpoint, device=runtime.device)
    if payload.get("crop_focus_policy") != crop_policy:
        raise ValueError("B20 visualization config crop policy differs from checkpoint")

    root = Path(config["data_root"])
    gold, variable_index, metadata_stats = _load_gold_surface(config, root)
    prediction, offsets = _predict_gold(
        model, config, root, gold, variable_index, runtime, crop_policy
    )
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
        targets=np.asarray(
            [[float(gold.iloc[row_idx][target]) for target in TARGETS]], dtype=np.float32
        ),
        train=False,
        crop_focus_policy=crop_policy,
    )
    item = one_ds[0]
    volumes_views = item["volumes"]
    present = item["present"].unsqueeze(0).to(runtime.device)
    series_meta = item["series_meta"].unsqueeze(0).to(runtime.device)

    view_probs = _view_probabilities(
        model, volumes_views, present, series_meta, runtime, target_idx
    )
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
        requested_pair = (int(args.series_index), int(args.slice_index))
        if requested_pair not in pairs:
            raise ValueError(f"requested series/slice pair {requested_pair} is not active")
        selected_row = pairs.index(requested_pair)
    series_idx, slice_idx = pairs[selected_row]
    record = variable_index[uid][series_idx]

    image = volumes[0, series_idx, slice_idx, 1].detach().float().cpu().numpy()
    cam_up = F.interpolate(
        cams[selected_row][None, None], size=image.shape, mode="bilinear", align_corners=False
    )[0, 0].detach().float().cpu().numpy()
    cam = _normalize_cam(cam_up)
    mask = cam >= float(args.cam_threshold)

    max_importance = float(importance.max().detach().cpu())
    rankings = []
    for row, ((series_number, slice_number), score) in enumerate(
        zip(pairs, importance.detach().float().cpu().numpy())
    ):
        series_record = variable_index[uid][series_number]
        rankings.append(
            {
                "cam_row": row,
                "series_index": int(series_number),
                "slice_index": int(slice_number),
                "series_uid": str(series_record["series_uid"]),
                "plane": str(series_record["plane"]),
                "cam_importance": float(score),
                "cam_importance_relative": float(score / max(max_importance, 1e-12)),
            }
        )

    out_dir = Path(args.out_dir)
    title = (
        f"B20 crop-only | {TARGETS[target_idx]} | truth=1 | TTA p={tta_probability:.3f} | "
        f"view {view_offset:+d} p={explained_probability:.3f} | {record['plane']} | "
        f"series {series_idx} | slice {slice_idx}"
    )
    stem = _save_visuals(out_dir, uid, target_idx, image, cam, mask, title)
    pd.DataFrame(rankings).sort_values(
        "cam_importance", ascending=False, ignore_index=True
    ).to_csv(out_dir / f"{stem}_slice_rankings.csv", index=False)

    metadata = {
        "experiment": "B20_crop_only_gradcam_visualization",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_epoch": int(payload.get("selected_epoch", -1)),
        "crop_focus_policy": crop_policy,
        "cosine_mask_used": False,
        "study_uid": uid,
        "target": TARGETS[target_idx],
        "target_slug": TARGET_SLUGS[target_idx],
        "expert_truth": truth,
        "frozen_three_view_tta_probability": tta_probability,
        "tta_offsets": list(offsets),
        "per_view_probabilities": {
            str(offset): float(prob) for offset, prob in zip(offsets, view_probs)
        },
        "explained_view_offset": view_offset,
        "explained_view_probability": float(explained_probability),
        "cam_layer": args.cam_layer,
        "cam_threshold": float(args.cam_threshold),
        "forced_series_slice": args.series_index is not None,
        "selected_series_index": int(series_idx),
        "selected_series_uid": str(record["series_uid"]),
        "selected_plane": str(record["plane"]),
        "selected_slice_index": int(slice_idx),
        "mask_pixel_fraction": float(mask.mean()),
        "metadata_repair": metadata_stats,
        "interpretation": "Grad-CAM model localization, not ground-truth lesion segmentation.",
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        {
            "selected_uid": uid,
            "target": TARGETS[target_idx],
            "truth": truth,
            "tta_probability": tta_probability,
            "view_probabilities": {str(o): float(p) for o, p in zip(offsets, view_probs)},
            "explained_view_offset": view_offset,
            "series_index": series_idx,
            "slice_index": slice_idx,
            "series_uid": str(record["series_uid"]),
            "plane": str(record["plane"]),
            "forced_series_slice": args.series_index is not None,
        }
    )
    print(out_dir / f"{stem}_panel.png")


if __name__ == "__main__":
    main()
