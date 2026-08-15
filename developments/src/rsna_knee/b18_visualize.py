"""Visualize one correctly detected B18 expert case with a Grad-CAM mask.

This is an explanation/localization mask for the study-level B18 classifier,
not a ground-truth lesion segmentation.
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
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_1_gold_eval import predict_b12_1
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b18_fisher_selection import load_b18_checkpoint, require_b18_contract
from .constants import SLUG_TO_DISPLAY, TARGETS, TARGET_SLUGS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .runtime import autocast, resolve_runtime


CAM_LAYERS = {
    "7x7": 7,
    "14x14": 5,
    "28x28": 3,
}


def _resolve_target(value: str | None) -> int | None:
    if value is None:
        return None
    raw = value.strip()
    if raw in TARGETS:
        return TARGETS.index(raw)
    lower = raw.lower()
    for j, target in enumerate(TARGETS):
        if target.lower() == lower:
            return j
    if lower in SLUG_TO_DISPLAY:
        return TARGETS.index(SLUG_TO_DISPLAY[lower])
    raise ValueError(
        f"unknown target {value!r}. Use one of: "
        + ", ".join(f"{t} ({s})" for t, s in zip(TARGETS, TARGET_SLUGS))
    )


def _load_gold_surface(config: dict, root: Path):
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("expected the complete 58-study expert-labelled surface")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    variable_index = build_variable_series_index(series, uids)
    if any(len(variable_index.get(uid, [])) == 0 for uid in uids):
        raise ValueError("an expert study has zero eligible MRI series")
    return gold, variable_index, metadata_stats


def _predict_gold(model, config, root, gold, variable_index, runtime):
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B18 visualization requires frozen TTA offsets [-1,0,1]")

    uids = gold["StudyInstanceUID"].tolist()
    ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 24_100_000),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != uids:
        raise RuntimeError("expert prediction order changed")
    return prediction, offsets


def _choose_success(
    gold: pd.DataFrame,
    prediction: np.ndarray,
    *,
    target_idx: int | None,
    uid: str | None,
    probability_threshold: float,
):
    truth = gold[TARGETS].to_numpy(np.float64)
    candidate = truth > 0.5

    if target_idx is not None:
        target_mask = np.zeros_like(candidate, dtype=bool)
        target_mask[:, target_idx] = True
        candidate &= target_mask

    if uid is not None:
        uid = str(uid)
        rows = gold["StudyInstanceUID"].astype(str).to_numpy() == uid
        if not rows.any():
            raise ValueError(f"--uid {uid!r} is not one of the 58 expert-labelled studies")
        candidate &= rows[:, None]

    successful = candidate & (prediction >= float(probability_threshold))
    if not successful.any():
        positive_scores = np.where(candidate, prediction, -np.inf)
        flat_best = int(np.argmax(positive_scores))
        best = float(positive_scores.reshape(-1)[flat_best])
        if not np.isfinite(best):
            raise ValueError("the requested UID/target has no expert-positive cell")
        i, j = np.unravel_index(flat_best, positive_scores.shape)
        raise RuntimeError(
            "no true-positive case meets "
            f"--probability-threshold={probability_threshold:.3f}. "
            f"Best expert-positive candidate: UID={gold.iloc[i]['StudyInstanceUID']}, "
            f"target={TARGETS[j]!r}, probability={best:.4f}."
        )

    scores = np.where(successful, prediction, -np.inf)
    flat = int(np.argmax(scores))
    return np.unravel_index(flat, scores.shape)


@torch.no_grad()
def _view_probabilities(model, volumes_views, present, series_meta, runtime, target_idx):
    probs = []
    for view in range(volumes_views.shape[0]):
        with autocast(runtime):
            logits = model(
                volumes_views[view].unsqueeze(0).to(runtime.device, non_blocking=True),
                present,
                series_meta,
            )
        probs.append(float(torch.sigmoid(logits.float())[0, target_idx].cpu()))
    return probs


def _gradcam_for_study(
    model,
    volumes,
    present,
    series_meta,
    *,
    target_idx: int,
    runtime,
    cam_layer: str,
):
    """Return CAMs for every active series/slice and their positive importance."""
    layer_index = CAM_LAYERS[cam_layer]
    if len(model.encoder.features) <= layer_index:
        raise RuntimeError(
            f"ConvNeXt features has {len(model.encoder.features)} modules; "
            f"cannot use layer index {layer_index}"
        )

    # No model parameter needs a gradient. The hook returns the same feature
    # values as a detached leaf requiring gradients, so Grad-CAM can backprop
    # from the study-level target without retaining the frozen encoder graph
    # below the selected layer.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records: list[dict] = []

    def capture(_module, _inputs, output):
        leaf = output.detach().requires_grad_(True)
        record = {"activation": leaf, "gradient": None}

        def save_gradient(gradient, rec=record):
            rec["gradient"] = gradient.detach()

        leaf.register_hook(save_gradient)
        records.append(record)
        return leaf

    handle = model.encoder.features[layer_index].register_forward_hook(capture)
    model.eval()
    try:
        with torch.enable_grad():
            with autocast(runtime):
                logits = model(volumes, present, series_meta)
            target_logit = logits[0, target_idx]
            probability = float(torch.sigmoid(target_logit.float()).detach().cpu())
            target_logit.backward()
    finally:
        handle.remove()

    if not records or any(record["gradient"] is None for record in records):
        raise RuntimeError("Grad-CAM hook did not receive complete activation gradients")

    activation = torch.cat(
        [record["activation"].detach().float() for record in records], dim=0
    )
    gradient = torch.cat([record["gradient"].float() for record in records], dim=0)
    if activation.shape != gradient.shape or activation.ndim != 4:
        raise RuntimeError(
            f"unexpected Grad-CAM shapes: {tuple(activation.shape)} vs {tuple(gradient.shape)}"
        )

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * activation).sum(dim=1))
    importance = cams.flatten(1).mean(dim=1)

    active_series = torch.nonzero(present[0] > 0, as_tuple=False).flatten().tolist()
    n_slices = int(volumes.shape[2])
    pairs = [(int(k), int(s)) for k in active_series for s in range(n_slices)]
    if len(pairs) != int(cams.shape[0]):
        raise RuntimeError(
            f"CAM row mapping mismatch: {len(pairs)} slice pairs vs {cams.shape[0]} activations"
        )
    return cams, importance, probability, pairs


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    maximum = float(cam.max())
    if maximum <= 0:
        raise RuntimeError("selected Grad-CAM is all zero; try another case or CAM layer")
    return cam / maximum


def _save_visuals(out_dir, uid, target_idx, image, cam, mask, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{uid}_{TARGET_SLUGS[target_idx]}"

    image = np.clip(image.astype(np.float32), 0, 1)
    cam = np.clip(cam.astype(np.float32), 0, 1)
    mask = mask.astype(bool)

    plt.imsave(out_dir / f"{stem}_original.png", image, cmap="gray", vmin=0, vmax=1)
    plt.imsave(out_dir / f"{stem}_gradcam.png", cam, cmap="turbo", vmin=0, vmax=1)
    plt.imsave(
        out_dir / f"{stem}_mask.png",
        mask.astype(np.float32),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    gray_rgb = np.repeat(image[..., None], 3, axis=2)
    masked = gray_rgb * 0.18
    highlight = np.zeros_like(gray_rgb)
    highlight[..., 0] = 1.0
    masked[mask] = 0.55 * gray_rgb[mask] + 0.45 * highlight[mask]
    plt.imsave(out_dir / f"{stem}_masked.png", np.clip(masked, 0, 1))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original center channel")

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(cam, cmap="turbo", alpha=0.48, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Thresholded CAM mask")

    axes[3].imshow(masked)
    if mask.any() and (~mask).any():
        axes[3].contour(
            mask.astype(np.float32), levels=[0.5], colors=["yellow"], linewidths=1.2
        )
    axes[3].set_title("Masked model evidence")

    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_dir / f"{stem}_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-b18-visualize",
        description="Plot one correctly detected B18 expert-positive case with Grad-CAM.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target display name or slug, e.g. ACL or medial_meniscus.",
    )
    parser.add_argument(
        "--uid",
        default=None,
        help="Optional expert StudyInstanceUID. Otherwise choose the strongest true positive.",
    )
    parser.add_argument("--probability-threshold", type=float, default=0.50)
    parser.add_argument("--view-offset", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--cam-layer", choices=sorted(CAM_LAYERS), default="14x14")
    parser.add_argument("--cam-threshold", type=float, default=0.60)
    parser.add_argument(
        "--out-dir", default="runs/b18_fisher_selection/visualization"
    )
    args = parser.parse_args()

    if not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("--probability-threshold must be in [0,1]")
    if not 0.0 < args.cam_threshold < 1.0:
        raise ValueError("--cam-threshold must be in (0,1)")

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    require_b18_contract(config)

    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, payload = load_b18_checkpoint(args.checkpoint, device=runtime.device)

    root = Path(config["data_root"])
    gold, variable_index, metadata_stats = _load_gold_surface(config, root)
    prediction, offsets = _predict_gold(
        model, config, root, gold, variable_index, runtime
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
    print(
        {
            "selected_uid": uid,
            "target": TARGETS[target_idx],
            "expert_truth": truth,
            "frozen_tta_probability": tta_probability,
            "successful_true_positive": bool(
                truth > 0.5 and tta_probability >= args.probability_threshold
            ),
        }
    )

    one_ds = VariableSeriesKneeDataset(
        [uid],
        variable_index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=np.asarray(
            [[float(gold.iloc[row_idx][target]) for target in TARGETS]],
            dtype=np.float32,
        ),
        train=False,
    )
    item = one_ds[0]
    volumes_views = item["volumes"]
    present = item["present"].unsqueeze(0).to(runtime.device)
    series_meta = item["series_meta"].unsqueeze(0).to(runtime.device)

    view_probs = _view_probabilities(
        model, volumes_views, present, series_meta, runtime, target_idx
    )
    if args.view_offset is None:
        view_idx = int(np.argmax(view_probs))
    else:
        view_idx = offsets.index(int(args.view_offset))
    view_offset = int(offsets[view_idx])

    print(
        {
            "tta_view_probabilities": {
                str(offset): float(prob) for offset, prob in zip(offsets, view_probs)
            },
            "explained_view_offset": view_offset,
        }
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    best_row = int(torch.argmax(importance).item())
    series_idx, slice_idx = pairs[best_row]
    record = variable_index[uid][series_idx]

    image = (
        volumes[0, series_idx, slice_idx, 1]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    cam_up = (
        F.interpolate(
            cams[best_row][None, None],
            size=image.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
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
                "fluid_id": int(series_record["fluid_id"]),
                "fat_id": int(series_record["fat_id"]),
                "cam_importance": float(score),
                "cam_importance_relative": float(
                    score / max(max_importance, 1e-12)
                ),
            }
        )

    out_dir = Path(args.out_dir)
    title = (
        f"B18 | {TARGETS[target_idx]} | truth=1 | TTA p={tta_probability:.3f} | "
        f"view {view_offset:+d} p={explained_probability:.3f} | "
        f"{record['plane']} | series {series_idx} | slice {slice_idx}"
    )
    stem = _save_visuals(
        out_dir, uid, target_idx, image, cam, mask, title
    )

    rankings_path = out_dir / f"{stem}_slice_rankings.csv"
    pd.DataFrame(rankings).sort_values(
        "cam_importance", ascending=False, ignore_index=True
    ).to_csv(rankings_path, index=False)

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_epoch": int(payload.get("selected_epoch", -1)),
        "study_uid": uid,
        "target": TARGETS[target_idx],
        "target_slug": TARGET_SLUGS[target_idx],
        "expert_truth": truth,
        "successful_true_positive_threshold": float(args.probability_threshold),
        "frozen_three_view_tta_probability": tta_probability,
        "tta_offsets": list(offsets),
        "per_view_probabilities": {
            str(offset): float(prob) for offset, prob in zip(offsets, view_probs)
        },
        "explained_view_offset": view_offset,
        "explained_view_probability": float(explained_probability),
        "cam_layer": args.cam_layer,
        "cam_threshold": float(args.cam_threshold),
        "mask_pixel_fraction": float(mask.mean()),
        "selected_series_index": int(series_idx),
        "selected_series_uid": str(record["series_uid"]),
        "selected_plane": str(record["plane"]),
        "selected_fluid_id": int(record["fluid_id"]),
        "selected_fat_id": int(record["fat_id"]),
        "selected_slice_index": int(slice_idx),
        "n_sampled_slices_per_series": int(volumes.shape[2]),
        "series_count": int(volumes.shape[1]),
        "metadata_repair": metadata_stats,
        "interpretation": (
            "Grad-CAM explanation/localization mask from the B18 study-level classifier; "
            "not a radiologist-drawn or ground-truth lesion segmentation."
        ),
        "governance": (
            "Post-selection visualization only; do not use this expert-set visualization "
            "to retune B18 or claim independent validation."
        ),
    }
    metadata_path = out_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved:")
    for suffix in (
        "original.png",
        "gradcam.png",
        "mask.png",
        "masked.png",
        "panel.png",
        "slice_rankings.csv",
        "metadata.json",
    ):
        print(out_dir / f"{stem}_{suffix}")


if __name__ == "__main__":
    main()
