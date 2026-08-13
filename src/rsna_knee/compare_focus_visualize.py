"""Same-view/same-series/same-slice Grad-CAM comparison for B18/B19/B20."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import VariableSeriesKneeDataset
from .b18_fisher_selection import load_b18_checkpoint, require_b18_contract
from .b18_visualize import (
    CAM_LAYERS,
    _gradcam_for_study,
    _load_gold_surface,
    _normalize_cam,
    _resolve_target,
    _view_probabilities,
)
from .b19_joint_focus import (
    JointFocusedVariableSeriesKneeDataset,
    load_b19_checkpoint,
    require_b19_contract,
)
from .b20_crop_focus import (
    CropFocusedVariableSeriesKneeDataset,
    load_b20_checkpoint,
    require_b20_contract,
)
from .constants import TARGETS, TARGET_SLUGS
from .runtime import resolve_runtime


def _dataset_item(kind, config, root, uid, variable_index, offsets, truth_row, policy=None):
    kwargs = dict(
        study_uids=[uid],
        series_records=variable_index,
        config=make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=np.asarray([truth_row], dtype=np.float32),
        train=False,
    )
    if kind == "b18":
        ds = VariableSeriesKneeDataset(
            [uid], variable_index,
            kwargs["config"], targets=kwargs["targets"], train=False
        )
    elif kind == "b19":
        ds = JointFocusedVariableSeriesKneeDataset(
            [uid], variable_index, kwargs["config"], targets=kwargs["targets"],
            train=False, joint_focus_policy=policy
        )
    elif kind == "b20":
        ds = CropFocusedVariableSeriesKneeDataset(
            [uid], variable_index, kwargs["config"], targets=kwargs["targets"],
            train=False, crop_focus_policy=policy
        )
    else:
        raise ValueError(kind)
    return ds[0]


def _explain(model, item, variable_index, uid, target_idx, runtime, offsets, view_offset, cam_layer):
    # Visualization/inference probabilities must be deterministic. The shared
    # _view_probabilities helper does not change model mode itself, so enforce
    # eval() here before both probability and Grad-CAM passes.
    model.eval()
    volumes_views = item["volumes"]
    present = item["present"].unsqueeze(0).to(runtime.device)
    series_meta = item["series_meta"].unsqueeze(0).to(runtime.device)
    view_probs = _view_probabilities(model, volumes_views, present, series_meta, runtime, target_idx)
    view_idx = offsets.index(int(view_offset))
    volumes = volumes_views[view_idx].unsqueeze(0).to(runtime.device)
    cams, importance, explained_probability, pairs = _gradcam_for_study(
        model, volumes, present, series_meta,
        target_idx=target_idx, runtime=runtime, cam_layer=cam_layer
    )
    expected_view_probability = float(view_probs[view_idx])
    consistency_delta = abs(float(explained_probability) - expected_view_probability)
    if consistency_delta > 0.02:
        raise RuntimeError(
            "view-probability bookkeeping mismatch after eval-mode enforcement: "
            f"offset={view_offset}, direct={expected_view_probability:.6f}, "
            f"Grad-CAM={float(explained_probability):.6f}, delta={consistency_delta:.6f}"
        )
    return {
        "volumes": volumes,
        "cams": cams,
        "importance": importance,
        "pairs": pairs,
        "view_probs": view_probs,
        "tta_probability": float(np.mean(view_probs)),
        "explained_probability": float(explained_probability),
        "expected_view_probability": expected_view_probability,
        "view_probability_consistency_abs_delta": consistency_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-focus-compare")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--b18-config", default="configs/b18_fisher_selection.yaml")
    parser.add_argument("--b18-checkpoint", required=True)
    parser.add_argument("--b19-config", default="configs/b19_joint_focus.yaml")
    parser.add_argument("--b19-checkpoint", required=True)
    parser.add_argument("--b20-config", default="configs/b20_crop_focus.yaml")
    parser.add_argument("--b20-checkpoint", default=None)
    parser.add_argument("--view-offset", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--series-index", type=int, default=None)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--reference-model", choices=["b18", "b19", "b20"], default="b18")
    parser.add_argument("--cam-layer", choices=sorted(CAM_LAYERS), default="28x28")
    parser.add_argument("--cam-threshold", type=float, default=0.65)
    parser.add_argument("--out", default="runs/focus_comparison/b18_b19_b20_same_slice.png")
    args = parser.parse_args()

    if (args.series_index is None) != (args.slice_index is None):
        raise ValueError("--series-index and --slice-index must be supplied together")
    if not 0.0 < args.cam_threshold < 1.0:
        raise ValueError("--cam-threshold must be in (0,1)")
    if args.reference_model == "b20" and not args.b20_checkpoint:
        raise ValueError("--reference-model b20 requires --b20-checkpoint")

    configs = {}
    for key, path in (("b18", args.b18_config), ("b19", args.b19_config), ("b20", args.b20_config)):
        config = _read_config(path)
        config = dict(config)
        config["data_root"] = args.data_root
        configs[key] = config

    require_b18_contract(configs["b18"])
    p19 = require_b19_contract(configs["b19"])
    p20 = require_b20_contract(configs["b20"])

    runtime = resolve_runtime(configs["b18"])
    print(runtime.describe())
    models = {}
    models["b18"], payload18 = load_b18_checkpoint(args.b18_checkpoint, device=runtime.device)
    models["b19"], payload19 = load_b19_checkpoint(args.b19_checkpoint, device=runtime.device)
    payloads = {"b18": payload18, "b19": payload19}
    if args.b20_checkpoint:
        models["b20"], payload20 = load_b20_checkpoint(args.b20_checkpoint, device=runtime.device)
        payloads["b20"] = payload20
    for model in models.values():
        model.eval()

    root = Path(args.data_root)
    gold, variable_index, _ = _load_gold_surface(configs["b18"], root)
    uid = str(args.uid)
    uid_mask = gold["StudyInstanceUID"].astype(str).to_numpy() == uid
    positions = np.flatnonzero(uid_mask)
    if positions.size == 0:
        raise ValueError("UID is not on the 58-study expert surface")
    row_idx = int(positions[0])
    target_idx = _resolve_target(args.target)
    if target_idx is None:
        raise ValueError("--target is required")
    truth = float(gold.iloc[row_idx][TARGETS[target_idx]])
    if truth <= 0.5:
        raise ValueError(f"requested expert case is not positive for {TARGETS[target_idx]}")
    truth_row = [float(gold.iloc[row_idx][target]) for target in TARGETS]

    offsets = tuple(int(x) for x in configs["b18"].get("b7_eval_tta_offsets", [-1, 0, 1]))
    items = {
        "b18": _dataset_item("b18", configs["b18"], root, uid, variable_index, offsets, truth_row),
        "b19": _dataset_item("b19", configs["b19"], root, uid, variable_index, offsets, truth_row, p19),
    }
    if "b20" in models:
        items["b20"] = _dataset_item("b20", configs["b20"], root, uid, variable_index, offsets, truth_row, p20)

    # Determine one common TTA view. If not forced, use the reference model's
    # highest deterministic evaluation-mode probability, then hold that exact
    # offset fixed for all models.
    ref = args.reference_model
    ref_item = items[ref]
    ref_present = ref_item["present"].unsqueeze(0).to(runtime.device)
    ref_meta = ref_item["series_meta"].unsqueeze(0).to(runtime.device)
    models[ref].eval()
    ref_probs = _view_probabilities(models[ref], ref_item["volumes"], ref_present, ref_meta, runtime, target_idx)
    view_offset = int(args.view_offset) if args.view_offset is not None else int(offsets[int(np.argmax(ref_probs))])

    explanations = {
        key: _explain(model, items[key], variable_index, uid, target_idx, runtime, offsets, view_offset, args.cam_layer)
        for key, model in models.items()
    }

    # Determine one common source pair. If not forced, use the strongest CAM row
    # from the reference model, then force that same series/slice in every model.
    if args.series_index is None:
        ref_exp = explanations[ref]
        ref_row = int(torch.argmax(ref_exp["importance"]).item())
        common_pair = ref_exp["pairs"][ref_row]
    else:
        common_pair = (int(args.series_index), int(args.slice_index))
    for key, exp in explanations.items():
        if common_pair not in exp["pairs"]:
            raise ValueError(f"common pair {common_pair} is not active for {key}")

    series_idx, slice_idx = common_pair
    record = variable_index[uid][series_idx]
    model_order = [key for key in ("b18", "b19", "b20") if key in models]
    fig, axes = plt.subplots(len(model_order), 4, figsize=(16, 4.4 * len(model_order)), squeeze=False, constrained_layout=True)
    summary = {}

    labels = {
        "b18": "B18 full FOV",
        "b19": "B19 crop + cosine mask",
        "b20": "B20 crop only",
    }
    for r, key in enumerate(model_order):
        exp = explanations[key]
        row = exp["pairs"].index(common_pair)
        image = exp["volumes"][0, series_idx, slice_idx, 1].detach().float().cpu().numpy()
        cam_up = F.interpolate(
            exp["cams"][row][None, None], size=image.shape, mode="bilinear", align_corners=False
        )[0, 0].detach().float().cpu().numpy()
        cam = _normalize_cam(cam_up)
        mask = cam >= float(args.cam_threshold)
        gray_rgb = np.repeat(np.clip(image, 0, 1)[..., None], 3, axis=2)
        masked = gray_rgb * 0.18
        highlight = np.zeros_like(gray_rgb)
        highlight[..., 0] = 1.0
        masked[mask] = 0.55 * gray_rgb[mask] + 0.45 * highlight[mask]

        axes[r, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{labels[key]} — input")
        axes[r, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(cam, cmap="turbo", alpha=0.48, vmin=0, vmax=1)
        axes[r, 1].set_title(
            f"Grad-CAM | TTA p={exp['tta_probability']:.3f} | view p={exp['explained_probability']:.3f}"
        )
        axes[r, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title(f"CAM ≥ {args.cam_threshold:.2f}")
        axes[r, 3].imshow(masked)
        if mask.any() and (~mask).any():
            axes[r, 3].contour(mask.astype(np.float32), levels=[0.5], colors=["yellow"], linewidths=1.2)
        axes[r, 3].set_title("Masked model evidence")
        for c in range(4):
            axes[r, c].axis("off")

        summary[key] = {
            "selected_epoch": int(payloads[key].get("selected_epoch", -1)),
            "tta_probability": float(exp["tta_probability"]),
            "per_view_probabilities": {str(o): float(p) for o, p in zip(offsets, exp["view_probs"])},
            "explained_view_probability": float(exp["explained_probability"]),
            "expected_view_probability": float(exp["expected_view_probability"]),
            "view_probability_consistency_abs_delta": float(exp["view_probability_consistency_abs_delta"]),
            "mask_pixel_fraction": float(mask.mean()),
        }

    fig.suptitle(
        f"Same-source Grad-CAM comparison | {TARGETS[target_idx]} | truth=1 | UID={uid}\n"
        f"view {view_offset:+d} | {record['plane']} | series {series_idx} | slice {slice_idx} | "
        f"seriesUID={record['series_uid']}",
        fontsize=11,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "study_uid": uid,
        "target": TARGETS[target_idx],
        "target_slug": TARGET_SLUGS[target_idx],
        "expert_truth": truth,
        "common_view_offset": view_offset,
        "common_series_index": int(series_idx),
        "common_slice_index": int(slice_idx),
        "common_series_uid": str(record["series_uid"]),
        "common_plane": str(record["plane"]),
        "reference_model": ref,
        "pair_was_forced": args.series_index is not None,
        "view_was_forced": args.view_offset is not None,
        "cam_layer": args.cam_layer,
        "cam_threshold": float(args.cam_threshold),
        "models": summary,
        "interpretation": "same source series/sampled slice and TTA offset across models; deterministic eval-mode probabilities; Grad-CAM is model localization, not lesion segmentation",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(output)
    print(metadata_path)


if __name__ == "__main__":
    main()
