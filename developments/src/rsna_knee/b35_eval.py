"""Expert-58 diagnostic for B35 target-conditioned spatial residual.

The evaluator reports the frozen B34 base and B35 candidate on the exact same
[-1,0,+1] centre-offset views produced by the B35 sampler.  Because the first
16 centres in every view exactly reproduce the historical B34 centres, the base
score is an internal check that the new data path did not change the old model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b20_crop_focus import b20_crop_focus_policy
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime
from .b35_target_spatial_residual import (
    B35_VERSION,
    B35SpatialDataset,
    B35TargetSpatialResidual,
    collate_b35,
)

B35_EVAL_OFFSETS = (-1, 0, 1)
B35_EXPECTED_GOLD_STUDIES = 58
B35_EXPECTED_GOLD_SERIES = 336


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_b35_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
    base_checkpoint: str | Path | None = None,
):
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("version") != B35_VERSION:
        raise ValueError("not a B35 Phase-A checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B35 checkpoint unexpectedly used expert labels")
    recorded_base = Path(str(payload.get("base_checkpoint", "")))
    base_path = Path(base_checkpoint).resolve() if base_checkpoint else recorded_base
    if not base_path.is_file():
        raise FileNotFoundError(
            f"B35 base checkpoint not found at {base_path}; pass --base-checkpoint"
        )
    observed_sha = sha256_file(base_path)
    if observed_sha != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B35 base checkpoint fingerprint mismatch")
    base, base_payload = load_phase9_checkpoint(
        base_path,
        expected_arm="llm_fill",
        device="cpu",
    )
    model = B35TargetSpatialResidual(
        base,
        grid_size=int(payload.get("head_spec", {}).get("grid_size", 3)),
    )
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload, base_payload


@torch.no_grad()
def evaluate_b35(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path | None = None,
) -> dict:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, payload, base_payload = load_b35_checkpoint(
        checkpoint,
        device=runtime.device,
        base_checkpoint=base_checkpoint,
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B35_EXPECTED_GOLD_STUDIES:
        raise ValueError("B35 evaluation requires the complete 58-study expert surface")
    if gold[TARGETS].isna().any().any():
        raise ValueError("B35 evaluation requires all 696 expert labels")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    uids = gold["StudyInstanceUID"].tolist()
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B35 expert surface contains a study with zero eligible series")
    if sum(counts) != B35_EXPECTED_GOLD_SERIES:
        raise ValueError(
            f"B35 expected {B35_EXPECTED_GOLD_SERIES} expert series, got {sum(counts)}"
        )

    crop_policy = b20_crop_focus_policy(config)
    dataset_config = make_b7_dataset_config(config, root, train=False)
    dataset_config.tta_center_offsets = ()
    ds = B35SpatialDataset(
        uids,
        index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=B35_EVAL_OFFSETS,
        targets=gold[TARGETS].to_numpy(np.float32),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b35,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 45_200_000),
    )

    base_blocks, candidate_blocks = [], []
    scored_uids: list[str] = []
    equivalence_error = 0.0
    attention_max_sum = np.zeros(len(TARGETS), dtype=np.float64)
    attention_entropy_sum = np.zeros(len(TARGETS), dtype=np.float64)
    attention_count = 0

    for batch_index, batch in enumerate(loader):
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        position = batch["slice_position"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        if volumes.ndim != 7:
            raise RuntimeError("B35 evaluation expects [B,V,K,S,C,H,W]")

        base_views, candidate_views = [], []
        for view in range(volumes.shape[1]):
            frame = volumes[:, view]
            pos = position[:, view]
            with autocast(runtime):
                out = model(frame, present, meta, pos)
            base_views.append(torch.sigmoid(out.base_logits.float()))
            candidate_views.append(torch.sigmoid(out.logits.float()))

            attention = out.attention.float().clamp_min(1e-12)
            entropy = -(attention * attention.log()).sum(dim=-1)
            denom = np.log(max(int(attention.shape[-1]), 2))
            entropy = entropy / float(denom)
            attention_max_sum += attention.max(dim=-1).values.sum(dim=0).cpu().numpy()
            attention_entropy_sum += entropy.sum(dim=0).cpu().numpy()
            attention_count += int(attention.shape[0])

            if batch_index == 0 and view == 1:
                equivalence_error = model.base_equivalence_error(frame, present, meta)

        base_blocks.append(torch.stack(base_views).mean(dim=0).cpu().numpy())
        candidate_blocks.append(
            torch.stack(candidate_views).mean(dim=0).cpu().numpy()
        )
        scored_uids.extend(str(x) for x in batch["study_uid"])

    if scored_uids != uids:
        raise RuntimeError("B35 expert evaluation changed study order")
    truth = gold[TARGETS].to_numpy(np.float64)
    base_prediction = np.concatenate(base_blocks, axis=0)
    candidate_prediction = np.concatenate(candidate_blocks, axis=0)
    base_macro, base_target = macro_auc_from_arrays(truth, base_prediction)
    candidate_macro, candidate_target = macro_auc_from_arrays(truth, candidate_prediction)

    per_target = {}
    for target, base_auc, candidate_auc in zip(TARGETS, base_target, candidate_target):
        per_target[target] = {
            "base_auc": float(base_auc),
            "b35_auc": float(candidate_auc),
            "delta": float(candidate_auc - base_auc),
        }

    focal = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion", "Fracture"]
    focal_base = float(np.mean([per_target[t]["base_auc"] for t in focal]))
    focal_b35 = float(np.mean([per_target[t]["b35_auc"] for t in focal]))
    gate = model.head.state()
    attention_diag = {
        target: {
            "mean_max_attention": float(attention_max_sum[i] / max(attention_count, 1)),
            "mean_normalized_entropy": float(
                attention_entropy_sum[i] / max(attention_count, 1)
            ),
        }
        for i, target in enumerate(TARGETS)
    }

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "base_checkpoint": str(payload.get("base_checkpoint")),
        "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
        "base_completed_epochs": int(base_payload.get("completed_epochs", -1)),
        "evaluation_role": "reused expert development diagnostic; not independent test evidence",
        "n_studies": len(uids),
        "tta_offsets": list(B35_EVAL_OFFSETS),
        "base_reconstruction_max_abs_error": float(equivalence_error),
        "base_macro_auc": float(base_macro),
        "b35_macro_auc": float(candidate_macro),
        "macro_delta": float(candidate_macro - base_macro),
        "focal_six": focal,
        "focal_six_base_mean_auc": focal_base,
        "focal_six_b35_mean_auc": focal_b35,
        "focal_six_delta": float(focal_b35 - focal_base),
        "per_target": per_target,
        "head": gate,
        "attention": attention_diag,
        "crop_policy": crop_policy,
        "metadata_repair": metadata_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate B35 on the reused 58-study expert surface")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint")
    ap.add_argument("--out", default="runs/b35_target_spatial_v1/expert58.json")
    args = ap.parse_args()
    config = dict(_read_config(args.config))
    result = evaluate_b35(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out)


if __name__ == "__main__":
    main()
