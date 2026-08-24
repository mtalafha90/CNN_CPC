"""B44 diagnostic: frozen-B42 nested 32 -> 64 centre coverage audit.

Diagnostic only.  This module does not train, tune, select, blend, threshold, or
promote a model.  It asks whether the weak B42 targets are limited by the fixed
32-centre slice coverage.

The 64-centre construction is deliberately nested: its first 32 centres are
*exactly* the historical B42 centres for the same series and TTA offset.  Thirty
two additional deterministic centres are then appended from a denser 64-centre
grid.  Therefore the original B42 image samples, first-16 B34 base path, crop,
constant-area geometry, encoder, sparse head, top-k=8, and TTA offsets
[-1, 0, +1] remain fixed; only additional local evidence locations are exposed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b35_target_spatial_residual import B35_BASE_SLICES, B35_DENSE_SLICES, _extra_centers, b35_centers
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b37_highres_sparse_mil import _native_center_crop
from .b42_constant_area_aspect_sparse_eval import load_b42_checkpoint
from .b42_constant_area_aspect_sparse_mil import (
    B42_EXPERT58_ROOT,
    B42ConstantAreaAspectDataset,
    collate_b42,
    require_b42_contract,
    resize_triplets_constant_area,
)
from .b43_target_series_plane_audit import B43_AUDIT_ROOT, _lme, _score_tokens
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .dicom import _normalise_volume, find_series_dir
from .evaluation import macro_auc_from_arrays
from .runtime import autocast, resolve_runtime

B44_AUDIT_VERSION = "b44_frozen_b42_nested_32_to_64_center_coverage_audit_v1"
B44_AUDIT_ROOT = f"{B42_EXPERT58_ROOT}/coverage_32_vs_64_audit"
B44_DENSE_SLICES = 64
B44_ADDED_SLICES = B44_DENSE_SLICES - B35_DENSE_SLICES
B44_AUDIT_LOADER_SEED_OFFSET = 54_100_000
B44_AUDIT_ROLE = (
    "reused post-B42 Expert-58 mechanistic coverage diagnostic; not independent "
    "test evidence, not a tuning set, and not a B44/B43 promotion criterion"
)
B44_FOCUS_TARGETS = ("ACL", "MCL", "Contusion", "Fracture")


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def b44_nested_centers_64(
    n_frames: int,
    *,
    gap: int = 1,
    center_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 64 centres with the historical B42 32 centres as an exact prefix."""
    base32, _ = b35_centers(
        int(n_frames),
        gap=int(gap),
        center_offset=int(center_offset),
        base_slices=B35_BASE_SLICES,
        dense_slices=B35_DENSE_SLICES,
    )
    dense64_candidates, _ = b35_centers(
        int(n_frames),
        gap=int(gap),
        center_offset=int(center_offset),
        base_slices=B35_BASE_SLICES,
        dense_slices=B44_DENSE_SLICES,
    )
    extras = _extra_centers(base32, dense64_candidates, B44_ADDED_SLICES)
    combined = np.concatenate((base32, extras)).astype(np.int64, copy=False)
    if combined.shape != (B44_DENSE_SLICES,):
        raise RuntimeError("B44 nested-centre construction did not produce 64 centres")
    if not np.array_equal(combined[:B35_DENSE_SLICES], base32):
        raise RuntimeError("B44 changed the historical B42 first 32 centres")
    denom = float(max(int(n_frames) - 1, 1))
    position = combined.astype(np.float32) / denom
    return combined, position


def preprocess_dense_triplets_b44_64(
    raw: np.ndarray,
    *,
    gap: int = 1,
    center_offset: int = 0,
    crop_fraction: float = 0.90,
) -> tuple[torch.Tensor, np.ndarray]:
    """Construct nested 64-centre B42 triplets with unchanged image preprocessing."""
    if int(gap) < 1:
        raise ValueError("B44 2.5D gap must be positive")
    normalized = _normalise_volume(raw)
    centers, position = b44_nested_centers_64(
        len(normalized),
        gap=int(gap),
        center_offset=int(center_offset),
    )
    offsets = np.asarray([-int(gap), 0, int(gap)], dtype=np.int64)
    index = np.clip(
        centers[:, None] + offsets[None, :],
        0,
        len(normalized) - 1,
    )
    triplets = normalized[index].astype(np.float32, copy=False)
    cropped = _native_center_crop(triplets, float(crop_fraction))
    images = resize_triplets_constant_area(cropped)
    if int(images.shape[0]) != B44_DENSE_SLICES:
        raise RuntimeError("B44 preprocessing did not return 64 centres")
    return images, position


class B44Nested64CoverageDataset(B42ConstantAreaAspectDataset):
    """B42 dataset with the exact original 32 centres plus 32 appended centres."""

    def _zero_b42(self) -> tuple[torch.Tensor, torch.Tensor]:
        views = len(self.center_offsets)
        image = torch.zeros(
            views,
            B44_DENSE_SLICES,
            3,
            448,
            448,
            dtype=torch.float32,
        )
        position = torch.zeros(views, B44_DENSE_SLICES, dtype=torch.float32)
        return image, position

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        path = find_series_dir(
            self.config.data_root,
            self.config.split,
            uid,
            str(series_uid),
        )
        if path is None:
            if self.config.strict_dicom:
                raise FileNotFoundError(f"missing series {uid}/{series_uid}")
            image, position = self._zero_b42()
            return image, position, 0.0
        try:
            raw = self._read_volume(path, plane.lower())
            images, positions = [], []
            for offset in self.center_offsets:
                image, position = preprocess_dense_triplets_b44_64(
                    raw,
                    gap=int(self.config.triplet_gap),
                    center_offset=int(offset),
                    crop_fraction=float(self.crop_focus_policy["crop_fraction"]),
                )
                images.append(image)
                positions.append(torch.from_numpy(position))
            return torch.stack(images), torch.stack(positions), 1.0
        except Exception:
            if self.config.strict_dicom:
                raise
            image, position = self._zero_b42()
            return image, position, 0.0


def _encode_ragged_64(model, volumes: list[torch.Tensor], present: torch.Tensor):
    """Use the frozen B42 encoder on 64 centres without changing any parameters."""
    if present.ndim == 2:
        if int(present.shape[0]) != 1:
            raise ValueError("B44 audit processes one study at a time")
        present_flat = present[0]
    elif present.ndim == 1:
        present_flat = present
    else:
        raise ValueError("B44 present mask must be [K] or [1,K]")
    if len(volumes) != int(present_flat.numel()):
        raise ValueError("B44 volumes/present count mismatch")

    global_rows: list[torch.Tensor | None] = []
    spatial_rows: list[torch.Tensor | None] = []
    template_global = template_spatial = None
    for series_tensor, flag in zip(volumes, present_flat):
        if series_tensor.ndim != 4 or int(series_tensor.shape[0]) != B44_DENSE_SLICES:
            raise ValueError("B44 series must be [64,3,H,W]")
        if float(flag.detach().item()) <= 0:
            global_rows.append(None)
            spatial_rows.append(None)
            continue
        global_series, spatial_series = model._encode_rect_group(series_tensor)
        global_rows.append(global_series)
        spatial_rows.append(spatial_series)
        if template_global is None:
            template_global, template_spatial = global_series, spatial_series

    if template_global is None or template_spatial is None:
        raise RuntimeError("B44 study has no readable MRI series")
    for index in range(len(global_rows)):
        if global_rows[index] is None:
            global_rows[index] = torch.zeros_like(template_global)
            spatial_rows[index] = torch.zeros_like(template_spatial)
    global_feature = torch.stack([x for x in global_rows if x is not None], dim=0).unsqueeze(0)
    spatial = torch.stack([x for x in spatial_rows if x is not None], dim=0).unsqueeze(0)
    return global_feature, spatial


def _read_b43_series_baseline(path: Path, uids: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"B44 requires B43 32-centre evidence table: {path}")
    frame = pd.read_csv(path)
    required = {
        "StudyInstanceUID", "target", "target_index", "truth", "series_uid",
        "plane", "series_top1_evidence", "base_probability", "combined_probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"B43 baseline evidence columns missing: {sorted(missing)}")
    if set(frame["StudyInstanceUID"].astype(str).unique()) != set(uids):
        raise RuntimeError("B44 B43-baseline Expert58 UID surface changed")
    return frame


def _aggregate_series_rows(view_rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "StudyInstanceUID", "target", "target_index", "truth", "series_index",
        "series_uid", "plane", "plane_id", "height", "width", "rectangular",
    ]
    values = [
        "base_probability", "combined_probability", "local_logit",
        "series_top1_evidence", "series_topk_lme", "series_mean_evidence",
    ]
    return view_rows.groupby(keys, sort=False, dropna=False)[values].mean().reset_index()


def _plane_signal(series_rows: pd.DataFrame, *, centers: int) -> pd.DataFrame:
    plane_best = (
        series_rows.sort_values(
            ["StudyInstanceUID", "target_index", "plane", "series_top1_evidence"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        .groupby(["StudyInstanceUID", "target_index", "plane"], as_index=False, sort=False)
        .head(1)
    )
    rows = []
    for (target, target_index, plane), z in plane_best.groupby(
        ["target", "target_index", "plane"], sort=False
    ):
        positive = z.loc[z["truth"] > 0.5]
        negative = z.loc[z["truth"] <= 0.5]
        auc = float("nan")
        if len(positive) and len(negative):
            auc = float(roc_auc_score(z["truth"], z["series_top1_evidence"]))
        rows.append(
            {
                "target": target,
                "target_index": int(target_index),
                "plane": plane,
                "centers": int(centers),
                "n_studies": int(len(z)),
                "n_positive": int(len(positive)),
                "n_negative": int(len(negative)),
                "positive_mean_best_evidence": float(positive["series_top1_evidence"].mean()),
                "negative_mean_best_evidence": float(negative["series_top1_evidence"].mean()),
                "evidence_separation": float(
                    positive["series_top1_evidence"].mean()
                    - negative["series_top1_evidence"].mean()
                ),
                "plane_specific_evidence_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target_index", "plane"], kind="mergesort"
    ).reset_index(drop=True)


def _study_target_prediction_from_series(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    keys = ["StudyInstanceUID", "target", "target_index", "truth"]
    out = frame.groupby(keys, sort=False, dropna=False)[column].agg(["mean", "min", "max"]).reset_index()
    delta = np.abs(out["max"].to_numpy(float) - out["min"].to_numpy(float))
    if len(delta) and float(delta.max()) > 1e-6:
        raise RuntimeError(f"{column} is not study-target invariant across series")
    return out[keys + ["mean"]].rename(columns={"mean": column})


def audit_b44_coverage_64(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    b43_series_baseline: str | Path = f"{B43_AUDIT_ROOT}/series_evidence_tta_mean.csv",
    out_root: str | Path = B44_AUDIT_ROOT,
) -> dict:
    """Evaluate frozen B42 with 64 nested centres and compare to recorded 32-centre evidence."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["b7_eval_batch_size"] = 1
    crop_policy = require_b42_contract(settings)
    root = Path(settings["data_root"])

    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B44 requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B44 Expert58 MRI series surface changed")

    baseline32 = _read_b43_series_baseline(Path(b43_series_baseline), uids)
    baseline32_base = _study_target_prediction_from_series(baseline32, "base_probability")
    baseline32_combined = _study_target_prediction_from_series(baseline32, "combined_probability")
    baseline32_signal = _plane_signal(baseline32, centers=B35_DENSE_SLICES)

    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    dcfg = make_b7_dataset_config(settings, root, train=False)
    dcfg.tta_center_offsets = ()
    dataset = B44Nested64CoverageDataset(
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
            seed=int(settings.get("seed", 2026)) + B44_AUDIT_LOADER_SEED_OFFSET
        ),
    )

    model, payload = load_b42_checkpoint(
        checkpoint,
        base_checkpoint=base_checkpoint,
        device=runtime.device,
    )
    model.eval()
    top_k = int(model.head.top_k)
    n_regions = int(model.head.n_regions)
    temperature = float(model.head.temperature)
    if top_k != 8:
        raise RuntimeError("B44 coverage diagnostic requires frozen B42 top-k=8")

    view_rows: list[dict] = []
    base_prediction_rows: list[np.ndarray] = []
    combined_prediction_rows: list[np.ndarray] = []
    scored_uids: list[str] = []

    with torch.no_grad():
        for batch_index, items in enumerate(loader, start=1):
            if len(items) != 1:
                raise RuntimeError("B44 audit requires one ragged study per batch")
            item = items[0]
            uid = str(item["study_uid"])
            scored_uids.append(uid)
            study_index = batch_index - 1
            records = index[uid]
            present_cpu = item["present"]
            present = present_cpu.to(runtime.device, non_blocking=True).unsqueeze(0)
            meta = item["series_meta"].to(runtime.device, non_blocking=True).unsqueeze(0)
            position_all = item["slice_position"].to(runtime.device, non_blocking=True)
            if position_all.shape != (len(records), len(B37_EVAL_OFFSETS), B44_DENSE_SLICES):
                raise RuntimeError(f"B44 TTA position shape changed: {tuple(position_all.shape)}")

            base_views: list[torch.Tensor] = []
            combined_views: list[torch.Tensor] = []
            for view_index, center_offset in enumerate(B37_EVAL_OFFSETS):
                volumes = [
                    series_tensor[view_index].to(runtime.device, non_blocking=True)
                    for series_tensor in item["volumes"]
                ]
                if any(int(volume.shape[0]) != B44_DENSE_SLICES for volume in volumes):
                    raise RuntimeError("B44 dataset returned a non-64-centre series")
                position = position_all[:, view_index].unsqueeze(0)
                with autocast(runtime):
                    global_feature, spatial = _encode_ragged_64(model, volumes, present)
                    base_logits = model._base_logits_from_global(global_feature, present, meta)
                    local_logits, _, _ = model.head(spatial, present, meta, position)
                    score = _score_tokens(model, spatial, present, meta, position)
                gate = model.head.effective_gate().to(device=runtime.device, dtype=local_logits.dtype)
                combined_logits = base_logits.float() + gate[None, :] * local_logits.float()
                base_probability = torch.sigmoid(base_logits.float())[0]
                combined_probability = torch.sigmoid(combined_logits)[0]
                base_views.append(base_probability.detach().cpu())
                combined_views.append(combined_probability.detach().cpu())

                per_series_tokens = B44_DENSE_SLICES * n_regions
                for target_index, target in enumerate(TARGETS):
                    target_score = score[0, target_index]
                    for series_index, (record, geom, flag) in enumerate(
                        zip(records, item["geometry"], present_cpu)
                    ):
                        if float(flag.item()) <= 0:
                            continue
                        start = series_index * per_series_tokens
                        end = start + per_series_tokens
                        finite = target_score[start:end]
                        if finite.numel() != per_series_tokens or not torch.isfinite(finite).all():
                            raise RuntimeError("B44 invalid evidence tokens inside present series")
                        top_values = torch.topk(
                            finite,
                            k=min(top_k, int(finite.numel())),
                            largest=True,
                            sorted=True,
                        ).values
                        plane_id = int(record["plane_id"])
                        plane = {1: "Sagittal", 2: "Coronal", 3: "Axial"}.get(
                            plane_id, f"ID{plane_id}"
                        )
                        view_rows.append(
                            {
                                "StudyInstanceUID": uid,
                                "target": target,
                                "target_index": target_index,
                                "truth": float(truth[study_index, target_index]),
                                "view_index": view_index,
                                "center_offset": int(center_offset),
                                "series_index": series_index,
                                "series_uid": str(record["series_uid"]),
                                "plane": plane,
                                "plane_id": plane_id,
                                "height": int(geom["height"]),
                                "width": int(geom["width"]),
                                "rectangular": bool(int(geom["height"]) != int(geom["width"])),
                                "base_probability": float(base_probability[target_index].item()),
                                "combined_probability": float(combined_probability[target_index].item()),
                                "local_logit": float(local_logits[0, target_index].detach().float().item()),
                                "series_top1_evidence": float(top_values[0].detach().float().item()),
                                "series_topk_lme": float(_lme(top_values, temperature).detach().item()),
                                "series_mean_evidence": float(finite.float().mean().detach().item()),
                            }
                        )

                del volumes, position, global_feature, spatial, base_logits, local_logits, score
                del combined_logits, base_probability, combined_probability

            base_prediction_rows.append(torch.stack(base_views).mean(dim=0).numpy())
            combined_prediction_rows.append(torch.stack(combined_views).mean(dim=0).numpy())
            del item, items, present_cpu, present, meta, position_all, base_views, combined_views
            _release()
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"[B44 frozen 64-centre coverage] {batch_index}/{len(loader)}", flush=True)

    if scored_uids != uids:
        raise RuntimeError("B44 Expert58 study order changed")
    base64_prediction = np.stack(base_prediction_rows).astype(np.float64)
    combined64_prediction = np.stack(combined_prediction_rows).astype(np.float64)
    if not np.isfinite(base64_prediction).all() or not np.isfinite(combined64_prediction).all():
        raise RuntimeError("B44 produced non-finite predictions")

    view64 = pd.DataFrame(view_rows)
    expected_view_rows = B18_EXPECTED_GOLD_SERIES * len(TARGETS) * len(B37_EVAL_OFFSETS)
    if len(view64) != expected_view_rows:
        raise RuntimeError(f"B44 expected {expected_view_rows} view rows, got {len(view64)}")
    series64 = _aggregate_series_rows(view64)
    if len(series64) != B18_EXPECTED_GOLD_SERIES * len(TARGETS):
        raise RuntimeError("B44 64-centre TTA-mean series row count changed")

    base32_matrix = (
        baseline32_base.pivot(index="StudyInstanceUID", columns="target", values="base_probability")
        .reindex(index=uids, columns=TARGETS)
        .to_numpy(np.float64)
    )
    base_reproduction_delta = float(np.abs(base64_prediction - base32_matrix).max())
    if base_reproduction_delta > 1e-6:
        raise RuntimeError(
            "B44 changed the frozen first-16/B34 base path: "
            f"max|delta|={base_reproduction_delta}"
        )

    signal64 = _plane_signal(series64, centers=B44_DENSE_SLICES)
    signal_compare = baseline32_signal.merge(
        signal64,
        on=["target", "target_index", "plane", "n_studies", "n_positive", "n_negative"],
        suffixes=("_32", "_64"),
        validate="one_to_one",
    )
    signal_compare["auc_delta_64_minus_32"] = (
        signal_compare["plane_specific_evidence_auc_64"]
        - signal_compare["plane_specific_evidence_auc_32"]
    )
    signal_compare["evidence_separation_delta_64_minus_32"] = (
        signal_compare["evidence_separation_64"]
        - signal_compare["evidence_separation_32"]
    )

    macro64, per_target64 = macro_auc_from_arrays(truth, combined64_prediction)
    baseline32_combined_matrix = (
        baseline32_combined.pivot(index="StudyInstanceUID", columns="target", values="combined_probability")
        .reindex(index=uids, columns=TARGETS)
        .to_numpy(np.float64)
    )
    macro32, per_target32 = macro_auc_from_arrays(truth, baseline32_combined_matrix)
    target_compare = pd.DataFrame(
        {
            "target": TARGETS,
            "target_index": np.arange(len(TARGETS), dtype=np.int64),
            "combined_auc_32": np.asarray(per_target32, dtype=np.float64),
            "combined_auc_64": np.asarray(per_target64, dtype=np.float64),
        }
    )
    target_compare["combined_auc_delta_64_minus_32"] = (
        target_compare["combined_auc_64"] - target_compare["combined_auc_32"]
    )
    target_compare["focus_target"] = target_compare["target"].isin(B44_FOCUS_TARGETS)

    focus_compare = signal_compare.loc[
        signal_compare["target"].isin(B44_FOCUS_TARGETS)
    ].copy()

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    view_path = out / "series_evidence_64_by_view.csv"
    series_path = out / "series_evidence_64_tta_mean.csv"
    signal_path = out / "target_plane_signal_32_vs_64.csv"
    focus_path = out / "focus_target_plane_signal_32_vs_64.csv"
    target_path = out / "target_combined_auc_32_vs_64.csv"
    view64.to_csv(view_path, index=False)
    series64.to_csv(series_path, index=False)
    signal_compare.to_csv(signal_path, index=False)
    focus_compare.to_csv(focus_path, index=False)
    target_compare.to_csv(target_path, index=False)

    summary = {
        "version": B44_AUDIT_VERSION,
        "evaluation_role": B44_AUDIT_ROLE,
        "source_model": "B42 frozen fixed-E2",
        "checkpoint": str(Path(checkpoint).resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "expert_studies": B18_EXPECTED_GOLD_STUDIES,
        "expert_series": B18_EXPECTED_GOLD_SERIES,
        "historical_centers": B35_DENSE_SLICES,
        "diagnostic_centers": B44_DENSE_SLICES,
        "added_centers": B44_ADDED_SLICES,
        "nested_first_32_exact": True,
        "first_16_base_path_unchanged": True,
        "base_prediction_max_abs_delta_64_vs_32": base_reproduction_delta,
        "tta_center_offsets": list(B37_EVAL_OFFSETS),
        "top_k_unchanged": top_k,
        "grid_size_unchanged": int(model.head.grid_size),
        "macro_auc_32_recorded": float(macro32),
        "macro_auc_64_diagnostic": float(macro64),
        "macro_auc_delta_64_minus_32": float(macro64 - macro32),
        "focus_targets": list(B44_FOCUS_TARGETS),
        "metadata_repair": metadata_stats,
        "outputs": {
            "series_evidence_64_by_view": str(view_path),
            "series_evidence_64_tta_mean": str(series_path),
            "target_plane_signal_32_vs_64": str(signal_path),
            "focus_target_plane_signal_32_vs_64": str(focus_path),
            "target_combined_auc_32_vs_64": str(target_path),
        },
        "interpretation_guardrail": (
            "Expert-58 is reused diagnostic evidence only.  The 64-centre result may "
            "identify a coverage mechanism but must not be used as independent test "
            "evidence or as a B42/B43/B44 promotion criterion."
        ),
    }
    summary_path = out / "audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("B44 FROZEN 32->64 CENTER COVERAGE AUDIT: PASS", flush=True)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--b43-series-baseline",
        default=f"{B43_AUDIT_ROOT}/series_evidence_tta_mean.csv",
    )
    parser.add_argument("--out-root", default=B44_AUDIT_ROOT)
    args = parser.parse_args(argv)
    config = _read_config(args.config)
    audit_b44_coverage_64(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        b43_series_baseline=args.b43_series_baseline,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B44_ADDED_SLICES",
    "B44_AUDIT_ROLE",
    "B44_AUDIT_ROOT",
    "B44_AUDIT_VERSION",
    "B44_DENSE_SLICES",
    "B44Nested64CoverageDataset",
    "audit_b44_coverage_64",
    "b44_nested_centers_64",
    "preprocess_dense_triplets_b44_64",
]
