"""B43 precursor diagnostic: target x series x plane evidence audit for frozen B42.

This module is deliberately diagnostic-only.  It does not train, select, tune,
blend, threshold, or promote a model.  It reuses the post-B42 Expert-58 surface
to answer a narrower mechanistic question: for each pathology, which acquired
MRI series and anatomical plane supply the strongest B42 sparse-MIL evidence,
and does removing that series materially change the frozen study prediction?

The frozen B42 endpoint and its three evaluation offsets [-1, 0, +1] are kept
unchanged.  Series are never re-encoded for leave-one-series-out diagnostics:
the already-computed B42 global/spatial features are reused and only the frozen
B34 aggregation mask plus frozen B42 sparse head are reevaluated.  Therefore the
leave-one-out quantities measure aggregation/routing sensitivity rather than a
new model.
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

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b42_constant_area_aspect_sparse_eval import load_b42_checkpoint
from .b42_constant_area_aspect_sparse_mil import (
    B42_EXPERT58_ROOT,
    B42ConstantAreaAspectDataset,
    collate_b42,
    require_b42_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import macro_auc_from_arrays
from .runtime import autocast, resolve_runtime

B43_AUDIT_VERSION = "b43_target_series_plane_evidence_audit_v1"
B43_AUDIT_ROOT = f"{B42_EXPERT58_ROOT}/target_series_plane_audit"
B43_AUDIT_LOADER_SEED_OFFSET = 53_100_000
B43_AUDIT_ROLE = (
    "reused post-B42 Expert-58 mechanistic diagnostic; not independent test evidence, "
    "not a tuning set, and not a B43 training/promotion criterion"
)

PLANE_NAMES = {0: "Unknown", 1: "Sagittal", 2: "Coronal", 3: "Axial"}
FLAG_NAMES = {0: "Unknown", 1: "False", 2: "True"}


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sigmoid_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().item())
    return float(1.0 / (1.0 + math.exp(-float(value))))


def decode_sparse_index(
    flat_index: int,
    *,
    n_slices: int,
    n_regions: int,
    grid_size: int,
) -> tuple[int, int, int, int, int]:
    """Decode flattened B36/B42 token index into series/slice/grid coordinates."""
    index = int(flat_index)
    slices = int(n_slices)
    regions = int(n_regions)
    grid = int(grid_size)
    if index < 0 or slices < 1 or regions < 1 or grid < 1 or grid * grid != regions:
        raise ValueError("invalid sparse-index geometry")
    per_series = slices * regions
    series_index = index // per_series
    within_series = index % per_series
    slice_index = within_series // regions
    region_index = within_series % regions
    region_row = region_index // grid
    region_col = region_index % grid
    return series_index, slice_index, region_index, region_row, region_col


def _lme(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """B36/B42 log-mean-exp pooling on an already selected vector."""
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("log-mean-exp requires a non-empty 1-D vector")
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("temperature must be positive")
    return tau * (torch.logsumexp(values.float() / tau, dim=0) - math.log(float(values.numel())))


def _score_tokens(model, spatial, present, series_meta, slice_position):
    """Return the exact frozen B42 token evidence surface [1,T,N]."""
    tokens, invalid = model.head._tokens(spatial, present, series_meta, slice_position)
    score = torch.einsum(
        "bnd,td->btn",
        tokens,
        model.head.evidence_weight.to(dtype=tokens.dtype),
    ) + model.head.evidence_bias.to(dtype=tokens.dtype)[None, :, None]
    return score.masked_fill(invalid[:, None, :], float("-inf"))


def _read_reference_prediction(path: Path, uids: list[str]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"B43 audit requires the already-recorded B42 Expert-58 prediction file: {path}"
        )
    frame = pd.read_csv(path)
    required = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != required:
        raise ValueError("B42 reference prediction columns changed")
    if frame["StudyInstanceUID"].astype(str).tolist() != uids:
        raise RuntimeError("B42 reference prediction UID order changed")
    prediction = frame[TARGETS].to_numpy(np.float64)
    if prediction.shape != (len(uids), len(TARGETS)) or not np.isfinite(prediction).all():
        raise RuntimeError("invalid B42 reference prediction matrix")
    return prediction


def _aggregate_series_rows(view_rows: pd.DataFrame, top_k: int) -> pd.DataFrame:
    keys = [
        "StudyInstanceUID",
        "target",
        "target_index",
        "truth",
        "series_index",
        "series_uid",
        "plane",
        "plane_id",
        "fluid_sensitive",
        "fluid_id",
        "fat_suppression",
        "fat_id",
        "height",
        "width",
        "rectangular",
    ]
    means = [
        "base_probability",
        "combined_probability",
        "local_logit",
        "gate_effective",
        "gated_local_logit",
        "series_top1_evidence",
        "series_topk_lme",
        "series_mean_evidence",
        "leave_one_out_base_probability",
        "leave_one_out_combined_probability",
        "leave_one_out_logit_delta",
        "leave_one_out_probability_delta",
    ]
    grouped = view_rows.groupby(keys, sort=False, dropna=False)
    out = grouped[means].mean().reset_index()
    selected = grouped["selected_count"].sum().reset_index(name="selected_count_three_views")
    top1 = grouped["is_global_top1_series"].sum().reset_index(name="global_top1_views")
    out = out.merge(selected, on=keys, validate="one_to_one")
    out = out.merge(top1, on=keys, validate="one_to_one")
    out["selected_fraction_three_views"] = out["selected_count_three_views"] / float(
        len(B37_EVAL_OFFSETS) * int(top_k)
    )
    out["global_top1_fraction"] = out["global_top1_views"] / float(len(B37_EVAL_OFFSETS))
    return out


def _strongest_series_table(series_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = series_rows.sort_values(
        ["StudyInstanceUID", "target_index", "series_top1_evidence", "series_index"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    strongest = ordered.groupby(["StudyInstanceUID", "target_index"], sort=False).head(1).copy()
    strongest = strongest.rename(
        columns={
            "series_index": "strongest_series_index",
            "series_uid": "strongest_series_uid",
            "plane": "strongest_plane",
            "plane_id": "strongest_plane_id",
            "fluid_sensitive": "strongest_fluid_sensitive",
            "fat_suppression": "strongest_fat_suppression",
            "series_top1_evidence": "strongest_series_top1_evidence",
            "series_topk_lme": "strongest_series_topk_lme",
            "leave_one_out_logit_delta": "strongest_series_leave_one_out_logit_delta",
            "leave_one_out_probability_delta": "strongest_series_leave_one_out_probability_delta",
        }
    )
    keep = [
        "StudyInstanceUID",
        "target",
        "target_index",
        "truth",
        "base_probability",
        "combined_probability",
        "local_logit",
        "gate_effective",
        "strongest_series_index",
        "strongest_series_uid",
        "strongest_plane",
        "strongest_plane_id",
        "strongest_fluid_sensitive",
        "strongest_fat_suppression",
        "strongest_series_top1_evidence",
        "strongest_series_topk_lme",
        "strongest_series_leave_one_out_logit_delta",
        "strongest_series_leave_one_out_probability_delta",
        "global_top1_fraction",
        "selected_fraction_three_views",
    ]
    return strongest[keep].reset_index(drop=True)


def _plane_summary(strongest: pd.DataFrame) -> pd.DataFrame:
    counts = (
        strongest.groupby(["target", "target_index", "truth", "strongest_plane"], sort=False)
        .size()
        .reset_index(name="strongest_count")
    )
    totals = (
        strongest.groupby(["target", "target_index", "truth"], sort=False)
        .size()
        .reset_index(name="study_target_count")
    )
    counts = counts.merge(totals, on=["target", "target_index", "truth"], validate="many_to_one")
    counts["strongest_fraction"] = counts["strongest_count"] / counts["study_target_count"]
    return counts.sort_values(
        ["target_index", "truth", "strongest_fraction", "strongest_plane"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def audit_target_series_plane(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B43_AUDIT_ROOT,
    reference_predictions: str | Path | None = None,
) -> dict:
    """Run the post-B42 Expert-58 target/series/plane mechanistic audit."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["b7_eval_batch_size"] = 1
    crop_policy = require_b42_contract(settings)
    root = Path(settings["data_root"])

    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B43 audit requires the complete reused 58-study expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B43 audit Expert-58 MRI series surface changed")

    if reference_predictions is None:
        reference_predictions = Path(B42_EXPERT58_ROOT) / "b42_combined_predictions.csv"
    reference = _read_reference_prediction(Path(reference_predictions), uids)

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
            seed=int(settings.get("seed", 2026)) + B43_AUDIT_LOADER_SEED_OFFSET
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
    grid_size = int(model.head.grid_size)
    temperature = float(model.head.temperature)
    gate = model.head.effective_gate().detach().float().cpu().numpy()

    view_rows: list[dict] = []
    prediction_rows: list[np.ndarray] = []
    scored_uids: list[str] = []

    with torch.no_grad():
        for batch_index, items in enumerate(loader, start=1):
            if len(items) != 1:
                raise RuntimeError("B43 audit requires one ragged study per batch")
            item = items[0]
            uid = str(item["study_uid"])
            scored_uids.append(uid)
            study_index = batch_index - 1
            records = index[uid]

            present_cpu = item["present"]
            present = present_cpu.to(runtime.device, non_blocking=True)
            meta = item["series_meta"].to(runtime.device, non_blocking=True)
            position_all = item["slice_position"].to(runtime.device, non_blocking=True)
            if position_all.ndim != 3 or int(position_all.shape[1]) != len(B37_EVAL_OFFSETS):
                raise RuntimeError("B43 audit TTA position shape changed")
            if len(records) != len(item["volumes"]):
                raise RuntimeError("B43 audit series-record order changed")

            combined_views: list[torch.Tensor] = []
            for view_index, center_offset in enumerate(B37_EVAL_OFFSETS):
                volumes = [
                    series_tensor[view_index].to(runtime.device, non_blocking=True)
                    for series_tensor in item["volumes"]
                ]
                position = position_all[:, view_index]

                with autocast(runtime):
                    global_feature, spatial = model._encode_ragged_study(volumes, present)
                    base_logits = model._base_logits_from_global(global_feature, present, meta)
                    local_logits, top_indices, top_values = model.head(
                        spatial, present, meta, position
                    )
                    score = _score_tokens(model, spatial, present, meta, position)

                gate_device = model.head.effective_gate().to(
                    device=runtime.device, dtype=local_logits.dtype
                )
                combined_logits = base_logits.float() + gate_device[None, :] * local_logits.float()
                base_probability = torch.sigmoid(base_logits.float())[0]
                combined_probability = torch.sigmoid(combined_logits)[0]
                combined_views.append(combined_probability.detach().cpu())

                verify_values, verify_indices = torch.topk(
                    score,
                    k=top_k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
                if not torch.equal(verify_indices, top_indices):
                    raise RuntimeError("B43 token-score reconstruction changed B42 top-k indices")
                max_top_value_delta = float(
                    torch.max(torch.abs(verify_values.float() - top_values.float())).item()
                )
                if max_top_value_delta > 1e-5:
                    raise RuntimeError(
                        f"B43 token-score reconstruction changed B42 top-k values: {max_top_value_delta}"
                    )

                valid_series = int((present_cpu > 0).sum().item())
                loo_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                if valid_series > 1:
                    for series_index, flag in enumerate(present_cpu):
                        if float(flag.item()) <= 0:
                            continue
                        present_loo = present.clone()
                        present_loo[series_index] = 0.0
                        with autocast(runtime):
                            loo_base_logits = model._base_logits_from_global(
                                global_feature, present_loo, meta
                            )
                            loo_local_logits, _, _ = model.head(
                                spatial, present_loo, meta, position
                            )
                        loo_combined_logits = loo_base_logits.float() + gate_device[None, :] * loo_local_logits.float()
                        loo_cache[series_index] = (
                            loo_base_logits.detach().float()[0],
                            loo_combined_logits.detach().float()[0],
                        )

                slices = int(spatial.shape[2])
                per_series_tokens = slices * n_regions
                if slices != 32:
                    raise RuntimeError(f"B43 expected 32 dense slices, got {slices}")

                for target_index, target in enumerate(TARGETS):
                    target_score = score[0, target_index]
                    selected = top_indices[0, target_index]
                    selected_series = torch.div(
                        selected, per_series_tokens, rounding_mode="floor"
                    )
                    full_local = float(local_logits[0, target_index].detach().float().item())
                    full_base_logit = float(base_logits[0, target_index].detach().float().item())
                    full_combined_logit = float(combined_logits[0, target_index].detach().float().item())

                    for series_index, (record, geom, flag) in enumerate(
                        zip(records, item["geometry"], present_cpu)
                    ):
                        if float(flag.item()) <= 0:
                            continue
                        start = series_index * per_series_tokens
                        end = start + per_series_tokens
                        series_score = target_score[start:end]
                        finite = series_score[torch.isfinite(series_score)]
                        if finite.numel() != per_series_tokens:
                            raise RuntimeError("B43 encountered invalid tokens inside a present series")
                        series_top_values = torch.topk(
                            finite,
                            k=min(top_k, int(finite.numel())),
                            largest=True,
                            sorted=True,
                        ).values
                        series_top1 = float(series_top_values[0].detach().float().item())
                        series_topk_lme = float(_lme(series_top_values, temperature).detach().item())
                        series_mean = float(finite.float().mean().detach().item())
                        selected_count = int((selected_series == series_index).sum().item())
                        is_top1_series = bool(int(selected_series[0].item()) == series_index)

                        if series_index in loo_cache:
                            loo_base_vec, loo_combined_vec = loo_cache[series_index]
                            loo_base_logit = float(loo_base_vec[target_index].item())
                            loo_combined_logit = float(loo_combined_vec[target_index].item())
                            loo_base_probability = _sigmoid_scalar(loo_base_logit)
                            loo_combined_probability = _sigmoid_scalar(loo_combined_logit)
                            loo_logit_delta = full_combined_logit - loo_combined_logit
                            loo_probability_delta = float(combined_probability[target_index].item()) - loo_combined_probability
                        else:
                            loo_base_probability = float("nan")
                            loo_combined_probability = float("nan")
                            loo_logit_delta = float("nan")
                            loo_probability_delta = float("nan")

                        meta_ids = record
                        plane_id = int(meta_ids["plane_id"])
                        fluid_id = int(meta_ids["fluid_id"])
                        fat_id = int(meta_ids["fat_id"])
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
                                "plane": PLANE_NAMES.get(plane_id, f"ID{plane_id}"),
                                "plane_id": plane_id,
                                "fluid_sensitive": FLAG_NAMES.get(fluid_id, f"ID{fluid_id}"),
                                "fluid_id": fluid_id,
                                "fat_suppression": FLAG_NAMES.get(fat_id, f"ID{fat_id}"),
                                "fat_id": fat_id,
                                "height": int(geom["height"]),
                                "width": int(geom["width"]),
                                "rectangular": bool(int(geom["height"]) != int(geom["width"])),
                                "base_probability": float(base_probability[target_index].item()),
                                "combined_probability": float(combined_probability[target_index].item()),
                                "base_logit": full_base_logit,
                                "local_logit": full_local,
                                "gate_effective": float(gate[target_index]),
                                "gated_local_logit": float(gate[target_index] * full_local),
                                "series_top1_evidence": series_top1,
                                "series_topk_lme": series_topk_lme,
                                "series_mean_evidence": series_mean,
                                "selected_count": selected_count,
                                "is_global_top1_series": is_top1_series,
                                "leave_one_out_base_probability": loo_base_probability,
                                "leave_one_out_combined_probability": loo_combined_probability,
                                "leave_one_out_logit_delta": loo_logit_delta,
                                "leave_one_out_probability_delta": loo_probability_delta,
                            }
                        )

                del (
                    volumes,
                    position,
                    global_feature,
                    spatial,
                    base_logits,
                    local_logits,
                    top_indices,
                    top_values,
                    score,
                    verify_values,
                    verify_indices,
                    combined_logits,
                    base_probability,
                    combined_probability,
                    loo_cache,
                )

            prediction_rows.append(torch.stack(combined_views, dim=0).mean(dim=0).numpy())
            del item, items, present_cpu, present, meta, position_all, combined_views
            _release()
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"[B43 target-series-plane audit] {batch_index}/{len(loader)}", flush=True)

    if scored_uids != uids:
        raise RuntimeError("B43 audit study order changed")
    prediction = np.stack(prediction_rows, axis=0).astype(np.float64)
    prediction_delta = np.abs(prediction - reference)
    max_prediction_delta = float(prediction_delta.max())
    if max_prediction_delta > 1e-6:
        raise RuntimeError(
            f"B43 audit failed frozen B42 prediction reproduction: max|delta|={max_prediction_delta}"
        )

    view_frame = pd.DataFrame(view_rows)
    expected_view_rows = B18_EXPECTED_GOLD_SERIES * len(TARGETS) * len(B37_EVAL_OFFSETS)
    if len(view_frame) != expected_view_rows:
        raise RuntimeError(
            f"B43 view evidence row count changed: expected {expected_view_rows}, got {len(view_frame)}"
        )
    series_frame = _aggregate_series_rows(view_frame, top_k)
    expected_series_rows = B18_EXPECTED_GOLD_SERIES * len(TARGETS)
    if len(series_frame) != expected_series_rows:
        raise RuntimeError("B43 aggregated series evidence row count changed")
    strongest = _strongest_series_table(series_frame)
    if len(strongest) != B18_EXPECTED_GOLD_STUDIES * len(TARGETS):
        raise RuntimeError("B43 strongest-series table row count changed")
    plane_summary = _plane_summary(strongest)

    macro_auc, per_target_auc = macro_auc_from_arrays(truth, prediction)
    target_summary_rows = []
    for target_index, target in enumerate(TARGETS):
        subset = strongest.loc[strongest["target_index"] == target_index]
        positive = subset.loc[subset["truth"] > 0.5]
        negative = subset.loc[subset["truth"] <= 0.5]
        target_summary_rows.append(
            {
                "target": target,
                "target_index": target_index,
                "auc": float(per_target_auc[target_index]),
                "n_positive": int(len(positive)),
                "n_negative": int(len(negative)),
                "positive_mean_strongest_top1_evidence": float(positive["strongest_series_top1_evidence"].mean()),
                "negative_mean_strongest_top1_evidence": float(negative["strongest_series_top1_evidence"].mean()),
                "positive_mean_strongest_loo_probability_delta": float(
                    positive["strongest_series_leave_one_out_probability_delta"].mean()
                ),
                "negative_mean_strongest_loo_probability_delta": float(
                    negative["strongest_series_leave_one_out_probability_delta"].mean()
                ),
                "positive_mean_combined_probability": float(positive["combined_probability"].mean()),
                "negative_mean_combined_probability": float(negative["combined_probability"].mean()),
            }
        )
    target_summary = pd.DataFrame(target_summary_rows)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    view_path = out / "series_evidence_by_view.csv"
    series_path = out / "series_evidence_tta_mean.csv"
    strongest_path = out / "strongest_series_by_study_target.csv"
    plane_path = out / "target_plane_summary.csv"
    target_path = out / "target_summary.csv"
    view_frame.to_csv(view_path, index=False)
    series_frame.to_csv(series_path, index=False)
    strongest.to_csv(strongest_path, index=False)
    plane_summary.to_csv(plane_path, index=False)
    target_summary.to_csv(target_path, index=False)

    summary = {
        "version": B43_AUDIT_VERSION,
        "evaluation_role": B43_AUDIT_ROLE,
        "source_model": "B42 frozen fixed-E2",
        "checkpoint": str(Path(checkpoint).resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "expert_studies": B18_EXPECTED_GOLD_STUDIES,
        "expert_series": B18_EXPECTED_GOLD_SERIES,
        "targets": len(TARGETS),
        "tta_center_offsets": list(B37_EVAL_OFFSETS),
        "top_k": top_k,
        "grid_size": grid_size,
        "regions_per_slice": n_regions,
        "dense_slices": 32,
        "macro_auc_reproduced": float(macro_auc),
        "reference_prediction_max_abs_delta": max_prediction_delta,
        "view_evidence_rows": int(len(view_frame)),
        "tta_mean_series_rows": int(len(series_frame)),
        "strongest_series_rows": int(len(strongest)),
        "metadata_repair": metadata_stats,
        "outputs": {
            "series_evidence_by_view": str(view_path),
            "series_evidence_tta_mean": str(series_path),
            "strongest_series_by_study_target": str(strongest_path),
            "target_plane_summary": str(plane_path),
            "target_summary": str(target_path),
        },
        "interpretation_guardrail": (
            "Use this audit only to diagnose whether evidence exists in individual series and "
            "whether frozen aggregation is sensitive to those series. Do not choose B42/B43 "
            "submission parameters from Expert-58."
        ),
    }
    summary_path = out / "audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("B43 TARGET-SERIES-PLANE AUDIT: PASS", flush=True)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=B43_AUDIT_ROOT)
    parser.add_argument("--reference-predictions", default=None)
    args = parser.parse_args(argv)
    config = _read_config(args.config)
    audit_target_series_plane(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        reference_predictions=args.reference_predictions,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B43_AUDIT_ROLE",
    "B43_AUDIT_ROOT",
    "B43_AUDIT_VERSION",
    "audit_target_series_plane",
    "decode_sparse_index",
]
