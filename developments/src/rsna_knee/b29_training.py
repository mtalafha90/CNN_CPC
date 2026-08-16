"""B29 fixed-E2 training on the exact historical B20 gradient surface.

B29 keeps B20's learned series token and tests one new representation hypothesis:
one learned slice summary may be an information bottleneck. A second deterministic
learned softmax summary is mixed with the historical token through an exactly
zero-initialised feature-wise tanh gate.

Everything else remains B20: B6 supervision, 3,120 studies, 17,475 series,
14,123 supervised cells, frozen B16 report-aligned encoder, 90% crop, optimizer,
augmentation, loader seed, five-epoch scheduler horizon, and fixed E2 endpoint.
Expert labels are not read during training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import B17_HEAD_LR, encoder_state_sha256, freeze_encoder
from .b18_fisher_selection import B18_CANDIDATE_EPOCHS
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset, require_b20_contract
from .b29_complementary_series_pool import (
    B29_EXPECTED_GATE_PARAMETERS,
    B29_EXPECTED_NEW_PARAMETERS,
    B29_EXPECTED_QUERY_PARAMETERS,
    B29_RESIDUAL_VERSION,
    b29_model_spec,
    build_b29_model,
)
from .budget import RuntimeBudget
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B29_TRAIN_VERSION = "1.0.0"
B29_EXPERIMENT = "B29_zero_gated_complementary_learned_series_summary"
B29_VARIANT = "b20_plus_zero_init_featurewise_complementary_softmax_summary_fixed_e2_v1"
B29_FIXED_EPOCHS = 2
B29_EXPECTED_STUDIES = 3120
B29_EXPECTED_SERIES = 17475
B29_EXPECTED_CELLS = 14123
B29_EXPECTED_POS = 6871
B29_EXPECTED_NEG = 7252
B29_MAX_HARD_HOURS = 8.25
B29_MIN_RESERVE_MINUTES = 30.0


def train_b29(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b29_complementary_series_pool",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)  # exact historical B20 construction seed
    runtime = resolve_runtime(config)
    print(runtime.describe())

    max_hours = min(float(config.get("runtime_budget_hours", 8.5)), B29_MAX_HARD_HOURS)
    reserve_minutes = max(
        float(config.get("runtime_reserve_minutes", 10.0)),
        B29_MIN_RESERVE_MINUTES,
    )
    budget = RuntimeBudget(max_hours=max_hours, reserve_minutes=reserve_minutes)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv")).copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    study_uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)
    study_uids = [str(uid) for uid in study_uids]

    if len(study_uids) != B29_EXPECTED_STUDIES:
        raise RuntimeError(f"B29 requires {B29_EXPECTED_STUDIES} studies")
    active_cells = int((weights > 0).sum())
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    if (active_cells, positive_cells, negative_cells) != (
        B29_EXPECTED_CELLS,
        B29_EXPECTED_POS,
        B29_EXPECTED_NEG,
    ):
        raise RuntimeError("B29 B6 supervision surface drifted from historical B20")

    gold_by_uid = torch.as_tensor(gold_mask(train).to_numpy(dtype=bool), dtype=torch.bool)
    uid_to_row = {uid: i for i, uid in enumerate(train["StudyInstanceUID"].tolist())}
    if any(bool(gold_by_uid[uid_to_row[uid]]) for uid in study_uids):
        raise RuntimeError("expert-gold study unexpectedly entered B29 gradients")

    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B29 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, study_uids)
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B29 reconstructed series surface does not match frozen B13 SHA-256")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != B29_EXPECTED_SERIES:
        raise ValueError("B29 requires the exact 17,475-series B20 gradient surface")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B29 reconstructed series surface failed viability")

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(study_uids) / batch_size))
    if expected_batches != 1560:
        raise RuntimeError("B29 exact B20 surface must yield 1,560 batches per epoch")
    target_multiplier = target_balance_multipliers(weights)

    train_ds = CropFocusedVariableSeriesKneeDataset(
        study_uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 19_100_000),  # exact historical B20 loader seed
    )

    spec = b29_model_spec(config, normalize_input=True)
    model = build_b29_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)

    state_initial = model.complementary_state()
    if state_initial["new_parameter_count"] != B29_EXPECTED_NEW_PARAMETERS:
        raise RuntimeError("B29 new parameter count changed")
    if state_initial["query_parameter_count"] != B29_EXPECTED_QUERY_PARAMETERS:
        raise RuntimeError("B29 complementary query parameter count changed")
    if state_initial["gate_parameter_count"] != B29_EXPECTED_GATE_PARAMETERS:
        raise RuntimeError("B29 complementary gate parameter count changed")
    if state_initial["gate_raw_max_abs"] != 0.0 or state_initial["gate_effective_max_abs"] != 0.0:
        raise RuntimeError("B29 complementary residual gate must start at exactly zero")

    model = model.to(runtime.device)
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params or any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("B29 frozen-encoder contract failed")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", B17_HEAD_LR))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B18_CANDIDATE_EPOCHS,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    clip = float(config.get("b7_grad_clip", 1.0))

    print(
        f"[B29] studies={len(study_uids)} series={B29_EXPECTED_SERIES} "
        f"cells={active_cells} | new_params={B29_EXPECTED_NEW_PARAMETERS} "
        f"(query={B29_EXPECTED_QUERY_PARAMETERS}, gate={B29_EXPECTED_GATE_PARAMETERS}) "
        f"| fixed E{B29_FIXED_EPOCHS}"
    )
    print(
        "[B29] token=A+tanh(g)*(C-A) | A=B20 learned-attention token | "
        "C=deterministic learned softmax slice summary | gate=zero init"
    )
    print(
        f"[B29] hard_runtime={max_hours:.2f} h | reserve={reserve_minutes:.0f} min "
        "| expert labels=not read"
    )

    history: list[dict] = []
    for epoch in range(1, B29_FIXED_EPOCHS + 1):
        epoch_started = time.monotonic()
        model.train()
        model.encoder.eval()
        loss_sum = 0.0
        steps = seen_studies = seen_series = seen_cells = seen_pos = seen_neg = 0
        gate_grad_seen = False
        query_grad_seen = False

        for batch in loader:
            if not budget.can_start(120.0):
                raise RuntimeError("B29 runtime budget expired before complete fixed-E2")
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            meta = batch["series_meta"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present, meta)
                loss = target_balanced_weak_bce(logits, target, weight, multiplier_t)
            scaler.scale(loss).backward()
            if any(p.grad is not None for p in model.encoder.parameters()):
                raise RuntimeError("B29 detected an encoder gradient")

            gate_grad = model.complementary_gate.grad
            query_grad = model.complementary_query.grad
            if gate_grad is None or not torch.isfinite(gate_grad).all():
                raise RuntimeError("B29 complementary gate gradient missing or non-finite")
            if query_grad is None or not torch.isfinite(query_grad).all():
                raise RuntimeError("B29 complementary query gradient missing or non-finite")
            gate_grad_seen = gate_grad_seen or bool(torch.count_nonzero(gate_grad).item() > 0)
            # At the very first zero-gate backward the query gradient is expected
            # to be zero. It must become data-coupled after the gate moves.
            query_grad_seen = query_grad_seen or bool(torch.count_nonzero(query_grad).item() > 0)

            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(head_params, clip)
            scaler.step(optimizer)
            scaler.update()

            active = weight > 0
            loss_sum += float(loss.item())
            steps += 1
            seen_studies += int(volumes.shape[0])
            seen_series += int(present.sum().item())
            seen_cells += int(active.sum().item())
            seen_pos += int((active & (target > 0.5)).sum().item())
            seen_neg += int((active & (target < 0.5)).sum().item())

        scheduler.step()
        epoch_seconds = time.monotonic() - epoch_started
        if encoder_state_sha256(model.encoder) != encoder_sha_initial:
            raise RuntimeError("B29 encoder changed despite freezing")
        full = (
            steps == expected_batches
            and seen_studies == B29_EXPECTED_STUDIES
            and seen_series == B29_EXPECTED_SERIES
            and seen_cells == B29_EXPECTED_CELLS
            and seen_pos == B29_EXPECTED_POS
            and seen_neg == B29_EXPECTED_NEG
        )
        if not full:
            raise RuntimeError(f"B29 epoch {epoch} did not complete the frozen surface")
        if not gate_grad_seen:
            raise RuntimeError(f"B29 epoch {epoch} never produced a nonzero gate gradient")
        if not query_grad_seen:
            raise RuntimeError(f"B29 epoch {epoch} never coupled the complementary query to the loss")

        state = model.complementary_state()
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(steps, 1),
            "head_lr": float(optimizer.param_groups[0]["lr"]),
            "batches": steps,
            "studies": seen_studies,
            "series_instances": seen_series,
            "supervision_cells": seen_cells,
            "positive_cells": seen_pos,
            "negative_cells": seen_neg,
            "epoch_seconds": round(float(epoch_seconds), 1),
            "full_coverage": True,
            "gate_nonzero_gradient_seen": True,
            "query_nonzero_gradient_seen": True,
            "complementary_pool": state,
        }
        history.append(row)
        print(
            f"[B29] E{epoch} loss={row['loss']:.10f} | {epoch_seconds/60:.1f} min "
            f"| gate|max|={state['gate_effective_max_abs']:.6f} "
            f"| gate|mean|={state['gate_effective_mean_abs']:.6f} "
            f"| qcos(primary)={state['query_cosine_to_primary']:+.4f}"
        )

    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("B29 encoder fingerprint changed")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "b29_model.pt"
    state_final = model.complementary_state()
    payload = {
        "experiment": B29_EXPERIMENT,
        "variant": B29_VARIANT,
        "b29_training_version": B29_TRAIN_VERSION,
        "b29_residual_version": B29_RESIDUAL_VERSION,
        "fixed_endpoint": True,
        "selected_epoch": B29_FIXED_EPOCHS,
        "completed_epochs": B29_FIXED_EPOCHS,
        "scheduler_horizon": B18_CANDIDATE_EPOCHS,
        "model_spec": spec,
        "model_state": model.state_dict(),
        "config": config,
        "crop_focus_enabled": True,
        "crop_focus_policy": crop_policy,
        "b19_cosine_mask_used": False,
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "study_uids": list(study_uids),
        "history": history,
        "complementary_pool_initial": state_initial,
        "complementary_pool_final": state_final,
        "supervision": supervision,
        "b6_policy": b6_policy,
        "b6_audit": b6_audit,
        "metadata_repair": metadata_stats,
        "series_summary": series_summary,
        "b6_root": str(Path(b6_root).resolve()),
        "series_policy": str(Path(series_policy_path).resolve()),
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "gold_studies_used_in_gradient": 0,
        "expert_checkpoint_selection": False,
        "evaluation_status": "not yet evaluated; fixed-E2 checkpoint",
        "runtime_budget": budget.to_dict(),
    }
    torch.save(payload, checkpoint)

    audit = {
        "experiment": B29_EXPERIMENT,
        "variant": B29_VARIANT,
        "checkpoint": str(checkpoint),
        "fixed_epoch": B29_FIXED_EPOCHS,
        "training_studies": B29_EXPECTED_STUDIES,
        "series_instances": B29_EXPECTED_SERIES,
        "supervision_cells": B29_EXPECTED_CELLS,
        "positive_cells": B29_EXPECTED_POS,
        "negative_cells": B29_EXPECTED_NEG,
        "new_parameter_count": B29_EXPECTED_NEW_PARAMETERS,
        "query_parameter_count": B29_EXPECTED_QUERY_PARAMETERS,
        "gate_parameter_count": B29_EXPECTED_GATE_PARAMETERS,
        "complementary_pool_initial": state_initial,
        "complementary_pool_final": state_final,
        "encoder_sha256": encoder_sha_final,
        "history": history,
        "runtime_budget": budget.to_dict(),
        "gold_used": False,
    }
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "complementary_pool_state.json").write_text(
        json.dumps(state_final, indent=2), encoding="utf-8"
    )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint)
    return checkpoint


def load_b29_checkpoint(path: str | Path, *, device: torch.device | str = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B29_EXPERIMENT or payload.get("variant") != B29_VARIANT:
        raise ValueError("not a B29 fixed-E2 complementary-series-pool checkpoint")
    if payload.get("fixed_endpoint") is not True or int(payload.get("completed_epochs", -1)) != B29_FIXED_EPOCHS:
        raise ValueError("B29 checkpoint must be the complete fixed-E2 endpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B29 checkpoint does not certify zero gold gradients")
    if payload.get("expert_checkpoint_selection") is not False:
        raise ValueError("B29 checkpoint unexpectedly used expert checkpoint selection")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("B29 encoder fingerprint changed")
    final_state = payload.get("complementary_pool_final") or {}
    if int(final_state.get("new_parameter_count", -1)) != B29_EXPECTED_NEW_PARAMETERS:
        raise ValueError("B29 checkpoint new-parameter contract changed")

    model = build_b29_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("B29 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("B29 zero-gated complementary learned series summary")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b29_complementary_series_pool")
    args = parser.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    train_b29(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
