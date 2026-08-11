"""Matched downstream training for the frozen weak-holdout-v2 B13 control and B15.

Both arms use the B13 hierarchical architecture, B6 v1.2.1 supervision policy,
all-real-series mapping, optimizer, augmentations, four epochs and TTA contract.
They train on the exact same 2,497 v2 weak-train studies.

The only model difference is encoder initialization:
- control: torchvision ConvNeXt-Tiny ImageNet-1K V1;
- B15: the frozen B15 same-study knee-MRI SSL encoder, itself initialized from
  the same ImageNet weights.

The 623 v2 holdout studies are excluded from downstream optimization in both arms.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    B7_MIN_CONFIDENCE,
    B7_NEGATIVE_TARGET,
    B7_NEGATIVE_WEIGHT,
    B7_POSITIVE_TARGET,
    B7_POSITIVE_WEIGHT,
    make_b7_dataset_config,
    load_frozen_b6_export,
    prepare_b7_supervision,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    audit_variable_series_surface,
    collate_variable_series,
)
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import (
    B13_INITIALIZATION,
    B13_INPUT_NORMALIZATION,
    B13_SERIES_SIGNATURE,
    _load_imagenet_encoder_state,
)
from .b15_ssl import (
    B15_SSL_OBJECTIVE,
    B15_SSL_VARIANT,
    WEAK_V2_MANIFEST_SHA256,
    WEAK_V2_SURFACE,
    load_b15_ssl_encoder,
    load_frozen_v2_manifest,
)
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B13_V2_VARIANT = "b13_imagenet_b6_hierarchical_weak_v2_control_v1"
B13_V2_EXPERIMENT = "B13_v2_matched_control"
B15_VARIANT = "b15_imagenet_mri_ssl_b6_hierarchical_weak_v2_v1"
B15_EXPERIMENT = "B15_mri_ssl_downstream"


def _require_close(config: dict, key: str, expected: float) -> None:
    value = float(config.get(key, expected))
    if not np.isclose(value, expected, atol=1e-12, rtol=0):
        raise ValueError(f"v2 downstream freezes {key}={expected}; got {value}")


def require_v2_downstream_contract(config: dict) -> None:
    integer_contract = {
        "seed": 2026,
        "requested_gpus": 1,
        "b7_n_slices": 16,
        "b7_image_size": 224,
        "b7_triplet_gap": 1,
        "b7_batch_size": 2,
        "b7_encoder_batch_size": 24,
        "b7_transformer_layers": 2,
        "b7_transformer_heads": 8,
        "b7_pathology_layers": 1,
        "b12_1_series_pool_heads": 8,
        "b7_epochs": 4,
        "b7_eval_batch_size": 2,
        "b7_n_bootstrap": 5000,
    }
    for key, expected in integer_contract.items():
        if int(config.get(key, expected)) != expected:
            raise ValueError(f"v2 downstream freezes {key}={expected}")
    float_contract = {
        "b7_dropout": 0.25,
        "b7_transformer_ff_mult": 2.0,
        "b7_encoder_lr": 1e-5,
        "b7_head_lr": 1e-4,
        "b7_min_lr": 1e-6,
        "b7_weight_decay": 1e-4,
        "b7_grad_clip": 1.0,
        "b7_noise_std": 0.02,
        "b7_slice_dropout": 0.08,
        "b7_center_jitter": 2.0,
        "b7_rotation_deg": 5.0,
        "b7_translate_frac": 0.03,
        "b7_scale_jitter": 0.05,
        "b7_gamma_jitter": 0.12,
        "b7_bias_field_strength": 0.08,
        "b7_min_confidence": B7_MIN_CONFIDENCE,
        "b7_positive_target": B7_POSITIVE_TARGET,
        "b7_negative_target": B7_NEGATIVE_TARGET,
        "b7_positive_weight": B7_POSITIVE_WEIGHT,
        "b7_negative_weight": B7_NEGATIVE_WEIGHT,
    }
    for key, expected in float_contract.items():
        _require_close(config, key, expected)
    bool_contract = {
        "competition_mode": True,
        "b7_gradient_checkpointing": True,
        "allow_external_pretrained": True,
        "pretrained": True,
    }
    for key, expected in bool_contract.items():
        if bool(config.get(key, expected)) is not expected:
            raise ValueError(f"v2 downstream freezes {key}={expected}")
    if bool(config.get("b12_use_physical_scale", False)):
        raise ValueError("v2 downstream freezes legacy direct resize")
    if tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])) != (1, 2):
        raise ValueError("v2 downstream freezes train gaps [1,2]")
    if tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1])) != (-1, 0, 1):
        raise ValueError("v2 downstream freezes TTA [-1,0,1]")


def _subset_v2_supervision(train, b6_frame, manifest):
    all_uids, all_targets, all_weights, all_summary = prepare_b7_supervision(train, b6_frame)
    if len(all_uids) != 3120 or int((all_weights > 0).sum()) != 14123:
        raise ValueError("v2 downstream requires exact frozen B6 active surface")
    row_by_uid = {str(uid): i for i, uid in enumerate(all_uids)}
    train_uids = manifest.loc[manifest["split"] == "train", "StudyInstanceUID"].astype(str).tolist()
    holdout_uids = manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str).tolist()
    missing = [uid for uid in train_uids + holdout_uids if uid not in row_by_uid]
    if missing:
        raise ValueError(f"v2 manifest contains {len(missing)} UID(s) outside active B6 surface")
    train_idx = np.asarray([row_by_uid[uid] for uid in train_uids], dtype=int)
    holdout_idx = np.asarray([row_by_uid[uid] for uid in holdout_uids], dtype=int)
    if len(train_idx) != 2497 or len(holdout_idx) != 623:
        raise ValueError("v2 downstream train/holdout counts changed")
    if set(train_uids).intersection(holdout_uids):
        raise ValueError("v2 train/holdout UID overlap")
    targets = all_targets[train_idx]
    weights = all_weights[train_idx]
    summary = {
        "all_active_studies": 3120,
        "all_usable_cells": 14123,
        "training_studies": int(len(train_uids)),
        "holdout_studies_excluded": int(len(holdout_uids)),
        "training_cells": int((weights > 0).sum()),
        "training_positive_cells": int(((weights > 0) & (targets > 0.5)).sum()),
        "training_negative_cells": int(((weights > 0) & (targets < 0.5)).sum()),
        "target_balance_scope": "recomputed from v2 weak-train studies only; holdout labels excluded",
        "frozen_b6_summary": all_summary,
    }
    return all_uids, train_uids, targets, weights, summary


def _load_candidate_encoder(model, ssl_checkpoint: str | Path) -> dict:
    payload = load_b15_ssl_encoder(ssl_checkpoint)
    model.encoder.load_state_dict(payload["encoder"], strict=True)
    return payload


def train_v2_downstream(
    config: dict,
    *,
    mode: str,
    b6_root: str | Path,
    series_policy_path: str | Path,
    weak_holdout_root: str | Path,
    out_root: str | Path,
    ssl_checkpoint: str | Path | None = None,
) -> Path:
    if mode not in {"control", "b15"}:
        raise ValueError("mode must be 'control' or 'b15'")
    if mode == "b15" and ssl_checkpoint is None:
        raise ValueError("B15 downstream requires --ssl-checkpoint")
    if mode == "control" and ssl_checkpoint is not None:
        raise ValueError("B13-v2 control must not receive an SSL checkpoint")
    validate_competition_config(config, purpose="train")
    require_v2_downstream_contract(config)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 16_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    _, manifest = load_frozen_v2_manifest(weak_holdout_root)

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    all_uids, train_uids, targets, weights, supervision = _subset_v2_supervision(
        train, b6_frame, manifest
    )

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    full_summary, full_index = audit_variable_series_surface(series, all_uids)
    if full_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("v2 downstream reconstructed full B13 series SHA mismatch")
    if int(full_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("v2 downstream requires frozen 17,475-series full mapping")
    frozen_summary = series_policy.get("series_summary", {})
    if frozen_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("supplied series policy is not frozen B12/B13 policy")
    variable_index = {uid: full_index[uid] for uid in train_uids}
    if any(not variable_index[uid] for uid in train_uids):
        raise ValueError("v2 weak-train study has zero eligible series")
    expected_series = int(sum(len(variable_index[uid]) for uid in train_uids))

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(train_uids) / batch_size))
    supervision.update(
        {
            "eligible_series_expected_per_full_epoch": expected_series,
            "full_coverage_batches_per_epoch": expected_batches,
            "full_series_signature_sha256": B13_SERIES_SIGNATURE,
            "weak_holdout_surface": WEAK_V2_SURFACE,
            "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        }
    )

    ds = VariableSeriesKneeDataset(
        train_uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 16_100_000),
    )

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    ssl_payload = None
    if mode == "control":
        _load_imagenet_encoder_state(model)
        variant = B13_V2_VARIANT
        experiment = B13_V2_EXPERIMENT
        initialization = B13_INITIALIZATION
        initialization_detail = "direct ImageNet-1K V1"
    else:
        ssl_payload = _load_candidate_encoder(model, ssl_checkpoint)
        variant = B15_VARIANT
        experiment = B15_EXPERIMENT
        initialization = B15_SSL_VARIANT
        initialization_detail = (
            f"{B13_INITIALIZATION} -> {B15_SSL_OBJECTIVE} on competition knee MRI"
        )
    model = model.to(runtime.device)

    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(config.get("b7_encoder_lr", 1e-5))},
            {"params": head_params, "lr": float(config.get("b7_head_lr", 1e-4))},
        ],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b7_epochs", 4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / ("b13_v2_control.pt" if mode == "control" else "b15_model.pt")
    policy = {
        "variant": variant,
        "experiment": experiment,
        "mode": mode,
        "architecture": "B13 hierarchical one-token-per-series",
        "initialization": initialization,
        "initialization_detail": initialization_detail,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "weak_holdout_surface": WEAK_V2_SURFACE,
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "weak_holdout_studies_used_in_gradient": 0,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_for_early_stopping": False,
        "training_studies": len(train_uids),
        "training_series": expected_series,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "series_policy": series_policy,
        "supervision": supervision,
        "target_balance_multiplier": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "model_spec": spec,
        "metadata_repair": metadata_stats,
        "matched_comparison": (
            "control and B15 share identical v2 weak-train UIDs, downstream architecture, "
            "B6 policy, target-balancing derivation, optimizer, augmentation, epochs and TTA; "
            "encoder initialization is the only model difference"
        ),
    }
    if ssl_payload is not None:
        policy["ssl_checkpoint"] = str(Path(ssl_checkpoint).resolve())
        policy["ssl_variant"] = ssl_payload.get("variant")
        policy["ssl_experiment"] = ssl_payload.get("experiment")
        policy["ssl_objective"] = ssl_payload.get("objective")
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(
        json.dumps(supervision, indent=2), encoding="utf-8"
    )

    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print(f"[budget] stopping {experiment} before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = pos_seen = neg_seen = 0
        series_seen = max_series = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print(f"[budget] stopping {experiment} before wall-clock reserve")
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present, series_meta)
                loss = target_balanced_weak_bce(
                    logits, target, weight, target_multiplier_t
                )
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            active = weight > 0
            loss_sum += float(loss.item())
            steps += 1
            study_draws += int(volumes.shape[0])
            active_cells += int(active.sum().item())
            pos_seen += int((active & (target > 0.5)).sum().item())
            neg_seen += int((active & (target < 0.5)).sum().item())
            series_seen += int((present > 0).sum().item())
            max_series = max(max_series, int(volumes.shape[1]))

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError(f"{experiment} completed no training batches")
        scheduler.step()
        full_study = steps == expected_batches and study_draws == len(ds)
        full_series = full_study and series_seen == expected_series
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_full_coverage_batches": expected_batches,
            "study_draws": int(study_draws),
            "expected_full_coverage_studies": len(ds),
            "active_supervision_cells_seen": int(active_cells),
            "expected_active_supervision_cells": int((weights > 0).sum()),
            "positive_cells_seen": int(pos_seen),
            "expected_positive_cells": int(((weights > 0) & (targets > 0.5)).sum()),
            "negative_cells_seen": int(neg_seen),
            "expected_negative_cells": int(((weights > 0) & (targets < 0.5)).sum()),
            "series_instances_seen": int(series_seen),
            "expected_series_instances": expected_series,
            "max_series_in_any_batch": int(max_series),
            "full_coverage": bool(full_study),
            "full_series_coverage": bool(full_series),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        checkpoint_payload = {
            **policy,
            "model_state": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "config": config,
            "completed_epochs": len(history),
            "history": history,
            "budget": budget.to_dict(),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != 4 or not all(
        row["full_coverage"]
        and row["full_series_coverage"]
        and not row["budget_limited"]
        for row in history
    ):
        print(
            f"[warning] {experiment} did not complete four exact v2 weak-train passes; "
            "do not run weak-holdout evaluation"
        )
    return checkpoint_path


def load_v2_downstream_checkpoint(
    checkpoint: str | Path,
    *,
    expected_mode: str | None = None,
    device: torch.device | str = "cpu",
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mode = payload.get("mode")
    if mode not in {"control", "b15"}:
        raise ValueError("not a v2 downstream control/B15 checkpoint")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"expected mode={expected_mode}, got {mode}")
    expected_variant = B13_V2_VARIANT if mode == "control" else B15_VARIANT
    if payload.get("variant") != expected_variant:
        raise ValueError("v2 downstream variant mismatch")
    if payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("v2 downstream checkpoint manifest SHA mismatch")
    if int(payload.get("weak_holdout_studies_used_in_gradient", -1)) != 0:
        raise ValueError("v2 downstream checkpoint does not certify holdout exclusion")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("v2 downstream checkpoint does not certify gold exclusion")
    history = payload.get("history", [])
    if int(payload.get("completed_epochs", -1)) != 4 or len(history) != 4:
        raise ValueError("v2 downstream checkpoint requires four epochs")
    if not all(
        bool(row.get("full_coverage"))
        and bool(row.get("full_series_coverage"))
        and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise ValueError("v2 downstream checkpoint lacks four complete passes")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("v2 downstream checkpoint missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload
