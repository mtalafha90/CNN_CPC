"""B20: B18 recipe with crop-only knee-focused preprocessing.

B20 follows the completed B19 observation that the cosine/vignette spatial mask
can itself become a shortcut. The only spatial intervention in B20-v1 is a
centered 90% field-of-view crop resized back to 224x224. No multiplicative mask,
black border, cosine taper, or crop jitter is used.

Everything else follows B18/B19: completed B16 report-aligned encoder, frozen
ConvNeXt, full B6 supervision surface, five candidate epochs, and one global
58-study expert macro-AUC checkpoint-selection statistic.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
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
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    audit_variable_series_surface,
    build_variable_series_index,
    collate_variable_series,
)
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import (
    B17_ENCODER_LR,
    B17_HEAD_LR,
    encoder_state_sha256,
    freeze_encoder,
)
from .b18_fisher_selection import (
    B18_CANDIDATE_EPOCHS,
    B18_EXPECTED_GOLD_SERIES,
    B18_EXPECTED_GOLD_STUDIES,
    B18_SELECTION_METRIC,
    B18_TIE_BREAK,
    require_b18_contract,
    select_best_epoch,
)
from .budget import RuntimeBudget
from .constants import TARGETS
from .crop_focus import (
    CROP_FOCUS_VERSION,
    DEFAULT_CROP_FOCUS_POLICY,
    apply_crop_focus,
    validate_crop_focus_policy,
)
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import macro_auc_from_arrays
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B20_VARIANT = "b20_crop_only_b18_global_expert_epoch_selection_v1"
B20_EXPERIMENT = "B20_crop_only_joint_focus"


class CropFocusedVariableSeriesKneeDataset(VariableSeriesKneeDataset):
    """Variable-series MRI dataset with B20's deterministic crop-only focus."""

    def __init__(self, *args, crop_focus_policy: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_focus_policy = validate_crop_focus_policy(crop_focus_policy)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["volumes"] = apply_crop_focus(item["volumes"], self.crop_focus_policy)
        return item


def b20_crop_focus_policy(config: dict) -> dict:
    if bool(config.get("b20_crop_focus_enabled", False)) is not True:
        raise ValueError("B20 requires b20_crop_focus_enabled=true")
    policy = {
        "version": str(config.get("b20_crop_focus_version", CROP_FOCUS_VERSION)),
        "crop_fraction": float(
            config.get(
                "b20_crop_focus_crop_fraction",
                DEFAULT_CROP_FOCUS_POLICY["crop_fraction"],
            )
        ),
    }
    policy = validate_crop_focus_policy(policy)
    for key, expected in DEFAULT_CROP_FOCUS_POLICY.items():
        if key == "version":
            if policy[key] != expected:
                raise ValueError(f"B20-v1 freezes {key}={expected!r}")
        elif not np.isclose(float(policy[key]), float(expected), atol=1e-12, rtol=0):
            raise ValueError(f"B20-v1 freezes {key}={expected}; got {policy[key]}")
    return policy


def require_b20_contract(config: dict) -> dict:
    """Require the complete B18 optimization/selection recipe plus crop-only focus."""
    require_b18_contract(config)
    if bool(config.get("b19_joint_focus_enabled", False)):
        raise ValueError("B20 must not enable the B19 cosine joint-focus transform")
    return b20_crop_focus_policy(config)


def _expert_loader(config, root, train, series, runtime, crop_policy):
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B20 requires the complete 58-study / 696-cell expert surface")
    uids = gold["StudyInstanceUID"].tolist()
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B20 expert surface contains a study with zero eligible series")
    if int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError(
            f"B20 expected {B18_EXPECTED_GOLD_SERIES} expert series, got {sum(counts)}"
        )
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B20 expert selection freezes TTA at [-1,0,1]")
    ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
        crop_focus_policy=crop_policy,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("b7_eval_batch_size", 2)),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 27_100_000),
    )
    return {
        "uids": uids,
        "truth": gold[TARGETS].to_numpy(np.float64),
        "loader": loader,
        "series_counts": counts,
        "tta_offsets": list(offsets),
    }


def _candidate_payload(
    *, model, config, spec, history, selection_history, policy, budget,
    encoder_sha_initial, model_epoch,
):
    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("B20 encoder changed despite frozen-encoder contract")
    return {
        **policy,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "model_epoch": int(model_epoch),
        "training_history_through_model_epoch": history,
        "selection_history_through_model_epoch": selection_history,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": encoder_sha_final,
        "budget": budget.to_dict(),
    }


def train_b20(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b20_crop_focus",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_policy = require_b20_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    # Reuse the B18/B19 seed/construction path so crop preprocessing is the
    # intended independent variable rather than random model initialization.
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(
        f"[B20] initialization={B16_REPORT_SSL_VARIANT} | encoder=frozen | "
        f"crop_focus={crop_policy} | no cosine mask | expert labels=selection only"
    )

    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)
    positive_cells = int(((weights > 0) & (targets > 0.5)).sum())
    negative_cells = int(((weights > 0) & (targets < 0.5)).sum())
    if (
        len(uids) != 3120
        or int((weights > 0).sum()) != 14123
        or positive_cells != 6871
        or negative_cells != 7252
    ):
        raise ValueError("B20 must retain the exact B18/B19 B6 supervision surface")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    frozen_summary = series_policy.get("series_summary", {})
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B20 reconstructed series surface does not match frozen B13 SHA-256")
    if frozen_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B20 supplied series policy is not the frozen B12/B13 policy")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("B20 requires exactly 17,475 B6-gradient series")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B20 reconstructed series surface no longer passes viability")

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    batches_per_epoch = int(math.ceil(len(uids) / batch_size))
    if batches_per_epoch != 1560:
        raise ValueError("B20 full B6 surface must produce 1,560 batches/epoch")
    expected_series = 17475

    train_ds = CropFocusedVariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
        crop_focus_policy=crop_policy,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 19_100_000),
    )
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    if not head_params:
        raise RuntimeError("B20 found no trainable non-encoder parameters")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("B20 encoder still has trainable parameters")

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
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    out = Path(out_root)
    candidates_dir = out / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out / "b20_model.pt"

    supervision = dict(supervision)
    supervision.update(
        {
            "training_studies": len(uids),
            "training_cells": int((weights > 0).sum()),
            "training_positive_cells": positive_cells,
            "training_negative_cells": negative_cells,
            "eligible_series_expected_per_full_epoch": expected_series,
            "full_coverage_batches_per_epoch": batches_per_epoch,
            "series_signature_sha256": series_summary["series_signature_sha256"],
        }
    )
    policy = {
        "variant": B20_VARIANT,
        "experiment": B20_EXPERIMENT,
        "status": "B20-v1 crop-only recipe frozen before first run",
        "architecture": "unchanged B18/B19 hierarchical series-token model",
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_training_mode": False,
        "encoder_optimizer_membership": False,
        "trainable_encoder_parameters": 0,
        "external_pretrained": True,
        "full_report_alignment": True,
        "training_studies": 3120,
        "training_series": 17475,
        "training_supervision_cells": 14123,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_for_early_stopping": False,
        "gold_checkpoint_selection": True,
        "gold_selection_studies": B18_EXPECTED_GOLD_STUDIES,
        "gold_selection_cells": B18_EXPECTED_GOLD_STUDIES * len(TARGETS),
        "gold_selection_series": B18_EXPECTED_GOLD_SERIES,
        "selection_metric": B18_SELECTION_METRIC,
        "selection_scope": "one global 12-target macro AUC only",
        "selection_tie_break": B18_TIE_BREAK,
        "candidate_epochs": B18_CANDIDATE_EPOCHS,
        "selected_gold_score_is_validation_evidence": False,
        "additional_label_smoothing": 0.0,
        "robust_loss": "none",
        "crop_focus_enabled": True,
        "crop_focus_policy": crop_policy,
        "b19_cosine_mask_used": False,
        "crop_jitter_used": False,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "series_policy": series_policy,
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "scientific_question": (
            "whether retaining the B19 central 90% knee crop while removing its artificial "
            "cosine/vignette border preserves diagnostic focus without creating a new shortcut"
        ),
        "single_change_vs_b19": (
            "remove the multiplicative cosine/vignette window; retain the same centered 90% "
            "crop and resize, with all optimization, supervision, architecture and selection fixed"
        ),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")

    history: list[dict] = []
    selection_history: list[dict] = []
    cycle_times: list[float] = []

    for epoch in range(B18_CANDIDATE_EPOCHS):
        if cycle_times and not budget.can_start(float(np.median(cycle_times)) * 1.20):
            raise RuntimeError("B20 cannot safely start the next complete train+selection cycle")
        cycle_start = time.monotonic()
        train_start = time.monotonic()
        model.train()
        model.encoder.eval()
        if model.encoder.training:
            raise RuntimeError("B20 encoder unexpectedly entered training mode")

        loss_sum = 0.0
        steps = study_draws = active_cells = pos_seen = neg_seen = series_seen = 0
        max_series = 0
        budget_limited = False
        for batch in train_loader:
            if not budget.can_start(120.0):
                budget_limited = True
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present, series_meta)
                loss = target_balanced_weak_bce(logits, target, weight, target_multiplier_t)
            scaler.scale(loss).backward()
            if any(parameter.grad is not None for parameter in model.encoder.parameters()):
                raise RuntimeError("B20 detected an encoder gradient despite freezing")
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(head_params, clip)
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

        train_seconds = time.monotonic() - train_start
        if steps == 0:
            raise RuntimeError("B20 completed no training batches")
        scheduler.step()
        encoder_sha_epoch = encoder_state_sha256(model.encoder)
        if encoder_sha_epoch != encoder_sha_initial:
            raise RuntimeError(f"B20 encoder state changed during epoch {epoch + 1}")

        full_study = steps == batches_per_epoch and study_draws == len(train_ds)
        full_series = full_study and series_seen == expected_series
        full_cells = (
            active_cells == int((weights > 0).sum())
            and pos_seen == positive_cells
            and neg_seen == negative_cells
        )
        if budget_limited or not full_study or not full_series or not full_cells:
            raise RuntimeError(f"B20 epoch {epoch + 1} did not complete the exact B18 surface")

        select_start = time.monotonic()
        pred_uids, prediction = predict_b12_1(model, expert["loader"], runtime)
        selection_seconds = time.monotonic() - select_start
        if pred_uids != expert["uids"]:
            raise RuntimeError("B20 expert prediction order changed")
        macro_auc, per_target = macro_auc_from_arrays(expert["truth"], prediction)
        if not np.isfinite(macro_auc) or not np.isfinite(per_target).all():
            raise RuntimeError("B20 expert selection requires all 12 target AUCs")

        selection_row = {
            "epoch": epoch + 1,
            "expert_selection_macro_auc": float(macro_auc),
            "selection_metric": B18_SELECTION_METRIC,
            "n_expert_studies": B18_EXPECTED_GOLD_STUDIES,
            "n_expert_targets": len(TARGETS),
            "all_12_target_aucs_defined": True,
            "selection_seconds": float(selection_seconds),
            "per_target_values_intentionally_not_logged": True,
        }
        selection_history.append(selection_row)
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": B17_ENCODER_LR,
            "head_lr": float(optimizer.param_groups[0]["lr"]),
            "training_seconds": float(train_seconds),
            "expert_selection_seconds": float(selection_seconds),
            "expert_selection_macro_auc": float(macro_auc),
            "batches": int(steps),
            "expected_full_coverage_batches": batches_per_epoch,
            "study_draws": int(study_draws),
            "expected_full_coverage_studies": len(train_ds),
            "active_supervision_cells_seen": int(active_cells),
            "expected_active_supervision_cells": int((weights > 0).sum()),
            "positive_cells_seen": int(pos_seen),
            "expected_positive_cells": positive_cells,
            "negative_cells_seen": int(neg_seen),
            "expected_negative_cells": negative_cells,
            "series_instances_seen": int(series_seen),
            "expected_series_instances": expected_series,
            "max_series_in_any_batch": int(max_series),
            "encoder_frozen": True,
            "encoder_training_mode": False,
            "encoder_gradients_detected": False,
            "encoder_sha256": encoder_sha_epoch,
            "full_coverage": True,
            "full_series_coverage": True,
            "budget_limited": False,
            "crop_focus_enabled": True,
            "cosine_mask_used": False,
        }
        history.append(row)
        print(row)
        print(
            {
                "b20_expert_selection_epoch": epoch + 1,
                "global_macro_auc": float(macro_auc),
                "selection_only_not_validation": True,
            }
        )

        candidate_path = candidates_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            _candidate_payload(
                model=model,
                config=config,
                spec=spec,
                history=list(history),
                selection_history=list(selection_history),
                policy=policy,
                budget=budget,
                encoder_sha_initial=encoder_sha_initial,
                model_epoch=epoch + 1,
            ),
            candidate_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out / "selection_history.json").write_text(
            json.dumps(selection_history, indent=2), encoding="utf-8"
        )
        cycle_times.append(time.monotonic() - cycle_start)

    if len(history) != B18_CANDIDATE_EPOCHS:
        raise RuntimeError("B20 requires all five complete candidate epochs")

    best = select_best_epoch(selection_history)
    selected_epoch = int(best["epoch"])
    selected_macro = float(best["expert_selection_macro_auc"])
    selected_candidate = candidates_dir / f"epoch_{selected_epoch}.pt"
    candidate_payload = torch.load(selected_candidate, map_location="cpu", weights_only=False)
    selected_payload = {
        **candidate_payload,
        "status": "B20 selected checkpoint; expert set consumed for global epoch selection",
        "candidate_epochs_completed": B18_CANDIDATE_EPOCHS,
        "selected_epoch": selected_epoch,
        "selected_expert_selection_macro_auc": selected_macro,
        "selection_history": selection_history,
        "training_history_all_candidates": history,
        "selection_decision": (
            "highest global 12-target expert-set macro AUC across epochs 1..5; ties choose earliest"
        ),
        "selected_score_role": "checkpoint selection statistic only; not validation/test evidence",
        "independent_evaluation_required": True,
    }
    torch.save(selected_payload, selected_path)
    summary = {
        "variant": B20_VARIANT,
        "experiment": B20_EXPERIMENT,
        "crop_focus_policy": crop_policy,
        "selection_history": selection_history,
        "selected_epoch": selected_epoch,
        "selected_expert_selection_macro_auc": selected_macro,
        "score_is_validation_evidence": False,
        "selected_checkpoint": str(selected_path),
    }
    (out / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[B20 selection]", json.dumps(summary, indent=2))
    print(selected_path)
    return selected_path


def load_b20_checkpoint(checkpoint: str | Path, *, device: torch.device | str = "cpu"):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B20_VARIANT or payload.get("experiment") != B20_EXPERIMENT:
        raise ValueError("not a selected B20 crop-only checkpoint")
    if payload.get("crop_focus_enabled") is not True:
        raise ValueError("B20 checkpoint does not certify crop-only preprocessing")
    validate_crop_focus_policy(payload.get("crop_focus_policy", {}))
    if bool(payload.get("b19_cosine_mask_used", True)):
        raise ValueError("B20 checkpoint unexpectedly certifies B19 cosine masking")
    if int(payload.get("candidate_epochs_completed", -1)) != B18_CANDIDATE_EPOCHS:
        raise ValueError("B20 selected checkpoint requires all five candidate epochs")
    selection_history = payload.get("selection_history", [])
    best = select_best_epoch(selection_history)
    if int(best["epoch"]) != int(payload.get("selected_epoch", -1)):
        raise ValueError("B20 selected checkpoint violates frozen global selection rule")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("B20 checkpoint encoder fingerprint changed")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B20 checkpoint missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("B20 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b20_crop_focus")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b20(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
