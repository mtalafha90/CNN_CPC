"""B18: B17 frozen training with global expert-guided epoch selection.

B18 intentionally keeps the complete B17 optimization/data/model recipe and
changes only checkpoint selection. Five exact B6-only training epochs are run.
After each epoch, the repeatedly reused 58-study expert set is evaluated with
fixed [-1,0,1] TTA and ONE scalar is exposed for selection: global 12-target
macro ROC AUC. The epoch with the highest global value is retained; exact ties
resolve to the earliest epoch.

The expert labels never enter gradients. Because they select the checkpoint,
the selected expert-set score is a selection statistic, not validation evidence.
No per-target epoch selection, generic label smoothing, robust loss, architecture
change, resolution change, weak-v2 gate, or B16/B17 target mixing is allowed.
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
    B17_EPOCHS,
    B17_HEAD_LR,
    encoder_state_sha256,
    freeze_encoder,
    require_b17_contract,
)
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import macro_auc_from_arrays
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

B18_VARIANT = "b18_fisher_global_expert_epoch_selection_v1"
B18_EXPERIMENT = "B18_fisher_expert_guided_epoch_selection"
B18_CANDIDATE_EPOCHS = 5
B18_SELECTION_METRIC = "macro_auc_12_target_global"
B18_TIE_BREAK = "earliest_epoch"
B18_EXPECTED_GOLD_STUDIES = 58
B18_EXPECTED_GOLD_SERIES = 336


def require_b18_contract(config: dict) -> None:
    """Require the unchanged B17 recipe plus B18's one selection intervention."""
    require_b17_contract(config)
    if int(config.get("b18_candidate_epochs", B18_CANDIDATE_EPOCHS)) != B18_CANDIDATE_EPOCHS:
        raise ValueError(f"B18 freezes b18_candidate_epochs={B18_CANDIDATE_EPOCHS}")
    if int(config.get("b7_epochs", B18_CANDIDATE_EPOCHS)) != B18_CANDIDATE_EPOCHS:
        raise ValueError("B18 requires exactly five B6-only candidate epochs")
    if bool(config.get("b18_expert_selection", True)) is not True:
        raise ValueError("B18 requires expert-guided checkpoint selection")
    if str(config.get("b18_selection_metric", B18_SELECTION_METRIC)) != B18_SELECTION_METRIC:
        raise ValueError(f"B18 freezes selection metric to {B18_SELECTION_METRIC}")
    if str(config.get("b18_selection_tie_break", B18_TIE_BREAK)) != B18_TIE_BREAK:
        raise ValueError(f"B18 freezes selection tie-break to {B18_TIE_BREAK}")
    if bool(config.get("b18_save_candidate_checkpoints", True)) is not True:
        raise ValueError("B18 requires auditable per-epoch candidate checkpoints")
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B18 expert selection freezes TTA at [-1,0,1]")
    if int(config.get("b7_eval_batch_size", 2)) != 2:
        raise ValueError("B18 expert selection freezes evaluation batch size at 2")


def select_best_epoch(selection_history: list[dict]) -> dict:
    """Return the highest global macro-AUC entry; numerical ties prefer earlier epochs."""
    if len(selection_history) != B18_CANDIDATE_EPOCHS:
        raise ValueError(f"B18 selection requires exactly {B18_CANDIDATE_EPOCHS} epoch entries")
    rows = []
    seen_epochs = set()
    for row in selection_history:
        epoch = int(row.get("epoch", -1))
        score = float(row.get("expert_selection_macro_auc", np.nan))
        if epoch < 1 or epoch > B18_CANDIDATE_EPOCHS or epoch in seen_epochs:
            raise ValueError("B18 selection history has invalid/duplicate epoch entries")
        if not np.isfinite(score):
            raise ValueError("B18 selection history contains a non-finite global macro AUC")
        seen_epochs.add(epoch)
        rows.append((epoch, score, row))
    if seen_epochs != set(range(1, B18_CANDIDATE_EPOCHS + 1)):
        raise ValueError("B18 selection history must contain epochs 1..5 exactly once")
    best_score = max(score for _, score, _ in rows)
    tied = [item for item in rows if np.isclose(item[1], best_score, atol=1e-12, rtol=0)]
    return min(tied, key=lambda item: item[0])[2]


def _make_candidate_payload(
    *,
    model,
    config,
    spec,
    history,
    selection_history,
    policy,
    budget,
    encoder_sha_initial: str,
    model_epoch: int,
) -> dict:
    encoder_sha_final = encoder_state_sha256(model.encoder)
    if encoder_sha_final != encoder_sha_initial:
        raise RuntimeError("B18 encoder changed despite the frozen-encoder contract")
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


def _prepare_expert_selection_loader(config: dict, root: Path, train, series, runtime):
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B18 requires the complete 58-study / 696-cell expert selection surface")
    uids = gold["StudyInstanceUID"].tolist()
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("B18 expert selection contains a study with zero eligible series")
    if int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError(
            f"B18 expected {B18_EXPECTED_GOLD_SERIES} expert-series instances, got {sum(counts)}"
        )
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    ds = VariableSeriesKneeDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        targets=gold[TARGETS].to_numpy(np.float32),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("b7_eval_batch_size", 2)),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 22_100_000),
    )
    return {
        "uids": uids,
        "truth": gold[TARGETS].to_numpy(np.float64),
        "loader": loader,
        "series_counts": counts,
        "tta_offsets": list(offsets),
    }


def train_b18(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b18_fisher_selection",
) -> Path:
    validate_competition_config(config, purpose="train")
    require_b18_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    # Match B17 construction/shuffle seeds exactly so B18 is, as far as the
    # runtime permits, the same five-epoch trajectory with a different selection rule.
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(
        f"[B18] initialization={B16_REPORT_SSL_VARIANT} | encoder=frozen | "
        "expert labels=checkpoint selection only"
    )

    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))

    # Exact B17 B6-only gradient surface.
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
        raise ValueError("B18 must retain the exact B17 3,120-study / 14,123-cell B6 surface")

    series_policy = _load_series_policy(series_policy_path)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    frozen_summary = series_policy.get("series_summary", {})
    if series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B18 reconstructed B6 series surface does not match frozen B13 SHA-256")
    if frozen_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B18 supplied series policy is not the frozen B12/B13 policy")
    if int(series_summary.get("eligible_recognized_plane_series", -1)) != 17475:
        raise ValueError("B18 requires exactly 17,475 B6-gradient series")
    if series_summary.get("viability_passed") is not True:
        raise ValueError("B18 reconstructed series surface no longer passes viability")

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    batches_per_epoch = int(math.ceil(len(uids) / batch_size))
    if batches_per_epoch != 1560:
        raise ValueError("B18 full B6 surface must produce 1,560 batches/epoch")
    expected_series = 17475
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

    train_ds = VariableSeriesKneeDataset(
        uids,
        variable_index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 19_100_000),
    )
    expert = _prepare_expert_selection_loader(config, root, train, series, runtime)

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    if not head_params:
        raise RuntimeError("B18 found no trainable non-encoder parameters")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("B18 encoder still has trainable parameters")
    trainable_head_parameters = sum(parameter.numel() for parameter in head_params)

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
    selected_path = out / "b18_model.pt"

    policy = {
        "variant": B18_VARIANT,
        "experiment": B18_EXPERIMENT,
        "status": "B18 recipe frozen before expert-guided epoch selection run",
        "architecture": "B13/B16/B17 hierarchical learned one-token-per-series aggregation",
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "initialization_detail": report_payload.get("initialization_detail"),
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_training_mode": False,
        "encoder_optimizer_membership": False,
        "runtime_encoder_gradient_checkpointing": False,
        "trainable_encoder_parameters": 0,
        "trainable_head_parameters": int(trainable_head_parameters),
        "external_pretrained": True,
        "full_report_alignment": True,
        "training_studies": 3120,
        "training_series": 17475,
        "training_supervision_cells": 14123,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_for_early_stopping": True,
        "gold_checkpoint_selection": True,
        "gold_selection_studies": B18_EXPECTED_GOLD_STUDIES,
        "gold_selection_cells": B18_EXPECTED_GOLD_STUDIES * len(TARGETS),
        "gold_selection_series": B18_EXPECTED_GOLD_SERIES,
        "gold_selection_tta_offsets": expert["tta_offsets"],
        "selection_metric": B18_SELECTION_METRIC,
        "selection_scope": "one global 12-target macro AUC only",
        "selection_tie_break": B18_TIE_BREAK,
        "candidate_epochs": B18_CANDIDATE_EPOCHS,
        "training_stopped_early": False,
        "additional_label_smoothing": 0.0,
        "b6_targets_already_soft": {"positive": 0.85, "negative": 0.05},
        "robust_loss": "none",
        "weak_v2_used_for_selection": False,
        "target_specific_epoch_selection": False,
        "target_specific_model_mixing": False,
        "selected_gold_score_is_validation_evidence": False,
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
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
        "scientific_question": (
            "whether expert-guided selection among five short frozen-encoder B6-only epochs identifies "
            "a better-transfer checkpoint than always taking epoch five"
        ),
        "single_change_vs_b17": (
            "evaluate the repeatedly reused 58-study expert set after each of the same five fixed B6-only "
            "epochs and retain the epoch with the highest global 12-target macro AUC; no expert gradients"
        ),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")

    history: list[dict] = []
    selection_history: list[dict] = []
    cycle_times: list[float] = []
    budget_exhausted = False

    for epoch in range(B18_CANDIDATE_EPOCHS):
        # Require room for a full training+selection cycle based on prior cycles.
        if cycle_times and not budget.can_start(float(np.median(cycle_times)) * 1.20):
            raise RuntimeError("B18 cannot safely start the next complete train+expert-selection cycle")
        cycle_start = time.monotonic()
        train_start = time.monotonic()
        model.train()
        model.encoder.eval()
        if model.encoder.training:
            raise RuntimeError("B18 encoder unexpectedly entered training mode")

        loss_sum = 0.0
        steps = study_draws = active_cells = pos_seen = neg_seen = series_seen = 0
        max_series = 0
        for batch in train_loader:
            # Keep enough reserve to finish the current batch; the global 8.5 h
            # budget is generous relative to the predeclared five-cycle run.
            if not budget.can_start(120.0):
                budget_exhausted = True
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
                raise RuntimeError("B18 detected an encoder gradient despite freezing")
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
        if budget_exhausted:
            raise RuntimeError("B18 budget ended inside a training epoch; no expert-selected model is valid")
        if steps == 0:
            raise RuntimeError("B18 completed no training batches")

        scheduler.step()
        encoder_sha_epoch = encoder_state_sha256(model.encoder)
        if encoder_sha_epoch != encoder_sha_initial:
            raise RuntimeError(f"B18 encoder state changed during epoch {epoch + 1}")
        full_study = steps == batches_per_epoch and study_draws == len(train_ds)
        full_series = full_study and series_seen == expected_series
        if not full_study or not full_series:
            raise RuntimeError(f"B18 epoch {epoch + 1} did not complete the exact B17 training surface")

        # Expert-set checkpoint selection only. predict_b12_1 is @torch.no_grad().
        select_start = time.monotonic()
        pred_uids, prediction = predict_b12_1(model, expert["loader"], runtime)
        selection_seconds = time.monotonic() - select_start
        if pred_uids != expert["uids"]:
            raise RuntimeError("B18 expert-selection prediction order changed")
        macro_auc, per_target = macro_auc_from_arrays(expert["truth"], prediction)
        if not np.isfinite(macro_auc) or not np.isfinite(per_target).all() or len(per_target) != len(TARGETS):
            raise RuntimeError("B18 expert selection requires all 12 target AUCs to be defined")

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
        }
        history.append(row)
        print(row)
        print(
            {
                "b18_expert_selection_epoch": epoch + 1,
                "global_macro_auc": float(macro_auc),
                "selection_only_not_validation": True,
            }
        )

        candidate_path = candidates_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            _make_candidate_payload(
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

    if len(history) != B18_CANDIDATE_EPOCHS or len(selection_history) != B18_CANDIDATE_EPOCHS:
        raise RuntimeError("B18 requires all five complete training/selection cycles")
    if not all(
        row["full_coverage"]
        and row["full_series_coverage"]
        and row["encoder_frozen"]
        and not row["encoder_training_mode"]
        and not row["encoder_gradients_detected"]
        and row["encoder_sha256"] == encoder_sha_initial
        and not row["budget_limited"]
        for row in history
    ):
        raise RuntimeError("B18 training history violates the frozen exact-coverage contract")

    best = select_best_epoch(selection_history)
    selected_epoch = int(best["epoch"])
    selected_macro = float(best["expert_selection_macro_auc"])
    selected_candidate = candidates_dir / f"epoch_{selected_epoch}.pt"
    candidate_payload = torch.load(selected_candidate, map_location="cpu", weights_only=False)
    selected_payload = {
        **candidate_payload,
        "status": "B18 selected checkpoint; expert set consumed for global epoch selection",
        "candidate_epochs_completed": B18_CANDIDATE_EPOCHS,
        "selected_epoch": selected_epoch,
        "selected_expert_selection_macro_auc": selected_macro,
        "selection_history": selection_history,
        "training_history_all_candidates": history,
        "selection_decision": (
            "highest global 12-target expert-set macro AUC across epochs 1..5; numerical ties choose "
            "the earliest epoch"
        ),
        "selected_score_role": "checkpoint selection statistic only; not validation/test evidence",
        "independent_evaluation_required": True,
        "next_evaluation_surface": "Kaggle hidden test / another genuinely independent set",
    }
    torch.save(selected_payload, selected_path)

    selection_summary = {
        "variant": B18_VARIANT,
        "experiment": B18_EXPERIMENT,
        "candidate_epochs": B18_CANDIDATE_EPOCHS,
        "selection_metric": B18_SELECTION_METRIC,
        "tie_break": B18_TIE_BREAK,
        "selection_history": selection_history,
        "selected_epoch": selected_epoch,
        "selected_expert_selection_macro_auc": selected_macro,
        "gold_labels_used_in_gradient": False,
        "gold_checkpoint_selection": True,
        "score_is_validation_evidence": False,
        "selected_checkpoint": str(selected_path),
    }
    (out / "selection.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")
    print("[B18 selection]", json.dumps(selection_summary, indent=2))
    print(selected_path)
    return selected_path


def load_b18_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B18_VARIANT or payload.get("experiment") != B18_EXPERIMENT:
        raise ValueError("not a selected B18 expert-guided checkpoint")
    if payload.get("initialization") != B16_REPORT_SSL_VARIANT:
        raise ValueError("B18 checkpoint initialization mismatch")
    if payload.get("input_normalization") != B13_INPUT_NORMALIZATION:
        raise ValueError("B18 checkpoint normalization mismatch")
    if payload.get("encoder_frozen") is not True:
        raise ValueError("B18 checkpoint does not certify a frozen encoder")
    if int(payload.get("trainable_encoder_parameters", -1)) != 0:
        raise ValueError("B18 checkpoint contains trainable encoder parameters")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B18 checkpoint does not certify zero expert-gradient use")
    if bool(payload.get("gold_checkpoint_selection", False)) is not True:
        raise ValueError("B18 requires expert checkpoint selection")
    if bool(payload.get("gold_labels_for_early_stopping", False)) is not True:
        raise ValueError("B18 checkpoint must acknowledge expert-guided selection")
    if int(payload.get("training_studies", -1)) != 3120:
        raise ValueError("B18 checkpoint must use the exact 3,120-study B6 gradient surface")
    if int(payload.get("training_series", -1)) != 17475:
        raise ValueError("B18 checkpoint must use the exact 17,475-series B6 gradient surface")
    if float(payload.get("additional_label_smoothing", -1.0)) != 0.0:
        raise ValueError("B18-v1 adds no generic label smoothing")
    if payload.get("robust_loss") != "none":
        raise ValueError("B18-v1 uses no robust-loss variant")
    if int(payload.get("candidate_epochs_completed", -1)) != B18_CANDIDATE_EPOCHS:
        raise ValueError("B18 selected checkpoint requires all five candidate epochs")
    selected_epoch = int(payload.get("selected_epoch", -1))
    if selected_epoch not in range(1, B18_CANDIDATE_EPOCHS + 1):
        raise ValueError("B18 selected epoch is invalid")
    selection_history = payload.get("selection_history", [])
    best = select_best_epoch(selection_history)
    if int(best["epoch"]) != selected_epoch:
        raise ValueError("B18 selected checkpoint does not match frozen global selection rule")
    if not np.isclose(
        float(best["expert_selection_macro_auc"]),
        float(payload.get("selected_expert_selection_macro_auc", np.nan)),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("B18 selected macro AUC metadata mismatch")
    history = payload.get("training_history_all_candidates", [])
    if len(history) != B18_CANDIDATE_EPOCHS or not all(
        bool(row.get("full_coverage"))
        and bool(row.get("full_series_coverage"))
        and bool(row.get("encoder_frozen"))
        and not bool(row.get("encoder_training_mode"))
        and not bool(row.get("encoder_gradients_detected"))
        and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise ValueError("B18 checkpoint lacks five complete candidate training epochs")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("B18 checkpoint encoder fingerprint changed")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B18 checkpoint missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("B18 reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b18")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b18_fisher_selection")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b18(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
