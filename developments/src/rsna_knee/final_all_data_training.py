"""Final production training using every competition study across the learning pipeline.

The final model is intentionally not another development experiment. It consumes
the 58 expert-labelled studies that were previously reserved for repeated gold
development evaluation, so those studies can no longer provide an honest
performance estimate for this model.

Data roles:
- 4,349 non-gold studies: already used by completed B16 full-report MRI/text alignment;
- 3,120 B6-active non-gold studies: B6 positive/negated pathology supervision;
- 1,229 B6-inactive non-gold studies: representation learning only (no invented labels);
- 58 expert studies: true 12-target 0/1 supervision in this final downstream fit.

The completed B16 report-aligned encoder is frozen exactly as in B17. Only the
unchanged hierarchical series/pathology head is trained for five fixed epochs.
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
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b17_training import encoder_state_sha256, freeze_encoder, require_b17_contract
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

FINAL_VARIANT = "final_all_data_frozen_report_encoder_v1"
FINAL_EXPERIMENT = "FINAL_all_data_production"
FINAL_EPOCHS = 5
FINAL_ENCODER_LR = 0.0
FINAL_HEAD_LR = 1e-4
FINAL_GOLD_WEIGHT = 1.0
FINAL_TRAINING_STUDIES = 3178
FINAL_B6_STUDIES = 3120
FINAL_GOLD_STUDIES = 58
FINAL_B6_CELLS = 14123
FINAL_GOLD_CELLS = 696
FINAL_ACTIVE_CELLS = FINAL_B6_CELLS + FINAL_GOLD_CELLS
FINAL_B6_SERIES = 17475
FINAL_GOLD_SERIES = 336
FINAL_TRAINING_SERIES = FINAL_B6_SERIES + FINAL_GOLD_SERIES
FINAL_BATCHES = 1589


def _require_close(config: dict, key: str, expected: float) -> None:
    value = float(config.get(key, expected))
    if not np.isclose(value, expected, atol=1e-12, rtol=0):
        raise ValueError(f"final production freezes {key}={expected}; got {value}")


def require_final_contract(config: dict) -> None:
    """Freeze final-production choices before hidden-test evaluation."""
    if int(config.get("b7_epochs", FINAL_EPOCHS)) != FINAL_EPOCHS:
        raise ValueError(f"final production freezes b7_epochs={FINAL_EPOCHS}")
    if int(config.get("b7_max_batches_per_epoch", FINAL_BATCHES)) != FINAL_BATCHES:
        raise ValueError(f"final production freezes b7_max_batches_per_epoch={FINAL_BATCHES}")
    _require_close(config, "b7_encoder_lr", FINAL_ENCODER_LR)
    _require_close(config, "b7_head_lr", FINAL_HEAD_LR)
    if bool(config.get("final_include_gold", True)) is not True:
        raise ValueError("final production requires final_include_gold=true")
    _require_close(config, "final_gold_weight", FINAL_GOLD_WEIGHT)
    if bool(config.get("final_encoder_frozen", True)) is not True:
        raise ValueError("final production requires final_encoder_frozen=true")
    _require_close(config, "final_additional_label_smoothing", 0.0)
    if str(config.get("final_robust_loss", "none")).lower() != "none":
        raise ValueError("final production freezes final_robust_loss=none")

    # Reuse the exact B17/B13 architecture, augmentation and optimization
    # contract by restoring only the deliberate final-surface difference.
    shadow = dict(config)
    shadow["b7_max_batches_per_epoch"] = 1560
    shadow["b17_encoder_frozen"] = True
    shadow["b17_label_smoothing"] = 0.0
    shadow["b17_robust_loss"] = "none"
    require_b17_contract(shadow)


def prepare_final_supervision(train_df, b6_frame):
    """Combine frozen B6 non-gold supervision with all 58 expert labels.

    Gold labels are hard 0/1 targets with base weight 1.0. B6 targets/weights
    remain exactly 0.85/0.05 and 0.50/1.00. No target-specific gold upweighting
    and no labels for B6-inactive report-only studies are introduced.
    """
    b6_uids, b6_targets, b6_weights, b6_summary = prepare_b7_supervision(train_df, b6_frame)
    if len(b6_uids) != FINAL_B6_STUDIES or int((b6_weights > 0).sum()) != FINAL_B6_CELLS:
        raise ValueError("final production requires the exact frozen 3,120-study B6 surface")

    gold = train_df.loc[gold_mask(train_df), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != FINAL_GOLD_STUDIES:
        raise ValueError(f"final production expected {FINAL_GOLD_STUDIES} expert studies")
    if gold[TARGETS].isna().any().any():
        raise ValueError("final production requires all 12 expert labels for every gold study")
    gold_values = gold[TARGETS].to_numpy(dtype=np.float32)
    if not np.isin(gold_values, [0.0, 1.0]).all():
        raise ValueError("expert labels must be binary 0/1")
    gold_weights = np.full_like(gold_values, FINAL_GOLD_WEIGHT, dtype=np.float32)
    gold_uids = gold["StudyInstanceUID"].tolist()

    if set(b6_uids).intersection(gold_uids):
        raise ValueError("B6 and expert study pools overlap unexpectedly")

    uids = list(b6_uids) + list(gold_uids)
    targets = np.concatenate([b6_targets, gold_values], axis=0)
    weights = np.concatenate([b6_weights, gold_weights], axis=0)
    if len(uids) != FINAL_TRAINING_STUDIES:
        raise ValueError("final production must contain exactly 3,178 supervised studies")
    if int((weights > 0).sum()) != FINAL_ACTIVE_CELLS:
        raise ValueError("final production must contain exactly 14,819 supervised cells")

    summary = {
        "pipeline_competition_studies": int(len(train_df)),
        "representation_non_gold_studies": 4349,
        "b6_active_supervised_studies": int(len(b6_uids)),
        "b6_inactive_representation_only_studies": 1229,
        "expert_supervised_studies": int(len(gold_uids)),
        "downstream_training_studies": int(len(uids)),
        "b6_supervision_cells": int((b6_weights > 0).sum()),
        "expert_supervision_cells": int(gold_values.size),
        "downstream_supervision_cells": int((weights > 0).sum()),
        "expert_positive_cells": int((gold_values > 0.5).sum()),
        "expert_negative_cells": int((gold_values < 0.5).sum()),
        "expert_base_weight": FINAL_GOLD_WEIGHT,
        "b6": b6_summary,
        "all_4407_studies_have_training_role": bool(len(train_df) == 4407),
        "performance_estimate_on_58_gold_forbidden_after_training": True,
    }
    return uids, targets, weights, summary, b6_uids, gold_uids


def _checkpoint_payload(*, model, config, spec, history, policy, budget, encoder_sha_initial):
    final_sha = encoder_state_sha256(model.encoder)
    if final_sha != encoder_sha_initial:
        raise RuntimeError("final-production encoder changed despite frozen contract")
    return {
        **policy,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "config": config,
        "completed_epochs": len(history),
        "history": history,
        "encoder_sha256_initial": encoder_sha_initial,
        "encoder_sha256_final": final_sha,
        "budget": budget.to_dict(),
    }


def train_final_all_data(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/final_all_data",
) -> Path:
    validate_competition_config(config, purpose="train")
    require_final_contract(config)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    # Reuse the B17 downstream initialization path so the only data-side change
    # from B17 is consumption of the 58 expert-labelled studies.
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(f"[FINAL] initialization={B16_REPORT_SSL_VARIANT} | encoder=frozen | gold=consumed")

    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    if len(train) != 4407:
        raise ValueError("final production requires the complete 4,407-study training release")
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, targets, weights, supervision, b6_uids, gold_uids = prepare_final_supervision(train, b6_frame)

    series_policy = _load_series_policy(series_policy_path)
    frozen_summary = series_policy.get("series_summary", {})
    if frozen_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("final production requires the frozen B12/B13 B6 series policy")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    b6_series_summary, _ = audit_variable_series_surface(series, b6_uids)
    if b6_series_summary.get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("final production reconstructed B6 series surface changed")
    if int(b6_series_summary.get("eligible_recognized_plane_series", -1)) != FINAL_B6_SERIES:
        raise ValueError("final production requires exactly 17,475 B6-active series")

    variable_index = build_variable_series_index(series, uids)
    zero = [uid for uid in uids if not variable_index.get(uid)]
    if zero:
        raise ValueError(f"final production found {len(zero)} supervised study/studies with zero eligible series")
    gold_series = int(sum(len(variable_index[uid]) for uid in gold_uids))
    expected_series = int(sum(len(variable_index[uid]) for uid in uids))
    if gold_series != FINAL_GOLD_SERIES or expected_series != FINAL_TRAINING_SERIES:
        raise ValueError(
            f"final production expected {FINAL_GOLD_SERIES} gold / {FINAL_TRAINING_SERIES} total series; "
            f"got {gold_series} / {expected_series}"
        )

    target_multiplier = target_balance_multipliers(weights)
    batch_size = int(config.get("b7_batch_size", 2))
    batches_per_epoch = int(math.ceil(len(uids) / batch_size))
    if batches_per_epoch != FINAL_BATCHES:
        raise ValueError(f"final production must produce {FINAL_BATCHES} batches/epoch")
    supervision.update(
        {
            "eligible_series_expected_per_full_epoch": expected_series,
            "b6_series": FINAL_B6_SERIES,
            "gold_series": gold_series,
            "full_coverage_batches_per_epoch": batches_per_epoch,
            "b6_series_signature_sha256": b6_series_summary["series_signature_sha256"],
        }
    )

    ds = VariableSeriesKneeDataset(
        uids,
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
        **runtime.loader_kwargs(seed=seed + 19_100_000),
    )

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"], strict=True)
    freeze_encoder(model)
    model.gradient_checkpointing = False
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    if not head_params:
        raise RuntimeError("final production found no trainable hierarchy/head parameters")
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("final-production encoder still has trainable parameters")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", FINAL_HEAD_LR))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FINAL_EPOCHS,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "final_model.pt"
    policy = {
        "variant": FINAL_VARIANT,
        "experiment": FINAL_EXPERIMENT,
        "status": "final production; 58-study gold development surface consumed by training",
        "architecture": "B17 frozen report-aligned encoder + unchanged hierarchical series/pathology head",
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "initialization_detail": report_payload.get("initialization_detail"),
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "encoder_optimizer_membership": False,
        "encoder_training_mode": False,
        "runtime_encoder_gradient_checkpointing": False,
        "encoder_sha256_initial": encoder_sha_initial,
        "training_studies": FINAL_TRAINING_STUDIES,
        "training_series": FINAL_TRAINING_SERIES,
        "training_cells": FINAL_ACTIVE_CELLS,
        "b6_training_studies": FINAL_B6_STUDIES,
        "gold_training_studies": FINAL_GOLD_STUDIES,
        "gold_studies_used_in_gradient": FINAL_GOLD_STUDIES,
        "gold_labels_used_in_gradient": True,
        "gold_labels_for_early_stopping": False,
        "gold_checkpoint_selection": False,
        "gold_evaluation_after_training_permitted": False,
        "gold_base_weight": FINAL_GOLD_WEIGHT,
        "additional_label_smoothing": 0.0,
        "robust_loss": "none",
        "fixed_epochs": FINAL_EPOCHS,
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
        "all_training_studies_accounted_for": {
            "total": 4407,
            "B16_report_representation_non_gold": 4349,
            "B6_active_downstream": 3120,
            "B6_inactive_representation_only": 1229,
            "expert_gold_downstream": 58,
        },
        "selection_policy": (
            "B17 selected the training recipe before gold consumption. This final model adds all 58 "
            "expert labels once and must go directly to independent hidden competition evaluation."
        ),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")

    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    expected_positive = int(((weights > 0) & (targets > 0.5)).sum())
    expected_negative = int(((weights > 0) & (targets < 0.5)).sum())

    for epoch in range(FINAL_EPOCHS):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping final production before next epoch")
            break
        start = time.monotonic()
        model.train()
        model.encoder.eval()
        if model.encoder.training:
            raise RuntimeError("final-production encoder unexpectedly entered training mode")

        loss_sum = 0.0
        steps = study_draws = active_cells = pos_seen = neg_seen = series_seen = 0
        max_series = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping final-production batches before wall-clock reserve")
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
            if any(p.grad is not None for p in model.encoder.parameters()):
                raise RuntimeError("final production detected an encoder gradient")
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

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("final production completed no training batches")
        scheduler.step()
        encoder_sha_epoch = encoder_state_sha256(model.encoder)
        if encoder_sha_epoch != encoder_sha_initial:
            raise RuntimeError(f"final-production encoder changed during epoch {epoch + 1}")
        full_study = steps == FINAL_BATCHES and study_draws == FINAL_TRAINING_STUDIES
        full_series = full_study and series_seen == FINAL_TRAINING_SERIES
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": 0.0,
            "head_lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_full_coverage_batches": FINAL_BATCHES,
            "study_draws": int(study_draws),
            "expected_full_coverage_studies": FINAL_TRAINING_STUDIES,
            "active_supervision_cells_seen": int(active_cells),
            "expected_active_supervision_cells": FINAL_ACTIVE_CELLS,
            "positive_cells_seen": int(pos_seen),
            "expected_positive_cells": expected_positive,
            "negative_cells_seen": int(neg_seen),
            "expected_negative_cells": expected_negative,
            "series_instances_seen": int(series_seen),
            "expected_series_instances": FINAL_TRAINING_SERIES,
            "max_series_in_any_batch": int(max_series),
            "encoder_frozen": True,
            "encoder_training_mode": False,
            "encoder_gradients_detected": False,
            "encoder_sha256": encoder_sha_epoch,
            "full_coverage": bool(full_study),
            "full_series_coverage": bool(full_series),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        torch.save(
            _checkpoint_payload(
                model=model,
                config=config,
                spec=spec,
                history=history,
                policy=policy,
                budget=budget,
                encoder_sha_initial=encoder_sha_initial,
            ),
            checkpoint_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != FINAL_EPOCHS or not all(
        row["full_coverage"]
        and row["full_series_coverage"]
        and row["encoder_frozen"]
        and not row["encoder_training_mode"]
        and not row["encoder_gradients_detected"]
        and row["encoder_sha256"] == encoder_sha_initial
        and not row["budget_limited"]
        for row in history
    ):
        print("[warning] final production did not complete five exact full passes; do not submit")
    return checkpoint_path


def load_final_checkpoint(checkpoint: str | Path, *, device: torch.device | str = "cpu"):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != FINAL_VARIANT or payload.get("experiment") != FINAL_EXPERIMENT:
        raise ValueError("not a final all-data production checkpoint")
    if payload.get("initialization") != B16_REPORT_SSL_VARIANT:
        raise ValueError("final checkpoint initialization mismatch")
    if payload.get("input_normalization") != B13_INPUT_NORMALIZATION:
        raise ValueError("final checkpoint normalization mismatch")
    if payload.get("encoder_frozen") is not True:
        raise ValueError("final checkpoint does not certify frozen encoder")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != FINAL_GOLD_STUDIES:
        raise ValueError("final checkpoint must consume all 58 expert studies")
    if payload.get("gold_labels_used_in_gradient") is not True:
        raise ValueError("final checkpoint does not certify expert-label training")
    if bool(payload.get("gold_labels_for_early_stopping", True)):
        raise ValueError("final checkpoint must not use gold early stopping")
    if bool(payload.get("gold_checkpoint_selection", True)):
        raise ValueError("final checkpoint must not use gold checkpoint selection")
    if bool(payload.get("gold_evaluation_after_training_permitted", True)):
        raise ValueError("final checkpoint must forbid post-training gold evaluation")
    if int(payload.get("training_studies", -1)) != FINAL_TRAINING_STUDIES:
        raise ValueError("final checkpoint training-study count mismatch")
    if int(payload.get("training_series", -1)) != FINAL_TRAINING_SERIES:
        raise ValueError("final checkpoint training-series count mismatch")
    if int(payload.get("training_cells", -1)) != FINAL_ACTIVE_CELLS:
        raise ValueError("final checkpoint supervision-cell count mismatch")
    history = payload.get("history", [])
    if int(payload.get("completed_epochs", -1)) != FINAL_EPOCHS or len(history) != FINAL_EPOCHS:
        raise ValueError("final production requires five completed epochs")
    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("final checkpoint encoder fingerprint changed")
    if not all(
        bool(row.get("full_coverage"))
        and bool(row.get("full_series_coverage"))
        and bool(row.get("encoder_frozen"))
        and not bool(row.get("encoder_training_mode"))
        and not bool(row.get("encoder_gradients_detected"))
        and str(row.get("encoder_sha256")) == initial_sha
        and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise ValueError("final checkpoint lacks five exact frozen-encoder passes")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("final checkpoint missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("final reconstructed encoder fingerprint mismatch")
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-final")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/final_all_data")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_final_all_data(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
