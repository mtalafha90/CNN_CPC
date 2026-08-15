"""B24 trainer: the B20 recipe with the supervision source as the only variable.

Both arms run through this module. The mode selects which labeller's targets and
weights are loaded; nothing else in the path depends on it, which is what makes
the comparison a single-variable one.

The encoder is the weak-v2-safe B16-v2 checkpoint rather than the historical
B16. That is not a recipe change -- it is the B21 lesson: historical B16 trained
on all 4,349 report studies, which includes the weak-v2 holdout, so a model
built on it may not be scored there. B24 is scored on both weak surfaces, so it
needs the safe encoder to make either legal.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    _read_config,
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b12_1_hierarchical import b12_1_model_spec, build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION, B13_SERIES_SIGNATURE
from .b16_v2_report_ssl import (
    B16_V2_REPORT_VARIANT,
    load_b16_v2_report_encoder,
)
from .b17_training import encoder_state_sha256, freeze_encoder
from .b21_dataset import make_matched_crop_dataset
from .b24_protocol import (
    B24_CROP_FRACTION,
    B24_EXPERIMENT,
    B24_FIXED_EPOCHS,
    B24_HEAD_LR,
    B24_SCHEDULER_HORIZON,
    MODES,
    mode_identity,
    require_b24_contract,
    require_passed_labeller_gate,
)
from .b24_supervision import arm_supervision, build_matched_surface, format_surface
from .budget import RuntimeBudget
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime


def train_b24(
    config: dict,
    *,
    mode: str,
    b6_root: str | Path,
    b23_root: str | Path,
    b23_holdout_root: str | Path,
    weak_holdout_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    out_root: str | Path,
) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    validate_competition_config(config, purpose="train")
    require_b24_contract(config)
    variant, supervision_name = mode_identity(mode)

    # Refuse to train on labels that never passed their own audit.
    gate = require_passed_labeller_gate(Path(b23_holdout_root) / "weak_holdout.json")

    report_payload = load_b16_v2_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 24_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    surface = build_matched_surface(
        config,
        b6_root=b6_root,
        b23_root=b23_root,
        weak_holdout_root=weak_holdout_root,
        b23_holdout_root=b23_holdout_root,
    )
    print(format_surface(surface))
    study_uids, targets, weights = arm_supervision(surface, mode)
    print(
        f"\n[{B24_EXPERIMENT}] mode={mode} | supervision={supervision_name} | "
        f"encoder=frozen {B16_V2_REPORT_VARIANT} | fixed E{B24_FIXED_EPOCHS} | "
        f"no checkpoint selection"
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series_policy = _load_series_policy(series_policy_path)
    if series_policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B24 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, _metadata_stats = backfill_series_metadata(series, root, split="train")
    _summary, full_index = audit_variable_series_surface(series, study_uids)
    variable_index = {uid: full_index[uid] for uid in study_uids}
    if any(not variable_index[uid] for uid in study_uids):
        raise ValueError("a B24 training study has zero eligible series")
    expected_series = int(sum(len(variable_index[uid]) for uid in study_uids))

    batch_size = int(config.get("b7_batch_size", 2))
    expected_batches = int(math.ceil(len(study_uids) / batch_size))
    target_multiplier = target_balance_multipliers(weights)

    dataset_config = make_b7_dataset_config(config, root, train=True)
    train_ds = make_matched_crop_dataset(
        "control",  # B20's post-resize crop geometry, frozen for B24
        study_uids,
        variable_index,
        dataset_config,
        crop_fraction=B24_CROP_FRACTION,
        targets=targets,
        weights=weights,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed),
    )

    spec = b12_1_model_spec(config, normalize_input=True)
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.encoder.load_state_dict(report_payload["encoder"])
    freeze_encoder(model)
    encoder_sha_initial = encoder_state_sha256(model.encoder)
    model = model.to(runtime.device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("B24 requires a frozen encoder")
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": float(config.get("b7_head_lr", B24_HEAD_LR))}],
        weight_decay=float(config.get("b7_weight_decay", 1e-4)),
    )
    # Keep B20's five-epoch cosine horizon so the first two epochs follow the
    # identical learning-rate trajectory; only the stopping point is earlier.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=B24_SCHEDULER_HORIZON,
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10)),
    )

    multiplier_t = torch.as_tensor(target_multiplier, dtype=torch.float32, device=runtime.device)
    history = []
    for epoch in range(1, B24_FIXED_EPOCHS + 1):
        model.train()
        epoch_started = time.monotonic()
        loss_sum, steps, seen_series, seen_cells = 0.0, 0, 0, 0
        for batch in loader:
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head_params, float(config.get("b7_grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            steps += 1
            seen_series += int(present.sum().item())
            seen_cells += int((weight > 0).sum().item())
        scheduler.step()

        epoch_seconds = time.monotonic() - epoch_started
        full_coverage = steps == expected_batches and seen_series == expected_series
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(steps, 1),
                "batches": steps,
                "expected_batches": expected_batches,
                "series_instances_seen": seen_series,
                "expected_series_instances": expected_series,
                "supervision_cells_seen": seen_cells,
                "head_lr": float(optimizer.param_groups[0]["lr"]),
                # Persisted so a later run can be budgeted from a measured
                # rate rather than an estimate.
                "epoch_seconds": round(float(epoch_seconds), 1),
                "seconds_per_study": round(float(epoch_seconds / max(len(study_uids), 1)), 4),
                "full_coverage": bool(full_coverage),
                "budget_limited": bool(budget.remaining_work_seconds <= 0.0),
            }
        )
        print(
            f"[B24:{mode}] epoch {epoch} loss {loss_sum / max(steps, 1):.10f} "
            f"coverage={full_coverage} | {epoch_seconds / 60:.1f} min "
            f"({epoch_seconds / max(len(study_uids), 1):.3f} s/study)"
        )
        if not full_coverage:
            raise RuntimeError(f"B24 epoch {epoch} did not achieve exact full coverage")

    if encoder_state_sha256(model.encoder) != encoder_sha_initial:
        raise RuntimeError("B24 encoder changed despite the frozen-encoder contract")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"b24_{mode}_model.pt"
    torch.save(
        {
            "experiment": B24_EXPERIMENT,
            "variant": variant,
            "mode": mode,
            "supervision": supervision_name,
            "model_spec": spec,
            "model_state": model.state_dict(),
            "encoder_sha256_initial": encoder_sha_initial,
            "initialization": B16_V2_REPORT_VARIANT,
            "encoder_frozen": True,
            "completed_epochs": B24_FIXED_EPOCHS,
            "fixed_endpoint": True,
            "expert_selection": False,
            "scheduler_horizon": B24_SCHEDULER_HORIZON,
            "crop_fraction": B24_CROP_FRACTION,
            "study_uids": list(study_uids),
            "surface_diagnostics": surface["diagnostics"],
            "labeller_gate": gate,
            "history": history,
        },
        checkpoint,
    )
    (out / "history.json").write_text(
        json.dumps({"mode": mode, "history": history, "surface": surface["diagnostics"]}, indent=2),
        encoding="utf-8",
    )
    print(f"[B24:{mode}] wrote {checkpoint}")
    return checkpoint


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--b23-root", required=True)
    parser.add_argument("--b23-holdout-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default=None)


def _run(mode: str) -> None:
    parser = argparse.ArgumentParser(description=f"B24 {mode} arm")
    _add_args(parser)
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    train_b24(
        config,
        mode=mode,
        b6_root=args.b6_root,
        b23_root=args.b23_root,
        b23_holdout_root=args.b23_holdout_root,
        weak_holdout_root=args.weak_holdout_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root or f"runs/b24_supervision/{mode}",
    )


def main_control() -> None:
    _run("b6_control")


def main() -> None:
    _run("b23_candidate")


if __name__ == "__main__":  # pragma: no cover
    main()
