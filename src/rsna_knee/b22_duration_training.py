"""Five-epoch full-B6 pre-resize-crop duration audit after B21 failed acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .b7_weak_supervision import _read_config, seed_everything
from .b13_training import B13_INPUT_NORMALIZATION
from .b16_report_ssl import (
    B16_REPORT_SSL_EXPERIMENT,
    B16_REPORT_SSL_OBJECTIVE,
    B16_REPORT_SSL_VARIANT,
    load_b16_report_encoder,
)
from .b21_full_setup import prepare_b21_full_surface
from .b22_duration_contract import require_b22_duration_contract
from .b22_duration_epoch import run_b22_epoch
from .b22_duration_model import build_b22_training_state
from .b22_duration_protocol import (
    B22_CROP_FRACTION,
    B22_EPOCHS,
    B22_EXPERIMENT,
    B22_SCHEDULER_HORIZON,
    B22_TRAIN_CELLS,
    B22_TRAIN_SERIES,
    B22_TRAIN_STUDIES,
    B22_VARIANT,
    require_failed_b21_acceptance,
)
from .budget import RuntimeBudget
from .policy import validate_competition_config
from .runtime import resolve_runtime


def train_b22_duration(
    config: dict,
    *,
    b6_root: str | Path,
    series_policy_path: str | Path,
    report_ssl_checkpoint: str | Path,
    b21_acceptance: str | Path,
    out_root: str | Path = "runs/b22_duration_audit",
) -> Path:
    validate_competition_config(config, purpose="train")
    crop_fraction = require_b22_duration_contract(config)
    failed_b21 = require_failed_b21_acceptance(b21_acceptance)
    report_payload = load_b16_report_encoder(report_ssl_checkpoint)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 19_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    print(
        "[B22 duration] full B6 | pre-resize crop 0.90 | historical B16 frozen | "
        "epochs 1-5 saved | no gold during training"
    )
    surface = prepare_b21_full_surface(
        config,
        b6_root=b6_root,
        series_policy_path=series_policy_path,
        runtime=runtime,
    )
    model, spec, head_params, optimizer, scheduler, scaler, encoder_sha = (
        build_b22_training_state(config, report_payload, runtime)
    )
    target_multiplier_t = torch.from_numpy(surface["target_multiplier"]).to(runtime.device)
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    clip = float(config.get("b7_grad_clip", 1.0))

    out = Path(out_root)
    candidates = out / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    policy = {
        "variant": B22_VARIANT,
        "experiment": B22_EXPERIMENT,
        "role": "post-hoc duration audit; not a promotion experiment",
        "working_model_remains": "B20_crop_only_joint_focus",
        "architecture": "historical B20 hierarchical one-token-per-series model",
        "initialization": B16_REPORT_SSL_VARIANT,
        "initialization_experiment": B16_REPORT_SSL_EXPERIMENT,
        "initialization_objective": B16_REPORT_SSL_OBJECTIVE,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "encoder_frozen": True,
        "training_studies": B22_TRAIN_STUDIES,
        "training_series": B22_TRAIN_SERIES,
        "training_supervision_cells": B22_TRAIN_CELLS,
        "training_epochs": B22_EPOCHS,
        "scheduler_horizon_epochs": B22_SCHEDULER_HORIZON,
        "crop_fraction": crop_fraction,
        "crop_stage": "native_array_pre_resize",
        "normalization_support": "cropped native field before percentile normalization",
        "output_resolution": 224,
        "gold_evaluation_during_training": False,
        "checkpoint_selection_during_training": False,
        "b21_acceptance_path": str(Path(b21_acceptance).resolve()),
        "b21_failed_macro_auc": float(failed_b21["b21_candidate"]["macro_auc"]),
        "b21_failed_promotion": True,
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": surface["b6_audit"].get("b6_version"),
        "b6_policy": surface["b6_policy"],
        "series_policy": surface["series_policy"],
        "supervision": surface["supervision"],
        "metadata_repair": surface["metadata_stats"],
        "report_ssl_checkpoint": str(Path(report_ssl_checkpoint).resolve()),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    history = []
    for epoch in range(1, B22_EPOCHS + 1):
        row = run_b22_epoch(
            model=model,
            loader=surface["loader"],
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            head_params=head_params,
            target_multiplier_t=target_multiplier_t,
            runtime=runtime,
            budget=budget,
            encoder_sha_initial=encoder_sha,
            clip=clip,
            epoch_number=epoch,
        )
        history.append(row)
        payload = {
            **policy,
            "model_state": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "model_spec": spec,
            "config": config,
            "completed_epochs": epoch,
            "model_epoch": epoch,
            "history": list(history),
            "encoder_sha256_initial": encoder_sha,
            "encoder_sha256_final": row["encoder_sha256"],
            "budget": budget.to_dict(),
        }
        path = candidates / f"epoch_{epoch}.pt"
        torch.save(payload, path)
        print(row)
        print({"checkpoint": str(path), "epoch": epoch})

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b22-duration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--b21-acceptance", required=True)
    parser.add_argument("--out-root", default="runs/b22_duration_audit")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b22_duration(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        b21_acceptance=args.b21_acceptance,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
