"""B15 MRI-domain SSL on the frozen weak-holdout-v2 development split.

This stage adapts the ImageNet ConvNeXt-Tiny encoder to competition knee MRI
without using report/B6/gold labels. The SSL image pool is deliberately stricter
than the downstream weak-training pool: every one of the 58 gold studies and
every frozen v2 weak-holdout study is excluded from SSL optimization.

Positive pairs are multi-instance 2.5D examples from the same knee study,
including different real MRI acquisitions and spatial positions. Different
studies in the mini-batch act as negatives. This is MICLe-style same-study
contrastive adaptation, not an exact reproduction of a particular published
implementation.
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

from .b7_weak_supervision import _read_config, make_b7_dataset_config, seed_everything
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b13_training import B13_INITIALIZATION, B13_INPUT_NORMALIZATION
from .budget import RuntimeBudget
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .model import ConvNeXtSliceEncoder
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import _contrastive_same_study, ssl_position_indices

B15_SSL_VARIANT = "b15_imagenet_knee_mri_same_study_contrastive_v1"
B15_SSL_EXPERIMENT = "B15_mri_domain_ssl"
B15_SSL_OBJECTIVE = "same_study_multi_instance_contrastive_2p5d_v1"
WEAK_V2_SURFACE = "weak_b6_holdout_v2"
WEAK_V2_MANIFEST_SHA256 = "1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1"


def load_frozen_v2_manifest(weak_holdout_root: str | Path):
    import pandas as pd

    root = Path(weak_holdout_root)
    json_path = root / "weak_holdout.json"
    csv_path = root / "weak_holdout_manifest.csv"
    if not json_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(
            f"B15 requires frozen v2 artifacts: {json_path} and {csv_path}"
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("surface") != WEAK_V2_SURFACE:
        raise ValueError("B15 requires weak_b6_holdout_v2")
    if payload.get("manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("B15 weak-holdout manifest SHA mismatch")
    if payload.get("status") != "FROZEN before B15/control training":
        raise ValueError("B15 requires the frozen pre-training v2 status")
    if payload.get("uses_gold_labels") is not False:
        raise ValueError("B15 v2 manifest does not certify zero gold-label use")
    if payload.get("uses_model_predictions") is not False:
        raise ValueError("B15 v2 manifest does not certify zero model-prediction use")
    if int(payload.get("train_studies", -1)) != 2497:
        raise ValueError("B15 v2 manifest must contain 2,497 weak-train studies")
    if int(payload.get("holdout_studies", -1)) != 623:
        raise ValueError("B15 v2 manifest must contain 623 holdout studies")
    if int(payload.get("report_group_overlap", -1)) != 0:
        raise ValueError("B15 v2 manifest contains report-group leakage")

    manifest = pd.read_csv(csv_path)
    required = {"StudyInstanceUID", "report_group", "split"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"v2 manifest missing columns: {missing}")
    manifest = manifest.copy()
    manifest["StudyInstanceUID"] = manifest["StudyInstanceUID"].astype(str)
    if manifest["StudyInstanceUID"].duplicated().any():
        raise ValueError("v2 manifest contains duplicate study UIDs")
    if set(manifest["split"]) != {"train", "holdout"}:
        raise ValueError("v2 manifest must contain train and holdout rows")
    if int((manifest["split"] == "train").sum()) != 2497:
        raise ValueError("v2 CSV train count mismatch")
    if int((manifest["split"] == "holdout").sum()) != 623:
        raise ValueError("v2 CSV holdout count mismatch")
    train_groups = set(manifest.loc[manifest["split"] == "train", "report_group"].astype(str))
    holdout_groups = set(manifest.loc[manifest["split"] == "holdout", "report_group"].astype(str))
    if train_groups.intersection(holdout_groups):
        raise ValueError("v2 CSV has report-group overlap")
    return payload, manifest


def b15_ssl_study_pool(train_df, manifest) -> tuple[list[str], dict]:
    """Return non-gold, non-v2-holdout studies for label-free B15 SSL."""
    train = train_df.copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    gold_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))
    if len(gold_uids) != 58:
        raise ValueError(f"B15 expected 58 gold studies, found {len(gold_uids)}")
    holdout_uids = set(
        manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str)
    )
    if len(holdout_uids) != 623:
        raise ValueError("B15 expected 623 v2 holdout studies")
    non_gold = train.loc[~gold_mask(train), "StudyInstanceUID"].astype(str).tolist()
    pool = [uid for uid in non_gold if uid not in holdout_uids]
    expected = len(non_gold) - len(holdout_uids)
    if len(non_gold) != 4349 or expected != 3726 or len(pool) != 3726:
        raise ValueError(
            f"B15 SSL pool contract changed: non_gold={len(non_gold)}, "
            f"holdout={len(holdout_uids)}, pool={len(pool)}"
        )
    if gold_uids.intersection(pool) or holdout_uids.intersection(pool):
        raise ValueError("B15 SSL pool leaks gold or v2 holdout studies")
    return pool, {
        "competition_non_gold_studies": 4349,
        "excluded_gold_studies": 58,
        "excluded_v2_holdout_studies": 623,
        "candidate_ssl_studies": 3726,
    }


class B15MRIRepresentationLearner(nn.Module):
    def __init__(self, projection_dim: int = 256) -> None:
        super().__init__()
        self.encoder = ConvNeXtSliceEncoder(
            3, pretrained_weights=True, normalize_input=True
        )
        d = self.encoder.out_dim
        self.projector = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, int(projection_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return nn.functional.normalize(self.projector(feat), dim=-1)


def _variable_ssl_examples(
    volumes: torch.Tensor,
    present: torch.Tensor,
    positions_per_series: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten active [B,K,S,C,H,W] examples while preserving study identity."""
    if volumes.ndim != 6:
        raise ValueError(f"expected [B,K,S,C,H,W], got {tuple(volumes.shape)}")
    b, k, s, c, h, w = volumes.shape
    if present.shape != (b, k):
        raise ValueError("present mask does not match variable-series batch")
    positions = ssl_position_indices(s, int(positions_per_series)).to(volumes.device)
    p = int(positions.numel())
    selected = volumes.index_select(2, positions)
    active = present.to(dtype=torch.bool).unsqueeze(-1).expand(b, k, p).reshape(-1)
    x = selected.reshape(b * k * p, c, h, w)[active]
    study_ids = (
        torch.arange(b, device=volumes.device)
        .view(b, 1, 1)
        .expand(b, k, p)
        .reshape(-1)[active]
    )
    return x, study_ids


def _require_ssl_contract(config: dict) -> None:
    expected_int = {
        "seed": 2026,
        "requested_gpus": 1,
        "b15_ssl_n_slices": 5,
        "b15_ssl_positions_per_series": 2,
        "b15_ssl_batch_size": 2,
        "b15_ssl_epochs": 4,
        "b15_ssl_projection_dim": 256,
    }
    for key, expected in expected_int.items():
        value = int(config.get(key, expected))
        if value != expected:
            raise ValueError(f"B15 SSL freezes {key}={expected}; got {value}")
    expected_float = {
        "b15_ssl_encoder_lr": 5e-5,
        "b15_ssl_projector_lr": 5e-4,
        "b15_ssl_min_lr": 1e-6,
        "b15_ssl_weight_decay": 1e-4,
        "b15_ssl_temperature": 0.15,
        "b15_ssl_grad_clip": 1.0,
    }
    for key, expected in expected_float.items():
        value = float(config.get(key, expected))
        if not np.isclose(value, expected, atol=1e-12, rtol=0):
            raise ValueError(f"B15 SSL freezes {key}={expected}; got {value}")
    if tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])) != (1, 2):
        raise ValueError("B15 SSL freezes train gap choices [1,2]")
    if int(config.get("b7_center_jitter", 2)) != 2:
        raise ValueError("B15 SSL freezes center jitter +/-2")


def pretrain_b15_ssl(
    config: dict,
    *,
    weak_holdout_root: str | Path,
    out_root: str | Path = "runs/b15_mri_ssl",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_ssl_contract(config)
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 15_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    weak_payload, manifest = load_frozen_v2_manifest(weak_holdout_root)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    ssl_uids, pool_stats = b15_ssl_study_pool(train, manifest)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, ssl_uids)
    zero = [uid for uid in ssl_uids if not variable_index.get(uid)]
    if zero:
        raise ValueError(
            f"B15 SSL found {len(zero)} non-gold/non-holdout studies with zero eligible real series"
        )
    expected_series = int(sum(len(variable_index[uid]) for uid in ssl_uids))

    ds_config = make_b7_dataset_config(config, root, train=True)
    ds_config.n_slices = int(config.get("b15_ssl_n_slices", 5))
    ds_config.slice_dropout = 0.0
    ds = VariableSeriesKneeDataset(
        ssl_uids,
        variable_index,
        ds_config,
        train=True,
    )
    batch_size = int(config.get("b15_ssl_batch_size", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 15_100_000),
    )
    expected_batches = int(math.ceil(len(ds) / batch_size))

    model = B15MRIRepresentationLearner(
        projection_dim=int(config.get("b15_ssl_projection_dim", 256))
    ).to(runtime.device)
    encoder_params = list(model.encoder.parameters())
    projector_params = list(model.projector.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(config.get("b15_ssl_encoder_lr", 5e-5))},
            {"params": projector_params, "lr": float(config.get("b15_ssl_projector_lr", 5e-4))},
        ],
        weight_decay=float(config.get("b15_ssl_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b15_ssl_epochs", 4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b15_ssl_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    temperature = float(config.get("b15_ssl_temperature", 0.15))
    clip = float(config.get("b15_ssl_grad_clip", 1.0))
    positions_per_series = int(config.get("b15_ssl_positions_per_series", 2))
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "b15_ssl_encoder.pt"
    policy = {
        "variant": B15_SSL_VARIANT,
        "experiment": B15_SSL_EXPERIMENT,
        "objective": B15_SSL_OBJECTIVE,
        "initialization": B13_INITIALIZATION,
        "input_normalization": B13_INPUT_NORMALIZATION,
        "weak_holdout_surface": WEAK_V2_SURFACE,
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "ssl_data_policy": (
            "all 4,349 non-gold competition studies minus all 623 frozen v2 holdout studies; "
            "no report/B6/gold labels used"
        ),
        "uses_gold_labels": False,
        "uses_b6_labels": False,
        "uses_report_labels": False,
        "uses_v2_holdout_images": False,
        "pool": pool_stats,
        "eligible_real_series": expected_series,
        "metadata_repair": metadata_stats,
        "weak_holdout_metadata": weak_payload,
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B15 SSL before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = series_seen = active_examples = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B15 SSL batches before wall-clock reserve")
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            x, study_ids = _variable_ssl_examples(
                volumes, present, positions_per_series=positions_per_series
            )
            if len(torch.unique(study_ids)) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                z = model(x)
                loss = _contrastive_same_study(z, study_ids, temperature=temperature)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            steps += 1
            study_draws += int(volumes.shape[0])
            series_seen += int((present > 0).sum().item())
            active_examples += int(x.shape[0])

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("B15 SSL completed no training batches")
        scheduler.step()
        full_coverage = (
            steps == expected_batches
            and study_draws == len(ds)
            and series_seen == expected_series
        )
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "projector_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_batches": int(expected_batches),
            "study_draws": int(study_draws),
            "expected_studies": int(len(ds)),
            "series_instances_seen": int(series_seen),
            "expected_series_instances": int(expected_series),
            "active_2p5d_examples": int(active_examples),
            "full_coverage": bool(full_coverage),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        torch.save(
            {
                "variant": B15_SSL_VARIANT,
                "experiment": B15_SSL_EXPERIMENT,
                "objective": B15_SSL_OBJECTIVE,
                "encoder": model.encoder.state_dict(),
                "config": config,
                "completed_epochs": len(history),
                "history": history,
                "initialization": B13_INITIALIZATION,
                "input_normalization": B13_INPUT_NORMALIZATION,
                "weak_holdout_surface": WEAK_V2_SURFACE,
                "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
                "ssl_studies": len(ds),
                "ssl_series": expected_series,
                "gold_studies_used": 0,
                "v2_holdout_studies_used": 0,
                "b6_labels_used": False,
                "report_labels_used": False,
                "budget": budget.to_dict(),
            },
            checkpoint_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != epochs or not all(
        row["full_coverage"] and not row["budget_limited"] for row in history
    ):
        print(
            "[warning] B15 SSL did not complete the frozen four full passes; "
            "do not start B15 downstream training"
        )
    return checkpoint_path


def load_b15_ssl_encoder(checkpoint: str | Path) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B15_SSL_VARIANT:
        raise ValueError("not a frozen B15 MRI-SSL checkpoint")
    if payload.get("objective") != B15_SSL_OBJECTIVE:
        raise ValueError("B15 SSL objective mismatch")
    if payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("B15 SSL checkpoint used a different v2 manifest")
    history = payload.get("history", [])
    if int(payload.get("completed_epochs", -1)) != 4 or len(history) != 4:
        raise ValueError("B15 SSL requires four completed epochs")
    if not all(
        bool(row.get("full_coverage")) and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise ValueError("B15 SSL checkpoint is not four complete unbudgeted passes")
    if int(payload.get("gold_studies_used", -1)) != 0:
        raise ValueError("B15 SSL checkpoint does not certify gold exclusion")
    if int(payload.get("v2_holdout_studies_used", -1)) != 0:
        raise ValueError("B15 SSL checkpoint does not certify v2 image exclusion")
    if payload.get("b6_labels_used") is not False or payload.get("report_labels_used") is not False:
        raise ValueError("B15 SSL checkpoint used forbidden labels")
    if not isinstance(payload.get("encoder"), dict):
        raise ValueError("B15 SSL checkpoint missing encoder state")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b15-ssl")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--out-root", default="runs/b15_mri_ssl")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = pretrain_b15_ssl(
        config,
        weak_holdout_root=args.weak_holdout_root,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
