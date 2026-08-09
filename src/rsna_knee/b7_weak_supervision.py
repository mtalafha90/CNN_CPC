"""B7: B5-initialized MRI training on frozen B6 weak labels.

B7-v1 is a fixed development experiment:
- MRI encoder initialized from the competition-only B5 checkpoint;
- 12-target cross-sequence/pathology-query model;
- supervision from frozen B6 v1.2.1 report labels only;
- no gold labels in the training loss or early stopping;
- asymmetric soft targets/weights motivated by the completed B6 gold audit;
- target-balanced weak BCE so common report findings do not dominate training.

The B6 gold audit informed the *global* B7-v1 supervision policy, so later gold
performance is a development estimate rather than pristine independent validation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .budget import RuntimeBudget
from .constants import DUAL_STREAMS, TARGETS
from .data import (
    backfill_series_metadata,
    build_series_index,
    gold_mask,
    load_series_csv,
    load_train_csv,
)
from .dataset import DatasetConfig, KneeStudyDataset
from .model import KneeMILNet
from .policy import validate_competition_config
from .report_ssl import B5_VARIANT
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import SSL_SOURCE

B7_VARIANT = "b7_b5_init_b6_asymmetric_weak_v1"
B7_REQUIRED_B6_VERSION = "1.2.1"

# Frozen B7-v1 supervision contract. These values must not be tuned after looking
# at B7 gold performance; a changed policy must be named as a new experiment.
B7_MIN_CONFIDENCE = 0.75
B7_POSITIVE_TARGET = 0.85
B7_NEGATIVE_TARGET = 0.05
B7_POSITIVE_WEIGHT = 0.50
B7_NEGATIVE_WEIGHT = 1.00


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _require_frozen_policy(config: dict) -> None:
    expected = {
        "b7_min_confidence": B7_MIN_CONFIDENCE,
        "b7_positive_target": B7_POSITIVE_TARGET,
        "b7_negative_target": B7_NEGATIVE_TARGET,
        "b7_positive_weight": B7_POSITIVE_WEIGHT,
        "b7_negative_weight": B7_NEGATIVE_WEIGHT,
    }
    for key, value in expected.items():
        configured = float(config.get(key, value))
        if not np.isclose(configured, value, atol=1e-12, rtol=0):
            raise ValueError(
                f"B7-v1 policy is frozen: {key} must remain {value}, got {configured}. "
                "Use a new experiment name for a changed supervision policy."
            )


def load_frozen_b6_export(b6_root: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    root = Path(b6_root)
    targets_path = root / "training_targets.csv"
    policy_path = root / "policy.json"
    audit_path = root / "audit.json"
    for path in (targets_path, policy_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"B7 requires B6 artifact: {path}")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    version = str(policy.get("version", audit.get("b6_version", "")))
    if version != B7_REQUIRED_B6_VERSION or str(audit.get("b6_version")) != B7_REQUIRED_B6_VERSION:
        raise ValueError(
            f"B7-v1 requires frozen B6 v{B7_REQUIRED_B6_VERSION}; "
            f"policy={version!r}, audit={audit.get('b6_version')!r}"
        )
    if int(audit.get("gold_rows_in_training_targets", -1)) != 0:
        raise ValueError("B6 audit does not certify zero gold rows in training_targets.csv")
    audited_threshold = float(audit.get("min_confidence_for_usable_cell", np.nan))
    if not np.isclose(audited_threshold, B7_MIN_CONFIDENCE, atol=1e-12, rtol=0):
        raise ValueError(
            f"B7-v1 requires the audited B6 threshold {B7_MIN_CONFIDENCE}, "
            f"got {audited_threshold}"
        )

    frame = pd.read_csv(targets_path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError("B6 training_targets.csv is missing StudyInstanceUID")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B6 training_targets.csv contains duplicate StudyInstanceUID values")
    return frame, policy, audit


def prepare_b7_supervision(
    train_df: pd.DataFrame,
    b6_frame: pd.DataFrame,
    *,
    min_confidence: float = B7_MIN_CONFIDENCE,
    positive_target: float = B7_POSITIVE_TARGET,
    negative_target: float = B7_NEGATIVE_TARGET,
    positive_weight: float = B7_POSITIVE_WEIGHT,
    negative_weight: float = B7_NEGATIVE_WEIGHT,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Join B6 report-only rows to official training metadata and map to B7 labels."""
    if not np.isclose(min_confidence, B7_MIN_CONFIDENCE):
        raise ValueError("B7-v1 min_confidence is frozen")
    if not np.isclose(positive_target, B7_POSITIVE_TARGET):
        raise ValueError("B7-v1 positive_target is frozen")
    if not np.isclose(negative_target, B7_NEGATIVE_TARGET):
        raise ValueError("B7-v1 negative_target is frozen")
    if not np.isclose(positive_weight, B7_POSITIVE_WEIGHT):
        raise ValueError("B7-v1 positive_weight is frozen")
    if not np.isclose(negative_weight, B7_NEGATIVE_WEIGHT):
        raise ValueError("B7-v1 negative_weight is frozen")

    train = train_df.copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    b6 = b6_frame.copy()
    b6["StudyInstanceUID"] = b6["StudyInstanceUID"].astype(str)

    gold = gold_mask(train)
    gold_uids = set(train.loc[gold, "StudyInstanceUID"])
    overlap = gold_uids.intersection(set(b6["StudyInstanceUID"]))
    if overlap:
        raise ValueError(f"B7 supervision contains {len(overlap)} gold study UID(s)")

    non_gold = train.loc[~gold, ["StudyInstanceUID"]].copy()
    expected = set(non_gold["StudyInstanceUID"])
    actual = set(b6["StudyInstanceUID"])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if missing or extra:
        raise ValueError(
            f"B6/non-gold UID mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    ordered = non_gold.merge(b6, on="StudyInstanceUID", how="left", validate="one_to_one")
    n = len(ordered)
    y = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    w = np.zeros((n, len(TARGETS)), dtype=np.float32)
    per_target: dict[str, dict] = {}

    for j, target in enumerate(TARGETS):
        state_col = f"{target}__state"
        conf_col = f"{target}__confidence"
        missing_cols = [name for name in (state_col, conf_col) if name not in ordered.columns]
        if missing_cols:
            raise ValueError(f"B6 export missing columns for {target}: {missing_cols}")

        state = ordered[state_col].fillna("").astype(str).to_numpy()
        conf = pd.to_numeric(ordered[conf_col], errors="coerce").fillna(0.0).to_numpy(float)
        positive = (state == "positive") & (conf >= min_confidence)
        negative = (state == "negated") & (conf >= min_confidence)

        y[positive, j] = float(positive_target)
        y[negative, j] = float(negative_target)
        w[positive, j] = float(positive_weight)
        w[negative, j] = float(negative_weight)

        per_target[target] = {
            "positive_cells": int(positive.sum()),
            "negative_cells": int(negative.sum()),
            "usable_cells": int((positive | negative).sum()),
            "base_weight_sum": float(w[:, j].sum()),
        }

    active = w.sum(axis=1) > 0
    summary = {
        "report_only_rows": int(n),
        "active_studies": int(active.sum()),
        "inactive_studies_zero_usable_cells": int((~active).sum()),
        "usable_cells": int((w > 0).sum()),
        "positive_cells": int(((w > 0) & (y > 0.5)).sum()),
        "negative_cells": int(((w > 0) & (y < 0.5)).sum()),
        "targets": per_target,
    }
    return (
        ordered.loc[active, "StudyInstanceUID"].astype(str).tolist(),
        y[active],
        w[active],
        summary,
    )


def target_balance_multipliers(weights: np.ndarray) -> np.ndarray:
    """Scale targets so each contributes equal expected total supervision mass."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(TARGETS):
        raise ValueError("weights must have shape [N,12]")
    mass = weights.sum(axis=0)
    valid = mass > 0
    if not valid.all():
        missing = [TARGETS[j] for j in np.flatnonzero(~valid)]
        raise ValueError(f"B7 has no usable supervision for target(s): {missing}")
    mean_mass = float(mass.mean())
    return (mean_mass / mass).astype(np.float32)


def target_balanced_weak_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    target_multiplier: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != weight.shape:
        raise ValueError("logits/target/weight shapes must match")
    if logits.ndim != 2 or logits.shape[1] != len(TARGETS):
        raise ValueError("B7 logits must have shape [B,12]")
    multiplier = torch.as_tensor(
        target_multiplier, dtype=logits.dtype, device=logits.device
    )
    if multiplier.shape != (len(TARGETS),):
        raise ValueError("target_multiplier must have shape [12]")
    effective = weight * multiplier[None, :]
    denominator = effective.sum()
    if float(denominator.detach().item()) <= 0:
        return logits.sum() * 0.0
    cell = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    return (cell * effective).sum() / denominator.clamp_min(1e-8)


def load_b5_encoder_payload(checkpoint: str | Path) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"B7 B5 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("B5 checkpoint must be a dictionary payload")
    if payload.get("source") != SSL_SOURCE:
        raise ValueError(
            f"B7 requires B5 source={SSL_SOURCE!r}; got {payload.get('source')!r}"
        )
    if payload.get("variant") != B5_VARIANT:
        raise ValueError(
            f"B7 requires B5 variant={B5_VARIANT!r}; got {payload.get('variant')!r}"
        )
    if int(payload.get("gold_studies_used", -1)) != 0:
        raise ValueError("B7 requires a B5 encoder trained with zero gold studies")
    if bool(payload.get("external_image_pretraining", True)):
        raise ValueError("B7-v1 requires B5 with no external image pretraining")
    state = payload.get("encoder")
    if not isinstance(state, dict):
        raise ValueError("B5 checkpoint has no encoder state_dict")
    return payload


def make_b7_dataset_config(
    config: dict,
    root: str | Path,
    *,
    train: bool,
    tta_offsets: tuple[int, ...] = (),
) -> DatasetConfig:
    return DatasetConfig(
        data_root=str(root),
        split="train",
        n_slices=int(config.get("b7_n_slices", 16)),
        image_size=int(config.get("b7_image_size", 224)),
        noise_std=float(config.get("b7_noise_std", 0.02)) if train else 0.0,
        slice_dropout=float(config.get("b7_slice_dropout", 0.08)) if train else 0.0,
        triplet_gap=int(config.get("b7_triplet_gap", 1)),
        strict_dicom=bool(config.get("strict_dicom", False)) if train else bool(
            config.get("strict_dicom_inference", True)
        ),
        tta_center_offsets=tuple(int(x) for x in tta_offsets),
        train_gap_choices=tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])),
        center_jitter=int(config.get("b7_center_jitter", 2)) if train else 0,
        rotation_deg=float(config.get("b7_rotation_deg", 5.0)) if train else 0.0,
        translate_frac=float(config.get("b7_translate_frac", 0.03)) if train else 0.0,
        scale_jitter=float(config.get("b7_scale_jitter", 0.05)) if train else 0.0,
        gamma_jitter=float(config.get("b7_gamma_jitter", 0.12)) if train else 0.0,
        bias_field_strength=float(config.get("b7_bias_field_strength", 0.08)) if train else 0.0,
        series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
    )


def b7_model_spec(config: dict, *, normalize_input: bool) -> dict:
    return {
        "architecture": "cross_sequence_pathology_queries_v1",
        "n_streams": len(DUAL_STREAMS),
        "n_slices": int(config.get("b7_n_slices", 16)),
        "in_channels": 3,
        "image_size": int(config.get("b7_image_size", 224)),
        "triplet_gap": int(config.get("b7_triplet_gap", 1)),
        "dropout": float(config.get("b7_dropout", 0.25)),
        "normalize_input": bool(normalize_input),
        "encoder_batch_size": int(config.get("b7_encoder_batch_size", 24)),
        "gradient_checkpointing": bool(config.get("b7_gradient_checkpointing", True)),
        "transformer_layers": int(config.get("b7_transformer_layers", 2)),
        "transformer_heads": int(config.get("b7_transformer_heads", 8)),
        "transformer_ff_mult": float(config.get("b7_transformer_ff_mult", 2.0)),
        "pathology_layers": int(config.get("b7_pathology_layers", 1)),
    }


def build_b7_model(spec: dict, *, encoder_state: dict | None = None) -> KneeMILNet:
    model = KneeMILNet(
        int(spec["n_streams"]),
        int(spec["n_slices"]),
        in_channels=int(spec.get("in_channels", 3)),
        pretrained_weights=False,
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        transformer_layers=int(spec["transformer_layers"]),
        transformer_heads=int(spec["transformer_heads"]),
        transformer_ff_mult=float(spec["transformer_ff_mult"]),
        pathology_layers=int(spec["pathology_layers"]),
    )
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model


def _checkpoint_payload(
    *,
    model: KneeMILNet,
    config: dict,
    spec: dict,
    history: list[dict],
    b5_checkpoint: Path,
    b5_payload: dict,
    b6_root: Path,
    b6_audit: dict,
    supervision: dict,
    target_multiplier: np.ndarray,
    metadata_stats: dict,
    budget: RuntimeBudget,
) -> dict:
    return {
        "variant": B7_VARIANT,
        "source": SSL_SOURCE,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "b6_gold_audit_informed_global_policy": True,
        "b5_checkpoint": str(b5_checkpoint.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b5_completed_epochs": b5_payload.get("completed_epochs"),
        "b6_root": str(b6_root.resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "supervision_policy": {
            "min_confidence": B7_MIN_CONFIDENCE,
            "positive_target": B7_POSITIVE_TARGET,
            "negative_target": B7_NEGATIVE_TARGET,
            "positive_weight": B7_POSITIVE_WEIGHT,
            "negative_weight": B7_NEGATIVE_WEIGHT,
            "uncertain_weight": 0.0,
            "unmentioned_weight": 0.0,
            "target_balancing": "inverse total B7 base supervision mass per target",
        },
        "target_balance_multiplier": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b7(
    config: dict,
    *,
    b5_checkpoint: str | Path,
    b6_root: str | Path,
    out_root: str | Path = "runs/b7_weak_supervision",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_frozen_policy(config)
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 7_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, b6_policy, b6_audit = load_frozen_b6_export(b6_root)
    uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, uids, mode="dual")

    has_mri = np.asarray(
        [
            any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS)
            for uid in uids
        ],
        dtype=bool,
    )
    if not has_mri.any():
        raise ValueError("B7 found no active weakly labelled studies with MRI series")
    supervision["active_studies_before_mri_filter"] = int(len(uids))
    supervision["studies_without_any_selected_mri_series"] = int((~has_mri).sum())
    uids = [uid for uid, keep in zip(uids, has_mri) if keep]
    targets = targets[has_mri]
    weights = weights[has_mri]
    supervision["training_studies"] = int(len(uids))
    supervision["training_usable_cells"] = int((weights > 0).sum())

    target_multiplier = target_balance_multipliers(weights)
    for j, target in enumerate(TARGETS):
        supervision["targets"][target]["training_base_weight_sum"] = float(weights[:, j].sum())
        supervision["targets"][target]["target_balance_multiplier"] = float(target_multiplier[j])
        supervision["targets"][target]["balanced_weight_sum"] = float(
            (weights[:, j] * target_multiplier[j]).sum()
        )

    ds = KneeStudyDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
    )
    batch_size = int(config.get("b7_batch_size", 2))
    if batch_size < 1:
        raise ValueError("b7_batch_size must be >=1")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **runtime.loader_kwargs(seed=seed + 7_100_000),
    )

    b5_path = Path(b5_checkpoint)
    b5_payload = load_b5_encoder_payload(b5_path)
    b5_config = b5_payload.get("config", {})
    normalize_input = bool(b5_config.get("normalize_input", False))
    spec = b7_model_spec(config, normalize_input=normalize_input)
    model = build_b7_model(spec, encoder_state=b5_payload["encoder"]).to(runtime.device)

    encoder_lr = float(config.get("b7_encoder_lr", 1e-5))
    head_lr = float(config.get("b7_head_lr", 1e-4))
    weight_decay = float(config.get("b7_weight_decay", 1e-4))
    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    epochs = int(config.get("b7_epochs", 4))
    max_batches = int(config.get("b7_max_batches_per_epoch", 500))
    if epochs < 1 or max_batches < 1:
        raise ValueError("b7_epochs and b7_max_batches_per_epoch must be >=1")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b7_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b7_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "b7_model.pt"
    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False

    (outdir / "supervision_plan.json").write_text(
        json.dumps(supervision, indent=2), encoding="utf-8"
    )
    policy_payload = {
        "experiment": "B7",
        "variant": B7_VARIANT,
        "status": "training recipe frozen before B7 gold evaluation",
        "b5_initialization": str(b5_path.resolve()),
        "b5_variant": b5_payload.get("variant"),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "b6_policy": b6_policy,
        "gold_labels_in_training_loss": False,
        "gold_labels_for_early_stopping": False,
        "b6_gold_audit_informed_global_policy": True,
        "fixed_epochs": epochs,
        "max_batches_per_epoch": max_batches,
        "supervision_policy": {
            "min_confidence": B7_MIN_CONFIDENCE,
            "positive_target": B7_POSITIVE_TARGET,
            "negative_target": B7_NEGATIVE_TARGET,
            "positive_weight": B7_POSITIVE_WEIGHT,
            "negative_weight": B7_NEGATIVE_WEIGHT,
            "uncertain_weight": 0.0,
            "unmentioned_weight": 0.0,
        },
        "target_balancing": {
            target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)
        },
        "model_spec": spec,
    }
    (outdir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2), encoding="utf-8"
    )

    for epoch in range(epochs):
        if epoch_times:
            estimate = float(np.median(epoch_times))
            if not budget.can_start(estimate * 1.20):
                print("[budget] stopping B7 before next epoch")
                break

        epoch_start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = 0
        study_draws = 0
        active_cells = 0
        positive_cells = 0
        negative_cells = 0

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B7 batches before wall-clock reserve")
                break

            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present)
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
            positive_cells += int((active & (target > 0.5)).sum().item())
            negative_cells += int((active & (target < 0.5)).sum().item())

        epoch_seconds = time.monotonic() - epoch_start
        epoch_times.append(epoch_seconds)
        if steps == 0:
            raise RuntimeError("B7 completed no training batches inside the runtime budget")
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(epoch_seconds),
            "batches": int(steps),
            "study_draws": int(study_draws),
            "active_supervision_cells_seen": int(active_cells),
            "positive_cells_seen": int(positive_cells),
            "negative_cells_seen": int(negative_cells),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)

        payload = _checkpoint_payload(
            model=model,
            config=config,
            spec=spec,
            history=history,
            b5_checkpoint=b5_path,
            b5_payload=b5_payload,
            b6_root=Path(b6_root),
            b6_audit=b6_audit,
            supervision=supervision,
            target_multiplier=target_multiplier,
            metadata_stats=metadata_stats,
            budget=budget,
        )
        torch.save(payload, checkpoint_path)
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if budget_exhausted:
            break

    if not history:
        raise RuntimeError("B7 did not complete an epoch")
    return checkpoint_path


def load_b7_checkpoint(
    checkpoint: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[KneeMILNet, dict]:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B7_VARIANT:
        raise ValueError(f"not a {B7_VARIANT} checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B7 checkpoint does not certify zero gold-gradient studies")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B7 checkpoint is missing model_spec/model_state")
    model = build_b7_model(spec)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b7")
    parser.add_argument("--config", required=True)
    parser.add_argument("--b5-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b7_weak_supervision")
    args = parser.parse_args()

    config = _read_config(args.config)
    path = train_b7(
        config,
        b5_checkpoint=args.b5_checkpoint,
        b6_root=args.b6_root,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
