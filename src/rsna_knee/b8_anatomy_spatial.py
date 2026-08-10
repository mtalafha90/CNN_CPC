"""B8: B7.1-initialized anatomy-aware spatial pathology-query MRI model.

B8 is a new named development experiment. It keeps the frozen B6 v1.2.1
weak-supervision policy and full 3,120-study corpus coverage from B7.1, but
changes the MRI tokenization from one globally pooled token per sampled slice
to a 2x2 spatial grid per slice. Fixed gentle stream/slice priors bias pathology
query attention without hard-cropping or gold-derived target-specific tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    B7_VARIANT,
    _read_config,
    _require_frozen_policy,
    load_frozen_b6_export,
    make_b7_dataset_config,
    prepare_b7_supervision,
    seed_everything,
    target_balance_multipliers,
    target_balanced_weak_bce,
)
from .budget import RuntimeBudget
from .constants import DUAL_STREAMS, TARGETS
from .data import backfill_series_metadata, build_series_index, load_series_csv, load_train_csv
from .dataset import KneeStudyDataset
from .model import SpatialAnatomyKneeMILNet
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import SSL_SOURCE

B8_VARIANT = "b8_b71_init_spatial_anatomy_v1"
B8_REQUIRED_B71_EXPERIMENT = "B7.1_full_coverage"
B8_REQUIRED_B71_BATCHES = 1560
B8_REQUIRED_B71_EPOCHS = 4
B8_REQUIRED_TRAINING_STUDIES = 3120
B8_REQUIRED_USABLE_CELLS = 14123

# Predeclared anatomy/sequence priors. These are based on general MRI anatomy and
# sequence sensitivity, not on the 58-study target-level B7.1 results.
B8_PREFERRED_STREAMS: dict[str, tuple[str, ...]] = {
    "ACL": ("sagittal_fluid", "sagittal_structural", "coronal_fluid"),
    "MCL": ("coronal_fluid", "coronal_structural"),
    "Medial Meniscus": ("sagittal_fluid", "sagittal_structural", "coronal_fluid", "coronal_structural"),
    "Lateral Meniscus": ("sagittal_fluid", "sagittal_structural", "coronal_fluid", "coronal_structural"),
    "Medial OA": ("coronal_structural", "coronal_fluid", "sagittal_structural"),
    "Lateral OA": ("coronal_structural", "coronal_fluid", "sagittal_structural"),
    "PF OA": ("axial_fluid", "axial_structural", "sagittal_structural"),
    "Effusion": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Synovitis": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Baker's": ("sagittal_fluid", "axial_fluid", "coronal_fluid"),
    "Contusion": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Fracture": ("sagittal_structural", "coronal_structural", "axial_structural"),
}

# A broad center preference is safe for focal internal structures. Diffuse/fluid
# findings are intentionally left slice-neutral. Sigma is in normalized slice
# coordinates [-1,1].
B8_SLICE_SIGMA: dict[str, float | None] = {
    "ACL": 0.34,
    "MCL": 0.48,
    "Medial Meniscus": 0.42,
    "Lateral Meniscus": 0.42,
    "Medial OA": 0.52,
    "Lateral OA": 0.52,
    "PF OA": 0.52,
    "Effusion": None,
    "Synovitis": None,
    "Baker's": None,
    "Contusion": None,
    "Fracture": None,
}


def build_anatomy_attention_bias(
    *,
    n_slices: int,
    spatial_grid_size: int,
    strength: float = 1.0,
    nonpreferred_stream_prior: float = 0.75,
    slice_prior_floor: float = 0.80,
) -> torch.Tensor:
    """Return fixed additive attention logits shaped [12, K*S*R].

    Preferred streams have prior 1.0; other streams retain substantial mass
    rather than being masked. Focal structures get a broad central-slice prior.
    The 2x2 within-slice regions receive equal fixed prior; their positional
    embeddings are learned so the model can discover spatial evidence without
    assuming canonical left/right/anterior/posterior image orientation.
    """
    n_slices = int(n_slices)
    grid = int(spatial_grid_size)
    strength = float(strength)
    nonpreferred_stream_prior = float(nonpreferred_stream_prior)
    slice_prior_floor = float(slice_prior_floor)
    if n_slices < 1 or grid < 1:
        raise ValueError("n_slices and spatial_grid_size must be >=1")
    if not 0 < nonpreferred_stream_prior <= 1:
        raise ValueError("nonpreferred_stream_prior must be in (0,1]")
    if not 0 < slice_prior_floor <= 1:
        raise ValueError("slice_prior_floor must be in (0,1]")
    if strength < 0:
        raise ValueError("strength must be non-negative")

    r = grid * grid
    positions = torch.linspace(-1.0, 1.0, n_slices, dtype=torch.float32)
    rows: list[torch.Tensor] = []
    for target in TARGETS:
        preferred = set(B8_PREFERRED_STREAMS[target])
        stream_prior = torch.tensor(
            [1.0 if stream in preferred else nonpreferred_stream_prior for stream in DUAL_STREAMS],
            dtype=torch.float32,
        )
        sigma = B8_SLICE_SIGMA[target]
        if sigma is None:
            slice_prior = torch.ones(n_slices, dtype=torch.float32)
        else:
            gaussian = torch.exp(-0.5 * (positions / float(sigma)) ** 2)
            slice_prior = slice_prior_floor + (1.0 - slice_prior_floor) * gaussian
        joint = stream_prior[:, None] * slice_prior[None, :]
        log_bias = strength * torch.log(joint.clamp_min(1e-6))
        rows.append(log_bias[:, :, None].expand(len(DUAL_STREAMS), n_slices, r).reshape(-1))
    return torch.stack(rows, dim=0)


def load_b71_payload(checkpoint: str | Path) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"B8 B7.1 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("B7.1 checkpoint must be a dictionary payload")
    if payload.get("variant") != B7_VARIANT:
        raise ValueError(f"B8 requires B7 implementation variant {B7_VARIANT!r}")
    if payload.get("source") != SSL_SOURCE:
        raise ValueError("B8 requires a competition-training-data checkpoint")
    if int(payload.get("completed_epochs", -1)) != B8_REQUIRED_B71_EPOCHS:
        raise ValueError("B8 requires the completed four-epoch B7.1 checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B8 requires B7.1 with zero gold-gradient studies")
    if int(payload.get("gold_studies_used_for_early_stopping", -1)) != 0:
        raise ValueError("B8 requires B7.1 with zero gold early-stopping studies")
    config = payload.get("config", {})
    if str(config.get("b7_experiment_name")) != B8_REQUIRED_B71_EXPERIMENT:
        raise ValueError("B8 requires the named B7.1_full_coverage checkpoint")
    if int(config.get("b7_max_batches_per_epoch", -1)) != B8_REQUIRED_B71_BATCHES:
        raise ValueError("B8 requires B7.1 full 1560-batch epoch coverage")
    supervision = payload.get("supervision", {})
    if int(supervision.get("training_studies", -1)) != B8_REQUIRED_TRAINING_STUDIES:
        raise ValueError("B8 requires the audited 3,120-study B7.1 training pool")
    if int(supervision.get("training_usable_cells", -1)) != B8_REQUIRED_USABLE_CELLS:
        raise ValueError("B8 requires the audited 14,123-cell B7.1 supervision pool")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("B7.1 checkpoint is missing model_state")
    return payload


def b8_model_spec(config: dict, *, normalize_input: bool) -> dict:
    n_slices = int(config.get("b7_n_slices", 16))
    grid = int(config.get("b8_spatial_grid_size", 2))
    return {
        "architecture": "spatial_anatomy_pathology_queries_v1",
        "n_streams": len(DUAL_STREAMS),
        "n_slices": n_slices,
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
        "spatial_grid_size": grid,
        "anatomy_prior_strength": float(config.get("b8_anatomy_prior_strength", 1.0)),
        "nonpreferred_stream_prior": float(config.get("b8_nonpreferred_stream_prior", 0.75)),
        "slice_prior_floor": float(config.get("b8_slice_prior_floor", 0.80)),
    }


def build_b8_model(spec: dict, *, b71_state: dict | None = None) -> SpatialAnatomyKneeMILNet:
    bias = build_anatomy_attention_bias(
        n_slices=int(spec["n_slices"]),
        spatial_grid_size=int(spec["spatial_grid_size"]),
        strength=float(spec["anatomy_prior_strength"]),
        nonpreferred_stream_prior=float(spec["nonpreferred_stream_prior"]),
        slice_prior_floor=float(spec["slice_prior_floor"]),
    )
    model = SpatialAnatomyKneeMILNet(
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
        spatial_grid_size=int(spec["spatial_grid_size"]),
        anatomy_attention_bias=bias,
    )
    if b71_state is not None:
        result = model.load_state_dict(b71_state, strict=False)
        allowed_missing = {"region_embedding", "anatomy_attention_bias"}
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if missing != allowed_missing or unexpected:
            raise ValueError(
                f"unexpected B7.1->B8 state mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
    return model


def _checkpoint_payload(
    *,
    model,
    config,
    spec,
    history,
    b71_checkpoint,
    b71_payload,
    b6_root,
    b6_audit,
    supervision,
    target_multiplier,
    metadata_stats,
    budget,
) -> dict:
    return {
        "variant": B8_VARIANT,
        "source": SSL_SOURCE,
        "model_state": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "model_spec": spec,
        "config": config,
        "completed_epochs": len(history),
        "gold_studies_used_in_gradient": 0,
        "gold_studies_used_for_early_stopping": 0,
        "b6_gold_audit_informed_global_policy": True,
        "initialization_checkpoint": str(Path(b71_checkpoint).resolve()),
        "initialization_variant": b71_payload.get("variant"),
        "initialization_experiment": b71_payload.get("config", {}).get("b7_experiment_name"),
        "b6_root": str(Path(b6_root).resolve()),
        "b6_version": b6_audit.get("b6_version"),
        "anatomy_prior": {
            "preferred_streams": {k: list(v) for k, v in B8_PREFERRED_STREAMS.items()},
            "slice_sigma": B8_SLICE_SIGMA,
            "strength": float(spec["anatomy_prior_strength"]),
            "nonpreferred_stream_prior": float(spec["nonpreferred_stream_prior"]),
            "slice_prior_floor": float(spec["slice_prior_floor"]),
            "in_plane_fixed_region_prior": "uniform; region embeddings learned",
        },
        "supervision_policy": b71_payload.get("supervision_policy"),
        "target_balance_multiplier": {target: float(target_multiplier[j]) for j, target in enumerate(TARGETS)},
        "supervision": supervision,
        "metadata_repair": metadata_stats,
        "history": history,
        "budget": budget.to_dict(),
    }


def train_b8(
    config: dict,
    *,
    b71_checkpoint: str | Path,
    b6_root: str | Path,
    out_root: str | Path = "runs/b8_spatial_anatomy",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_frozen_policy(config)
    if str(config.get("b8_experiment_name", "")) != "B8_spatial_anatomy_v1":
        raise ValueError("B8 requires b8_experiment_name=B8_spatial_anatomy_v1")
    seed = int(config.get("seed", 2026))
    seed_everything(seed + 8_000_000)
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
    has_mri = np.asarray([any(index.get(uid, {}).get(stream) for stream in DUAL_STREAMS) for uid in uids], dtype=bool)
    if not has_mri.any():
        raise ValueError("B8 found no active weakly labelled studies with MRI series")
    supervision["active_studies_before_mri_filter"] = int(len(uids))
    supervision["studies_without_any_selected_mri_series"] = int((~has_mri).sum())
    uids = [uid for uid, keep in zip(uids, has_mri) if keep]
    targets, weights = targets[has_mri], weights[has_mri]
    supervision["training_studies"] = int(len(uids))
    supervision["training_usable_cells"] = int((weights > 0).sum())
    if supervision["training_studies"] != B8_REQUIRED_TRAINING_STUDIES:
        raise ValueError("B8 training pool changed from the frozen B7.1 3,120-study pool")
    if supervision["training_usable_cells"] != B8_REQUIRED_USABLE_CELLS:
        raise ValueError("B8 usable-cell pool changed from frozen B7.1")

    target_multiplier = target_balance_multipliers(weights)
    for j, target in enumerate(TARGETS):
        supervision["targets"][target]["training_base_weight_sum"] = float(weights[:, j].sum())
        supervision["targets"][target]["target_balance_multiplier"] = float(target_multiplier[j])
        supervision["targets"][target]["balanced_weight_sum"] = float((weights[:, j] * target_multiplier[j]).sum())

    ds = KneeStudyDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=True),
        targets=targets,
        weights=weights,
        train=True,
    )
    batch_size = int(config.get("b8_batch_size", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **runtime.loader_kwargs(seed=seed + 8_100_000),
    )

    b71_path = Path(b71_checkpoint)
    b71_payload = load_b71_payload(b71_path)
    normalize_input = bool(b71_payload.get("model_spec", {}).get("normalize_input", False))
    spec = b8_model_spec(config, normalize_input=normalize_input)
    model = build_b8_model(spec, b71_state=b71_payload["model_state"]).to(runtime.device)

    encoder_lr = float(config.get("b8_encoder_lr", 1e-5))
    head_lr = float(config.get("b8_head_lr", 1e-4))
    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=float(config.get("b8_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b8_epochs", 4))
    max_batches = int(config.get("b8_max_batches_per_epoch", 1560))
    if epochs != 4 or max_batches != 1560 or batch_size != 2:
        raise ValueError("B8-v1 is frozen to 4 epochs, 1560 batches/epoch, batch_size=2")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b8_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    clip = float(config.get("b8_grad_clip", 1.0))
    target_multiplier_t = torch.from_numpy(target_multiplier).to(runtime.device)

    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "b8_model.pt"
    history: list[dict] = []
    epoch_times: list[float] = []
    budget_exhausted = False
    (outdir / "supervision_plan.json").write_text(json.dumps(supervision, indent=2), encoding="utf-8")
    policy_payload = {
        "experiment": "B8",
        "variant": B8_VARIANT,
        "status": "training recipe frozen before first B8 gold evaluation",
        "initialization": str(b71_path.resolve()),
        "initialization_experiment": B8_REQUIRED_B71_EXPERIMENT,
        "single_architecture_change": "2x2 within-slice spatial tokens plus fixed gentle pathology stream/slice attention priors",
        "gold_labels_in_training_loss": False,
        "gold_labels_for_early_stopping": False,
        "b6_gold_audit_informed_global_policy": True,
        "fixed_epochs": epochs,
        "max_batches_per_epoch": max_batches,
        "full_corpus_studies_per_epoch": B8_REQUIRED_TRAINING_STUDIES,
        "model_spec": spec,
        "preferred_streams": {k: list(v) for k, v in B8_PREFERRED_STREAMS.items()},
        "slice_sigma": B8_SLICE_SIGMA,
        "b6_policy": b6_policy,
    }
    (outdir / "policy.json").write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B8 before next epoch")
            break
        epoch_start = time.monotonic()
        model.train()
        loss_sum = 0.0
        steps = study_draws = active_cells = positive_cells = negative_cells = 0
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B8 batches before wall-clock reserve")
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            target = batch["target"].to(runtime.device, non_blocking=True)
            weight = batch["weight"].to(runtime.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                logits = model(volumes, present)
                loss = target_balanced_weak_bce(logits, target, weight, target_multiplier_t)
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
            raise RuntimeError("B8 completed no training batches inside the runtime budget")
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
        torch.save(
            _checkpoint_payload(
                model=model,
                config=config,
                spec=spec,
                history=history,
                b71_checkpoint=b71_path,
                b71_payload=b71_payload,
                b6_root=Path(b6_root),
                b6_audit=b6_audit,
                supervision=supervision,
                target_multiplier=target_multiplier,
                metadata_stats=metadata_stats,
                budget=budget,
            ),
            checkpoint_path,
        )
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if not history:
        raise RuntimeError("B8 did not complete an epoch")
    return checkpoint_path


def load_b8_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SpatialAnatomyKneeMILNet, dict]:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("variant") != B8_VARIANT:
        raise ValueError(f"not a {B8_VARIANT} checkpoint")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B8 checkpoint does not certify zero gold-gradient studies")
    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B8 checkpoint is missing model_spec/model_state")
    model = build_b8_model(spec)
    model.load_state_dict(state, strict=True)
    return model.to(device), payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b8")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="override data_root from YAML")
    parser.add_argument("--b71-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b8_spatial_anatomy")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_b8(
        config,
        b71_checkpoint=args.b71_checkpoint,
        b6_root=args.b6_root,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
