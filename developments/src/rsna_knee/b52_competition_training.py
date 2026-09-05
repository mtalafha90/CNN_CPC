"""B52: train the model, instead of perturbing an untrained one.

The public leaderboard top is `0.952`. This project's hidden score is `0.714`.
That gap is not a missing mechanism -- eight architecture experiments since B37
searched for one and none moved the hidden score. It is that the model has never
been trained.

The frozen contract every experiment since B37 inherited:

```text
b17_encoder_frozen: true          the network that reads pixels is frozen
b7_encoder_lr: 0.0                its learning rate is exactly zero
b37_encoder_trainable_stages: 1   one stage of five thaws, at 0.05x = 5e-6
b7_max_batches_per_epoch: 1560    3,120 studies per epoch, of 4,349
epochs                            2, fixed, no checkpoint selection
make_b7_dataset_config(train=False)   all nine augmentations zeroed
```

3,120 optimiser steps at batch size 2, no augmentation, pixel encoder fixed.
Measuring an architecture through that is measuring it through a floor.

Every one of those choices was right for its purpose. Determinism made runs
comparable; a fixed two-epoch endpoint made post-hoc selection impossible; the
frozen encoder isolated supervision from representation. They are why this
archive can state what it knows. They are also why it scores 0.714.

**Competition ranking and causal inference want opposite things.** B52 does not
replace the scientific line and takes nothing from it. It is a separate entry
point that does the ordinary things a competition entry does:

```text
encoder            all five tail stages train, not one
augmentation       on -- the nine settings the config already carries
schedule           cosine that actually completes, over real epochs
selection          best epoch on a held-out split, not a fixed epoch
split              B50's scanner-grouped gate: unseen scanners validate
```

## What this deliberately does not do

It does not touch B42's geometry, the sparse-MIL head, the supervision policy or
the label export. Those are held fixed so that if the score moves, the training
regime is what moved it. One change at a time still applies -- the change here is
"train the model", and it is one change.

## The selection surface

Validation uses report-derived labels on unseen scanners, not the 58 expert
studies. Two reasons. The expert set is 58 studies, resolves to about +/-0.03,
and has been inspected so many times it is spent as a selection surface. And
hidden scores have run consistently *above* Expert-58 across this project --
`0.694` against roughly `0.66`, `0.714` against `0.683` -- which suggests the
hidden labels behave more like the report-derived ones than like expert
adjudication. Selecting on the surface that resembles the target is the point.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b13_training import B13_SERIES_SIGNATURE
from .b17_training import encoder_state_sha256
from .b35_training import _require_base_checkpoint, sha256_file
from .b37_highres_sparse_training import (
    B37_GRAD_CLIP,
    B37_HEAD_LR,
    B37_WEIGHT_DECAY,
    _format_memory_state,
    _memory_state,
    _trim_host_memory,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    collate_b42,
    require_b42_contract,
)
from .b42_constant_area_aspect_sparse_training import (
    _batch_scales,
    _losses,
    _move_study,
)
from .b48_global_conditioned_sparse_training import (
    _config_sha256,
    _indices_for_split,
    _report_only_surface,
    _uid_sha256,
    b48_fill_artifacts,
)
from .b50_adapted_hierarchy_mil import B50AdaptedHierarchySparseMILResidual
from .b54_spacing_conditioned_mil import (
    B54SpacingConditionedMIL,
    assert_conditioning_will_train,
    conditioning_has_moved,
    losses_with_spacing,
    move_study_with_spacing,
)
from .b54_spacing_run import (
    attach_spacing,
    install_spacing_conditioning,
    preflight,
    spacing_summary,
    with_spacing,
)
from .b50_adapted_hierarchy_training import load_b50_selection_gate
from .b50_ordered_slice_selection_split import (
    B50_SPLIT_EXCLUDED,
    B50_SPLIT_SEEN,
    B50_SPLIT_TRAIN,
    B50_SPLIT_UNSEEN,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .encoder_finetune import MAX_TRAINABLE_STAGES
from .evaluation import fast_auc
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import make_scaler, resolve_runtime
from .training_resume import load_checkpoint, resume, save_checkpoint

B52_EXPERIMENT = "B52_COMPETITION_FULL_FINETUNE"
B52_VERSION = "b52_competition_full_finetune_v1"
B52_RUN_ROOT = "runs/086_Experiment_B52_competition_full_finetune"
B52_CHECKPOINT_NAME = "b52_best_model.pt"

# What B52 changes, and the frozen values it changes them from.
B52_DEFAULT_EPOCHS = 12
B52_DEFAULT_ENCODER_STAGES = MAX_TRAINABLE_STAGES
B52_DEFAULT_ENCODER_LR_SCALE = 0.10
B52_DEFAULT_HIERARCHY_LR_SCALE = 0.05
B52_INHERITED_EPOCHS = 2
B52_INHERITED_ENCODER_STAGES = 1

B52_PRIMARY_SPLIT = B50_SPLIT_UNSEEN
B52_TRAIN_SPLIT = B50_SPLIT_TRAIN

# The gate's `train` rows alone are about a third of the report-only population.
# For a competition run every row that is not the validation surface is training
# data: the seen-scanner comparator and the rows B48/B49 spent were withheld to
# keep a *selection* surface clean, and selection here happens only on unseen
# scanners. Nothing that validates B52 is trained on.
B52_FULL_TRAIN_SPLITS = (B50_SPLIT_TRAIN, B50_SPLIT_SEEN, B50_SPLIT_EXCLUDED)
B52_SEED = 2026
B52_CONSTRUCTION_SEED_OFFSET = 11
B52_LOADER_SEED_OFFSET = 29


def _read_config(path: str | Path) -> dict:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def b52_parameter_groups(
    model: B50AdaptedHierarchySparseMILResidual,
    *,
    head_lr: float,
    encoder_lr_scale: float,
    hierarchy_lr_scale: float,
) -> list[dict]:
    """Three rates: fresh head fastest, pretrained encoder and hierarchy slower.

    A pretrained feature is worth more than a randomly initialised head and is
    easily destroyed by the head's step size, so the encoder keeps a reduced
    rate even though far more of it now trains.
    """
    encoder = [p for p in model.base.encoder.parameters() if p.requires_grad]
    hierarchy = [p for p in model.hierarchy_parameters() if p.requires_grad]
    head = [p for p in model.head.parameters() if p.requires_grad]

    if not encoder:
        raise RuntimeError(
            "B52 trains the encoder; none of it requires gradients. Check "
            "encoder_trainable_stages."
        )

    groups = [{"params": head, "lr": float(head_lr), "name": "sparse_head"}]
    groups.append(
        {
            "params": encoder,
            "lr": float(head_lr) * float(encoder_lr_scale),
            "name": "encoder",
        }
    )
    if hierarchy:
        groups.append(
            {
                "params": hierarchy,
                "lr": float(head_lr) * float(hierarchy_lr_scale),
                "name": "study_hierarchy",
            }
        )

    seen: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise RuntimeError("a parameter reached the optimiser twice")
            seen.add(id(parameter))
    return groups


def select_train_and_validation(
    all_uids: list, domain_rows, train_splits: tuple = (B52_TRAIN_SPLIT,)
) -> tuple:
    """Split the report-only surface into training and unseen-scanner validation.

    `_indices_for_split` returns a NumPy array and already refuses an empty
    split, so the only check worth adding here is the one it cannot make: that
    no study appears on both sides. A leak there would raise the validation
    score and silently corrupt every checkpoint choice made from it.
    """
    names = tuple(train_splits or (B52_TRAIN_SPLIT,))
    if B52_PRIMARY_SPLIT in names:
        raise ValueError(
            f"{B52_PRIMARY_SPLIT} is the validation surface and cannot also train"
        )
    train_indices = np.unique(
        np.concatenate(
            [_indices_for_split(all_uids, domain_rows, name) for name in names]
        )
    )
    valid_indices = _indices_for_split(all_uids, domain_rows, B52_PRIMARY_SPLIT)

    train_uids = [all_uids[int(index)] for index in train_indices]
    valid_uids = [all_uids[int(index)] for index in valid_indices]

    overlap = sorted(set(train_uids) & set(valid_uids))
    if overlap:
        raise RuntimeError(
            f"B52 train and validation splits share {len(overlap)} studies "
            f"(for example {overlap[0]}); selection would be measured on training data"
        )
    return train_indices, valid_indices, train_uids, valid_uids


def masked_binary_targets(target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Report states as binary, NaN where nothing supervises the cell.

    The soft targets are 0.85 and 0.05, so the state boundary is 0.5 -- the same
    rule the scanner split was built with. Copied in behaviour from
    `b48_global_conditioned_sparse_eval._masked_target_matrix` so B52's numbers
    sit on the same scale as every other experiment's.
    """
    value = (np.asarray(target, dtype=np.float64) > 0.5).astype(np.float64)
    value[np.asarray(weight, dtype=np.float64) <= 0] = np.nan
    return value


def macro_auc(target: np.ndarray, weight: np.ndarray, prediction: np.ndarray) -> dict:
    """Per-target and macro AUC over the cells the reports actually supervise."""
    masked = masked_binary_targets(target, weight)
    per_target = {
        name: float(fast_auc(masked[:, index], prediction[:, index]))
        for index, name in enumerate(TARGETS)
    }
    defined = [value for value in per_target.values() if np.isfinite(value)]
    return {
        "macro_auc": float(np.mean(defined)) if defined else float("nan"),
        "per_target_auc": per_target,
        "targets_defined": len(defined),
    }


@torch.no_grad()
def evaluate_split(
    model,
    runtime,
    loader,
    multiplier_t,
    aux_weight: float,
    move=_move_study,
    losses=_losses,
) -> dict:
    """Score one split without touching gradients or the training mode flag."""
    was_training = model.training
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    losses: list[float] = []

    for items in loader:
        for item in items:
            tensors = move(item, runtime.device)
            out, total, _combined, _local = losses(
                model, runtime, tensors, multiplier_t, aux_weight
            )
            predictions.append(
                torch.sigmoid(out.logits.detach().float()).cpu().numpy().reshape(-1)
            )
            targets.append(item["target"].numpy().reshape(-1))
            weights.append(item["weight"].numpy().reshape(-1))
            losses.append(float(total.detach().item()))
        _trim_host_memory()

    model.train(was_training)
    scores = macro_auc(
        np.stack(targets), np.stack(weights), np.stack(predictions)
    )
    scores["loss"] = float(np.mean(losses)) if losses else float("nan")
    scores["studies"] = len(predictions)
    return scores


def _build_dataset(
    uids, index, dataset_config, crop_policy, targets, weights, spacing: bool = False
):
    # `with_spacing` returns a subclass, so every frozen contract that tests
    # for B42ConstantAreaAspectDataset still holds.
    cls = with_spacing(B42ConstantAreaAspectDataset) if spacing else (
        B42ConstantAreaAspectDataset
    )
    return cls(
        uids,
        index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )


def train_b52(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    epochs: int = B52_DEFAULT_EPOCHS,
    encoder_trainable_stages: int = B52_DEFAULT_ENCODER_STAGES,
    encoder_lr_scale: float = B52_DEFAULT_ENCODER_LR_SCALE,
    hierarchy_lr_scale: float = B52_DEFAULT_HIERARCHY_LR_SCALE,
    augment: bool = True,
    train_splits: tuple = (B52_TRAIN_SPLIT,),
    gradient_checkpointing: bool = True,
    seed: int = B52_SEED,
    spacing_geometry_csv: str | Path | None = None,
    out_root: str | Path = B52_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train to convergence and keep the best epoch on unseen scanners."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["seed"] = int(seed)

    if not 1 <= int(encoder_trainable_stages) <= MAX_TRAINABLE_STAGES:
        raise ValueError(
            f"B52 trains the encoder; stages must be 1..{MAX_TRAINABLE_STAGES}"
        )
    if int(epochs) < 1:
        raise ValueError("B52 needs at least one epoch")

    domain_payload, domain_rows, domain_meta = load_b50_selection_gate(domain_split)
    seed_everything(int(seed) + B52_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    print(
        f"[B52] epochs={epochs} (was {B52_INHERITED_EPOCHS}) "
        f"encoder_stages={encoder_trainable_stages} (was {B52_INHERITED_ENCODER_STAGES}) "
        f"augment={augment} (was False)",
        flush=True,
    )
    print(f"[B52] split sha={domain_meta['sha256']}", flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(
        root / settings.get("train_csv", "train.csv")
    ) != expected_train_sha:
        raise ValueError("B52 domain split source train.csv fingerprint mismatch")

    fill_artifacts = b48_fill_artifacts(labels_root)
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
    )

    train_indices, valid_indices, train_uids, valid_uids = select_train_and_validation(
        all_uids, domain_rows, tuple(train_splits)
    )
    print(
        f"[B52] training on {len(train_uids)} studies from {list(train_splits)}; "
        f"validating on {len(valid_uids)} from {B52_PRIMARY_SPLIT}",
        flush=True,
    )

    train_targets, train_weights = all_targets[train_indices], all_weights[train_indices]
    valid_targets, valid_weights = all_targets[valid_indices], all_weights[valid_indices]
    target_multiplier = target_balance_multipliers(train_weights)

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B52 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    _train_summary, train_index = audit_variable_series_surface(series, train_uids)
    _valid_summary, valid_index = audit_variable_series_surface(series, valid_uids)

    # B52 keeps B42's geometry exactly, so it asserts that contract rather than
    # reading the crop policy out of the config by hand. This is also what
    # returns the policy object the dataset expects.
    crop_policy = require_b42_contract(settings)

    # The one line that turns nine disabled augmentations back on.
    train_config = make_b7_dataset_config(settings, root, train=bool(augment))
    train_config.tta_center_offsets = ()
    valid_config = make_b7_dataset_config(settings, root, train=False)
    valid_config.tta_center_offsets = ()

    # B54: attach the measured slice spacing to every series record before the
    # datasets are built. Absent `--spacing-geometry-csv` this is skipped and
    # B52 runs exactly as it always has.
    use_spacing = spacing_geometry_csv is not None
    spacing_state: dict = {"enabled": False}
    if use_spacing:
        for index in (train_index, valid_index):
            stats = attach_spacing(
                index,
                series_geometry_csv=spacing_geometry_csv,
                data_root=root,
                split="train",
            )
            print(f"[B54] spacing {stats}", flush=True)
        spacing_state = {
            "enabled": True,
            "series_geometry_csv": str(spacing_geometry_csv),
            "train": spacing_summary(train_index),
            "validation": spacing_summary(valid_index),
        }

    train_dataset = _build_dataset(
        train_uids,
        train_index,
        train_config,
        crop_policy,
        train_targets,
        train_weights,
        spacing=use_spacing,
    )
    valid_dataset = _build_dataset(
        valid_uids,
        valid_index,
        valid_config,
        crop_policy,
        valid_targets,
        valid_weights,
        spacing=use_spacing,
    )
    batch_size = int(settings.get("b42_effective_batch", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=int(seed) + B52_LOADER_SEED_OFFSET),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=int(seed) + B52_LOADER_SEED_OFFSET),
    )

    residual_class = (
        B54SpacingConditionedMIL if use_spacing else B50AdaptedHierarchySparseMILResidual
    )
    model = residual_class(
        base_model,
        grid_size=int(settings["b37_grid_size"]),
        top_k=int(settings["b37_top_k"]),
        temperature=float(settings["b37_temperature"]),
        encoder_trainable_stages=int(encoder_trainable_stages),
        encoder_chunk_size=int(settings["b37_encoder_chunk_size"]),
        adapt_hierarchy=True,
    ).to(runtime.device)
    # Checkpointing recomputes the encoder forward during backward to save
    # memory. With five stages this run peaks near 1.4 GiB of a 16 GiB card, so
    # the memory it buys is not needed and the recompute is pure time.
    model.gradient_checkpointing = bool(gradient_checkpointing)
    # The conditioning is installed *after* base_model arrived with its
    # pretrained weights, which is the only safe order: installing first adds a
    # state-dict key the checkpoint does not have.
    if use_spacing:
        install_spacing_conditioning(model.base)
        gate = preflight(train_index, model=model)
        print(f"[B54] preflight {gate['passed']}", flush=True)
        if not gate["passed"]:
            raise RuntimeError(f"B54 preflight failed: {gate['problems']}")
    model.train()

    trainable = model.trainable_parameter_summary()
    print(f"[B52] trainable={trainable}", flush=True)

    head_lr = float(settings.get("b37_head_lr", B37_HEAD_LR))
    groups = b52_parameter_groups(
        model,
        head_lr=head_lr,
        encoder_lr_scale=float(encoder_lr_scale),
        hierarchy_lr_scale=float(hierarchy_lr_scale),
    )
    for group in groups:
        print(
            f"[B52]   {group['name']:<16} lr={group['lr']:.3e} "
            f"params={sum(p.numel() for p in group['params']):,}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(settings.get("b37_weight_decay", B37_WEIGHT_DECAY))
    )
    # Without this the conditioning could sit in no optimiser group at all,
    # stay at its zero initialisation for the whole run, and make the ablation
    # report no effect from a model never trained to use the spacing.
    if use_spacing:
        spacing_state["optimiser"] = assert_conditioning_will_train(model, optimizer)
    # T_max equals the epochs actually run, so the cosine completes. The frozen
    # contract used T_max=5 with a two-epoch endpoint, which stopped at 90.5%
    # of peak having never trained at a reduced rate.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=float(settings.get("b7_min_lr", 1e-6))
    )
    scaler = make_scaler(runtime)
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(settings["b37_local_aux_weight"])
    clip = float(settings.get("b37_grad_clip", B37_GRAD_CLIP))
    clipped = [p for group in groups for p in group["params"]]

    if preflight_only:
        items = [train_dataset[index] for index in range(min(2, len(train_dataset)))]
        optimizer.zero_grad(set_to_none=True)
        for item, scale in zip(items, _batch_scales(items, multiplier_cpu)):
            tensors = _move_study(item, runtime.device)
            _out, total, _combined, _local = _losses(
                model, runtime, tensors, multiplier_t, aux_weight
            )
            scaler.scale(total * float(scale)).backward()
        moved = sum(
            1
            for p in model.base.encoder.parameters()
            if p.requires_grad and p.grad is not None and torch.count_nonzero(p.grad) > 0
        )
        if moved == 0:
            raise RuntimeError("B52 preflight: no encoder gradient reached the encoder")
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[B52 preflight] PASS encoder tensors with gradient={moved} "
            f"{_format_memory_state(_memory_state(runtime))}",
            flush=True,
        )
        return None

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / B52_CHECKPOINT_NAME
    # The guard exists so a fresh run cannot quietly overwrite a finished one.
    # A resume is the one case where the best checkpoint legitimately already
    # exists: the run was interrupted after an epoch improved on it. Allow it
    # only when there is a recovery point beside it to resume from, so the
    # guard still catches an accidental re-run into a finished directory.
    if checkpoint_path.exists() and load_checkpoint(out) is None:
        raise FileExistsError(f"B52 will not overwrite {checkpoint_path}")

    history: list[dict] = []
    best_macro = -float("inf")
    best_epoch = 0

    def _check_spacing_learned() -> None:
        """The conditioning starts at exactly zero, so it must not end there.

        `assert_conditioning_will_train` makes this near-impossible, and it is
        kept anyway because the failure it guards is silent: a finished run
        whose ablation reports no effect because the term was never learned.
        Mirrors B49's encoder-fingerprint check.
        """
        if not use_spacing:
            return
        moved = conditioning_has_moved(model)
        spacing_state["conditioning_moved"] = bool(moved)
        if not moved:
            raise RuntimeError(
                "B54 spacing conditioning is still exactly zero after training; "
                "the run did not test what it claims to"
            )

    move_study = move_study_with_spacing if use_spacing else _move_study
    compute_losses = losses_with_spacing if use_spacing else _losses

    # Resume before the loop. Runs here are already nineteen hours and B54 is
    # longer; without this a crash costs the whole attempt.
    resumed = resume(
        out,
        model=model,
        version=B52_VERSION,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    print(f"[B52] {resumed.describe()}", flush=True)
    if resumed.restored:
        history = list(resumed.history)

    for epoch in range(resumed.start_epoch, int(epochs) + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        total_sum = 0.0
        batches = 0

        for items in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scales = _batch_scales(items, multiplier_cpu)
            batch_total = 0.0
            for item, scale in zip(items, scales):
                tensors = move_study(item, runtime.device)
                _out, total, _combined, _local = compute_losses(
                    model, runtime, tensors, multiplier_t, aux_weight
                )
                scaler.scale(total * float(scale)).backward()
                batch_total += float(total.detach().item()) * float(scale)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clipped, clip)
            scaler.step(optimizer)
            scaler.update()
            total_sum += batch_total
            batches += 1
            _trim_host_memory()

        scheduler.step()
        scores = evaluate_split(
            model,
            runtime,
            valid_loader,
            multiplier_t,
            aux_weight,
            move=move_study,
            losses=compute_losses,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_sum / max(batches, 1),
            "validation_loss": scores["loss"],
            "validation_macro_auc": scores["macro_auc"],
            "validation_per_target_auc": scores["per_target_auc"],
            "targets_defined": scores["targets_defined"],
            "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
            "epoch_minutes": round((time.monotonic() - started) / 60.0, 1),
        }
        history.append(row)
        # Atomic, and complete: weights, optimiser, schedule, loss scale and
        # every generator. An epoch boundary is the only safe place to stop.
        save_checkpoint(
            out,
            epoch=epoch,
            model=model,
            version=B52_VERSION,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            extra={"spacing": spacing_state},
        )
        print(
            f"[B52] E{epoch:>2} train={row['train_loss']:.6f} "
            f"val={row['validation_loss']:.6f} "
            f"macroAUC={row['validation_macro_auc']:.6f} "
            f"({row['epoch_minutes']} min)",
            flush=True,
        )

        if np.isfinite(row["validation_macro_auc"]) and row["validation_macro_auc"] > best_macro:
            best_macro = float(row["validation_macro_auc"])
            best_epoch = epoch
            _check_spacing_learned()
            payload = {
                "experiment": B52_EXPERIMENT,
                "version": B52_VERSION,
                "selected_epoch": epoch,
                "selection_metric": f"macro_auc on {B52_PRIMARY_SPLIT}",
                "selection_value": best_macro,
                "epochs_planned": int(epochs),
                "seed": int(seed),
                "encoder_trainable_stages": int(encoder_trainable_stages),
                "encoder_lr_scale": float(encoder_lr_scale),
                "hierarchy_lr_scale": float(hierarchy_lr_scale),
                "augmentation_enabled": bool(augment),
                "train_splits": list(train_splits),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "head_lr": head_lr,
                "changed_from_frozen_contract": {
                    "epochs": [B52_INHERITED_EPOCHS, int(epochs)],
                    "encoder_trainable_stages": [
                        B52_INHERITED_ENCODER_STAGES, int(encoder_trainable_stages)
                    ],
                    "augmentation": [False, bool(augment)],
                    "checkpoint_selection": ["none; fixed epoch 2", "best validation epoch"],
                },
                "base_checkpoint": str(base_path),
                "base_checkpoint_sha256": sha256_file(base_path),
                "base_state": model.base.state_dict(),
                "head_state": model.head.state_dict(),
                "model_state": model.state(),
                "encoder_sha256_initial": encoder_initial_sha,
                "encoder_sha256_final": encoder_state_sha256(model.base.encoder),
                "spacing": spacing_state,
                "training_studies": len(train_uids),
                "validation_studies": len(valid_uids),
                "training_uids_sha256": _uid_sha256(train_uids),
                "gold_labels_used": False,
                "gold_studies_used_in_gradient": 0,
                "target_balance_multiplier": {
                    name: float(target_multiplier[index])
                    for index, name in enumerate(TARGETS)
                },
                "label_confidence": confidence,
                "fill_policy": fill_policy,
                "fill_audit": fill_audit,
                "fill_artifacts": fill_artifacts,
                "supervision": supervision,
                "series_policy_signature": B13_SERIES_SIGNATURE,
                "metadata_repair": metadata_stats,
                "domain_split_sha256": domain_meta["sha256"],
                "config_sha256": _config_sha256(settings),
                "source_sha256": {"training": sha256_file(Path(__file__))},
                "history": history,
                "governance": (
                    "B52 selects its checkpoint on a held-out report-labelled "
                    "split, which is competition practice and deliberately not "
                    "the frozen-endpoint policy the scientific line uses. Its "
                    "validation number is a selection statistic, not evidence of "
                    "an effect, and must not be quoted as one."
                ),
            }
            torch.save(payload, checkpoint_path)
            print(f"[B52]     new best at epoch {epoch}: {best_macro:.6f}", flush=True)

    if best_epoch == 0:
        raise RuntimeError("B52 finished without a usable validation score")

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[B52] best epoch {best_epoch} macroAUC {best_macro:.6f}", flush=True)
    print(checkpoint_path, flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("Train B52: full fine-tune with validation selection")
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--epochs", type=int, default=B52_DEFAULT_EPOCHS)
    parser.add_argument(
        "--encoder-stages", type=int, default=B52_DEFAULT_ENCODER_STAGES,
        help=f"1..{MAX_TRAINABLE_STAGES}; the frozen contract used 1",
    )
    parser.add_argument("--encoder-lr-scale", type=float, default=B52_DEFAULT_ENCODER_LR_SCALE)
    parser.add_argument("--hierarchy-lr-scale", type=float, default=B52_DEFAULT_HIERARCHY_LR_SCALE)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--all-data", action="store_true",
        help=(
            "train on every split except the unseen-scanner validation surface "
            f"({', '.join(B52_FULL_TRAIN_SPLITS)}) instead of the gate's train rows alone"
        ),
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="faster, uses more GPU memory; identical maths",
    )
    parser.add_argument("--seed", type=int, default=B52_SEED)
    parser.add_argument("--out-root", default=B52_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--spacing-geometry-csv",
        default=None,
        help=(
            "B54: series_geometry.csv from rsna_knee.slice_geometry_scan. Given "
            "it, the study hierarchy is conditioned on each series' measured "
            "slice spacing. Omitted, this runs exactly as B52 always has"
        ),
    )
    args = parser.parse_args()

    train_b52(
        _read_config(args.config),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        epochs=args.epochs,
        encoder_trainable_stages=args.encoder_stages,
        encoder_lr_scale=args.encoder_lr_scale,
        hierarchy_lr_scale=args.hierarchy_lr_scale,
        augment=not args.no_augment,
        train_splits=B52_FULL_TRAIN_SPLITS if args.all_data else (B52_TRAIN_SPLIT,),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        out_root=args.out_root,
        spacing_geometry_csv=args.spacing_geometry_csv,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B52_EXPERIMENT",
    "B52_FULL_TRAIN_SPLITS",
    "B52_RUN_ROOT",
    "B52_VERSION",
    "b52_parameter_groups",
    "select_train_and_validation",
    "evaluate_split",
    "macro_auc",
    "masked_binary_targets",
    "train_b52",
]
