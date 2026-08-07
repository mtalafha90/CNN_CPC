"""Train the exam-level model with cross-validation.

Usage::

    python -m rsnaknee.train --config configs/base.yaml --folds 0 1 2 3 4

Each fold writes a checkpoint, its out-of-fold predictions and a metric report
into ``paths.output_dir``. The out-of-fold file is what you should use to tune
blending weights and thresholds — never the validation score of a single fold,
which is far noisier than it looks at this dataset size.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .dataset import NUM_SERIES_TYPES, DatasetConfig, KneeExamDataset, collate_exams
from .folds import add_folds
from .losses import DistillationLoss, MultiLabelLoss, compute_pos_weight
from .metrics import evaluate, log_report
from .models import KneeExamModel, ModelConfig, ModelEma
from .schema import DataSchema, discover_schema
from .utils import AverageMeter, get_logger, read_json, seed_everything, write_json

LOGGER = get_logger()


def amp_dtype_from_string(name: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}.get(name, None)


def _make_grad_scaler(device: torch.device):
    """Create a gradient scaler, tolerating both the new and old torch APIs."""
    try:
        return torch.amp.GradScaler(device.type)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler()


def build_dataset_config(config: Config, training: bool) -> DatasetConfig:
    data = config.data
    return DatasetConfig(
        cache_dir=config.paths.cache_dir,
        image_size=data.image_size,
        depth=data.depth,
        max_series=data.max_series,
        augment=data.augment and training,
        horizontal_flip=data.horizontal_flip,
        rotate_degrees=data.rotate_degrees,
        scale_jitter=data.scale_jitter,
        intensity_jitter=data.intensity_jitter,
        noise_std=data.noise_std,
        series_dropout=data.series_dropout if training else 0.0,
        random_erase=data.random_erase if training else 0.0,
    )


def build_model(config: Config, num_labels: int) -> KneeExamModel:
    model_config = ModelConfig(
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        num_labels=num_labels,
        num_series_types=NUM_SERIES_TYPES,
        embed_dim=config.model.embed_dim,
        slice_layers=config.model.slice_layers,
        slice_heads=config.model.slice_heads,
        dropout=config.model.dropout,
        drop_path=config.model.drop_path,
        max_slices=config.data.depth,
        max_series=config.data.max_series,
        grad_checkpoint=config.model.grad_checkpoint,
        channels_last=config.train.channels_last,
    )
    return KneeExamModel(model_config)


def build_optimiser(model: KneeExamModel, config: Config) -> torch.optim.Optimizer:
    """Give the pre-trained backbone a smaller learning rate than the new head.

    The backbone already encodes useful features; the aggregation layers start
    from scratch. Training both at the same rate either destroys the pre-trained
    weights or starves the new layers.
    """
    backbone_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else head_params).append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": config.train.learning_rate * config.train.backbone_lr_scale,
            },
            {"params": head_params, "lr": config.train.learning_rate},
        ],
        weight_decay=config.train.weight_decay,
    )


def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up then cosine decay to 1% of the peak rate."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def masked_series_loss(
    criterion: MultiLabelLoss,
    series_logits: torch.Tensor,
    series_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Apply the exam labels to every real series as auxiliary supervision."""
    batch, series, num_labels = series_logits.shape
    expanded = labels.unsqueeze(1).expand(batch, series, num_labels)
    keep = series_mask.reshape(-1) > 0.5
    if keep.sum() == 0:
        return series_logits.sum() * 0.0
    flat_logits = series_logits.reshape(-1, num_labels)[keep]
    flat_labels = expanded.reshape(-1, num_labels)[keep]
    return criterion(flat_logits, flat_labels)


def run_epoch(
    model: KneeExamModel,
    loader: DataLoader,
    criterion: MultiLabelLoss,
    optimiser: torch.optim.Optimizer | None,
    device: torch.device,
    config: Config,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEma | None = None,
    epoch: int = 0,
    total_steps: int = 1,
    warmup_steps: int = 0,
    global_step: int = 0,
    distil: DistillationLoss | None = None,
) -> tuple[float, int]:
    """One training pass. Returns the mean loss and the updated global step."""
    model.train()
    meter = AverageMeter()
    amp_dtype = amp_dtype_from_string(config.train.amp_dtype)
    accumulate = max(1, config.train.accumulate)
    assert optimiser is not None

    optimiser.zero_grad(set_to_none=True)
    for index, batch in enumerate(loader):
        pixels = batch["pixels"].to(device, non_blocking=True)
        if config.train.channels_last:
            pixels = pixels.contiguous()
        series_type = batch["series_type"].to(device, non_blocking=True)
        series_mask = batch["series_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output = model(pixels, series_type, series_mask)
            loss = criterion(output["logits"], labels)
            if config.train.aux_weight > 0:
                loss = loss + config.train.aux_weight * masked_series_loss(
                    criterion, output["series_logits"], series_mask, labels
                )
            if distil is not None and "teacher" in batch:
                teacher = batch["teacher"].to(device, non_blocking=True)
                loss = loss + config.train.distil_weight * distil(output["logits"], teacher)

        scaled = loss / accumulate
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (index + 1) % accumulate == 0:
            multiplier = cosine_schedule(global_step, total_steps, warmup_steps)
            for group in optimiser.param_groups:
                group.setdefault("base_lr", group["lr"])
                group["lr"] = group["base_lr"] * multiplier

            if scaler is not None:
                scaler.unscale_(optimiser)
            if config.train.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
            if scaler is not None:
                scaler.step(optimiser)
                scaler.update()
            else:
                optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            global_step += 1
            if ema is not None:
                ema.update(model)

        meter.update(loss.item(), labels.shape[0])
        if index % 50 == 0:
            LOGGER.info(
                "epoch %d | step %d/%d | loss %.4f", epoch, index, len(loader), meter.avg
            )

    return meter.avg, global_step


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Return probabilities, labels (when present) and exam ids."""
    model.eval()
    amp_dtype = amp_dtype_from_string(config.train.amp_dtype)
    probabilities, labels, exam_ids = [], [], []

    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        series_type = batch["series_type"].to(device, non_blocking=True)
        series_mask = batch["series_mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output = model(pixels, series_type, series_mask)
        probabilities.append(torch.sigmoid(output["logits"].float()).cpu().numpy())
        exam_ids.extend(batch["exam_id"])
        if "labels" in batch:
            labels.append(batch["labels"].numpy())

    return (
        np.concatenate(probabilities),
        np.concatenate(labels) if labels else None,
        exam_ids,
    )


def prepare_frames(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, DataSchema]:
    """Load the labels, the cache manifest and the schema, and add folds."""
    output_dir = Path(config.paths.output_dir)
    schema_path = output_dir / "schema.json"
    if schema_path.exists():
        schema = DataSchema.from_dict(read_json(schema_path))
        LOGGER.info("Reusing schema from %s", schema_path)
    else:
        schema = discover_schema(
            config.paths.data_dir, config.paths.train_csv, config.paths.sample_submission_csv
        )
        write_json(schema_path, schema.to_dict())

    train_path = config.paths.train_csv or str(Path(config.paths.data_dir) / "train.csv")
    frame = pd.read_csv(train_path)
    frame[schema.id_column] = frame[schema.id_column].astype(str)

    labels = [label for label in schema.labels if label in frame.columns]
    if not labels:
        raise ValueError(
            "None of the schema labels are columns of the training CSV. Check schema.json."
        )
    # Fill missing annotations with 0; the loss masks true NaNs if you prefer to
    # keep them, but a missing finding is almost always an absent finding here.
    frame[labels] = frame[labels].fillna(0).astype(np.float32)

    manifest_path = Path(config.paths.cache_dir) / "series_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No cache manifest at {manifest_path}. Run rsnaknee.preprocess first."
        )
    manifest = pd.read_csv(manifest_path)
    manifest["exam_id"] = manifest["exam_id"].astype(str)

    frame = add_folds(
        frame,
        labels,
        group_column=schema.group_column,
        n_splits=config.data.n_folds,
        seed=config.data.seed,
    )
    schema.labels = labels
    return frame, manifest, schema


def attach_teacher(frame: pd.DataFrame, config: Config, schema: DataSchema) -> list[str]:
    """Merge text-teacher predictions onto the training frame, if they exist."""
    if config.train.distil_weight <= 0:
        return []
    teacher_path = Path(config.paths.output_dir) / "text_teacher_oof.csv"
    if not teacher_path.exists():
        LOGGER.warning(
            "distil_weight is set but %s does not exist; run rsnaknee.text first. "
            "Continuing without distillation.",
            teacher_path,
        )
        return []
    teacher = pd.read_csv(teacher_path)
    teacher[schema.id_column] = teacher[schema.id_column].astype(str)
    teacher_columns = [f"teacher_{label}" for label in schema.labels]
    rename = {label: f"teacher_{label}" for label in schema.labels if label in teacher.columns}
    teacher = teacher.rename(columns=rename)
    keep = [schema.id_column] + [c for c in teacher_columns if c in teacher.columns]
    merged = frame.merge(teacher[keep], on=schema.id_column, how="left")
    missing = [c for c in teacher_columns if c not in merged.columns]
    for column in missing:
        merged[column] = np.nan
    frame[teacher_columns] = merged[teacher_columns].fillna(0.5).to_numpy()
    LOGGER.info("Attached text-teacher predictions for distillation")
    return teacher_columns


def train_fold(
    fold: int,
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    schema: DataSchema,
    config: Config,
    teacher_columns: list[str],
) -> dict:
    """Train one fold and return its metric report."""
    seed_everything(config.data.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = frame[frame["fold"] != fold].reset_index(drop=True)
    valid_frame = frame[frame["fold"] == fold].reset_index(drop=True)
    LOGGER.info("Fold %d: %d train / %d validation exams", fold, len(train_frame), len(valid_frame))

    train_dataset = KneeExamDataset(
        train_frame,
        manifest,
        build_dataset_config(config, training=True),
        schema.id_column,
        schema.labels,
        training=True,
        teacher_columns=teacher_columns,
    )
    valid_dataset = KneeExamDataset(
        valid_frame,
        manifest,
        build_dataset_config(config, training=False),
        schema.id_column,
        schema.labels,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_exams,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.data.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_exams,
        pin_memory=True,
    )

    model = build_model(config, schema.num_labels).to(device)
    if config.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if config.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimiser = build_optimiser(model, config)
    for group in optimiser.param_groups:
        group["base_lr"] = group["lr"]

    pos_weight = compute_pos_weight(
        train_frame[schema.labels].to_numpy(), config.train.pos_weight_max, device
    )
    criterion = MultiLabelLoss(
        pos_weight=pos_weight,
        focal_gamma=config.train.focal_gamma,
        label_smoothing=config.train.label_smoothing,
    ).to(device)
    distil = (
        DistillationLoss(config.train.distil_temperature).to(device)
        if teacher_columns and config.train.distil_weight > 0
        else None
    )

    use_fp16_scaler = config.train.amp_dtype == "fp16" and device.type == "cuda"
    scaler = _make_grad_scaler(device) if use_fp16_scaler else None
    ema = ModelEma(model, config.train.ema_decay) if config.train.ema_decay > 0 else None

    steps_per_epoch = max(1, len(train_loader) // max(1, config.train.accumulate))
    total_steps = steps_per_epoch * config.train.epochs
    warmup_steps = int(steps_per_epoch * config.train.warmup_epochs)

    best_score = -np.inf
    best_report: dict = {}
    best_predictions: np.ndarray | None = None
    patience = 0
    global_step = 0

    for epoch in range(config.train.epochs):
        loss, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            device,
            config,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            global_step=global_step,
            distil=distil,
        )

        evaluation_model = ema.module if ema is not None else model
        probabilities, labels, exam_ids = predict(evaluation_model, valid_loader, device, config)
        report = evaluate(labels, probabilities, schema.labels)
        report["epoch"] = epoch
        report["train_loss"] = loss
        log_report(report, prefix=f"fold {fold} epoch {epoch}:")

        score = report.get(config.train.metric_name, report["macro_auc"])
        if score > best_score:
            best_score, best_report, best_predictions = score, report, probabilities
            patience = 0
            state = {
                "model": (ema.module if ema is not None else model).state_dict(),
                "config": config.to_dict(),
                "schema": schema.to_dict(),
                "fold": fold,
                "score": float(score),
            }
            torch.save(state, output_dir / f"fold{fold}.pt")
            LOGGER.info("Fold %d: new best %s = %.5f", fold, config.train.metric_name, score)
        else:
            patience += 1
            if patience >= config.train.early_stop_patience:
                LOGGER.info("Fold %d: stopping early after %d epochs without gain", fold, patience)
                break

    if best_predictions is not None:
        oof = pd.DataFrame(best_predictions, columns=schema.labels)
        oof.insert(0, schema.id_column, exam_ids)
        oof["fold"] = fold
        oof.to_csv(output_dir / f"oof_fold{fold}.csv", index=False)
    write_json(output_dir / f"report_fold{fold}.json", best_report)
    return best_report


def combine_oof(config: Config, schema: DataSchema, frame: pd.DataFrame) -> dict | None:
    """Concatenate the fold predictions and score them as one set."""
    output_dir = Path(config.paths.output_dir)
    files = sorted(output_dir.glob("oof_fold*.csv"))
    if not files:
        return None
    oof = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    oof[schema.id_column] = oof[schema.id_column].astype(str)
    oof.to_csv(output_dir / "oof.csv", index=False)

    merged = frame[[schema.id_column] + schema.labels].merge(
        oof, on=schema.id_column, suffixes=("", "_pred")
    )
    truth = merged[schema.labels].to_numpy()
    predicted = merged[[f"{label}_pred" for label in schema.labels]].to_numpy()
    report = evaluate(truth, predicted, schema.labels)
    log_report(report, prefix="out-of-fold:")
    write_json(output_dir / "report_oof.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RSNA knee MRI model")
    parser.add_argument("--config", default=None, help="Path to a YAML config")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Folds to train")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, help="key.sub=value")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(Path(config.paths.output_dir) / "config.yaml")

    frame, manifest, schema = prepare_frames(config)
    teacher_columns = attach_teacher(frame, config, schema)

    folds = args.folds if args.folds is not None else list(range(config.data.n_folds))
    for fold in folds:
        train_fold(fold, frame, manifest, schema, config, teacher_columns)

    combine_oof(config, schema, frame)


if __name__ == "__main__":
    main()
