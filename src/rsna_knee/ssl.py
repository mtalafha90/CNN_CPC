"""In-domain self-supervised pretraining on non-gold knee MRI studies."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .budget import RuntimeBudget
from .data import backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import ConvNeXtSliceEncoder
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime

PLANE_LABELS = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
SEQUENCE_LABELS = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
SSL_SOURCE = "competition_training_data"


class MRIRepresentationLearner(nn.Module):
    def __init__(self, *, pretrained: bool = False, normalize_input: bool = True, projection_dim: int = 256):
        super().__init__()
        self.encoder = ConvNeXtSliceEncoder(3, pretrained_weights=pretrained, normalize_input=normalize_input)
        d = self.encoder.out_dim
        self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, projection_dim))
        self.plane_head = nn.Linear(d, 3)
        self.sequence_head = nn.Linear(d, 2)

    def forward(self, x):
        feat = self.encoder(x)
        return (
            feat,
            nn.functional.normalize(self.projector(feat), dim=-1),
            self.plane_head(feat),
            self.sequence_head(feat),
        )


def _contrastive_same_study(z: torch.Tensor, study_ids: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.sum() * 0.0
    logits = (z @ z.T) / float(temperature)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -1e4)
    positives = (study_ids[:, None] == study_ids[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = log_prob.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -per_anchor[valid].mean()


def pretrain_ssl(config: dict) -> Path:
    validate_competition_config(config, purpose="train")
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    runtime = resolve_runtime(config)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    non_gold = train.loc[~gold_mask(train), "StudyInstanceUID"].tolist()
    if not non_gold:
        raise ValueError("no non-gold studies available for SSL")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, _ = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, non_gold, mode="dual")
    ds = KneeStudyDataset(
        non_gold,
        index,
        DatasetConfig(
            data_root=str(root), split="train",
            n_slices=int(config.get("ssl_n_slices", 5)),
            image_size=int(config.get("image_size", 224)),
            noise_std=float(config.get("ssl_noise_std", 0.01)),
            slice_dropout=0.0,
            triplet_gap=int(config.get("triplet_gap", 1)),
            strict_dicom=bool(config.get("strict_dicom", False)),
            train_gap_choices=tuple(int(x) for x in config.get("train_gap_choices", [1, 2])),
            center_jitter=int(config.get("center_jitter", 2)),
            rotation_deg=float(config.get("rotation_deg", 5.0)),
            translate_frac=float(config.get("translate_frac", 0.03)),
            scale_jitter=float(config.get("scale_jitter", 0.05)),
            gamma_jitter=float(config.get("gamma_jitter", 0.12)),
            bias_field_strength=float(config.get("bias_field_strength", 0.08)),
            series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
        ),
        train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("ssl_batch_size", 4)),
        shuffle=True,
        drop_last=True,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 700_000),
    )

    model = MRIRepresentationLearner(
        pretrained=bool(config.get("pretrained", False)),
        normalize_input=bool(config.get("normalize_input", False)),
        projection_dim=int(config.get("ssl_projection_dim", 256)),
    ).to(runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("ssl_lr", 2e-4)),
        weight_decay=float(config.get("ssl_weight_decay", 1e-4)),
    )
    scaler = make_scaler(runtime)
    epochs = int(config.get("ssl_epochs", 4))
    temperature = float(config.get("ssl_temperature", 0.15))
    meta_weight = float(config.get("ssl_metadata_weight", 0.25))
    outdir = Path(config.get("ssl_output_dir", "runs/ssl"))
    checkpoint_path = outdir / "ssl_encoder.pt"
    outdir.mkdir(parents=True, exist_ok=True)

    history = []
    epoch_times = []
    for epoch in range(epochs):
        if epoch_times:
            estimate = float(torch.tensor(epoch_times).median().item())
            if not budget.can_start(estimate * 1.25):
                print("[budget] stopping SSL before the next epoch")
                break
        epoch_start = time.monotonic()
        model.train()
        total, steps = 0.0, 0
        for batch in loader:
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            b, k, s, c, h, w = volumes.shape
            active = present.reshape(-1) > 0
            if int(active.sum().item()) < 2:
                continue
            x = volumes[:, :, s // 2].reshape(b * k, c, h, w)[active]
            stream_idx = torch.arange(k, device=runtime.device).repeat(b)[active]
            study_ids = torch.arange(b, device=runtime.device).repeat_interleave(k)[active]
            plane = PLANE_LABELS.to(runtime.device)[stream_idx]
            sequence = SEQUENCE_LABELS.to(runtime.device)[stream_idx]

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                _, z, plane_logits, sequence_logits = model(x)
                contrast = _contrastive_same_study(z, study_ids, temperature)
                metadata_loss = nn.functional.cross_entropy(plane_logits, plane) + nn.functional.cross_entropy(
                    sequence_logits, sequence
                )
                loss = contrast + meta_weight * metadata_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            steps += 1

        epoch_seconds = time.monotonic() - epoch_start
        epoch_times.append(epoch_seconds)
        row = {"epoch": epoch + 1, "loss": total / max(steps, 1), "epoch_seconds": epoch_seconds}
        history.append(row)
        print(row)

    if not history:
        raise RuntimeError("SSL did not complete one epoch inside the runtime budget")
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "config": config,
            "source": SSL_SOURCE,
            "non_gold_studies": len(non_gold),
            "completed_epochs": len(history),
            "budget": budget.to_dict(),
        },
        checkpoint_path,
    )
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return checkpoint_path
