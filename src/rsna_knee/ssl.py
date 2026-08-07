"""In-domain self-supervised pretraining on non-gold knee MRI studies."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import backfill_series_metadata, build_series_index, gold_mask, load_series_csv, load_train_csv
from .dataset import DatasetConfig, KneeStudyDataset
from .model import ConvNeXtSliceEncoder
from .runtime import autocast, barrier, make_scaler, resolve_runtime

PLANE_LABELS = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
SEQUENCE_LABELS = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)


class MRIRepresentationLearner(nn.Module):
    def __init__(self, *, pretrained: bool = True, normalize_input: bool = True, projection_dim: int = 256):
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


def _global_embeddings(z: torch.Tensor, study_ids: torch.Tensor, runtime):
    """Differentiably gather variable active-stream counts using padding."""
    if not runtime.distributed:
        return z, study_ids

    local_n = torch.tensor([z.shape[0]], device=z.device, dtype=torch.long)
    size_parts = [torch.zeros_like(local_n) for _ in range(runtime.world_size)]
    dist.all_gather(size_parts, local_n)
    sizes = [int(x.item()) for x in size_parts]
    max_n = max(sizes)

    if z.shape[0] < max_n:
        z_pad = torch.cat([z, z.new_zeros((max_n - z.shape[0], z.shape[1]))], dim=0)
        id_pad = torch.cat([study_ids, study_ids.new_full((max_n - study_ids.shape[0],), -1)], dim=0)
    else:
        z_pad, id_pad = z, study_ids

    from torch.distributed.nn.functional import all_gather as differentiable_all_gather

    z_parts = list(differentiable_all_gather(z_pad))
    id_parts = [torch.empty_like(id_pad) for _ in range(runtime.world_size)]
    dist.all_gather(id_parts, id_pad)
    return (
        torch.cat([part[:n] for part, n in zip(z_parts, sizes)], dim=0),
        torch.cat([part[:n] for part, n in zip(id_parts, sizes)], dim=0),
    )


def _all_ranks_have_active_streams(active_count: int, runtime) -> bool:
    if not runtime.distributed:
        return active_count > 0
    flag = torch.tensor([1 if active_count > 0 else 0], device=runtime.device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def pretrain_ssl(config: dict) -> Path:
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
            data_root=str(root),
            split="train",
            n_slices=int(config.get("ssl_n_slices", 5)),
            image_size=int(config.get("image_size", 224)),
            noise_std=float(config.get("ssl_noise_std", 0.01)),
            slice_dropout=0.0,
            triplet_gap=int(config.get("triplet_gap", 1)),
            strict_dicom=bool(config.get("strict_dicom", False)),
        ),
        train=True,
    )
    sampler = (
        DistributedSampler(
            ds,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=int(config.get("seed", 2026)),
            drop_last=True,
        )
        if runtime.distributed
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("ssl_batch_size", 4)),
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=runtime.distributed,
        **runtime.loader_kwargs(),
    )

    model = MRIRepresentationLearner(
        pretrained=bool(config.get("pretrained", True)),
        normalize_input=bool(config.get("normalize_input", True)),
        projection_dim=int(config.get("ssl_projection_dim", 256)),
    ).to(runtime.device)
    if runtime.distributed:
        model = DDP(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("ssl_lr", 2e-4)),
        weight_decay=float(config.get("ssl_weight_decay", 1e-4)),
    )
    scaler = make_scaler(runtime)
    epochs = int(config.get("ssl_epochs", 10))
    temperature = float(config.get("ssl_temperature", 0.15))
    meta_weight = float(config.get("ssl_metadata_weight", 0.25))
    outdir = Path(config.get("ssl_output_dir", "runs/ssl"))
    checkpoint_path = outdir / "ssl_encoder.pt"
    if runtime.is_main:
        outdir.mkdir(parents=True, exist_ok=True)
    barrier(runtime)

    history = []
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        steps = 0
        for batch_idx, batch in enumerate(loader):
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            b, k, s, c, h, w = volumes.shape
            active = present.reshape(-1) > 0
            if not _all_ranks_have_active_streams(int(active.sum().item()), runtime):
                continue

            x = volumes[:, :, s // 2].reshape(b * k, c, h, w)[active]
            stream_idx = torch.arange(k, device=runtime.device).repeat(b)[active]
            # Rank/batch offsets make study IDs globally unique while retaining
            # identical IDs for multiple sequences from the same local study.
            study_ids = (
                torch.arange(b, device=runtime.device)
                + runtime.rank * 1_000_000
                + batch_idx * 10_000
            ).repeat_interleave(k)[active]
            plane = PLANE_LABELS.to(runtime.device)[stream_idx]
            sequence = SEQUENCE_LABELS.to(runtime.device)[stream_idx]

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                _, z, plane_logits, sequence_logits = model(x)
                z_global, ids_global = _global_embeddings(z, study_ids, runtime)
                contrast = _contrastive_same_study(z_global, ids_global, temperature)
                metadata_loss = nn.functional.cross_entropy(plane_logits, plane) + nn.functional.cross_entropy(
                    sequence_logits, sequence
                )
                loss = contrast + meta_weight * metadata_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            steps += 1

        row = {"epoch": epoch + 1, "loss": total / max(steps, 1)}
        if runtime.is_main:
            history.append(row)
            print(row)

    if runtime.is_main:
        base = model.module if isinstance(model, DDP) else model
        torch.save(
            {"encoder": base.encoder.state_dict(), "config": config, "non_gold_studies": len(non_gold)},
            checkpoint_path,
        )
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    barrier(runtime)
    return checkpoint_path
