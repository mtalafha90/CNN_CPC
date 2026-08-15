"""B5: competition-only image-report representation learning.

B5 changes the MRI representation rather than the downstream gold-label
classifier.  It excludes every gold study, fits a TF-IDF -> TruncatedSVD text
space only on the 4,349 report-only competition studies, and aligns MRI study
embeddings with their own report semantics while retaining the strong B1
image-image and acquisition-metadata SSL objectives.

No external language model or external image weights are used.  The report
branch is training-only; the saved artifact used downstream is an MRI encoder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader

from .budget import RuntimeBudget
from .data import (
    backfill_series_metadata,
    build_series_index,
    gold_mask,
    load_series_csv,
    load_train_csv,
    normalize_report,
    report_hash,
)
from .dataset import DatasetConfig, KneeStudyDataset
from .model import ConvNeXtSliceEncoder
from .policy import validate_competition_config
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import (
    PLANE_LABELS,
    SEQUENCE_LABELS,
    SSL_SOURCE,
    _contrastive_same_study,
    _ssl_examples,
)

B5_VARIANT = "b5_image_report_tfidf_svd"


class MRIReportRepresentationLearner(nn.Module):
    """Strong-SSL ConvNeXt plus image-image, metadata and report heads."""

    def __init__(
        self,
        *,
        report_dim: int,
        normalize_input: bool,
        image_projection_dim: int = 256,
    ) -> None:
        super().__init__()
        if report_dim < 2:
            raise ValueError("report_dim must be >=2")
        self.encoder = ConvNeXtSliceEncoder(
            3,
            pretrained_weights=False,
            normalize_input=bool(normalize_input),
        )
        d = self.encoder.out_dim
        self.image_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, int(image_projection_dim)),
        )
        self.report_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, int(report_dim)),
        )
        self.plane_head = nn.Linear(d, 3)
        self.sequence_head = nn.Linear(d, 2)

    def forward_examples(self, x: torch.Tensor):
        feat = self.encoder(x)
        image_z = nn.functional.normalize(self.image_projector(feat), dim=-1)
        return feat, image_z, self.plane_head(feat), self.sequence_head(feat)

    def project_report_space(self, study_feat: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.report_projector(study_feat), dim=-1)


def fit_report_semantics(
    reports,
    *,
    requested_dim: int = 256,
    max_features: int = 20_000,
    min_df: int = 2,
    seed: int = 2026,
):
    """Fit a competition-only TF-IDF/SVD report semantic space.

    Returns normalized dense report vectors, integer duplicate-report group IDs,
    the fitted sklearn objects, and compact diagnostics.  Empty reports are
    represented by an explicit token so every selected study remains defined.
    """
    texts = [normalize_report(str(x)) for x in reports]
    texts = [x if x else "__empty_report__" for x in texts]
    if len(texts) < 3:
        raise ValueError("B5 report semantics require at least three reports")
    if requested_dim < 2 or max_features < 2 or min_df < 1:
        raise ValueError("invalid B5 text-space configuration")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=int(max_features),
        min_df=int(min_df),
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(texts)
    max_dim = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_dim < 2:
        raise ValueError(
            f"B5 TF-IDF vocabulary is too small for SVD: shape={tfidf.shape}"
        )
    actual_dim = min(int(requested_dim), int(max_dim))
    svd = TruncatedSVD(n_components=actual_dim, random_state=int(seed))
    vectors = svd.fit_transform(tfidf).astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)

    hashes = [report_hash(x) for x in texts]
    group_lookup: dict[str, int] = {}
    group_ids = np.empty(len(hashes), dtype=np.int64)
    for i, value in enumerate(hashes):
        if value not in group_lookup:
            group_lookup[value] = len(group_lookup)
        group_ids[i] = group_lookup[value]

    stats = {
        "reports": int(len(texts)),
        "tfidf_features": int(tfidf.shape[1]),
        "requested_svd_dim": int(requested_dim),
        "actual_svd_dim": int(actual_dim),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "unique_report_groups": int(len(group_lookup)),
        "duplicate_report_rows": int(len(texts) - len(group_lookup)),
        "empty_reports": int(sum(x == "__empty_report__" for x in texts)),
        "max_features": int(max_features),
        "min_df": int(min_df),
        "ngram_range": [1, 2],
        "external_text_model": False,
    }
    return vectors, group_ids, vectorizer, svd, stats


def _aggregate_study_features(
    feat: torch.Tensor,
    study_ids: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool active 2.5D example features to one vector per study."""
    if feat.ndim != 2 or study_ids.ndim != 1 or len(feat) != len(study_ids):
        raise ValueError("invalid B5 feature/study-id shapes")
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")
    d = int(feat.shape[1])
    pooled = torch.zeros((batch_size, d), dtype=feat.dtype, device=feat.device)
    counts = torch.zeros((batch_size, 1), dtype=feat.dtype, device=feat.device)
    pooled.index_add_(0, study_ids, feat)
    counts.index_add_(
        0,
        study_ids,
        torch.ones((len(study_ids), 1), dtype=feat.dtype, device=feat.device),
    )
    valid = counts[:, 0] > 0
    pooled = pooled / counts.clamp_min(1.0)
    return pooled, valid


def _report_alignment_losses(
    image_z: torch.Tensor,
    report_z: torch.Tensor,
    report_groups: torch.Tensor,
    *,
    queue_z: torch.Tensor | None = None,
    queue_groups: torch.Tensor | None = None,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image->report contrast plus direct cosine alignment.

    Same normalized report hashes are masked as negatives, both inside the
    current batch and in the queue, to avoid penalizing duplicate report text.
    """
    if image_z.ndim != 2 or report_z.shape != image_z.shape:
        raise ValueError("image/report embeddings must have the same [B,D] shape")
    if report_groups.shape != (len(image_z),):
        raise ValueError("report_groups must have shape [B]")
    if len(image_z) < 1:
        raise ValueError("B5 report alignment requires at least one study")
    if temperature <= 0:
        raise ValueError("temperature must be >0")

    report_z = nn.functional.normalize(report_z, dim=-1)
    image_z = nn.functional.normalize(image_z, dim=-1)
    candidate_z = report_z
    candidate_groups = report_groups
    if queue_z is not None and len(queue_z):
        if queue_groups is None or queue_z.ndim != 2 or queue_z.shape[1] != image_z.shape[1]:
            raise ValueError("invalid B5 report queue")
        candidate_z = torch.cat([report_z, nn.functional.normalize(queue_z, dim=-1)], dim=0)
        candidate_groups = torch.cat([report_groups, queue_groups], dim=0)

    logits = (image_z @ candidate_z.T) / float(temperature)
    targets = torch.arange(len(image_z), device=image_z.device)
    same_group = report_groups[:, None].eq(candidate_groups[None, :])
    positive = torch.zeros_like(same_group)
    positive[torch.arange(len(image_z), device=image_z.device), targets] = True
    logits = logits.masked_fill(same_group & ~positive, -1e4)
    nce = nn.functional.cross_entropy(logits, targets)
    cosine = 1.0 - (image_z * report_z).sum(dim=-1).mean()
    return nce, cosine


def _update_report_queue(
    queue_z: torch.Tensor | None,
    queue_groups: torch.Tensor | None,
    new_z: torch.Tensor,
    new_groups: torch.Tensor,
    *,
    capacity: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if capacity <= 0:
        return None, None
    z = nn.functional.normalize(new_z.detach(), dim=-1)
    g = new_groups.detach()
    if queue_z is not None and len(queue_z):
        z = torch.cat([queue_z, z], dim=0)
        g = torch.cat([queue_groups, g], dim=0)
    if len(z) > int(capacity):
        z = z[-int(capacity):]
        g = g[-int(capacity):]
    return z, g


def _load_strong_ssl_encoder(model: MRIReportRepresentationLearner, checkpoint: str | Path) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"B5 initialization checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("source") != SSL_SOURCE:
        raise ValueError(
            f"B5 requires competition-only SSL initialization source={SSL_SOURCE!r}; "
            f"got {payload.get('source')!r}"
        )
    state = payload.get("encoder")
    if not isinstance(state, dict):
        raise ValueError("B5 initialization checkpoint has no encoder state_dict")
    model.encoder.load_state_dict(state, strict=True)
    return payload


def pretrain_report_ssl(config: dict, *, checkpoint: str | Path | None = None) -> Path:
    validate_competition_config(config, purpose="train")
    runtime = resolve_runtime(config)
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    non_gold_frame = train.loc[~gold_mask(train), ["StudyInstanceUID", "Report"]].copy()
    if non_gold_frame.empty:
        raise ValueError("B5 found no non-gold report-only studies")
    non_gold_frame["StudyInstanceUID"] = non_gold_frame["StudyInstanceUID"].astype(str)
    non_gold = non_gold_frame["StudyInstanceUID"].tolist()

    seed = int(config.get("seed", 2026))
    report_vectors, report_groups, vectorizer, svd, text_stats = fit_report_semantics(
        non_gold_frame["Report"].fillna("").tolist(),
        requested_dim=int(config.get("b5_report_dim", 256)),
        max_features=int(config.get("b5_tfidf_max_features", 20_000)),
        min_df=int(config.get("b5_tfidf_min_df", 2)),
        seed=seed,
    )
    report_dim = int(report_vectors.shape[1])
    uid_to_report = {uid: i for i, uid in enumerate(non_gold)}

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, non_gold, mode="dual")

    n_slices = int(config.get("b5_n_slices", config.get("ssl_n_slices", 9)))
    positions_per_stream = int(
        config.get("b5_positions_per_stream", config.get("ssl_positions_per_stream", 2))
    )
    if positions_per_stream < 1 or positions_per_stream > n_slices:
        raise ValueError("b5_positions_per_stream must be in [1,b5_n_slices]")

    ds = KneeStudyDataset(
        non_gold,
        index,
        DatasetConfig(
            data_root=str(root),
            split="train",
            n_slices=n_slices,
            image_size=int(config.get("image_size", 224)),
            noise_std=float(config.get("b5_noise_std", config.get("ssl_noise_std", 0.01))),
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
    batch_size = int(config.get("b5_batch_size", config.get("ssl_batch_size", 3)))
    if batch_size < 2:
        raise ValueError("B5 requires b5_batch_size >=2")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **runtime.loader_kwargs(seed=seed + 950_000),
    )

    init_checkpoint = checkpoint or config.get("ssl_encoder_checkpoint")
    if not init_checkpoint:
        raise ValueError("B5 requires --checkpoint or ssl_encoder_checkpoint")
    init_payload = torch.load(Path(init_checkpoint), map_location="cpu", weights_only=False)
    init_config = init_payload.get("config", {}) if isinstance(init_payload, dict) else {}
    normalize_input = bool(init_config.get("normalize_input", config.get("normalize_input", False)))

    model = MRIReportRepresentationLearner(
        report_dim=report_dim,
        normalize_input=normalize_input,
        image_projection_dim=int(config.get("b5_image_projection_dim", config.get("ssl_projection_dim", 256))),
    )
    init_payload = _load_strong_ssl_encoder(model, init_checkpoint)
    model = model.to(runtime.device)

    encoder_lr = float(config.get("b5_encoder_lr", 5e-5))
    head_lr = float(config.get("b5_head_lr", 2e-4))
    weight_decay = float(config.get("b5_weight_decay", config.get("ssl_weight_decay", 1e-4)))
    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    epochs = int(config.get("b5_epochs", 4))
    if epochs < 1:
        raise ValueError("b5_epochs must be >=1")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b5_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)

    max_batches = int(config.get("b5_max_batches_per_epoch", 1000))
    queue_capacity = int(config.get("b5_report_queue_size", 256))
    image_temperature = float(config.get("b5_image_temperature", config.get("ssl_temperature", 0.15)))
    report_temperature = float(config.get("b5_report_temperature", 0.10))
    image_weight = float(config.get("b5_image_weight", 1.0))
    metadata_weight = float(config.get("b5_metadata_weight", config.get("ssl_metadata_weight", 0.25)))
    report_weight = float(config.get("b5_report_weight", 0.5))
    cosine_weight = float(config.get("b5_report_cosine_weight", 0.25))
    if max_batches < 1 or queue_capacity < 0:
        raise ValueError("invalid B5 batch/queue configuration")

    outdir = Path(config.get("b5_output_dir", "runs/b5_report_ssl"))
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "b5_encoder.pt"
    joblib.dump(
        {"vectorizer": vectorizer, "svd": svd, "stats": text_stats},
        outdir / "report_text_space.joblib",
    )
    np.savez_compressed(
        outdir / "report_semantics.npz",
        study_uids=np.asarray(non_gold, dtype=str),
        report_vectors=report_vectors,
        report_group_ids=report_groups,
    )
    (outdir / "report_semantics.json").write_text(
        json.dumps(text_stats, indent=2), encoding="utf-8"
    )

    history: list[dict] = []
    epoch_times: list[float] = []
    total_study_draws = 0
    total_active_examples = 0
    total_batches = 0
    budget_exhausted = False
    queue_z: torch.Tensor | None = None
    queue_groups: torch.Tensor | None = None

    report_vectors_t = torch.from_numpy(report_vectors)
    report_groups_t = torch.from_numpy(report_groups)

    for epoch in range(epochs):
        if epoch_times:
            estimate = float(np.median(epoch_times))
            if not budget.can_start(estimate * 1.25):
                print("[budget] stopping B5 before the next epoch")
                break
        epoch_start = time.monotonic()
        model.train()
        sums = {
            "loss": 0.0,
            "image_contrast": 0.0,
            "metadata": 0.0,
            "report_nce": 0.0,
            "report_cosine": 0.0,
        }
        steps = 0
        epoch_study_draws = 0
        epoch_active_examples = 0

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B5 batches before the wall-clock reserve")
                break

            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            x, stream_idx, study_ids, _, _ = _ssl_examples(
                volumes,
                present,
                positions_per_stream=positions_per_stream,
            )
            if int(x.shape[0]) < 2:
                continue

            batch_report_indices = torch.tensor(
                [uid_to_report[str(uid)] for uid in batch["study_uid"]],
                dtype=torch.long,
            )
            report_target = report_vectors_t.index_select(0, batch_report_indices).to(
                runtime.device, non_blocking=True
            )
            group_target = report_groups_t.index_select(0, batch_report_indices).to(
                runtime.device, non_blocking=True
            )
            plane = PLANE_LABELS.to(runtime.device)[stream_idx]
            sequence = SEQUENCE_LABELS.to(runtime.device)[stream_idx]

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                feat, image_z, plane_logits, sequence_logits = model.forward_examples(x)
                image_contrast = _contrastive_same_study(
                    image_z,
                    study_ids,
                    temperature=image_temperature,
                )
                metadata_loss = nn.functional.cross_entropy(
                    plane_logits, plane
                ) + nn.functional.cross_entropy(sequence_logits, sequence)

                study_feat, valid = _aggregate_study_features(
                    feat,
                    study_ids,
                    batch_size=int(volumes.shape[0]),
                )
                report_image_z = model.project_report_space(study_feat[valid])
                report_nce, report_cosine = _report_alignment_losses(
                    report_image_z,
                    report_target[valid],
                    group_target[valid],
                    queue_z=queue_z,
                    queue_groups=queue_groups,
                    temperature=report_temperature,
                )
                loss = (
                    image_weight * image_contrast
                    + metadata_weight * metadata_loss
                    + report_weight * (report_nce + cosine_weight * report_cosine)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            queue_z, queue_groups = _update_report_queue(
                queue_z,
                queue_groups,
                report_target[valid].to(dtype=torch.float32),
                group_target[valid],
                capacity=queue_capacity,
            )

            sums["loss"] += float(loss.item())
            sums["image_contrast"] += float(image_contrast.item())
            sums["metadata"] += float(metadata_loss.item())
            sums["report_nce"] += float(report_nce.item())
            sums["report_cosine"] += float(report_cosine.item())
            steps += 1
            epoch_study_draws += int(volumes.shape[0])
            epoch_active_examples += int(x.shape[0])

        epoch_seconds = time.monotonic() - epoch_start
        epoch_times.append(epoch_seconds)
        if steps > 0:
            scheduler.step()
            total_study_draws += epoch_study_draws
            total_active_examples += epoch_active_examples
            total_batches += steps
            row = {
                "epoch": epoch + 1,
                **{key: value / steps for key, value in sums.items()},
                "encoder_lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
                "epoch_seconds": float(epoch_seconds),
                "batches": int(steps),
                "study_draws": int(epoch_study_draws),
                "active_2p5d_examples": int(epoch_active_examples),
                "queue_size": int(0 if queue_z is None else len(queue_z)),
                "budget_limited": bool(budget_exhausted),
            }
            history.append(row)
            print(row)
        if budget_exhausted:
            break

    if not history:
        raise RuntimeError("B5 did not complete one training batch inside the runtime budget")

    coverage = {
        "non_gold_studies": int(len(non_gold)),
        "gold_studies_excluded": int(gold_mask(train).sum()),
        "reports_used_for_text_fit": int(len(non_gold)),
        "total_study_draws": int(total_study_draws),
        "approx_corpus_passes": float(total_study_draws / max(len(non_gold), 1)),
        "total_batches": int(total_batches),
        "total_active_2p5d_examples": int(total_active_examples),
        "n_slices": int(n_slices),
        "positions_per_stream": int(positions_per_stream),
        "batch_size": int(batch_size),
        "report_queue_size": int(queue_capacity),
    }
    checkpoint_payload = {
        "encoder": model.encoder.state_dict(),
        "config": config,
        # Keep the established competition-only source contract so the B4
        # frozen-feature extractor can consume B5 without any special case.
        "source": SSL_SOURCE,
        "variant": B5_VARIANT,
        "report_supervision": "competition_training_reports_non_gold_only",
        "external_image_pretraining": False,
        "external_text_model": False,
        "gold_studies_used": 0,
        "initialized_from": str(Path(init_checkpoint).resolve()),
        "initialized_from_source": init_payload.get("source"),
        "initialized_from_completed_epochs": init_payload.get("completed_epochs"),
        "completed_epochs": len(history),
        "text_stats": text_stats,
        "coverage": coverage,
        "metadata_repair": metadata_stats,
        "loss_weights": {
            "image": image_weight,
            "metadata": metadata_weight,
            "report": report_weight,
            "report_cosine_within_report": cosine_weight,
        },
        "temperatures": {
            "image": image_temperature,
            "report": report_temperature,
        },
        "learning_rates": {
            "encoder": encoder_lr,
            "heads": head_lr,
        },
        "budget": budget.to_dict(),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (outdir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (outdir / "policy.json").write_text(
        json.dumps(
            {
                "candidate": "B5_image_report_representation",
                "variant": B5_VARIANT,
                "competition_only": True,
                "gold_studies_used": 0,
                "external_image_pretraining": False,
                "external_text_model": False,
                "text_encoder": "TF-IDF word 1-2 grams -> TruncatedSVD",
                "report_branch_used_at_inference": False,
                "initialization_checkpoint": str(Path(init_checkpoint).resolve()),
                "output_checkpoint": str(checkpoint_path.resolve()),
                "text_stats": text_stats,
                "coverage": coverage,
                "loss_weights": checkpoint_payload["loss_weights"],
                "temperatures": checkpoint_payload["temperatures"],
                "learning_rates": checkpoint_payload["learning_rates"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b5")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="strong competition-only SSL encoder checkpoint; defaults to config ssl_encoder_checkpoint",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="override b5_output_dir from config",
    )
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.out_root:
        config = dict(config)
        config["b5_output_dir"] = args.out_root
    path = pretrain_report_ssl(config, checkpoint=args.checkpoint)
    print(path)


if __name__ == "__main__":
    main()
