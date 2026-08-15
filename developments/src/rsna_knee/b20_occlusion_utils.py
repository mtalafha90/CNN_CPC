"""Utilities for B20 blur-occlusion sensitivity diagnostics."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def grid_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size < 1 or stride < 1:
        raise ValueError("patch size and stride must be positive")
    if patch_size > length:
        raise ValueError("patch size cannot exceed image dimension")
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def blur_tokens(tokens: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError("blur expects [N,C,H,W]")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("blur kernel must be an odd integer >= 3")
    radius = kernel_size // 2
    padded = F.pad(tokens, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def accumulate_patch_map(height: int, width: int, records: list[dict]):
    total = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float64)
    for row in records:
        y0, y1 = int(row["y0"]), int(row["y1"])
        x0, x1 = int(row["x0"]), int(row["x1"])
        total[y0:y1, x0:x1] += float(row["delta_probability"])
        coverage[y0:y1, x0:x1] += 1.0
    if (coverage == 0).any():
        raise RuntimeError("occlusion grid did not cover every pixel")
    return total / coverage, coverage


def head_from_slice_features(model, slice_features, present):
    tokens = model._pool_real_series(slice_features, present)
    padding = present <= 0
    empty = padding.all(dim=1)
    safe_padding = padding.clone()
    if empty.any():
        safe_padding[empty, 0] = False
        tokens = tokens.clone()
        tokens[empty, 0] = 0
    contextual = model.context(tokens, src_key_padding_mask=safe_padding)
    contextual = contextual.masked_fill(padding[:, :, None], 0.0)
    b = contextual.shape[0]
    queries = model.pathology_tokens[None, :, :].expand(b, -1, -1)
    queries = model.pathology_context(queries)
    attended, _ = model.cross_attention(
        queries, contextual, contextual, key_padding_mask=safe_padding, need_weights=False
    )
    queries = model.dropout(model.query_norm(queries + attended))
    logits = (queries * model.target_weight[None]).sum(dim=-1) + model.target_bias
    return torch.where(empty[:, None], model.target_bias[None], logits)


def encode_modified_tokens(model, tokens: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [model.encoder(chunk) for chunk in tokens.split(model.encoder_batch_size, dim=0)],
        dim=0,
    )


def metadata_vector(model, series_meta, series_idx: int):
    meta = series_meta[:, series_idx]
    return (
        model.plane_embedding(meta[:, 0].clamp(0, 3))
        + model.fluid_embedding(meta[:, 1].clamp(0, 2))
        + model.fat_embedding(meta[:, 2].clamp(0, 2))
    )[0]


def cached_probability(model, slice_features, present, target_idx: int) -> float:
    logits = head_from_slice_features(model, slice_features, present)
    return float(torch.sigmoid(logits.float())[0, target_idx].detach().cpu())


def positive_percentile_mask(values: np.ndarray, percentile: float = 80.0):
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if not np.any(positive > 0):
        return np.zeros_like(positive, dtype=bool)
    cutoff = float(np.percentile(positive[positive > 0], percentile))
    return positive >= cutoff


def pearson(a: np.ndarray, b: np.ndarray):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def iou(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def save_panel(path: Path, *, image, cam, occlusion, records, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.asarray(image, dtype=np.float32), 0, 1)
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0, 1)
    occ = np.asarray(occlusion, dtype=np.float64)
    positive = np.maximum(occ, 0.0)
    pmax = float(positive.max())
    pnorm = positive / pmax if pmax > 0 else positive
    limit = max(float(np.max(np.abs(occ))), 1e-8)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4.5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(cam, cmap="turbo", alpha=0.48, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(pnorm, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title("Occlusion support\npositive delta-p")
    signed = axes[3].imshow(occ, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[3].set_title("Signed occlusion delta-p")
    fig.colorbar(signed, ax=axes[3], fraction=0.046, pad=0.02)

    axes[4].imshow(image, cmap="gray", vmin=0, vmax=1)
    top = sorted(records, key=lambda r: float(r["delta_probability"]), reverse=True)[:5]
    for rank, row in enumerate(top, start=1):
        x0, y0 = int(row["x0"]), int(row["y0"])
        width, height = int(row["x1"]) - x0, int(row["y1"]) - y0
        axes[4].add_patch(plt.Rectangle((x0, y0), width, height, fill=False, linewidth=1.5))
        axes[4].text(x0 + 2, y0 + 10, str(rank), fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.7, "pad": 1})
    axes[4].set_title("Top 5 causal patches")

    axes[5].imshow(image, cmap="gray", vmin=0, vmax=1)
    cam_mask = positive_percentile_mask(cam, 80)
    occ_mask = positive_percentile_mask(positive, 80)
    if cam_mask.any() and (~cam_mask).any():
        axes[5].contour(cam_mask, levels=[0.5], linewidths=1.2)
    if occ_mask.any() and (~occ_mask).any():
        axes[5].contour(occ_mask, levels=[0.5], linewidths=1.2)
    axes[5].set_title("Top-20% regions\nCAM + causal support")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
