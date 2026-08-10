"""B12 variable-number-of-series MRI representation.

B7.1 compresses every knee into six semantic stream slots. B12 instead keeps
all acquired series with a repaired sagittal/coronal/axial plane, represents
each real series by its own slice tokens, and pads only within a mini-batch.
There is no learned acquisition-rank/slot embedding: series are identified only
by image content plus coarse observed metadata (plane, fluid sensitivity and
fat suppression), so repeated acquisitions remain separate without being
forced into false semantic slots.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import N_TARGETS
from .data import PLANES, build_series_index
from .dataset import KneeStudyDataset
from .model import ConvNeXtSliceEncoder

PLANE_TO_ID = {"Sagittal": 1, "Coronal": 2, "Axial": 3}
B12_SERIES_POLICY = "all_repaired_anatomical_series_v1"


def _flag_id(value) -> int:
    """0=unknown, 1=false, 2=true."""
    if pd.isna(value):
        return 0
    return 2 if bool(value) else 1


def build_variable_series_index(
    series_df: pd.DataFrame,
    studies: Iterable[str],
) -> dict[str, list[dict]]:
    """Return every repaired series with a recognized anatomical plane.

    No fluid/structural winner is selected. Ordering is deterministic only for
    reproducibility; the B12 model has no series-position embedding.
    """
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    work["SeriesInstanceUID"] = work["SeriesInstanceUID"].astype(str)
    grouped = {uid: part for uid, part in work.groupby("StudyInstanceUID", sort=False)}
    empty = work.iloc[0:0]
    result: dict[str, list[dict]] = {}
    for study in studies:
        uid = str(study)
        records: list[dict] = []
        part = grouped.get(uid, empty)
        for _, row in part.iterrows():
            plane = str(row.get("Anatomical_Plane", ""))
            plane_id = PLANE_TO_ID.get(plane, 0)
            if plane_id == 0:
                continue
            records.append(
                {
                    "series_uid": str(row["SeriesInstanceUID"]),
                    "plane": plane,
                    "plane_id": int(plane_id),
                    "fluid_id": _flag_id(row.get("Fluid_Sensitive")),
                    "fat_id": _flag_id(row.get("Fat_Suppression")),
                }
            )
        records.sort(
            key=lambda x: (
                int(x["plane_id"]),
                int(x["fluid_id"]),
                int(x["fat_id"]),
                str(x["series_uid"]),
            )
        )
        result[uid] = records
    return result


def variable_series_signature(index: dict[str, list[dict]], studies: Iterable[str]) -> str:
    rows = []
    for uid in studies:
        key = str(uid)
        rows.append(
            {
                "StudyInstanceUID": key,
                "series": [
                    {
                        "series_uid": str(r["series_uid"]),
                        "plane_id": int(r["plane_id"]),
                        "fluid_id": int(r["fluid_id"]),
                        "fat_id": int(r["fat_id"]),
                    }
                    for r in index.get(key, [])
                ],
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_variable_series_surface(
    series_df: pd.DataFrame,
    studies: Iterable[str],
) -> tuple[dict, dict[str, list[dict]]]:
    uids = [str(x) for x in studies]
    variable = build_variable_series_index(series_df, uids)
    legacy = build_series_index(series_df, uids, mode="dual")

    counts = np.asarray([len(variable[uid]) for uid in uids], dtype=np.int64)
    legacy_counts = []
    missing_legacy = 0
    extra_counts = []
    for uid in uids:
        legacy_set = {str(v) for v in legacy[uid].values() if v}
        variable_set = {str(r["series_uid"]) for r in variable[uid]}
        missing_legacy += len(legacy_set.difference(variable_set))
        legacy_counts.append(len(legacy_set))
        extra_counts.append(len(variable_set.difference(legacy_set)))
    legacy_counts_arr = np.asarray(legacy_counts, dtype=np.int64)
    extra_arr = np.asarray(extra_counts, dtype=np.int64)

    study_set = set(uids)
    relevant = series_df.loc[series_df["StudyInstanceUID"].astype(str).isin(study_set)]
    recognized = relevant["Anatomical_Plane"].isin(PLANES)
    unknown_plane = int((~recognized).sum())
    eligible_total = int(counts.sum())
    legacy_total = int(legacy_counts_arr.sum())
    extra_total = int(extra_arr.sum())
    extra_fraction = float(extra_total / max(legacy_total, 1))
    study_extra_fraction = float((extra_arr > 0).mean()) if len(extra_arr) else 0.0

    quantile_levels = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    q = np.quantile(counts, quantile_levels).tolist() if len(counts) else [0.0] * len(quantile_levels)
    summary = {
        "policy": B12_SERIES_POLICY,
        "studies": int(len(uids)),
        "series_rows_for_studies": int(len(relevant)),
        "eligible_recognized_plane_series": eligible_total,
        "excluded_unknown_plane_series": unknown_plane,
        "historical_dual_unique_series": legacy_total,
        "extra_series_retained": extra_total,
        "extra_series_fraction_vs_historical_unique": extra_fraction,
        "studies_with_extra_series": int((extra_arr > 0).sum()),
        "studies_with_extra_series_fraction": study_extra_fraction,
        "studies_with_zero_eligible_series": int((counts == 0).sum()),
        "historical_selected_series_missing_from_b12": int(missing_legacy),
        "series_per_study": {
            "min": int(counts.min()) if len(counts) else 0,
            "mean": float(counts.mean()) if len(counts) else 0.0,
            "q25": float(q[1]),
            "median": float(q[2]),
            "q75": float(q[3]),
            "q90": float(q[4]),
            "q95": float(q[5]),
            "q99": float(q[6]),
            "max": int(counts.max()) if len(counts) else 0,
        },
    }
    summary["viability_passed"] = bool(
        summary["studies_with_zero_eligible_series"] == 0
        and summary["historical_selected_series_missing_from_b12"] == 0
        and extra_fraction >= 0.05
        and study_extra_fraction >= 0.10
    )
    summary["viability_requirements"] = {
        "zero_studies_without_eligible_series": True,
        "zero_historical_selected_series_missing": True,
        "minimum_extra_series_fraction_vs_historical_unique": 0.05,
        "minimum_fraction_studies_with_extra_series": 0.10,
    }
    summary["series_signature_sha256"] = variable_series_signature(variable, uids)
    return summary, variable


class VariableSeriesKneeDataset(KneeStudyDataset):
    """Knee dataset returning the actual eligible series count for each study."""

    def __init__(self, study_uids, series_records, config, targets=None, weights=None, train=False):
        super().__init__(study_uids, {}, config, targets=targets, weights=weights, train=train)
        if config.physical_scale_policy is not None:
            raise ValueError("B12-v1 freezes legacy resize; physical-scale normalization is disabled")
        self.series_records = series_records
        missing = [uid for uid in self.study_uids if not self.series_records.get(uid)]
        if missing:
            raise ValueError(f"B12 has {len(missing)} study/studies with zero eligible series")

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        records = self.series_records[uid]
        volumes, present, meta = [], [], []
        for record in records:
            volume, flag = self._load(uid, record["series_uid"], "series")
            volumes.append(volume)
            present.append(flag)
            meta.append([record["plane_id"], record["fluid_id"], record["fat_id"]])
        stacked = torch.stack(volumes)
        if self.config.tta_center_offsets:
            # [K,V,S,C,H,W] -> [V,K,S,C,H,W]
            stacked = stacked.permute(1, 0, 2, 3, 4, 5).contiguous()
        item = {
            "study_uid": uid,
            "volumes": stacked,
            "present": torch.tensor(present, dtype=torch.float32),
            "series_meta": torch.tensor(meta, dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        if self.weights is not None:
            item["weight"] = torch.from_numpy(np.asarray(self.weights[idx], dtype=np.float32))
        return item


def collate_variable_series(batch: list[dict]) -> dict:
    """Pad only to the largest real series count in the current mini-batch."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_k = max(int(item["present"].shape[0]) for item in batch)
    first = batch[0]["volumes"]
    b = len(batch)
    if first.ndim == 5:  # [K,S,C,H,W]
        _, s, c, h, w = first.shape
        volumes = first.new_zeros((b, max_k, s, c, h, w))
        for i, item in enumerate(batch):
            k = item["volumes"].shape[0]
            volumes[i, :k] = item["volumes"]
    elif first.ndim == 6:  # [V,K,S,C,H,W]
        v, _, s, c, h, w = first.shape
        volumes = first.new_zeros((b, v, max_k, s, c, h, w))
        for i, item in enumerate(batch):
            if item["volumes"].shape[0] != v:
                raise ValueError("B12 batch contains inconsistent TTA view counts")
            k = item["volumes"].shape[1]
            volumes[i, :, :k] = item["volumes"]
    else:
        raise ValueError(f"unexpected B12 item volume shape {tuple(first.shape)}")

    present = torch.zeros((b, max_k), dtype=torch.float32)
    meta = torch.zeros((b, max_k, 3), dtype=torch.long)
    for i, item in enumerate(batch):
        k = item["present"].shape[0]
        present[i, :k] = item["present"]
        meta[i, :k] = item["series_meta"]

    out = {
        "study_uid": [str(item["study_uid"]) for item in batch],
        "volumes": volumes,
        "present": present,
        "series_meta": meta,
    }
    if all("target" in item for item in batch):
        out["target"] = torch.stack([item["target"] for item in batch])
    if all("weight" in item for item in batch):
        out["weight"] = torch.stack([item["weight"] for item in batch])
    return out


class VariableSeriesKneeMILNet(nn.Module):
    """B7.1 pathology-query model over a variable number of real MRI series."""

    def __init__(
        self,
        n_slices: int,
        *,
        in_channels: int = 3,
        pretrained_weights: bool = False,
        normalize_input: bool = False,
        dropout: float = 0.25,
        encoder_batch_size: int = 24,
        gradient_checkpointing: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_mult: float = 2.0,
        pathology_layers: int = 1,
    ) -> None:
        super().__init__()
        if n_slices < 1 or encoder_batch_size < 1:
            raise ValueError("n_slices and encoder_batch_size must be positive")
        self.n_slices = int(n_slices)
        self.in_channels = int(in_channels)
        self.encoder_batch_size = int(encoder_batch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder = ConvNeXtSliceEncoder(
            in_channels,
            pretrained_weights=pretrained_weights,
            normalize_input=normalize_input,
        )
        d = self.encoder.out_dim
        if d % int(transformer_heads) != 0:
            raise ValueError("transformer_heads must divide encoder feature dimension")

        self.slice_position = nn.Parameter(torch.randn(self.n_slices, d) * 0.02)
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)

        mri_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(transformer_heads),
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(mri_layer, num_layers=int(transformer_layers), norm=nn.LayerNorm(d))
        self.pathology_tokens = nn.Parameter(torch.randn(N_TARGETS, d) * 0.02)
        pathology_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(transformer_heads),
            dim_feedforward=int(d * float(transformer_ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pathology_context = nn.TransformerEncoder(
            pathology_layer, num_layers=int(pathology_layers), norm=nn.LayerNorm(d)
        )
        self.cross_attention = nn.MultiheadAttention(
            d, int(transformer_heads), dropout=float(dropout), batch_first=True
        )
        self.query_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.target_weight = nn.Parameter(torch.empty(N_TARGETS, d))
        self.target_bias = nn.Parameter(torch.zeros(N_TARGETS))
        nn.init.xavier_uniform_(self.target_weight)

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _encode_slices(self, volumes: torch.Tensor, present: torch.Tensor, series_meta: torch.Tensor) -> torch.Tensor:
        if volumes.ndim != 6:
            raise ValueError("B12 model expects [B,K,S,3,H,W]")
        b, k, s, c, h, w = volumes.shape
        if s != self.n_slices or c != self.in_channels:
            raise ValueError("B12 slice/channel contract mismatch")
        if present.shape != (b, k):
            raise ValueError("B12 present mask shape mismatch")
        if series_meta.shape != (b, k, 3):
            raise ValueError("B12 series_meta must have shape [B,K,3]")

        flat_series = volumes.reshape(b * k, s, c, h, w)
        active_indices = torch.nonzero(present.reshape(-1) > 0, as_tuple=False).flatten()
        d = self.encoder.out_dim
        if active_indices.numel() == 0:
            return volumes.new_zeros((b, k, s, d))
        active = flat_series.index_select(0, active_indices)
        flat = active.reshape(-1, c, h, w)
        encoded = torch.cat(
            [self._encode_chunk(chunk) for chunk in flat.split(self.encoder_batch_size, dim=0)],
            dim=0,
        ).reshape(active.shape[0], s, d)
        all_features = encoded.new_zeros((b * k, s, d)).index_copy(0, active_indices, encoded)
        features = all_features.reshape(b, k, s, d)

        plane = self.plane_embedding(series_meta[:, :, 0].clamp(0, 3))
        fluid = self.fluid_embedding(series_meta[:, :, 1].clamp(0, 2))
        fat = self.fat_embedding(series_meta[:, :, 2].clamp(0, 2))
        metadata = plane + fluid + fat
        mask = present[:, :, None, None].to(features.dtype)
        return (
            features
            + self.slice_position[None, None, :, :]
            + metadata[:, :, None, :]
        ) * mask

    def _mri_tokens(self, volumes, present, series_meta):
        features = self._encode_slices(volumes, present, series_meta)
        b, k, s, d = features.shape
        tokens = features.reshape(b, k * s, d)
        padding = (present <= 0)[:, :, None].expand(b, k, s).reshape(b, k * s)
        empty = padding.all(dim=1)
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        contextual = self.context(tokens, src_key_padding_mask=safe_padding)
        contextual = contextual.masked_fill(padding[:, :, None], 0.0)
        return contextual, safe_padding, empty

    def forward(self, volumes, present, series_meta):
        memory, padding, empty = self._mri_tokens(volumes, present, series_meta)
        b = memory.shape[0]
        queries = self.pathology_tokens[None, :, :].expand(b, -1, -1)
        queries = self.pathology_context(queries)
        attended, _ = self.cross_attention(
            queries, memory, memory, key_padding_mask=padding, need_weights=False
        )
        queries = self.dropout(self.query_norm(queries + attended))
        logits = (queries * self.target_weight[None, :, :]).sum(dim=-1) + self.target_bias
        return torch.where(empty[:, None], self.target_bias[None, :], logits)


def b12_model_spec(config: dict, *, normalize_input: bool) -> dict:
    return {
        "architecture": "variable_series_pathology_queries_v1",
        "series_policy": B12_SERIES_POLICY,
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
        "metadata_embeddings": "plane/fluid/fat categorical; no series-position embedding",
    }


def build_b12_model(spec: dict, *, encoder_state: dict | None = None) -> VariableSeriesKneeMILNet:
    if spec.get("architecture") != "variable_series_pathology_queries_v1":
        raise ValueError("not a B12 variable-series model spec")
    model = VariableSeriesKneeMILNet(
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
