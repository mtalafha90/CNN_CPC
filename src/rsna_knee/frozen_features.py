"""B4: frozen in-domain SSL features with nested low-capacity classifiers.

This module deliberately separates representation learning from gold-label
classification. The ConvNeXt encoder is loaded from the competition-only SSL
checkpoint, frozen, and used once to cache deterministic per-study features.
All PCA and logistic-regression fitting is then performed inside the existing
nested gold folds so outer OOF labels never influence model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from .constants import DUAL_STREAMS, TARGETS
from .data import (
    backfill_series_metadata,
    build_series_index,
    gold_mask,
    load_series_csv,
    load_test_csv,
    load_train_csv,
    make_balanced_gold_folds,
)
from .dataset import DatasetConfig, KneeStudyDataset
from .evaluation import bootstrap_macro_auc, fast_auc, macro_auc_from_arrays
from .model import ConvNeXtSliceEncoder
from .runtime import autocast, resolve_runtime
from .ssl import SSL_SOURCE


# Predeclared target-specific subsets. These are hypotheses based on anatomy and
# sequence sensitivity, not learned from OOF performance. The alternative
# feature mode always exposes all six streams.
TARGET_STREAM_SUBSETS: dict[str, tuple[str, ...]] = {
    "ACL": ("sagittal_fluid", "sagittal_structural"),
    "MCL": ("coronal_fluid", "coronal_structural"),
    "Medial Meniscus": (
        "sagittal_fluid", "sagittal_structural", "coronal_fluid", "coronal_structural"
    ),
    "Lateral Meniscus": (
        "sagittal_fluid", "sagittal_structural", "coronal_fluid", "coronal_structural"
    ),
    "Medial OA": (
        "sagittal_structural", "coronal_fluid", "coronal_structural", "axial_structural"
    ),
    "Lateral OA": (
        "sagittal_structural", "coronal_fluid", "coronal_structural", "axial_structural"
    ),
    "PF OA": ("axial_fluid", "axial_structural"),
    "Effusion": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Synovitis": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Baker's": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Contusion": ("sagittal_fluid", "coronal_fluid", "axial_fluid"),
    "Fracture": ("sagittal_structural", "coronal_structural", "axial_structural"),
}

DEFAULT_PCA_COMPONENTS = (4, 8, 12, 16)
DEFAULT_C_VALUES = (0.1, 1.0)
DEFAULT_FEATURE_MODES = ("all", "prior")
POOL_NAMES = ("mean", "std", "max")


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _load_frozen_encoder(checkpoint: str | Path, device: torch.device):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"SSL checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("source")
    if source != SSL_SOURCE:
        raise ValueError(
            f"B4 requires a competition-only SSL checkpoint with source={SSL_SOURCE!r}; got {source!r}"
        )
    ssl_config = payload.get("config", {})
    encoder = ConvNeXtSliceEncoder(
        3,
        pretrained_weights=False,
        normalize_input=bool(ssl_config.get("normalize_input", False)),
    )
    state = payload.get("encoder")
    if not isinstance(state, dict):
        raise ValueError("SSL checkpoint does not contain an encoder state_dict")
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(device), payload


@torch.no_grad()
def _encode_batch(
    encoder: ConvNeXtSliceEncoder,
    volumes: torch.Tensor,
    present: torch.Tensor,
    runtime,
    *,
    encoder_batch_size: int,
) -> torch.Tensor:
    """Return pooled features shaped [B, K, 3*D] on CPU."""
    if volumes.ndim != 6:
        raise ValueError(f"expected [B,K,S,C,H,W], got {tuple(volumes.shape)}")
    b, k, s, c, h, w = volumes.shape
    if present.shape != (b, k):
        raise ValueError(f"present mask shape {tuple(present.shape)} != {(b, k)}")
    if k != len(DUAL_STREAMS) or c != 3:
        raise ValueError("B4 feature extraction requires the six-stream 3-channel contract")
    if encoder_batch_size < 1:
        raise ValueError("encoder_batch_size must be >=1")

    flat = volumes.reshape(b * k * s, c, h, w)
    active = (
        present.to(dtype=torch.bool)
        .unsqueeze(-1)
        .expand(b, k, s)
        .reshape(-1)
    )
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    d = int(encoder.out_dim)
    encoded = torch.zeros((b * k * s, d), dtype=torch.float32)

    for index_chunk in active_indices.split(int(encoder_batch_size)):
        x = flat.index_select(0, index_chunk).to(runtime.device, non_blocking=True)
        with autocast(runtime):
            z = encoder(x)
        encoded.index_copy_(0, index_chunk.cpu(), z.float().cpu())

    z = encoded.reshape(b, k, s, d)
    mean = z.mean(dim=2)
    std = z.std(dim=2, unbiased=False)
    maximum = z.max(dim=2).values
    pooled = torch.cat([mean, std, maximum], dim=-1)
    pooled = pooled * present.cpu().to(dtype=pooled.dtype).unsqueeze(-1)
    return pooled


def extract_feature_cache(
    config: dict,
    *,
    out_path: str | Path,
    split: str = "train",
    scope: str = "gold",
    checkpoint: str | Path | None = None,
) -> Path:
    """Cache deterministic frozen-SSL study features.

    For the initial B4 OOF experiment use ``split=train, scope=gold``. Test/all
    extraction is supported so the same representation can later be used for
    submission inference without changing the feature contract.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if scope not in {"gold", "all"}:
        raise ValueError("scope must be gold or all")
    if split == "test" and scope == "gold":
        raise ValueError("test split has no gold scope")

    runtime = resolve_runtime(config)
    root = Path(config["data_root"])
    checkpoint = checkpoint or config.get("ssl_encoder_checkpoint")
    if not checkpoint:
        raise ValueError("B4 requires ssl_encoder_checkpoint or --checkpoint")
    encoder, ssl_payload = _load_frozen_encoder(checkpoint, runtime.device)

    if split == "train":
        frame = load_train_csv(root / config.get("train_csv", "train.csv"))
        if scope == "gold":
            frame = frame.loc[gold_mask(frame)].copy()
        series_name = config.get("train_series_csv", "train_series.csv")
    else:
        frame = load_test_csv(root / config.get("test_csv", "test.csv"))
        series_name = config.get("test_series_csv", "test_series.csv")

    uids = frame["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("no studies selected for B4 feature extraction")

    series = load_series_csv(root / series_name)
    series, metadata_stats = backfill_series_metadata(series, root, split=split)
    index = build_series_index(series, uids, mode="dual")

    dataset = KneeStudyDataset(
        uids,
        index,
        DatasetConfig(
            data_root=str(root),
            split=split,
            n_slices=int(config.get("b4_n_slices", config.get("n_slices", 16))),
            image_size=int(config.get("image_size", 224)),
            noise_std=0.0,
            slice_dropout=0.0,
            triplet_gap=int(config.get("triplet_gap", 1)),
            strict_dicom=bool(config.get("strict_dicom", False)),
            center_jitter=0,
            center_offset=0,
            rotation_deg=0.0,
            translate_frac=0.0,
            scale_jitter=0.0,
            gamma_jitter=0.0,
            bias_field_strength=0.0,
            series_cache_mb=int(config.get("series_cache_mb_per_worker", 256)),
        ),
        train=False,
    )
    batch_size = max(1, int(config.get("b4_feature_batch_size", 1)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 940_000),
    )
    encoder_batch_size = max(1, int(config.get("b4_encoder_batch_size", 48)))

    feature_rows: list[np.ndarray] = []
    present_rows: list[np.ndarray] = []
    output_uids: list[str] = []
    for batch_index, batch in enumerate(loader):
        pooled = _encode_batch(
            encoder,
            batch["volumes"],
            batch["present"],
            runtime,
            encoder_batch_size=encoder_batch_size,
        )
        feature_rows.append(pooled.numpy().astype(np.float32, copy=False))
        present_rows.append(batch["present"].numpy().astype(np.float32, copy=False))
        output_uids.extend(str(x) for x in batch["study_uid"])
        print(
            {
                "phase": "b4_extract",
                "batch": batch_index + 1,
                "studies_done": len(output_uids),
                "studies_total": len(dataset),
            }
        )

    features = np.concatenate(feature_rows, axis=0)
    present = np.concatenate(present_rows, axis=0)
    if features.shape[:2] != (len(output_uids), len(DUAL_STREAMS)):
        raise RuntimeError("B4 extracted feature shape does not match study/stream contract")
    if not np.isfinite(features).all() or not np.isfinite(present).all():
        raise RuntimeError("B4 feature cache contains non-finite values")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        study_uids=np.asarray(output_uids, dtype=str),
        features=features,
        present=present,
        stream_names=np.asarray(DUAL_STREAMS, dtype=str),
        pool_names=np.asarray(POOL_NAMES, dtype=str),
    )
    policy = {
        "candidate": "B4_frozen_ssl_classical",
        "split": split,
        "scope": scope,
        "studies": int(len(output_uids)),
        "feature_shape": list(features.shape),
        "stream_order": list(DUAL_STREAMS),
        "pooling": list(POOL_NAMES),
        "encoder_frozen": True,
        "encoder_trainable_parameters": 0,
        "ssl_encoder_checkpoint": str(Path(checkpoint).resolve()),
        "ssl_checkpoint_source": ssl_payload.get("source"),
        "ssl_completed_epochs": ssl_payload.get("completed_epochs"),
        "external_pretrained": False,
        "n_slices": int(dataset.config.n_slices),
        "image_size": int(dataset.config.image_size),
        "triplet_gap": int(dataset.config.triplet_gap),
        "metadata_repair": metadata_stats,
    }
    out.with_suffix(".json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(json.dumps(policy, indent=2))
    return out


def load_feature_cache(path: str | Path):
    with np.load(path, allow_pickle=False) as payload:
        uids = payload["study_uids"].astype(str)
        features = payload["features"].astype(np.float64)
        present = payload["present"].astype(np.float64)
        streams = payload["stream_names"].astype(str).tolist()
    if streams != list(DUAL_STREAMS):
        raise ValueError(f"feature stream order mismatch: {streams}")
    if features.ndim != 3 or features.shape[:2] != (len(uids), len(DUAL_STREAMS)):
        raise ValueError(f"invalid B4 feature shape: {features.shape}")
    if present.shape != (len(uids), len(DUAL_STREAMS)):
        raise ValueError(f"invalid B4 present shape: {present.shape}")
    if len(set(uids.tolist())) != len(uids):
        raise ValueError("B4 feature cache contains duplicate study UIDs")
    if not np.isfinite(features).all() or not np.isfinite(present).all():
        raise ValueError("B4 feature cache contains non-finite values")
    return uids, features, present


def _stream_indices(target: str, mode: str) -> list[int]:
    if mode == "all":
        return list(range(len(DUAL_STREAMS)))
    if mode != "prior":
        raise ValueError(f"unknown B4 feature mode: {mode}")
    names = TARGET_STREAM_SUBSETS[target]
    return [DUAL_STREAMS.index(name) for name in names]


def target_design_matrix(
    features: np.ndarray,
    present: np.ndarray,
    target: str,
    mode: str,
) -> np.ndarray:
    indices = _stream_indices(target, mode)
    x = np.asarray(features[:, indices, :], dtype=np.float64).reshape(len(features), -1)
    # Explicit missing-stream flags allow the low-capacity classifier to
    # distinguish an absent series from a true near-zero SSL representation.
    p = np.asarray(present[:, indices], dtype=np.float64)
    return np.concatenate([x, p], axis=1)


def _constant_probability(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    value = float(y.mean()) if y.size else 0.5
    return np.full(int(n), value, dtype=np.float64)


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    n_components: int,
    c_value: float,
    seed: int,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.float64)
    classes = np.unique(y_train[np.isfinite(y_train)])
    if len(classes) < 2:
        return _constant_probability(y_train, len(x_eval))
    maximum_components = min(int(x_train.shape[0]) - 1, int(x_train.shape[1]))
    if maximum_components < 1:
        return _constant_probability(y_train, len(x_eval))
    actual_components = min(int(n_components), maximum_components)
    estimator = Pipeline(
        [
            (
                "pca",
                PCA(
                    n_components=actual_components,
                    whiten=True,
                    svd_solver="full",
                ),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    estimator.fit(x_train, y_train.astype(int))
    return estimator.predict_proba(x_eval)[:, 1].astype(np.float64)


def _candidate_grid(
    pca_components: Iterable[int],
    c_values: Iterable[float],
    feature_modes: Iterable[str],
):
    for mode in feature_modes:
        if mode not in {"all", "prior"}:
            raise ValueError(f"unsupported feature mode: {mode}")
        for n_components in pca_components:
            if int(n_components) < 1:
                raise ValueError("PCA component counts must be >=1")
            for c_value in c_values:
                if float(c_value) <= 0:
                    raise ValueError("logistic C values must be >0")
                yield str(mode), int(n_components), float(c_value)


def nested_classical_oof(
    config: dict,
    *,
    feature_path: str | Path,
    out_root: str | Path,
    pca_components: Iterable[int] = DEFAULT_PCA_COMPONENTS,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    feature_modes: Iterable[str] = DEFAULT_FEATURE_MODES,
    n_bootstrap: int = 5000,
) -> dict:
    """Run target-wise nested B4 selection and untouched outer OOF scoring."""
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train)].copy()
    if gold[TARGETS].isna().any().any():
        raise ValueError("B4 initial experiment requires fully labelled gold studies")

    cache_uids, cache_features, cache_present = load_feature_cache(feature_path)
    cache_index = {uid: i for i, uid in enumerate(cache_uids.tolist())}
    missing = [uid for uid in gold["StudyInstanceUID"].astype(str) if uid not in cache_index]
    if missing:
        raise ValueError(f"B4 feature cache is missing {len(missing)} gold studies")
    order = np.asarray([cache_index[str(uid)] for uid in gold["StudyInstanceUID"]], dtype=int)
    features = cache_features[order]
    present = cache_present[order]
    y = gold[TARGETS].to_numpy(np.float64)

    seed = int(config.get("seed", 2026))
    n_folds = int(config.get("n_folds", 3))
    if n_folds < 3:
        raise ValueError("B4 nested OOF requires at least three folds")
    folds_full = make_balanced_gold_folds(train, n_splits=n_folds, seed=seed)
    fold_ids = folds_full.loc[gold.index].to_numpy(dtype=int)
    candidates = list(_candidate_grid(pca_components, c_values, feature_modes))
    if not candidates:
        raise ValueError("B4 candidate grid is empty")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    full_oof = np.full_like(y, np.nan, dtype=np.float64)
    fold_payloads: dict[str, dict] = {}

    for outer_fold in range(n_folds):
        inner_fold = int(config.get("inner_selection_fold", (outer_fold + 1) % n_folds))
        if inner_fold == outer_fold:
            raise ValueError("B4 inner fold must differ from outer fold")
        selection_train = (fold_ids != outer_fold) & (fold_ids != inner_fold)
        inner = fold_ids == inner_fold
        final_train = fold_ids != outer_fold
        outer = fold_ids == outer_fold
        if not selection_train.any() or not inner.any() or not outer.any():
            raise ValueError(f"empty nested partition for B4 outer fold {outer_fold}")

        outer_pred = np.zeros((int(outer.sum()), len(TARGETS)), dtype=np.float64)
        target_policy: dict[str, dict] = {}
        selected_inner_scores = []

        for j, target in enumerate(TARGETS):
            best = None
            for candidate_index, (mode, n_components, c_value) in enumerate(candidates):
                x = target_design_matrix(features, present, target, mode)
                pred = _fit_predict(
                    x[selection_train],
                    y[selection_train, j],
                    x[inner],
                    n_components=n_components,
                    c_value=c_value,
                    seed=seed + 10_000 * outer_fold + 100 * j + candidate_index,
                )
                score = fast_auc(y[inner, j], pred)
                if not np.isfinite(score):
                    continue
                row = {
                    "feature_mode": mode,
                    "pca_components": int(n_components),
                    "C": float(c_value),
                    "inner_auc": float(score),
                    "candidate_index": int(candidate_index),
                }
                if best is None or score > best[0] + 1e-12:
                    best = (float(score), row)

            if best is None:
                raise RuntimeError(f"no finite B4 inner AUC for target {target}, outer fold {outer_fold}")

            policy = best[1]
            x = target_design_matrix(features, present, target, policy["feature_mode"])
            outer_pred[:, j] = _fit_predict(
                x[final_train],
                y[final_train, j],
                x[outer],
                n_components=int(policy["pca_components"]),
                c_value=float(policy["C"]),
                seed=seed + 100_000 + 10_000 * outer_fold + j,
            )
            target_policy[target] = policy
            selected_inner_scores.append(float(policy["inner_auc"]))

        full_oof[outer] = outer_pred
        outer_score, outer_per_target = macro_auc_from_arrays(y[outer], outer_pred)
        fold_dir = out_root / f"fold{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_frame = pd.DataFrame(outer_pred, columns=TARGETS)
        fold_frame.insert(0, "StudyInstanceUID", gold.loc[outer, "StudyInstanceUID"].astype(str).to_numpy())
        fold_frame.to_csv(fold_dir / "oof.csv", index=False)

        bootstrap = bootstrap_macro_auc(
            y[outer],
            outer_pred,
            n_bootstrap=int(config.get("n_bootstrap", 2000)),
            seed=seed + outer_fold,
        )
        (fold_dir / "bootstrap.json").write_text(
            json.dumps(bootstrap.to_dict(), indent=2), encoding="utf-8"
        )
        selection_payload = {
            "candidate": "B4_frozen_ssl_classical",
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "selection_gold_train": int(selection_train.sum()),
            "inner_gold": int(inner.sum()),
            "final_gold_train": int(final_train.sum()),
            "outer_gold": int(outer.sum()),
            "inner_macro_auc_from_targetwise_selection": float(np.mean(selected_inner_scores)),
            "outer_macro_auc": float(outer_score),
            "outer_per_target_auc": {
                target: float(outer_per_target[j]) for j, target in enumerate(TARGETS)
            },
            "targets": target_policy,
        }
        (fold_dir / "selection.json").write_text(
            json.dumps(selection_payload, indent=2), encoding="utf-8"
        )
        fold_payloads[str(outer_fold)] = selection_payload
        print(
            {
                "phase": "b4_nested",
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "inner_macro_auc": selection_payload["inner_macro_auc_from_targetwise_selection"],
                "outer_macro_auc": float(outer_score),
            }
        )

    if not np.isfinite(full_oof).all():
        raise RuntimeError("B4 OOF matrix was not completely populated")
    combined = pd.DataFrame(full_oof, columns=TARGETS)
    combined.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].astype(str).to_numpy())
    combined.to_csv(out_root / "oof.csv", index=False)

    pooled = bootstrap_macro_auc(y, full_oof, n_bootstrap=int(n_bootstrap), seed=seed)
    evaluation = pooled.to_dict()
    (out_root / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    policy = {
        "candidate": "B4_frozen_ssl_classical",
        "encoder_frozen": True,
        "gold_labels_used_for_encoder": False,
        "feature_cache": str(Path(feature_path).resolve()),
        "pca_fit_scope": "nested_training_partition_only",
        "classifier_fit_scope": "nested_training_partition_only",
        "outer_labels_used_for_selection": False,
        "feature_modes": list(feature_modes),
        "pca_components": [int(x) for x in pca_components],
        "C_values": [float(x) for x in c_values],
        "target_stream_subsets": {k: list(v) for k, v in TARGET_STREAM_SUBSETS.items()},
        "folds": fold_payloads,
        "pooled_evaluation": evaluation,
    }
    (out_root / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(pooled.summary())
    return policy


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b4")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="cache deterministic frozen SSL study features")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--scope", choices=["gold", "all"], default="gold")
    p.add_argument("--checkpoint", default=None)

    p = sub.add_parser("nested", help="run leakage-safe nested PCA/logistic OOF")
    p.add_argument("--config", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out-root", default="runs/b4_frozen_ssl")
    p.add_argument("--pca-components", type=int, nargs="+", default=list(DEFAULT_PCA_COMPONENTS))
    p.add_argument("--C", dest="c_values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    p.add_argument("--feature-modes", nargs="+", default=list(DEFAULT_FEATURE_MODES))
    p.add_argument("--n-bootstrap", type=int, default=5000)

    args = parser.parse_args()
    config = _read_config(args.config)
    if args.command == "extract":
        print(
            extract_feature_cache(
                config,
                out_path=args.out,
                split=args.split,
                scope=args.scope,
                checkpoint=args.checkpoint,
            )
        )
        return

    nested_classical_oof(
        config,
        feature_path=args.features,
        out_root=args.out_root,
        pca_components=args.pca_components,
        c_values=args.c_values,
        feature_modes=args.feature_modes,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
