"""B4.1: shared-policy nested classical classifiers on frozen SSL features.

Compared with B4 target-wise selection, B4.1 selects one common
(feature mode, PCA dimension, logistic C) tuple per outer fold using only the
inner gold fold. The selected shared policy is then refit independently for the
12 targets on all non-outer gold studies and evaluated on the untouched outer
fold. This reduces selection variance from 12 independent hyperparameter
searches to one fold-level search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from .constants import TARGETS
from .data import gold_mask, load_train_csv, make_balanced_gold_folds
from .evaluation import bootstrap_macro_auc, macro_auc_from_arrays
from .frozen_features import (
    DEFAULT_C_VALUES,
    DEFAULT_FEATURE_MODES,
    DEFAULT_PCA_COMPONENTS,
    _candidate_grid,
    _fit_predict,
    load_feature_cache,
    target_design_matrix,
)


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _align_gold_features(config: dict, feature_path: str | Path):
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train)].copy()
    if gold[TARGETS].isna().any().any():
        raise ValueError("B4.1 requires fully labelled gold studies")

    cache_uids, cache_features, cache_present = load_feature_cache(feature_path)
    cache_index = {uid: i for i, uid in enumerate(cache_uids.tolist())}
    missing = [uid for uid in gold["StudyInstanceUID"].astype(str) if uid not in cache_index]
    if missing:
        raise ValueError(f"feature cache is missing {len(missing)} gold studies")
    order = np.asarray([cache_index[str(uid)] for uid in gold["StudyInstanceUID"]], dtype=int)
    return train, gold, cache_features[order], cache_present[order], gold[TARGETS].to_numpy(np.float64)


def _predict_candidate(
    features: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    mode: str,
    n_components: int,
    c_value: float,
    seed: int,
) -> np.ndarray:
    pred = np.zeros((int(eval_mask.sum()), len(TARGETS)), dtype=np.float64)
    for j, target in enumerate(TARGETS):
        x = target_design_matrix(features, present, target, mode)
        pred[:, j] = _fit_predict(
            x[train_mask],
            y[train_mask, j],
            x[eval_mask],
            n_components=int(n_components),
            c_value=float(c_value),
            seed=int(seed) + j,
        )
    return pred


def select_shared_candidate(
    features: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    selection_train: np.ndarray,
    inner: np.ndarray,
    *,
    candidates,
    seed: int,
) -> tuple[dict, np.ndarray]:
    """Choose one common candidate by inner macro AUC only."""
    best = None
    for candidate_index, (mode, n_components, c_value) in enumerate(candidates):
        inner_pred = _predict_candidate(
            features,
            present,
            y,
            selection_train,
            inner,
            mode=mode,
            n_components=n_components,
            c_value=c_value,
            seed=seed + 1000 * candidate_index,
        )
        macro, per_target = macro_auc_from_arrays(y[inner], inner_pred)
        if not np.isfinite(macro):
            continue
        row = {
            "feature_mode": str(mode),
            "pca_components": int(n_components),
            "C": float(c_value),
            "inner_macro_auc": float(macro),
            "inner_per_target_auc": {
                target: float(per_target[j]) for j, target in enumerate(TARGETS)
            },
            "candidate_index": int(candidate_index),
        }
        if best is None or macro > best[0] + 1e-12:
            best = (float(macro), row, inner_pred)

    if best is None:
        raise RuntimeError("no finite B4.1 shared candidate")
    return best[1], best[2]


def nested_shared_oof(
    config: dict,
    *,
    feature_path: str | Path,
    out_root: str | Path,
    pca_components: Iterable[int] = DEFAULT_PCA_COMPONENTS,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    feature_modes: Iterable[str] = DEFAULT_FEATURE_MODES,
    n_bootstrap: int = 5000,
) -> dict:
    train, gold, features, present, y = _align_gold_features(config, feature_path)

    seed = int(config.get("seed", 2026))
    n_folds = int(config.get("n_folds", 3))
    if n_folds < 3:
        raise ValueError("B4.1 nested OOF requires at least three folds")
    folds_full = make_balanced_gold_folds(train, n_splits=n_folds, seed=seed)
    fold_ids = folds_full.loc[gold.index].to_numpy(dtype=int)
    candidates = list(_candidate_grid(pca_components, c_values, feature_modes))
    if not candidates:
        raise ValueError("B4.1 candidate grid is empty")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    full_oof = np.full_like(y, np.nan, dtype=np.float64)
    fold_payloads: dict[str, dict] = {}

    for outer_fold in range(n_folds):
        inner_fold = int(config.get("inner_selection_fold", (outer_fold + 1) % n_folds))
        if inner_fold == outer_fold:
            raise ValueError("B4.1 inner fold must differ from outer fold")

        selection_train = (fold_ids != outer_fold) & (fold_ids != inner_fold)
        inner = fold_ids == inner_fold
        final_train = fold_ids != outer_fold
        outer = fold_ids == outer_fold
        if not selection_train.any() or not inner.any() or not outer.any():
            raise ValueError(f"empty nested partition for outer fold {outer_fold}")

        selected, _ = select_shared_candidate(
            features,
            present,
            y,
            selection_train,
            inner,
            candidates=candidates,
            seed=seed + 10_000 * outer_fold,
        )

        outer_pred = _predict_candidate(
            features,
            present,
            y,
            final_train,
            outer,
            mode=selected["feature_mode"],
            n_components=selected["pca_components"],
            c_value=selected["C"],
            seed=seed + 100_000 + 10_000 * outer_fold,
        )
        full_oof[outer] = outer_pred
        outer_score, outer_per_target = macro_auc_from_arrays(y[outer], outer_pred)

        fold_dir = out_root / f"fold{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_frame = pd.DataFrame(outer_pred, columns=TARGETS)
        fold_frame.insert(0, "StudyInstanceUID", gold.loc[outer, "StudyInstanceUID"].astype(str).to_numpy())
        fold_frame.to_csv(fold_dir / "oof.csv", index=False)

        bootstrap = bootstrap_macro_auc(
            y[outer], outer_pred,
            n_bootstrap=int(config.get("n_bootstrap", 2000)),
            seed=seed + outer_fold,
        )
        (fold_dir / "bootstrap.json").write_text(json.dumps(bootstrap.to_dict(), indent=2), encoding="utf-8")

        selection_payload = {
            "candidate": "B4_1_frozen_ssl_shared_policy",
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "selection_gold_train": int(selection_train.sum()),
            "inner_gold": int(inner.sum()),
            "final_gold_train": int(final_train.sum()),
            "outer_gold": int(outer.sum()),
            "shared_policy": selected,
            "outer_macro_auc": float(outer_score),
            "outer_per_target_auc": {
                target: float(outer_per_target[j]) for j, target in enumerate(TARGETS)
            },
        }
        (fold_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
        fold_payloads[str(outer_fold)] = selection_payload
        print({
            "phase": "b4_1_shared_nested",
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "policy": {
                "feature_mode": selected["feature_mode"],
                "pca_components": selected["pca_components"],
                "C": selected["C"],
            },
            "inner_macro_auc": selected["inner_macro_auc"],
            "outer_macro_auc": float(outer_score),
        })

    if not np.isfinite(full_oof).all():
        raise RuntimeError("B4.1 OOF matrix was not completely populated")

    combined = pd.DataFrame(full_oof, columns=TARGETS)
    combined.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].astype(str).to_numpy())
    combined.to_csv(out_root / "oof.csv", index=False)

    pooled = bootstrap_macro_auc(y, full_oof, n_bootstrap=int(n_bootstrap), seed=seed)
    evaluation = pooled.to_dict()
    (out_root / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    policy = {
        "candidate": "B4_1_frozen_ssl_shared_policy",
        "encoder_frozen": True,
        "gold_labels_used_for_encoder": False,
        "feature_cache": str(Path(feature_path).resolve()),
        "selection_unit": "one_shared_policy_per_outer_fold",
        "selection_criterion": "inner_macro_auc_only",
        "pca_fit_scope": "nested_training_partition_only",
        "classifier_fit_scope": "nested_training_partition_only",
        "outer_labels_used_for_selection": False,
        "feature_modes": list(feature_modes),
        "pca_components": [int(x) for x in pca_components],
        "C_values": [float(x) for x in c_values],
        "folds": fold_payloads,
        "pooled_evaluation": evaluation,
    }
    (out_root / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(pooled.summary())
    return policy


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b4-shared")
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-root", default="runs/b4_1_shared_ssl")
    parser.add_argument("--pca-components", type=int, nargs="+", default=list(DEFAULT_PCA_COMPONENTS))
    parser.add_argument("--C", dest="c_values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--feature-modes", nargs="+", default=list(DEFAULT_FEATURE_MODES))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    nested_shared_oof(
        _read_config(args.config),
        feature_path=args.features,
        out_root=args.out_root,
        pca_components=args.pca_components,
        c_values=args.c_values,
        feature_modes=args.feature_modes,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
