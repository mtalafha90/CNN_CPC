"""B4.3: target-wise two-fold cross-validated policy selection on frozen SSL features.

For each untouched outer gold fold, the remaining two folds are used in a
symmetric two-way selector.  Every candidate is evaluated by training on one
non-outer fold and predicting the other, then reversing the roles.  The two
held-out prediction blocks are concatenated and scored over all non-outer gold
studies.  Each pathology selects its own policy from this cross-validated score,
after which the selected model is refit on all non-outer gold studies and used
once on the untouched outer fold.

This keeps B4's pathology-specific flexibility while replacing its single
18--20 study inner selector with a 38--40 study cross-validated selector.  No
outer-fold labels are used for policy selection.
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
from .data import make_balanced_gold_folds
from .evaluation import bootstrap_macro_auc, fast_auc, macro_auc_from_arrays
from .frozen_features import (
    DEFAULT_C_VALUES,
    DEFAULT_FEATURE_MODES,
    DEFAULT_PCA_COMPONENTS,
    _candidate_grid,
    _fit_predict,
    target_design_matrix,
)
from .frozen_features_shared import _align_gold_features


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _two_way_cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    fold_a: int,
    fold_b: int,
    *,
    n_components: int,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned two-way held-out predictions over the two non-outer folds."""
    fold_a_mask = fold_ids == int(fold_a)
    fold_b_mask = fold_ids == int(fold_b)
    selected = fold_a_mask | fold_b_mask
    if not fold_a_mask.any() or not fold_b_mask.any():
        raise ValueError("B4.3 two-way CV requires two non-empty folds")

    selected_indices = np.flatnonzero(selected)
    local_position = {int(global_i): local_i for local_i, global_i in enumerate(selected_indices)}
    pred = np.full(selected_indices.size, np.nan, dtype=np.float64)

    pred_a = _fit_predict(
        x[fold_b_mask],
        y[fold_b_mask],
        x[fold_a_mask],
        n_components=int(n_components),
        c_value=float(c_value),
        seed=int(seed),
    )
    for global_i, value in zip(np.flatnonzero(fold_a_mask), pred_a):
        pred[local_position[int(global_i)]] = float(value)

    pred_b = _fit_predict(
        x[fold_a_mask],
        y[fold_a_mask],
        x[fold_b_mask],
        n_components=int(n_components),
        c_value=float(c_value),
        seed=int(seed) + 1,
    )
    for global_i, value in zip(np.flatnonzero(fold_b_mask), pred_b):
        pred[local_position[int(global_i)]] = float(value)

    if not np.isfinite(pred).all():
        raise RuntimeError("B4.3 two-way CV prediction vector was not fully populated")
    return selected_indices, pred


def select_target_candidate_crossval(
    features: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    outer_fold: int,
    target_index: int,
    *,
    candidates,
    seed: int,
) -> dict:
    """Select one target policy using only two-way CV on the non-outer folds."""
    non_outer_folds = sorted(set(int(x) for x in np.unique(fold_ids)) - {int(outer_fold)})
    if len(non_outer_folds) != 2:
        raise ValueError(
            f"B4.3 currently requires exactly three gold folds; outer={outer_fold}, non_outer={non_outer_folds}"
        )
    fold_a, fold_b = non_outer_folds
    target = TARGETS[int(target_index)]
    best = None

    for candidate_index, (mode, n_components, c_value) in enumerate(candidates):
        x = target_design_matrix(features, present, target, mode)
        selected_indices, pred = _two_way_cv_predictions(
            x,
            y[:, target_index],
            fold_ids,
            fold_a,
            fold_b,
            n_components=n_components,
            c_value=c_value,
            seed=int(seed) + 1000 * candidate_index,
        )
        score = fast_auc(y[selected_indices, target_index], pred)
        if not np.isfinite(score):
            continue
        row = {
            "feature_mode": str(mode),
            "pca_components": int(n_components),
            "C": float(c_value),
            "cv_auc": float(score),
            "candidate_index": int(candidate_index),
            "cv_folds": [int(fold_a), int(fold_b)],
            "cv_studies": int(len(selected_indices)),
        }
        # Fixed candidate order supplies deterministic tie-breaking.
        if best is None or score > best[0] + 1e-12:
            best = (float(score), row)

    if best is None:
        raise RuntimeError(f"no finite B4.3 CV AUC for target {target}, outer fold {outer_fold}")
    return best[1]


def nested_crossval_oof(
    config: dict,
    *,
    feature_path: str | Path,
    out_root: str | Path,
    pca_components: Iterable[int] = DEFAULT_PCA_COMPONENTS,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    feature_modes: Iterable[str] = DEFAULT_FEATURE_MODES,
    n_bootstrap: int = 5000,
) -> dict:
    """Run B4.3 target-wise two-way-CV selection and untouched outer OOF."""
    train, gold, features, present, y = _align_gold_features(config, feature_path)

    seed = int(config.get("seed", 2026))
    n_folds = int(config.get("n_folds", 3))
    if n_folds != 3:
        raise ValueError("B4.3 currently requires exactly three gold folds")
    folds_full = make_balanced_gold_folds(train, n_splits=n_folds, seed=seed)
    fold_ids = folds_full.loc[gold.index].to_numpy(dtype=int)
    candidates = list(_candidate_grid(pca_components, c_values, feature_modes))
    if not candidates:
        raise ValueError("B4.3 candidate grid is empty")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    full_oof = np.full_like(y, np.nan, dtype=np.float64)
    fold_payloads: dict[str, dict] = {}

    for outer_fold in range(n_folds):
        final_train = fold_ids != outer_fold
        outer = fold_ids == outer_fold
        non_outer_folds = sorted(set(int(x) for x in np.unique(fold_ids)) - {outer_fold})
        if not final_train.any() or not outer.any() or len(non_outer_folds) != 2:
            raise ValueError(f"invalid B4.3 partition for outer fold {outer_fold}")

        outer_pred = np.zeros((int(outer.sum()), len(TARGETS)), dtype=np.float64)
        target_policies: dict[str, dict] = {}
        cv_scores: list[float] = []

        for j, target in enumerate(TARGETS):
            selected = select_target_candidate_crossval(
                features,
                present,
                y,
                fold_ids,
                outer_fold,
                j,
                candidates=candidates,
                seed=seed + 100_000 * outer_fold + 10_000 * j,
            )
            target_policies[target] = selected
            cv_scores.append(float(selected["cv_auc"]))

            x = target_design_matrix(features, present, target, selected["feature_mode"])
            outer_pred[:, j] = _fit_predict(
                x[final_train],
                y[final_train, j],
                x[outer],
                n_components=int(selected["pca_components"]),
                c_value=float(selected["C"]),
                seed=seed + 1_000_000 + 100_000 * outer_fold + 10_000 * j,
            )

        full_oof[outer] = outer_pred
        outer_score, outer_per_target = macro_auc_from_arrays(y[outer], outer_pred)

        fold_dir = out_root / f"fold{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_frame = pd.DataFrame(outer_pred, columns=TARGETS)
        fold_frame.insert(
            0,
            "StudyInstanceUID",
            gold.loc[outer, "StudyInstanceUID"].astype(str).to_numpy(),
        )
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
            "candidate": "B4_3_frozen_ssl_targetwise_two_way_cv",
            "outer_fold": int(outer_fold),
            "cv_folds": [int(x) for x in non_outer_folds],
            "cv_gold": int(final_train.sum()),
            "final_gold_train": int(final_train.sum()),
            "outer_gold": int(outer.sum()),
            "mean_selected_cv_auc": float(np.mean(cv_scores)),
            "targets": target_policies,
            "outer_macro_auc": float(outer_score),
            "outer_per_target_auc": {
                target: float(outer_per_target[j]) for j, target in enumerate(TARGETS)
            },
        }
        (fold_dir / "selection.json").write_text(
            json.dumps(selection_payload, indent=2), encoding="utf-8"
        )
        fold_payloads[str(outer_fold)] = selection_payload
        print(
            {
                "phase": "b4_3_crossval_nested",
                "outer_fold": int(outer_fold),
                "cv_folds": [int(x) for x in non_outer_folds],
                "cv_gold": int(final_train.sum()),
                "mean_selected_cv_auc": float(np.mean(cv_scores)),
                "outer_macro_auc": float(outer_score),
            }
        )

    if not np.isfinite(full_oof).all():
        raise RuntimeError("B4.3 OOF matrix was not completely populated")

    combined = pd.DataFrame(full_oof, columns=TARGETS)
    combined.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].astype(str).to_numpy())
    combined.to_csv(out_root / "oof.csv", index=False)

    pooled = bootstrap_macro_auc(y, full_oof, n_bootstrap=int(n_bootstrap), seed=seed)
    evaluation = pooled.to_dict()
    (out_root / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    policy = {
        "candidate": "B4_3_frozen_ssl_targetwise_two_way_cv",
        "encoder_frozen": True,
        "gold_labels_used_for_encoder": False,
        "feature_cache": str(Path(feature_path).resolve()),
        "outer_labels_used_for_selection": False,
        "selection_unit": "target",
        "selection_method": "two_way_cross_validation_on_both_non_outer_folds",
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
    parser = argparse.ArgumentParser("rsna-knee-b4-crossval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-root", default="runs/b4_3_crossval_ssl")
    parser.add_argument("--pca-components", type=int, nargs="+", default=list(DEFAULT_PCA_COMPONENTS))
    parser.add_argument("--C", dest="c_values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--feature-modes", nargs="+", default=list(DEFAULT_FEATURE_MODES))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    nested_crossval_oof(
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
