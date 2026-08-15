"""B4.2: grouped-policy nested classifiers on frozen SSL features.

B4.2 sits between B4 target-wise policy selection and B4.1 one-policy-per-fold.
Four pathology groups are fixed a priori from anatomy/sequence context. For each
outer fold, each group selects one common (feature mode, PCA dimension,
logistic C) policy using only the inner gold fold. Individual targets still fit
separate logistic-regression coefficients after the group policy is selected.
The untouched outer fold is never used for policy selection.
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
from .evaluation import bootstrap_macro_auc, macro_auc_from_arrays
from .frozen_features import (
    DEFAULT_C_VALUES,
    DEFAULT_FEATURE_MODES,
    DEFAULT_PCA_COMPONENTS,
    _candidate_grid,
    _fit_predict,
    target_design_matrix,
)
from .frozen_features_shared import _align_gold_features


PATHOLOGY_GROUPS: dict[str, tuple[str, ...]] = {
    "ligament_meniscus": (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
    ),
    "osteoarthritis": (
        "Medial OA",
        "Lateral OA",
        "PF OA",
    ),
    "fluid_inflammatory": (
        "Effusion",
        "Synovitis",
        "Baker's",
    ),
    "osseous_injury": (
        "Contusion",
        "Fracture",
    ),
}


def _validate_groups(groups: dict[str, tuple[str, ...]] = PATHOLOGY_GROUPS) -> None:
    if not groups:
        raise ValueError("B4.2 pathology groups must not be empty")
    flattened = [target for targets in groups.values() for target in targets]
    if len(flattened) != len(set(flattened)):
        raise ValueError("B4.2 pathology groups overlap")
    if set(flattened) != set(TARGETS):
        missing = sorted(set(TARGETS).difference(flattened))
        extra = sorted(set(flattened).difference(TARGETS))
        raise ValueError(f"B4.2 pathology groups must partition TARGETS; missing={missing}, extra={extra}")
    if any(len(targets) == 0 for targets in groups.values()):
        raise ValueError("B4.2 pathology groups must be non-empty")


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _group_indices(targets: tuple[str, ...]) -> list[int]:
    return [TARGETS.index(target) for target in targets]


def _predict_group(
    features: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    targets: tuple[str, ...],
    *,
    mode: str,
    n_components: int,
    c_value: float,
    seed: int,
) -> np.ndarray:
    indices = _group_indices(targets)
    pred = np.zeros((int(eval_mask.sum()), len(indices)), dtype=np.float64)
    for local_j, global_j in enumerate(indices):
        target = TARGETS[global_j]
        x = target_design_matrix(features, present, target, mode)
        pred[:, local_j] = _fit_predict(
            x[train_mask],
            y[train_mask, global_j],
            x[eval_mask],
            n_components=int(n_components),
            c_value=float(c_value),
            seed=int(seed) + global_j,
        )
    return pred


def select_group_candidate(
    features: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    selection_train: np.ndarray,
    inner: np.ndarray,
    targets: tuple[str, ...],
    *,
    candidates,
    seed: int,
) -> dict:
    """Choose one shared candidate for a predefined pathology group."""
    indices = _group_indices(targets)
    best = None
    for candidate_index, (mode, n_components, c_value) in enumerate(candidates):
        inner_pred = _predict_group(
            features,
            present,
            y,
            selection_train,
            inner,
            targets,
            mode=mode,
            n_components=n_components,
            c_value=c_value,
            seed=seed + 1000 * candidate_index,
        )
        macro, per_target = macro_auc_from_arrays(y[inner][:, indices], inner_pred)
        if not np.isfinite(macro):
            continue
        row = {
            "feature_mode": str(mode),
            "pca_components": int(n_components),
            "C": float(c_value),
            "inner_group_macro_auc": float(macro),
            "inner_per_target_auc": {
                target: float(per_target[j]) for j, target in enumerate(targets)
            },
            "candidate_index": int(candidate_index),
        }
        # Candidate order is fixed and therefore provides deterministic tie-breaking.
        if best is None or macro > best[0] + 1e-12:
            best = (float(macro), row)
    if best is None:
        raise RuntimeError(f"no finite B4.2 group candidate for {targets}")
    return best[1]


def nested_grouped_oof(
    config: dict,
    *,
    feature_path: str | Path,
    out_root: str | Path,
    pca_components: Iterable[int] = DEFAULT_PCA_COMPONENTS,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    feature_modes: Iterable[str] = DEFAULT_FEATURE_MODES,
    n_bootstrap: int = 5000,
) -> dict:
    """Run leakage-safe grouped B4.2 policy selection and pooled outer OOF."""
    _validate_groups()
    train, gold, features, present, y = _align_gold_features(config, feature_path)

    seed = int(config.get("seed", 2026))
    n_folds = int(config.get("n_folds", 3))
    if n_folds < 3:
        raise ValueError("B4.2 nested OOF requires at least three folds")
    folds_full = make_balanced_gold_folds(train, n_splits=n_folds, seed=seed)
    fold_ids = folds_full.loc[gold.index].to_numpy(dtype=int)
    candidates = list(_candidate_grid(pca_components, c_values, feature_modes))
    if not candidates:
        raise ValueError("B4.2 candidate grid is empty")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    full_oof = np.full_like(y, np.nan, dtype=np.float64)
    fold_payloads: dict[str, dict] = {}

    for outer_fold in range(n_folds):
        inner_fold = int(config.get("inner_selection_fold", (outer_fold + 1) % n_folds))
        if inner_fold == outer_fold:
            raise ValueError("B4.2 inner fold must differ from outer fold")

        selection_train = (fold_ids != outer_fold) & (fold_ids != inner_fold)
        inner = fold_ids == inner_fold
        final_train = fold_ids != outer_fold
        outer = fold_ids == outer_fold
        if not selection_train.any() or not inner.any() or not outer.any():
            raise ValueError(f"empty B4.2 nested partition for outer fold {outer_fold}")

        outer_pred = np.zeros((int(outer.sum()), len(TARGETS)), dtype=np.float64)
        group_policies: dict[str, dict] = {}
        inner_group_scores: list[float] = []

        for group_index, (group_name, group_targets) in enumerate(PATHOLOGY_GROUPS.items()):
            selected = select_group_candidate(
                features,
                present,
                y,
                selection_train,
                inner,
                group_targets,
                candidates=candidates,
                seed=seed + 10_000 * outer_fold + 100_000 * group_index,
            )
            group_policies[group_name] = {
                "targets": list(group_targets),
                **selected,
            }
            inner_group_scores.append(float(selected["inner_group_macro_auc"]))

            group_outer = _predict_group(
                features,
                present,
                y,
                final_train,
                outer,
                group_targets,
                mode=selected["feature_mode"],
                n_components=int(selected["pca_components"]),
                c_value=float(selected["C"]),
                seed=seed + 1_000_000 + 10_000 * outer_fold + 100_000 * group_index,
            )
            for local_j, global_j in enumerate(_group_indices(group_targets)):
                outer_pred[:, global_j] = group_outer[:, local_j]

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
            "candidate": "B4_2_frozen_ssl_grouped_policy",
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "selection_gold_train": int(selection_train.sum()),
            "inner_gold": int(inner.sum()),
            "final_gold_train": int(final_train.sum()),
            "outer_gold": int(outer.sum()),
            "pathology_groups": group_policies,
            "mean_inner_group_macro_auc": float(np.mean(inner_group_scores)),
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
                "phase": "b4_2_grouped_nested",
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "policies": {
                    group: {
                        "feature_mode": payload["feature_mode"],
                        "pca_components": payload["pca_components"],
                        "C": payload["C"],
                    }
                    for group, payload in group_policies.items()
                },
                "mean_inner_group_macro_auc": float(np.mean(inner_group_scores)),
                "outer_macro_auc": float(outer_score),
            }
        )

    if not np.isfinite(full_oof).all():
        raise RuntimeError("B4.2 OOF matrix was not completely populated")

    combined = pd.DataFrame(full_oof, columns=TARGETS)
    combined.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].astype(str).to_numpy())
    combined.to_csv(out_root / "oof.csv", index=False)

    pooled = bootstrap_macro_auc(y, full_oof, n_bootstrap=int(n_bootstrap), seed=seed)
    evaluation = pooled.to_dict()
    (out_root / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    policy = {
        "candidate": "B4_2_frozen_ssl_grouped_policy",
        "encoder_frozen": True,
        "gold_labels_used_for_encoder": False,
        "feature_cache": str(Path(feature_path).resolve()),
        "outer_labels_used_for_selection": False,
        "selection_unit": "predefined_pathology_group",
        "pathology_groups": {name: list(targets) for name, targets in PATHOLOGY_GROUPS.items()},
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
    parser = argparse.ArgumentParser("rsna-knee-b4-grouped")
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-root", default="runs/b4_2_grouped_ssl")
    parser.add_argument("--pca-components", type=int, nargs="+", default=list(DEFAULT_PCA_COMPONENTS))
    parser.add_argument("--C", dest="c_values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--feature-modes", nargs="+", default=list(DEFAULT_FEATURE_MODES))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    nested_grouped_oof(
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
