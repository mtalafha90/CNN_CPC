"""Fold-safe competition-data-only report teacher ensemble.

The competition test surface is image-only, so radiology reports are used only
as a training teacher.  This module benchmarks and exports a stronger teacher
without external models or external data:

* deterministic clinical-rule states with fold-safe empirical calibration;
* word TF-IDF + balanced logistic regression;
* character TF-IDF + balanced logistic regression;
* target-wise reliability-weighted ensembling chosen only from inner cross-fit
  predictions inside each outer fold;
* a one-dimensional fold-safe probability calibration of the ensemble;
* target-wise confidence derived from cross-fit AUC and prediction certainty.

For outer fold ``k`` no gold report or label from fold ``k`` participates in
vectorizer fitting, classifier fitting, component weighting, or probability
calibration.  The resulting ``fold{k}/pseudo_labels.csv`` can therefore be used
as a leakage-safe teacher for image-model fold ``k`` once its benchmark has been
inspected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .calibration import fit_calibration
from .constants import TARGETS
from .data import add_report_groups, gold_mask, load_train_csv, make_balanced_gold_folds, normalize_report
from .evaluation import bootstrap_macro_auc, fast_auc, macro_auc_from_arrays
from .report_labels import state_dataframe

COMPONENTS = ("rules", "word", "char")


@dataclass(frozen=True)
class TextComponentSpec:
    name: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int = 2
    c: float = 2.0


WORD_SPEC = TextComponentSpec(
    name="word",
    analyzer="word",
    ngram_range=(1, 2),
    max_features=40000,
)
CHAR_SPEC = TextComponentSpec(
    name="char",
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=60000,
)


def _normalized_reports(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [normalize_report(text) for text in df["Report"].fillna("").astype(str)],
        dtype=object,
    )


def _corpus_mask_excluding_groups(df: pd.DataFrame, heldout_mask: np.ndarray) -> np.ndarray:
    """Exclude held-out report groups from unsupervised TF-IDF fitting too."""
    work = add_report_groups(df)
    heldout_mask = np.asarray(heldout_mask, dtype=bool)
    blocked = set(work.loc[heldout_mask, "report_group"].astype(str))
    if not blocked:
        return np.ones(len(work), dtype=bool)
    return ~work["report_group"].astype(str).isin(blocked).to_numpy()


def _constant_probability(y: np.ndarray, fallback: float = 0.5) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float(fallback)
    return float(np.clip(y.mean(), 0.01, 0.99))


def _fit_text_component(
    texts: np.ndarray,
    gold: np.ndarray,
    *,
    train_gold_mask: np.ndarray,
    corpus_mask: np.ndarray,
    predict_mask: np.ndarray,
    spec: TextComponentSpec,
) -> np.ndarray:
    """Fit one TF-IDF representation and 12 target-specific binary models."""
    train_gold_mask = np.asarray(train_gold_mask, dtype=bool)
    corpus_mask = np.asarray(corpus_mask, dtype=bool)
    predict_mask = np.asarray(predict_mask, dtype=bool)
    n_pred = int(predict_mask.sum())
    out = np.zeros((n_pred, len(TARGETS)), dtype=np.float32)

    vectorizer = TfidfVectorizer(
        analyzer=spec.analyzer,
        ngram_range=spec.ngram_range,
        min_df=spec.min_df,
        max_features=spec.max_features,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )

    try:
        vectorizer.fit(texts[corpus_mask].tolist())
        x_train = vectorizer.transform(texts[train_gold_mask].tolist())
        x_pred = vectorizer.transform(texts[predict_mask].tolist())
    except ValueError:
        # Tiny synthetic/test corpora can legitimately have an empty vocabulary.
        train_gold = gold[train_gold_mask]
        for j in range(len(TARGETS)):
            out[:, j] = _constant_probability(train_gold[:, j])
        return out

    train_gold = gold[train_gold_mask]
    for j in range(len(TARGETS)):
        y = train_gold[:, j]
        labelled = np.isfinite(y)
        fallback = _constant_probability(y)
        if labelled.sum() < 2 or np.unique(y[labelled]).size < 2:
            out[:, j] = fallback
            continue
        model = LogisticRegression(
            C=spec.c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=2026,
        )
        model.fit(x_train[labelled], y[labelled].astype(int))
        out[:, j] = model.predict_proba(x_pred)[:, 1].astype(np.float32)
    return out


def _rule_component(
    states: np.ndarray,
    gold: np.ndarray,
    *,
    train_gold_mask: np.ndarray,
    predict_mask: np.ndarray,
) -> np.ndarray:
    calibration = fit_calibration(states[train_gold_mask], gold[train_gold_mask])
    return calibration.apply(states[predict_mask]).astype(np.float32)


def _component_predictions(
    df: pd.DataFrame,
    texts: np.ndarray,
    states: np.ndarray,
    gold: np.ndarray,
    *,
    train_gold_mask: np.ndarray,
    heldout_gold_mask: np.ndarray,
    predict_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    corpus_mask = _corpus_mask_excluding_groups(df, heldout_gold_mask)
    return {
        "rules": _rule_component(
            states,
            gold,
            train_gold_mask=train_gold_mask,
            predict_mask=predict_mask,
        ),
        "word": _fit_text_component(
            texts,
            gold,
            train_gold_mask=train_gold_mask,
            corpus_mask=corpus_mask,
            predict_mask=predict_mask,
            spec=WORD_SPEC,
        ),
        "char": _fit_text_component(
            texts,
            gold,
            train_gold_mask=train_gold_mask,
            corpus_mask=corpus_mask,
            predict_mask=predict_mask,
            spec=CHAR_SPEC,
        ),
    }


def _auc_by_component(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [fast_auc(y_true[:, j], pred[:, j]) for j in range(len(TARGETS))],
            dtype=float,
        )
        for name, pred in predictions.items()
    }


def _target_weights(
    component_auc: dict[str, np.ndarray],
    *,
    adaptive_strength: float = 0.75,
) -> np.ndarray:
    """Return ``[component,target]`` weights shrunk toward equal weighting.

    Only AUC above 0.5 earns adaptive weight.  Shrinkage prevents a tiny inner
    sample from turning a noisy component ranking into an all-or-nothing choice.
    """
    if not 0.0 <= adaptive_strength <= 1.0:
        raise ValueError("adaptive_strength must be in [0,1]")
    auc = np.vstack([component_auc[name] for name in COMPONENTS]).astype(float)
    skill = np.maximum(np.nan_to_num(auc, nan=0.5) - 0.5, 0.0)
    equal = np.full_like(skill, 1.0 / len(COMPONENTS))
    adaptive = equal.copy()
    sums = skill.sum(axis=0)
    useful = sums > 1e-12
    adaptive[:, useful] = skill[:, useful] / sums[useful]
    weights = (1.0 - adaptive_strength) * equal + adaptive_strength * adaptive
    weights /= weights.sum(axis=0, keepdims=True)
    return weights


def _weighted_ensemble(predictions: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([predictions[name] for name in COMPONENTS], axis=0)
    return np.sum(stacked * weights[:, None, :], axis=0).astype(np.float32)


def _fit_probability_calibrators(scores: np.ndarray, y_true: np.ndarray) -> list[object]:
    calibrators: list[object] = []
    for j in range(len(TARGETS)):
        y = y_true[:, j]
        score = scores[:, j]
        labelled = np.isfinite(y) & np.isfinite(score)
        if labelled.sum() < 4 or np.unique(y[labelled]).size < 2:
            calibrators.append(_constant_probability(y[labelled]))
            continue
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=2026)
        model.fit(score[labelled, None], y[labelled].astype(int))
        calibrators.append(model)
    return calibrators


def _apply_probability_calibrators(scores: np.ndarray, calibrators: list[object]) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    for j, calibrator in enumerate(calibrators):
        if isinstance(calibrator, (float, int, np.floating)):
            out[:, j] = float(calibrator)
        else:
            out[:, j] = calibrator.predict_proba(scores[:, j, None])[:, 1].astype(np.float32)
    return np.clip(out, 0.001, 0.999)


def _teacher_confidence(probabilities: np.ndarray, reliability_auc: np.ndarray) -> np.ndarray:
    """Convert target-level cross-fit discrimination and cell certainty to weight.

    ``AUC=0.5`` produces zero confidence.  ``AUC=1`` permits full confidence,
    but only for predictions far from 0.5.  This deliberately avoids globally
    lowering a single pseudo-label threshold just to increase sample counts.
    """
    reliability = np.clip(2.0 * (np.nan_to_num(reliability_auc, nan=0.5) - 0.5), 0.0, 1.0)
    certainty = np.clip(2.0 * np.abs(probabilities - 0.5), 0.0, 1.0)
    return (certainty * reliability[None, :]).astype(np.float32)


def _inner_crossfit(
    df: pd.DataFrame,
    texts: np.ndarray,
    states: np.ndarray,
    gold: np.ndarray,
    folds: np.ndarray,
    *,
    outer_fold: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Cross-fit base teachers on the two non-outer gold folds."""
    is_gold = np.asarray(gold_mask(df), dtype=bool)
    train_outer_mask = is_gold & (folds != outer_fold)
    train_indices = np.flatnonzero(train_outer_mask)
    position = {idx: pos for pos, idx in enumerate(train_indices.tolist())}
    oof = {
        name: np.full((len(train_indices), len(TARGETS)), np.nan, dtype=np.float32)
        for name in COMPONENTS
    }

    remaining_folds = sorted(set(int(x) for x in folds[train_outer_mask]))
    for inner_holdout in remaining_folds:
        inner_val_mask = is_gold & (folds == inner_holdout)
        inner_train_mask = is_gold & (folds != outer_fold) & (folds != inner_holdout)
        heldout_mask = is_gold & ((folds == outer_fold) | (folds == inner_holdout))
        pred = _component_predictions(
            df,
            texts,
            states,
            gold,
            train_gold_mask=inner_train_mask,
            heldout_gold_mask=heldout_mask,
            predict_mask=inner_val_mask,
        )
        rows = [position[idx] for idx in np.flatnonzero(inner_val_mask)]
        for name in COMPONENTS:
            oof[name][rows] = pred[name]

    if any(not np.isfinite(oof[name]).all() for name in COMPONENTS):
        raise RuntimeError("incomplete inner report-teacher cross-fit predictions")

    y_train = gold[train_indices]
    component_auc = _auc_by_component(y_train, oof)
    weights = _target_weights(component_auc)
    raw_ensemble = _weighted_ensemble(oof, weights)
    reliability_auc = np.asarray(
        [fast_auc(y_train[:, j], raw_ensemble[:, j]) for j in range(len(TARGETS))],
        dtype=float,
    )
    return oof, weights, reliability_auc, component_auc


def _metrics_dict(values: Iterable[float]) -> dict[str, float | None]:
    return {
        target: (float(value) if np.isfinite(value) else None)
        for target, value in zip(TARGETS, values)
    }


def run_report_teacher_benchmark(
    train_csv: str | Path,
    *,
    out_dir: str | Path = "runs/report_teacher",
    n_folds: int = 3,
    seed: int = 2026,
    n_bootstrap: int = 2000,
) -> dict:
    """Benchmark and export a leakage-safe fold-specific report teacher."""
    if n_folds != 3:
        raise ValueError("current nested report-teacher benchmark requires exactly 3 folds")
    df = load_train_csv(train_csv)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    folds_series = make_balanced_gold_folds(df, n_splits=n_folds, seed=seed)
    folds = folds_series.to_numpy(dtype=int)
    is_gold = np.asarray(gold_mask(df), dtype=bool)
    if int(is_gold.sum()) == 0 or np.any(folds[is_gold] < 0):
        raise ValueError("gold fold assignment is incomplete")

    gold = df[TARGETS].to_numpy(dtype=np.float64)
    texts = _normalized_reports(df)
    states = state_dataframe(df)

    fold_assignment = pd.DataFrame(
        {
            "StudyInstanceUID": df["StudyInstanceUID"].astype(str),
            "is_gold": is_gold,
            "gold_fold": folds,
        }
    )
    fold_assignment.to_csv(out_root / "fold_assignments.csv", index=False)

    oof_rows: list[pd.DataFrame] = []
    fold_payload: dict[str, dict] = {}

    for outer_fold in range(n_folds):
        outer_mask = is_gold & (folds == outer_fold)
        train_gold_mask = is_gold & (folds != outer_fold)
        inner_pred, weights, reliability_auc, component_inner_auc = _inner_crossfit(
            df,
            texts,
            states,
            gold,
            folds,
            outer_fold=outer_fold,
        )
        inner_indices = np.flatnonzero(train_gold_mask)
        y_inner = gold[inner_indices]
        raw_inner = _weighted_ensemble(inner_pred, weights)
        calibrators = _fit_probability_calibrators(raw_inner, y_inner)
        calibrated_inner = _apply_probability_calibrators(raw_inner, calibrators)
        inner_ensemble_auc = np.asarray(
            [fast_auc(y_inner[:, j], calibrated_inner[:, j]) for j in range(len(TARGETS))],
            dtype=float,
        )

        # Final fold-specific teacher: fit only on non-outer gold, then predict
        # every study. Outer gold predictions are valid OOF teacher predictions;
        # non-gold predictions are the weak labels for MRI fold ``outer_fold``.
        all_mask = np.ones(len(df), dtype=bool)
        final_components = _component_predictions(
            df,
            texts,
            states,
            gold,
            train_gold_mask=train_gold_mask,
            heldout_gold_mask=outer_mask,
            predict_mask=all_mask,
        )
        raw_all = _weighted_ensemble(final_components, weights)
        probabilities = _apply_probability_calibrators(raw_all, calibrators)
        confidence = _teacher_confidence(probabilities, reliability_auc)

        fold_dir = out_root / f"fold{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pseudo = pd.DataFrame(
            {
                "StudyInstanceUID": df["StudyInstanceUID"].astype(str),
                "teacher_fold": outer_fold,
                "is_gold": is_gold,
                "is_outer_gold": outer_mask,
            }
        )
        for j, target in enumerate(TARGETS):
            pseudo[target] = probabilities[:, j]
            pseudo[f"{target}__confidence"] = confidence[:, j]
        pseudo.to_csv(fold_dir / "pseudo_labels.csv", index=False)

        outer_idx = np.flatnonzero(outer_mask)
        fold_oof = pd.DataFrame({"StudyInstanceUID": df.loc[outer_mask, "StudyInstanceUID"].astype(str)})
        for j, target in enumerate(TARGETS):
            fold_oof[target] = probabilities[outer_idx, j]
            for name in COMPONENTS:
                fold_oof[f"{target}__{name}"] = final_components[name][outer_idx, j]
            fold_oof[f"{target}__confidence"] = confidence[outer_idx, j]
        fold_oof["outer_fold"] = outer_fold
        oof_rows.append(fold_oof)

        fold_payload[str(outer_fold)] = {
            "n_train_gold": int(train_gold_mask.sum()),
            "n_outer_gold": int(outer_mask.sum()),
            "component_inner_auc": {
                name: _metrics_dict(component_inner_auc[name]) for name in COMPONENTS
            },
            "component_weights": {
                target: {name: float(weights[i, j]) for i, name in enumerate(COMPONENTS)}
                for j, target in enumerate(TARGETS)
            },
            "inner_ensemble_auc": _metrics_dict(inner_ensemble_auc),
            "inner_reliability_auc": _metrics_dict(reliability_auc),
            "pseudo_labels": str((fold_dir / "pseudo_labels.csv").resolve()),
        }
        (fold_dir / "teacher.json").write_text(
            json.dumps(fold_payload[str(outer_fold)], indent=2), encoding="utf-8"
        )

    oof = pd.concat(oof_rows, ignore_index=True).sort_values("StudyInstanceUID").reset_index(drop=True)
    oof.to_csv(out_root / "oof.csv", index=False)

    gold_table = df.loc[is_gold, ["StudyInstanceUID", *TARGETS]].copy()
    merged = gold_table.merge(oof, on="StudyInstanceUID", how="inner", suffixes=("", "_pred"))
    if len(merged) != int(is_gold.sum()):
        raise RuntimeError("report-teacher OOF does not cover every gold study exactly once")
    y_true = merged[TARGETS].to_numpy(dtype=float)
    ensemble_pred = merged[[f"{t}_pred" for t in TARGETS]].to_numpy(dtype=float)
    bootstrap = bootstrap_macro_auc(y_true, ensemble_pred, n_bootstrap=n_bootstrap, seed=seed)

    component_oof: dict[str, dict] = {}
    for name in COMPONENTS:
        pred = merged[[f"{t}__{name}" for t in TARGETS]].to_numpy(dtype=float)
        macro, per_target = macro_auc_from_arrays(y_true, pred)
        component_oof[name] = {
            "macro_auc": float(macro),
            "per_target_auc": _metrics_dict(per_target),
        }

    payload = {
        "method": "competition-data-only fold-safe report ensemble",
        "components": list(COMPONENTS),
        "n_studies": len(df),
        "n_gold": int(is_gold.sum()),
        "n_folds": n_folds,
        "seed": seed,
        "external_models": False,
        "external_data": False,
        "oof": bootstrap.to_dict(),
        "component_oof": component_oof,
        "folds": fold_payload,
    }
    (out_root / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
