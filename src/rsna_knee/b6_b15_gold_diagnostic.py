"""B6/B15 reused-gold diagnostic package.

This module performs no training and uses no GPU. It characterizes the frozen B6
report states on the already-reused 58-study gold development surface and
measures how B15 moved relative to the historical B13 development champion.

The package intentionally distinguishes:
1) coverage-conditioned teacher AUC on high-confidence positive/negated cells;
2) a full-surface state-only ranking baseline over all 58 x 12 gold cells;
3) state -> expert-truth frequencies;
4) B13 -> B15 movement on cells where the high-confidence B6 teacher is correct
   versus wrong.

None of these diagnostics is an independent validation result or a theoretical
performance ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs, fast_auc
from .report_labels import (
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
    STATES,
)

STATE_SOFT_SCORES = {
    STATE_POSITIVE: 0.85,
    STATE_NEGATED: 0.05,
    STATE_UNCERTAIN: 0.50,
    STATE_UNMENTIONED: 0.50,
}


def _finite_float(value: float | np.floating) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _read_gold_predictions(path: str | Path, ordered_uids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"StudyInstanceUID", *TARGETS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction file {path} missing columns: {missing}")
    frame = frame[["StudyInstanceUID", *TARGETS]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"prediction file {path} contains duplicate StudyInstanceUID values")
    if not np.isfinite(frame[TARGETS].to_numpy(dtype=float)).all():
        raise ValueError(f"prediction file {path} contains non-finite predictions")

    requested = [str(uid) for uid in ordered_uids]
    present = set(frame["StudyInstanceUID"])
    missing_uids = [uid for uid in requested if uid not in present]
    extra_uids = sorted(present.difference(requested))
    if missing_uids or extra_uids:
        raise ValueError(
            f"prediction file {path} does not match the exact 58-study gold surface: "
            f"missing={len(missing_uids)}, extra={len(extra_uids)}"
        )
    order = {uid: i for i, uid in enumerate(requested)}
    return frame.sort_values("StudyInstanceUID", key=lambda s: s.map(order)).reset_index(drop=True)


def _read_structured_gold(path: str | Path, gold: pd.DataFrame) -> pd.DataFrame:
    structured = pd.read_csv(path)
    required = {"StudyInstanceUID"}
    for target in TARGETS:
        required.update({f"{target}__state", f"{target}__confidence"})
    missing = sorted(required.difference(structured.columns))
    if missing:
        raise ValueError(f"structured label file {path} missing columns: {missing}")

    structured = structured.copy()
    structured["StudyInstanceUID"] = structured["StudyInstanceUID"].astype(str)
    if structured["StudyInstanceUID"].duplicated().any():
        raise ValueError("structured_labels.csv contains duplicate StudyInstanceUID values")

    keep = ["StudyInstanceUID"]
    for target in TARGETS:
        keep.extend([f"{target}__state", f"{target}__confidence"])
    merged = gold.merge(structured[keep], on="StudyInstanceUID", how="left", validate="one_to_one")
    if len(merged) != len(gold):
        raise AssertionError("structured-label merge changed gold row count")

    for target in TARGETS:
        state_col = f"{target}__state"
        conf_col = f"{target}__confidence"
        if merged[state_col].isna().any():
            raise ValueError(f"structured labels are missing gold states for {target}")
        unexpected = sorted(set(merged[state_col].astype(str).unique()).difference(STATES))
        if unexpected:
            raise ValueError(f"unexpected B6 states for {target}: {unexpected}")
        confidence = pd.to_numeric(merged[conf_col], errors="coerce")
        if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError(f"invalid B6 confidence values for {target}")
        merged[conf_col] = confidence.astype(float)
    return merged


def _macro_selective_auc(truth: np.ndarray, score: np.ndarray, eligible: np.ndarray) -> tuple[float, np.ndarray]:
    truth = np.asarray(truth, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if truth.shape != score.shape or truth.shape != eligible.shape or truth.ndim != 2:
        raise ValueError("selective AUC inputs must have identical 2D shapes")

    values = np.empty(truth.shape[1], dtype=np.float64)
    for j in range(truth.shape[1]):
        mask = eligible[:, j] & np.isfinite(truth[:, j]) & np.isfinite(score[:, j])
        values[j] = fast_auc(truth[mask, j], score[mask, j])
    finite = values[np.isfinite(values)]
    return (float(finite.mean()) if finite.size else float("nan")), values


def bootstrap_selective_macro_auc(
    truth: np.ndarray,
    score: np.ndarray,
    eligible: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict:
    point, per_target = _macro_selective_auc(truth, score, eligible)
    rng = np.random.default_rng(seed)
    strict_values: list[float] = []
    relaxed_values: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(truth), size=len(truth))
        macro, target_values = _macro_selective_auc(truth[idx], score[idx], eligible[idx])
        if np.isfinite(macro):
            relaxed_values.append(float(macro))
        if np.isfinite(target_values).all():
            strict_values.append(float(target_values.mean()))

    def interval(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        lo, hi = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
        return float(lo), float(hi)

    strict_lo, strict_hi = interval(strict_values)
    relaxed_lo, relaxed_hi = interval(relaxed_values)
    return {
        "macro_auc_over_defined_targets": _finite_float(point),
        "per_target_auc": {target: _finite_float(value) for target, value in zip(TARGETS, per_target)},
        "per_target_defined": {target: bool(np.isfinite(value)) for target, value in zip(TARGETS, per_target)},
        "n_targets_defined_on_full_set": int(np.isfinite(per_target).sum()),
        "strict_all_12_targets": {
            "ci_lower": strict_lo,
            "ci_upper": strict_hi,
            "n_valid_replicates": int(len(strict_values)),
            "valid_replicate_fraction": float(len(strict_values) / n_bootstrap),
        },
        "relaxed_defined_target_macro": {
            "ci_lower": relaxed_lo,
            "ci_upper": relaxed_hi,
            "n_valid_replicates": int(len(relaxed_values)),
            "valid_replicate_fraction": float(len(relaxed_values) / n_bootstrap),
        },
        "n_bootstrap": int(n_bootstrap),
        "interpretation": (
            "coverage-conditioned AUC on high-confidence positive/negated B6 cells only; "
            "not directly comparable as a ceiling to full-surface B13/B15 macro AUC"
        ),
    }


def _state_truth_rows(merged: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    rows: list[dict] = []
    pooled: dict[str, list[tuple[int, float]]] = {state: [] for state in STATES}

    for target in TARGETS:
        truth = pd.to_numeric(merged[target], errors="coerce").astype(int)
        states = merged[f"{target}__state"].astype(str)
        confidence = merged[f"{target}__confidence"].astype(float)
        for state in STATES:
            mask = states.eq(state)
            y = truth.loc[mask]
            c = confidence.loc[mask]
            n = int(mask.sum())
            positives = int(y.sum())
            negatives = int(n - positives)
            rows.append({
                "target": target,
                "state": state,
                "n": n,
                "gold_positive": positives,
                "gold_negative": negatives,
                "gold_positive_fraction": float(positives / n) if n else np.nan,
                "mean_confidence": float(c.mean()) if n else np.nan,
                "median_confidence": float(c.median()) if n else np.nan,
                "n_confidence_ge_threshold": int((c >= min_confidence).sum()) if n else 0,
            })
            pooled[state].extend((int(v), float(cv)) for v, cv in zip(y.tolist(), c.tolist()))

    for state in STATES:
        values = pooled[state]
        truth = np.asarray([v for v, _ in values], dtype=int)
        confidence = np.asarray([c for _, c in values], dtype=float)
        n = int(len(values))
        positives = int(truth.sum()) if n else 0
        rows.append({
            "target": "__POOLED__",
            "state": state,
            "n": n,
            "gold_positive": positives,
            "gold_negative": int(n - positives),
            "gold_positive_fraction": float(positives / n) if n else np.nan,
            "mean_confidence": float(confidence.mean()) if n else np.nan,
            "median_confidence": float(np.median(confidence)) if n else np.nan,
            "n_confidence_ge_threshold": int((confidence >= min_confidence).sum()) if n else 0,
        })
    return pd.DataFrame(rows)


def _alignment_cells(
    merged: pd.DataFrame,
    b13: pd.DataFrame,
    b15: pd.DataFrame,
    min_confidence: float,
) -> pd.DataFrame:
    b13_idx = b13.set_index("StudyInstanceUID")
    b15_idx = b15.set_index("StudyInstanceUID")
    rows: list[dict] = []

    for _, study in merged.iterrows():
        uid = str(study["StudyInstanceUID"])
        for target in TARGETS:
            state = str(study[f"{target}__state"])
            confidence = float(study[f"{target}__confidence"])
            if state not in (STATE_POSITIVE, STATE_NEGATED) or confidence < min_confidence:
                continue

            truth = int(float(study[target]))
            teacher_binary = int(state == STATE_POSITIVE)
            teacher_soft = float(STATE_SOFT_SCORES[state])
            p13 = float(b13_idx.at[uid, target])
            p15 = float(b15_idx.at[uid, target])
            delta = p15 - p13
            teacher_sign = 1.0 if teacher_binary == 1 else -1.0
            truth_sign = 1.0 if truth == 1 else -1.0
            rows.append({
                "StudyInstanceUID": uid,
                "target": target,
                "truth": truth,
                "state": state,
                "confidence": confidence,
                "teacher_binary": teacher_binary,
                "teacher_soft_score": teacher_soft,
                "teacher_correct": bool(teacher_binary == truth),
                "b13": p13,
                "b15": p15,
                "delta_b15_minus_b13": delta,
                "movement_toward_teacher": delta * teacher_sign,
                "movement_toward_truth": delta * truth_sign,
                "b13_abs_distance_to_teacher": abs(p13 - teacher_soft),
                "b15_abs_distance_to_teacher": abs(p15 - teacher_soft),
                "change_abs_distance_to_teacher": abs(p15 - teacher_soft) - abs(p13 - teacher_soft),
                "b13_abs_distance_to_truth": abs(p13 - truth),
                "b15_abs_distance_to_truth": abs(p15 - truth),
                "change_abs_distance_to_truth": abs(p15 - truth) - abs(p13 - truth),
            })
    return pd.DataFrame(rows)


def _summarize_alignment(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n_cells": 0,
            "n_studies": 0,
            "mean_movement_toward_teacher": None,
            "median_movement_toward_teacher": None,
            "fraction_move_toward_teacher": None,
            "mean_movement_toward_truth": None,
            "fraction_move_toward_truth": None,
            "mean_change_abs_distance_to_teacher": None,
            "mean_change_abs_distance_to_truth": None,
        }
    return {
        "n_cells": int(len(frame)),
        "n_studies": int(frame["StudyInstanceUID"].nunique()),
        "mean_movement_toward_teacher": float(frame["movement_toward_teacher"].mean()),
        "median_movement_toward_teacher": float(frame["movement_toward_teacher"].median()),
        "fraction_move_toward_teacher": float((frame["movement_toward_teacher"] > 0).mean()),
        "mean_movement_toward_truth": float(frame["movement_toward_truth"].mean()),
        "fraction_move_toward_truth": float((frame["movement_toward_truth"] > 0).mean()),
        "mean_change_abs_distance_to_teacher": float(frame["change_abs_distance_to_teacher"].mean()),
        "mean_change_abs_distance_to_truth": float(frame["change_abs_distance_to_truth"].mean()),
    }


def _cluster_bootstrap_alignment(frame: pd.DataFrame, *, n_bootstrap: int, seed: int) -> dict:
    if frame.empty:
        return {"n_bootstrap": n_bootstrap, "n_valid_replicates": 0, "metrics": {}}
    uids = frame["StudyInstanceUID"].drop_duplicates().astype(str).tolist()
    groups = {uid: frame.loc[frame["StudyInstanceUID"].astype(str).eq(uid)] for uid in uids}
    rng = np.random.default_rng(seed)
    metric_names = [
        "mean_movement_toward_teacher",
        "fraction_move_toward_teacher",
        "mean_movement_toward_truth",
        "fraction_move_toward_truth",
        "mean_change_abs_distance_to_teacher",
        "mean_change_abs_distance_to_truth",
    ]
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    point = _summarize_alignment(frame)

    for _ in range(n_bootstrap):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        replicate = pd.concat([groups[str(uid)] for uid in sampled], ignore_index=True)
        summary = _summarize_alignment(replicate)
        for name in metric_names:
            value = summary[name]
            if value is not None and np.isfinite(value):
                draws[name].append(float(value))

    out = {}
    for name in metric_names:
        values = np.asarray(draws[name], dtype=float)
        if values.size:
            lo, hi = np.percentile(values, [2.5, 97.5])
            out[name] = {
                "point": point[name],
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "n_valid_replicates": int(values.size),
            }
        else:
            out[name] = {"point": None, "ci_lower": None, "ci_upper": None, "n_valid_replicates": 0}
    return {
        "n_bootstrap": int(n_bootstrap),
        "cluster_unit": "StudyInstanceUID",
        "n_cluster_studies": int(len(uids)),
        "metrics": out,
    }


def _alignment_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target in TARGETS:
        part = frame.loc[frame["target"].eq(target)]
        wrong = part.loc[~part["teacher_correct"]]
        correct = part.loc[part["teacher_correct"]]
        rows.append({
            "target": target,
            "n_high_confidence": int(len(part)),
            "n_teacher_correct": int(len(correct)),
            "n_teacher_wrong": int(len(wrong)),
            "wrong_mean_movement_toward_teacher": float(wrong["movement_toward_teacher"].mean()) if len(wrong) else np.nan,
            "wrong_fraction_move_toward_teacher": float((wrong["movement_toward_teacher"] > 0).mean()) if len(wrong) else np.nan,
            "wrong_mean_change_abs_distance_to_truth": float(wrong["change_abs_distance_to_truth"].mean()) if len(wrong) else np.nan,
            "correct_mean_movement_toward_teacher": float(correct["movement_toward_teacher"].mean()) if len(correct) else np.nan,
            "correct_fraction_move_toward_teacher": float((correct["movement_toward_teacher"] > 0).mean()) if len(correct) else np.nan,
        })
    return pd.DataFrame(rows)


def run_b6_b15_gold_diagnostic(
    *,
    data_root: str | Path,
    structured_csv: str | Path,
    b13_predictions: str | Path,
    b15_predictions: str | Path,
    out_root: str | Path = "runs/b6_b15_gold_diagnostic",
    min_confidence: float = 0.75,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0,1]")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")

    root = Path(data_root)
    train = load_train_csv(root / "train.csv")
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58 or gold[TARGETS].isna().any().any():
        raise ValueError("diagnostic requires the complete 58-study gold development surface")

    merged = _read_structured_gold(structured_csv, gold)
    uids = gold["StudyInstanceUID"].tolist()
    b13 = _read_gold_predictions(b13_predictions, uids)
    b15 = _read_gold_predictions(b15_predictions, uids)
    truth = gold[TARGETS].to_numpy(dtype=np.float64)

    state_score = np.empty_like(truth, dtype=np.float64)
    eligible = np.zeros_like(truth, dtype=bool)
    for j, target in enumerate(TARGETS):
        states = merged[f"{target}__state"].astype(str)
        confidence = merged[f"{target}__confidence"].astype(float).to_numpy()
        state_score[:, j] = states.map(STATE_SOFT_SCORES).to_numpy(dtype=float)
        eligible[:, j] = states.isin([STATE_POSITIVE, STATE_NEGATED]).to_numpy() & (confidence >= min_confidence)

    selective_teacher = bootstrap_selective_macro_auc(
        truth, state_score, eligible, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    selective_teacher.update({
        "n_eligible_cells": int(eligible.sum()),
        "possible_gold_cells": int(truth.size),
        "coverage": float(eligible.mean()),
        "soft_scores": {"positive": STATE_SOFT_SCORES[STATE_POSITIVE], "negated": STATE_SOFT_SCORES[STATE_NEGATED]},
        "min_confidence": float(min_confidence),
    })

    full_state_result = bootstrap_macro_auc(truth, state_score, n_bootstrap=n_bootstrap, seed=seed + 2)
    full_state_baseline = full_state_result.to_dict()
    full_state_baseline.update({
        "surface_cells": int(truth.size),
        "state_scores": dict(STATE_SOFT_SCORES),
        "interpretation": (
            "descriptive state-only ranking baseline on the exact full gold surface; "
            "comparable as an AUC baseline to model scores but not a theoretical ceiling"
        ),
    })

    b13_array = b13[TARGETS].to_numpy(dtype=np.float64)
    b15_array = b15[TARGETS].to_numpy(dtype=np.float64)
    b13_result = bootstrap_macro_auc(truth, b13_array, n_bootstrap=n_bootstrap, seed=seed + 3).to_dict()
    b15_result = bootstrap_macro_auc(truth, b15_array, n_bootstrap=n_bootstrap, seed=seed + 4).to_dict()
    paired = compare_runs(truth, b13_array, b15_array, n_bootstrap=n_bootstrap, seed=seed + 5)

    state_audit = _state_truth_rows(merged, min_confidence)
    alignment = _alignment_cells(merged, b13, b15, min_confidence)
    by_target = _alignment_by_target(alignment)

    correct = alignment.loc[alignment["teacher_correct"]]
    wrong = alignment.loc[~alignment["teacher_correct"]]
    all_summary = _summarize_alignment(alignment)
    correct_summary = _summarize_alignment(correct)
    wrong_summary = _summarize_alignment(wrong)
    correct_bootstrap = _cluster_bootstrap_alignment(correct, n_bootstrap=n_bootstrap, seed=seed + 11)
    wrong_bootstrap = _cluster_bootstrap_alignment(wrong, n_bootstrap=n_bootstrap, seed=seed + 12)

    wrong_move_ci = wrong_bootstrap["metrics"].get("mean_movement_toward_teacher", {})
    wrong_truth_dist_ci = wrong_bootstrap["metrics"].get("mean_change_abs_distance_to_truth", {})
    evidence_flags = {
        "teacher_wrong_mean_movement_toward_teacher_positive": bool(
            wrong_summary["mean_movement_toward_teacher"] is not None and wrong_summary["mean_movement_toward_teacher"] > 0
        ),
        "teacher_wrong_movement_toward_teacher_ci_above_zero": bool(
            wrong_move_ci.get("ci_lower") is not None and wrong_move_ci["ci_lower"] > 0
        ),
        "teacher_wrong_mean_truth_distance_increased": bool(
            wrong_summary["mean_change_abs_distance_to_truth"] is not None and wrong_summary["mean_change_abs_distance_to_truth"] > 0
        ),
        "teacher_wrong_truth_distance_change_ci_above_zero": bool(
            wrong_truth_dist_ci.get("ci_lower") is not None and wrong_truth_dist_ci["ci_lower"] > 0
        ),
    }

    payload = {
        "experiment": "B6_B15_reused_gold_diagnostic_v1",
        "n_gold_studies": int(len(gold)),
        "n_gold_target_cells": int(truth.size),
        "development_only": True,
        "independent_validation": False,
        "uses_gpu": False,
        "performs_training": False,
        "changes_model_selection": False,
        "b6_min_confidence": float(min_confidence),
        "coverage_conditioned_teacher_auc": selective_teacher,
        "full_surface_state_only_baseline": full_state_baseline,
        "model_gold_reproduction": {
            "B13": b13_result,
            "B15": b15_result,
            "paired_B15_minus_B13": paired,
        },
        "noise_alignment": {
            "all_high_confidence_cells": all_summary,
            "teacher_correct_cells": correct_summary,
            "teacher_wrong_cells": wrong_summary,
            "teacher_correct_cluster_bootstrap": correct_bootstrap,
            "teacher_wrong_cluster_bootstrap": wrong_bootstrap,
            "evidence_flags": evidence_flags,
            "interpretation_rule": (
                "The strongest evidence for B6-error imitation is teacher-wrong cells moving "
                "toward the B6 class while simultaneously moving farther from expert truth. "
                "This remains a descriptive reused-gold diagnostic, not independent validation."
            ),
        },
        "caveats": [
            "Coverage-conditioned teacher AUC uses only high-confidence positive/negated B6 cells and is not a ceiling for full-surface model AUC.",
            "The full-surface state-only baseline assigns uncertain and unmentioned states 0.50 by frozen diagnostic convention; it is a descriptive baseline, not a learned teacher.",
            "B13 and B15 were trained on different B6 study surfaces, so their gold movement comparison is descriptive rather than a pure encoder-only causal contrast.",
            "Per-target results are diagnostic only and must not be used for target-specific model mixing or tuning.",
        ],
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    state_audit.to_csv(out / "state_truth_audit.csv", index=False)
    alignment.to_csv(out / "high_confidence_alignment_cells.csv", index=False)
    by_target.to_csv(out / "alignment_by_target.csv", index=False)
    (out / "diagnostic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("coverage-conditioned teacher macro AUC:", selective_teacher["macro_auc_over_defined_targets"], f"(coverage={selective_teacher['coverage']:.3f})")
    print("full-surface B6 state-only macro AUC:", full_state_baseline["macro_auc"], f"[{full_state_baseline['ci_lower']:.4f}, {full_state_baseline['ci_upper']:.4f}]")
    print("B13 / B15 reproduced macro AUC:", b13_result["macro_auc"], "/", b15_result["macro_auc"])
    print("high-confidence B6 cells:", len(alignment), "| correct:", len(correct), "| wrong:", len(wrong))
    if len(wrong):
        print("teacher-wrong mean B13->B15 movement toward B6:", wrong_summary["mean_movement_toward_teacher"], "| mean truth-distance change:", wrong_summary["mean_change_abs_distance_to_truth"])
    print(out / "diagnostic.json")
    print(out / "state_truth_audit.csv")
    print(out / "high_confidence_alignment_cells.csv")
    print(out / "alignment_by_target.csv")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b6-b15-diagnostic")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--structured", default="runs/b6_report_labels_v121/structured_labels.csv")
    parser.add_argument("--b13-predictions", default="runs/b13_imagenet/gold_eval/gold_predictions.csv")
    parser.add_argument("--b15-predictions", default="runs/b15_mri_ssl/gold_confirmation/gold_predictions.csv")
    parser.add_argument("--out-root", default="runs/b6_b15_gold_diagnostic")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    payload = run_b6_b15_gold_diagnostic(
        data_root=args.data_root,
        structured_csv=args.structured,
        b13_predictions=args.b13_predictions,
        b15_predictions=args.b15_predictions,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
