"""B23 labeller audit: a DESCRIPTIVE, POST-HOC labeller comparison.

**This is not a confirmatory test, and its interval is not an independence
claim.** The B23 prompt was written from `state_truth_audit.csv`, which
aggregates all 58 expert studies, so the labeller has already seen aggregate
information about every case it is scored on. A paired CI excluding zero
therefore cannot restore independence; it can only say that the two labellers
differ on a surface both were tuned against. Report it as a development
diagnostic and never as validation.

What the audit is still good for is large, structural differences -- coverage
going from 36% to 85%, or specificity moving off 0.61 -- which are far bigger
than the optimism the reuse can manufacture. Read it for those, not for a
two-point macro-AUC edge.


The 58 expert-labelled studies cannot rank near-neighbour MRI models -- the B22
duration trajectory moved 0.0439 within a single run in which nothing but the
epoch count changed, which is larger than the entire B13->B20 campaign. But that
same surface is perfectly adequate for measuring a *report labeller*, because
here the question is not "is model A better than model B by 0.002" but "does
this labeller agree with expert truth substantially more often than a regex".
The B6 regex leaves an enormous margin: specificity 0.6061, precision 0.6905,
coverage 0.3606.

This audit therefore reports, for B23 and frozen B6 side by side:

  * the confusion summary on usable cells (sensitivity/specificity/PPV/NPV);
  * coverage;
  * P(expert positive | state) for all four states;
  * the state-only ranking macro AUC on all 696 gold cells;
  * a paired study-cluster bootstrap of the state-only macro AUC difference.

The state-only scores are the frozen diagnostic values already used by the
B6/B15 gold diagnostic, so the B23 number is directly comparable to the B6
state-only baseline of 0.7025.

Auditing a labeller against gold is more defensible than selecting an MRI model
against it, because the margin over a 0.6061-specificity regex is large. It is
still not independent. Do not use this audit to pick a downstream checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .evaluation import macro_auc_from_arrays

# Frozen diagnostic state->score map, identical to the B6/B15 gold diagnostic so
# the resulting macro AUC is comparable to the recorded B6 baseline of 0.7025.
STATE_ONLY_SCORES = {
    "positive": 0.85,
    "negated": 0.05,
    "uncertain": 0.50,
    "unmentioned": 0.50,
}
DEFAULT_MIN_CONFIDENCE = 0.75


def _state_matrix(structured: pd.DataFrame, uids: list[str]) -> pd.DataFrame:
    frame = structured.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    frame = frame.set_index("StudyInstanceUID")
    missing = [uid for uid in uids if uid not in frame.index]
    if missing:
        raise ValueError(f"{len(missing)} gold studies absent from the labeller export")
    return frame.loc[uids]


def state_only_scores(states: pd.DataFrame) -> np.ndarray:
    """Map the four parser states onto the frozen diagnostic ranking scores."""
    out = np.full((len(states), len(TARGETS)), np.nan, dtype=np.float64)
    for j, target in enumerate(TARGETS):
        column = states[f"{target}__state"].astype(str).str.strip().str.lower()
        unknown = sorted(set(column.unique()).difference(STATE_ONLY_SCORES))
        if unknown:
            raise ValueError(f"target {target!r} has unknown states {unknown}")
        out[:, j] = column.map(STATE_ONLY_SCORES).to_numpy(dtype=np.float64)
    return out


def confusion_summary(
    truth: np.ndarray,
    states: pd.DataFrame,
    confidences: np.ndarray,
    *,
    min_confidence: float,
) -> dict:
    """Sensitivity/specificity/PPV/NPV over cells the downstream loss would use."""
    tp = fp = tn = fn = 0
    usable_cells = 0
    total_cells = 0
    for j, target in enumerate(TARGETS):
        state = states[f"{target}__state"].astype(str).str.strip().str.lower().to_numpy()
        conf = confidences[:, j]
        y = truth[:, j]
        labelled = np.isfinite(y)
        total_cells += int(labelled.sum())
        usable = labelled & np.isin(state, ("positive", "negated")) & (conf >= min_confidence)
        usable_cells += int(usable.sum())
        called_positive = usable & (state == "positive")
        called_negative = usable & (state == "negated")
        tp += int(np.sum(called_positive & (y == 1)))
        fp += int(np.sum(called_positive & (y == 0)))
        tn += int(np.sum(called_negative & (y == 0)))
        fn += int(np.sum(called_negative & (y == 1)))

    def _ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "positive_precision": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "balanced_accuracy": float(
            np.nanmean([_ratio(tp, tp + fn), _ratio(tn, tn + fp)])
        ),
        "usable_cells": usable_cells,
        "labelled_cells": total_cells,
        "coverage": _ratio(usable_cells, total_cells),
    }


def state_truth_table(truth: np.ndarray, states: pd.DataFrame) -> pd.DataFrame:
    """P(expert positive | state) for every target and state, plus a pooled row."""
    rows = []
    for j, target in enumerate(TARGETS):
        column = states[f"{target}__state"].astype(str).str.strip().str.lower().to_numpy()
        y = truth[:, j]
        for state in STATE_ONLY_SCORES:
            mask = (column == state) & np.isfinite(y)
            n = int(mask.sum())
            positives = int(np.sum(y[mask] == 1)) if n else 0
            rows.append(
                {
                    "target": target,
                    "state": state,
                    "n": n,
                    "gold_positive": positives,
                    "gold_negative": n - positives,
                    "p_gold_positive": float(positives / n) if n else float("nan"),
                }
            )
    table = pd.DataFrame(rows)
    pooled = (
        table.groupby("state", as_index=False)[["n", "gold_positive", "gold_negative"]]
        .sum()
        .assign(target="__pooled__")
    )
    pooled["p_gold_positive"] = pooled["gold_positive"] / pooled["n"].replace(0, np.nan)
    return pd.concat([table, pooled[table.columns]], ignore_index=True)


def paired_state_only_bootstrap(
    truth: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> dict:
    """Study-cluster paired bootstrap of the state-only macro AUC difference.

    Resamples whole studies so the 12 correlated target cells from one knee stay
    together, and rejects a replicate unless all 12 target AUCs are defined for
    both labellers -- the same strict estimand the weak-v2 surface uses.
    """
    rng = np.random.default_rng(seed)
    n = truth.shape[0]
    deltas = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        macro_a, per_a = macro_auc_from_arrays(truth[idx], score_a[idx])
        macro_b, per_b = macro_auc_from_arrays(truth[idx], score_b[idx])
        if not np.all(np.isfinite(per_a)) or not np.all(np.isfinite(per_b)):
            continue
        deltas.append(macro_b - macro_a)
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        raise RuntimeError("no usable bootstrap replicates")
    return {
        "median_difference": float(np.median(deltas)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "probability_candidate_better": float(np.mean(deltas > 0)),
        "valid_replicates": int(deltas.size),
        "requested_replicates": int(n_bootstrap),
    }


def audit_labeller(
    train_csv: str | Path,
    candidate_structured: str | Path,
    *,
    baseline_structured: str | Path | None = None,
    out_root: str | Path = "runs/b23_labeller_audit",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    n_bootstrap: int = 5000,
) -> dict:
    """Score one labeller -- and optionally a baseline -- against expert gold."""
    train = load_train_csv(train_csv)
    gold = train.loc[gold_mask(train)].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(dtype=np.float64)

    candidate = _state_matrix(pd.read_csv(candidate_structured), uids)
    cand_conf = np.column_stack(
        [candidate[f"{target}__confidence"].to_numpy(dtype=np.float64) for target in TARGETS]
    )
    cand_scores = state_only_scores(candidate)
    cand_macro, cand_per_target = macro_auc_from_arrays(truth, cand_scores)

    payload: dict = {
        "interpretation": (
            "DESCRIPTIVE / POST-HOC. The B23 prompt was designed using aggregate "
            "information from all 58 expert studies, so this comparison is a "
            "development diagnostic, not confirmatory validation, and the paired "
            "interval is not an independence claim."
        ),
        "confirmatory": False,
        "n_gold_studies": int(len(uids)),
        "n_gold_cells": int(np.isfinite(truth).sum()),
        "min_confidence": float(min_confidence),
        "candidate": {
            "state_only_macro_auc": float(cand_macro),
            "per_target_auc": {
                target: float(value) for target, value in zip(TARGETS, cand_per_target)
            },
            "confusion": confusion_summary(
                truth, candidate, cand_conf, min_confidence=min_confidence
            ),
        },
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    state_truth_table(truth, candidate).to_csv(out / "candidate_state_truth.csv", index=False)

    if baseline_structured is not None:
        baseline = _state_matrix(pd.read_csv(baseline_structured), uids)
        base_conf = np.column_stack(
            [baseline[f"{target}__confidence"].to_numpy(dtype=np.float64) for target in TARGETS]
        )
        base_scores = state_only_scores(baseline)
        base_macro, base_per_target = macro_auc_from_arrays(truth, base_scores)
        payload["baseline"] = {
            "state_only_macro_auc": float(base_macro),
            "per_target_auc": {
                target: float(value) for target, value in zip(TARGETS, base_per_target)
            },
            "confusion": confusion_summary(
                truth, baseline, base_conf, min_confidence=min_confidence
            ),
        }
        payload["raw_difference"] = float(cand_macro - base_macro)
        payload["paired_bootstrap"] = paired_state_only_bootstrap(
            truth, base_scores, cand_scores, n_bootstrap=n_bootstrap
        )
        state_truth_table(truth, baseline).to_csv(out / "baseline_state_truth.csv", index=False)

    (out / "labeller_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# Predeclared adoption thresholds, taken from the measured frozen B6 baseline.
B6_STATE_ONLY_MACRO_AUC = 0.7025
B6_COVERAGE = 0.3606
B6_SPECIFICITY = 0.6061


def load_labeller_audit(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "candidate" not in payload:
        raise ValueError(f"{path} is not a B23 labeller audit payload")
    return payload


def gate_status(payload: dict) -> dict:
    """Evaluate the predeclared B23 adoption gate against an audit payload.

    Every criterion must hold. The macro-AUC interval is included because it was
    predeclared, but the structural criteria -- coverage and specificity -- are
    the ones that carry the weight, since the prompt was written with aggregate
    knowledge of these same 58 studies and a small AUC edge could be optimism.
    """
    candidate = payload.get("candidate") or {}
    confusion = candidate.get("confusion") or {}
    boot = payload.get("paired_bootstrap") or {}
    reasons: list[str] = []

    macro = float(candidate.get("state_only_macro_auc", float("nan")))
    coverage = float(confusion.get("coverage", float("nan")))
    specificity = float(confusion.get("specificity", float("nan")))
    ci_low = float(boot.get("ci_low", float("nan")))

    if not (macro > B6_STATE_ONLY_MACRO_AUC):
        reasons.append(f"state-only macro AUC {macro:.4f} <= B6 {B6_STATE_ONLY_MACRO_AUC}")
    if not (coverage > B6_COVERAGE):
        reasons.append(f"coverage {coverage:.4f} <= B6 {B6_COVERAGE}")
    if not (specificity > B6_SPECIFICITY):
        reasons.append(f"specificity {specificity:.4f} <= B6 {B6_SPECIFICITY}")
    if not (ci_low > 0.0):
        reasons.append(
            "paired 95% CI does not exclude zero "
            f"(low={ci_low:.4f}); run the audit with --baseline to produce it"
        )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "state_only_macro_auc": macro,
        "coverage": coverage,
        "specificity": specificity,
        "paired_ci_low": ci_low,
        "evidence_type": "descriptive/post-hoc; not confirmatory validation",
    }


def format_audit(payload: dict) -> str:
    lines = [
        "B23 labeller audit -- DESCRIPTIVE / POST-HOC, not confirmatory",
        "  the prompt was designed from these same 58 studies in aggregate",
        f"  gold studies {payload['n_gold_studies']} | gold cells {payload['n_gold_cells']}",
        "",
    ]
    for name in ("baseline", "candidate"):
        block = payload.get(name)
        if block is None:
            continue
        conf = block["confusion"]
        lines.append(f"  {name}")
        lines.append(f"    state-only macro AUC  {block['state_only_macro_auc']:.10f}")
        lines.append(f"    sensitivity           {conf['sensitivity']:.4f}")
        lines.append(f"    specificity           {conf['specificity']:.4f}")
        lines.append(f"    positive precision    {conf['positive_precision']:.4f}")
        lines.append(f"    NPV                   {conf['npv']:.4f}")
        lines.append(f"    coverage              {conf['coverage']:.4f}")
        lines.append(f"    usable cells          {conf['usable_cells']} / {conf['labelled_cells']}")
        lines.append("")
    if "paired_bootstrap" in payload:
        boot = payload["paired_bootstrap"]
        lines.append(f"  raw difference        {payload['raw_difference']:+.10f}")
        lines.append(f"  paired median         {boot['median_difference']:+.10f}")
        lines.append(f"  95% paired CI         [{boot['ci_low']:+.10f},{boot['ci_high']:+.10f}]")
        lines.append(f"  P(candidate better)   {boot['probability_candidate_better']:.4f}")
        lines.append(
            f"  valid replicates      {boot['valid_replicates']}/{boot['requested_replicates']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a report labeller against expert gold")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--candidate", required=True, help="candidate structured_labels.csv")
    parser.add_argument("--baseline", default=None, help="frozen B6 structured_labels.csv")
    parser.add_argument("--out-root", default="runs/b23_labeller_audit")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    payload = audit_labeller(
        args.train_csv,
        args.candidate,
        baseline_structured=args.baseline,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
        n_bootstrap=args.n_bootstrap,
    )
    print(format_audit(payload))


if __name__ == "__main__":  # pragma: no cover
    main()
