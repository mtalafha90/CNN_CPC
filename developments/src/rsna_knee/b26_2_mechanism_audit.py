"""Post-hoc mechanism audit for the completed B26.2 experiment.

This module does not train a model and does not modify supervision.  It asks
what the 171 quality-approved Synovitis fills changed *inside the frozen B20
loss recipe*, and compares target co-occurrence on the weak-supervision surface
with the already-reused 58-study expert surface.

The expert contingency is explicitly post-hoc descriptive evidence only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import (
    load_frozen_b6_export,
    prepare_b7_supervision,
    target_balance_multipliers,
)
from .b26_2_training import apply_b26_2_fill_to_arrays
from .constants import TARGETS
from .data import gold_mask, load_train_csv

EXPERIMENT = "B26_2_posthoc_supervision_mechanism_audit_v1"
SYN = "Synovitis"
RELATED = ("Effusion", "Baker's")


def _target_mass_table(y: np.ndarray, w: np.ndarray) -> dict[str, dict]:
    mult = target_balance_multipliers(w)
    out: dict[str, dict] = {}
    for j, target in enumerate(TARGETS):
        active = w[:, j] > 0
        pos = active & (y[:, j] > 0.5)
        neg = active & (y[:, j] < 0.5)
        raw_pos = float(w[pos, j].sum())
        raw_neg = float(w[neg, j].sum())
        raw_total = raw_pos + raw_neg
        m = float(mult[j])
        eff_pos = raw_pos * m
        eff_neg = raw_neg * m
        eff_total = eff_pos + eff_neg
        out[target] = {
            "positive_cells": int(pos.sum()),
            "negative_cells": int(neg.sum()),
            "usable_cells": int(active.sum()),
            "raw_positive_weight_mass": raw_pos,
            "raw_negative_weight_mass": raw_neg,
            "raw_total_weight_mass": raw_total,
            "target_balance_multiplier": m,
            "effective_positive_mass_before_global_denominator": eff_pos,
            "effective_negative_mass_before_global_denominator": eff_neg,
            "effective_total_mass_before_global_denominator": eff_total,
            "within_target_positive_mass_fraction": (eff_pos / eff_total) if eff_total else None,
            "within_target_negative_mass_fraction": (eff_neg / eff_total) if eff_total else None,
        }
    total_effective = sum(x["effective_total_mass_before_global_denominator"] for x in out.values())
    for target in TARGETS:
        out[target]["normalized_total_loss_share"] = (
            out[target]["effective_total_mass_before_global_denominator"] / total_effective
            if total_effective else None
        )
    return out


def _contingency_from_supervision(
    y: np.ndarray,
    w: np.ndarray,
    a: str,
    b: str,
) -> dict:
    ja, jb = TARGETS.index(a), TARGETS.index(b)
    mask = (w[:, ja] > 0) & (w[:, jb] > 0)
    av = y[mask, ja] > 0.5
    bv = y[mask, jb] > 0.5
    return _binary_contingency(av, bv, label_a=a, label_b=b)


def _contingency_from_gold(frame: pd.DataFrame, a: str, b: str) -> dict:
    subset = frame[[a, b]].copy()
    mask = subset[a].notna() & subset[b].notna()
    av = subset.loc[mask, a].to_numpy(dtype=float) > 0.5
    bv = subset.loc[mask, b].to_numpy(dtype=float) > 0.5
    return _binary_contingency(av, bv, label_a=a, label_b=b)


def _binary_contingency(a, b, *, label_a: str, label_b: str) -> dict:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("binary contingency arrays must align")
    pp = int((a & b).sum())
    pn = int((a & ~b).sum())
    np_ = int((~a & b).sum())
    nn = int((~a & ~b).sum())
    n = int(a.size)
    denom = float((pp + pn) * (np_ + nn) * (pp + np_) * (pn + nn)) ** 0.5
    phi = ((pp * nn - pn * np_) / denom) if denom > 0 else None
    b_pos = pp + np_
    b_neg = pn + nn
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_both_defined": n,
        "a_pos_b_pos": pp,
        "a_pos_b_neg": pn,
        "a_neg_b_pos": np_,
        "a_neg_b_neg": nn,
        "phi": phi,
        "p_a_positive_given_b_positive": (pp / b_pos) if b_pos else None,
        "p_a_positive_given_b_negative": (pn / b_neg) if b_neg else None,
    }


def build_audit(
    *,
    train_csv: str | Path,
    b6_root: str | Path,
    filtered_candidates: str | Path,
) -> dict:
    train = load_train_csv(train_csv).copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

    b6_frame, _policy, _audit = load_frozen_b6_export(b6_root)
    uids, base_y, base_w, _summary = prepare_b7_supervision(train, b6_frame)
    uids = [str(x) for x in uids]
    if len(uids) != 3120 or int((base_w > 0).sum()) != 14123:
        raise RuntimeError("mechanism audit requires the exact historical B20/B6 surface")

    filtered = pd.read_csv(filtered_candidates, dtype={"StudyInstanceUID": str})
    final_y, final_w, fill_diag = apply_b26_2_fill_to_arrays(
        uids, base_y, base_w, filtered
    )
    if int(fill_diag["accepted_total"]) != 171:
        raise RuntimeError("mechanism audit requires the completed 171-cell B26.2 fill")

    base_mass = _target_mass_table(base_y, base_w)
    final_mass = _target_mass_table(final_y, final_w)
    syn_base = base_mass[SYN]
    syn_final = final_mass[SYN]

    expert = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    if len(expert) != 58:
        raise RuntimeError("expected the reused 58-study expert surface")

    cooccurrence = {}
    for related in RELATED:
        cooccurrence[related] = {
            "b6_supervision": _contingency_from_supervision(base_y, base_w, SYN, related),
            "b26_2_supervision": _contingency_from_supervision(final_y, final_w, SYN, related),
            "reused_expert_posthoc": _contingency_from_gold(expert, SYN, related),
        }

    result = {
        "experiment": EXPERIMENT,
        "role": "post-hoc mechanism audit; no training; no promotion decision",
        "working_model": "B20_crop_only_joint_focus",
        "b26_2_status": "not promoted after reused-expert diagnostic",
        "training_studies": len(uids),
        "base_usable_cells": int((base_w > 0).sum()),
        "final_usable_cells": int((final_w > 0).sum()),
        "fill_diagnostics": fill_diag,
        "synovitis_loss_mass": {
            "b6": syn_base,
            "b26_2": syn_final,
            "multiplier_ratio_b26_2_over_b6": (
                syn_final["target_balance_multiplier"] / syn_base["target_balance_multiplier"]
            ),
            "negative_mass_fraction_change": (
                syn_final["within_target_negative_mass_fraction"]
                - syn_base["within_target_negative_mass_fraction"]
            ),
        },
        "all_target_loss_mass": {
            "b6": base_mass,
            "b26_2": final_mass,
        },
        "cooccurrence": cooccurrence,
        "expert_surface_warning": (
            "The 58-study expert contingency is already reused development data and is included "
            "only to understand the completed negative result. It must not be treated as new "
            "validation or used to claim promotion."
        ),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser("B26.2 post-hoc supervision mechanism audit")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--filtered-candidates", required=True)
    ap.add_argument(
        "--out",
        default="runs/b26_2_training/mechanism_audit.json",
    )
    args = ap.parse_args()

    result = build_audit(
        train_csv=Path(args.data_root) / "train.csv",
        b6_root=args.b6_root,
        filtered_candidates=args.filtered_candidates,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    sb = result["synovitis_loss_mass"]["b6"]
    sf = result["synovitis_loss_mass"]["b26_2"]
    print("=" * 76)
    print("B26.2 SUPERVISION MECHANISM AUDIT -- POST-HOC, NO TRAINING")
    print("=" * 76)
    print(
        f"Synovitis B6      : {sb['positive_cells']} pos / {sb['negative_cells']} neg | "
        f"mult={sb['target_balance_multiplier']:.6f} | "
        f"neg mass fraction={sb['within_target_negative_mass_fraction']:.4f}"
    )
    print(
        f"Synovitis B26.2   : {sf['positive_cells']} pos / {sf['negative_cells']} neg | "
        f"mult={sf['target_balance_multiplier']:.6f} | "
        f"neg mass fraction={sf['within_target_negative_mass_fraction']:.4f}"
    )
    print()
    for related, block in result["cooccurrence"].items():
        print(f"Synovitis vs {related}")
        for surface in ("b6_supervision", "b26_2_supervision", "reused_expert_posthoc"):
            x = block[surface]
            print(
                f"  {surface:24s} n={x['n_both_defined']:4d} "
                f"phi={x['phi'] if x['phi'] is not None else float('nan'):+.4f} "
                f"P(Syn+|{related}+)=\n"
                f"    {x['p_a_positive_given_b_positive'] if x['p_a_positive_given_b_positive'] is not None else float('nan'):.4f} | "
                f"P(Syn+|{related}-)={x['p_a_positive_given_b_negative'] if x['p_a_positive_given_b_negative'] is not None else float('nan'):.4f}"
            )
        print()
    print("saved:", out)


if __name__ == "__main__":
    main()
