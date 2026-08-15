#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rsna_knee.b24_training as bt

from rsna_knee.b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    prepare_b7_supervision,
)
from rsna_knee.chatgpt_hybrid_supervision import (
    HYBRID_EXPERIMENT,
    load_hybrid_export,
)
from rsna_knee.b15_ssl import load_frozen_v2_manifest
from rsna_knee.b24_protocol import MODE_CANDIDATE, MODE_CONTROL
from rsna_knee.b24_supervision import surface_diagnostics
from rsna_knee.data import gold_mask, load_train_csv


EXPERIMENT = "B25X_chatgpt_hybrid_training_v1"
ARMS = ("control", "hybrid", "fill")


def build_hybrid_surface(
    config,
    *,
    b6_root,
    hybrid_root,
    weak_holdout_root=None,
    candidate_kind="hybrid",
):
    """Build one matched study surface for B6, Hybrid and B6+Hybrid-fill.

    Every arm uses the same sorted intersection of B6-active and hybrid-active
    studies, with the frozen weak-v2 holdout and all gold studies excluded.
    This keeps MRI exposure and batch membership comparable.  The ``fill`` arm
    preserves every B6 committed cell and adds hybrid supervision only where
    B6 is silent.
    """
    if candidate_kind not in ("hybrid", "fill"):
        raise ValueError("candidate_kind must be 'hybrid' or 'fill'")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    hybrid_frame, hybrid_policy, hybrid_audit = load_hybrid_export(hybrid_root)

    b6_uids, b6_y, b6_w, b6_summary = prepare_b7_supervision(train, b6_frame)
    hybrid_uids, hybrid_y, hybrid_w, hybrid_summary = prepare_b7_supervision(
        train, hybrid_frame
    )

    b6_index = {str(uid): i for i, uid in enumerate(b6_uids)}
    hybrid_index = {str(uid): i for i, uid in enumerate(hybrid_uids)}
    b6_active = set(b6_index)
    hybrid_active = set(hybrid_index)

    weak_holdout: set[str] = set()
    if weak_holdout_root is not None:
        _, manifest = load_frozen_v2_manifest(weak_holdout_root)
        weak_holdout = set(
            manifest.loc[
                manifest["split"] == "holdout", "StudyInstanceUID"
            ].astype(str)
        )

    gold_uids = set(
        train.loc[gold_mask(train), "StudyInstanceUID"].astype(str)
    )
    excluded = weak_holdout | gold_uids

    shared = sorted((b6_active & hybrid_active) - excluded)
    if len(shared) < 2:
        raise RuntimeError(
            f"B25X matched surface has only {len(shared)} studies"
        )

    y_b6 = np.stack([b6_y[b6_index[uid]] for uid in shared])
    w_b6 = np.stack([b6_w[b6_index[uid]] for uid in shared])
    y_hybrid = np.stack([hybrid_y[hybrid_index[uid]] for uid in shared])
    w_hybrid = np.stack([hybrid_w[hybrid_index[uid]] for uid in shared])

    fill_meta = None
    if candidate_kind == "hybrid":
        y_candidate = y_hybrid.copy()
        w_candidate = w_hybrid.copy()
    else:
        y_candidate = y_b6.copy()
        w_candidate = w_b6.copy()
        added = (w_b6 <= 0) & (w_hybrid > 0)
        y_candidate[added] = y_hybrid[added]
        w_candidate[added] = w_hybrid[added]

        b6_used = w_b6 > 0
        if not np.array_equal(y_candidate[b6_used], y_b6[b6_used]):
            raise RuntimeError("B25X fill arm changed a B6 target")
        if not np.array_equal(w_candidate[b6_used], w_b6[b6_used]):
            raise RuntimeError("B25X fill arm changed a B6 weight")

        fill_meta = {
            "policy": (
                "preserve every B6 committed cell exactly; add ChatGPT hybrid "
                "supervision only where B6 has zero supervision"
            ),
            "b6_cells_preserved": int(b6_used.sum()),
            "hybrid_only_cells_added": int(added.sum()),
            "b6_cells_dropped": 0,
            "b6_cells_overridden": 0,
            "final_usable_cells": int((w_candidate > 0).sum()),
        }

    diagnostics = surface_diagnostics(
        shared,
        y_b6,
        w_b6,
        y_candidate,
        w_candidate,
    )

    if set(shared) & excluded:
        raise RuntimeError("gold or weak-v2 study leaked into B25X training")

    return {
        "study_uids": shared,
        "control": {"targets": y_b6, "weights": w_b6},
        "candidate": {"targets": y_candidate, "weights": w_candidate},
        "diagnostics": diagnostics,
        "candidate_kind": candidate_kind,
        "fill_ablation": fill_meta,
        "excluded": {
            "gold": len(gold_uids),
            "weak_v2_holdout": len((b6_active & hybrid_active) & weak_holdout),
            "b23_holdout": 0,
        },
        "b6_active_studies": len(b6_active),
        "hybrid_active_studies": len(hybrid_active),
        "hybrid_cell_coverage": float(hybrid_audit.get("cell_coverage", float("nan"))),
        "hybrid_cache_file_sha256": str(hybrid_audit.get("cache_file_sha256", "")),
        "hybrid_source_provenance": str(hybrid_policy.get("source_provenance", "")),
        "b6_supervision": b6_summary,
        "hybrid_supervision": hybrid_summary,
    }


def format_hybrid_surface(surface):
    d = surface["diagnostics"]
    kind = surface["candidate_kind"]
    candidate_name = "Hybrid" if kind == "hybrid" else "B6+Hybrid-fill"
    lines = [
        "B25X ChatGPT-hybrid matched training surface",
        f"  candidate                   {candidate_name}",
        f"  shared studies              {d['studies']}",
        f"  possible cells              {d['possible_cells']}",
        "",
        f"  B6 usable cells             {d['control_usable_cells']} "
        f"({d['control_usable_cells']/d['possible_cells']:.1%})",
        f"  candidate usable cells      {d['candidate_usable_cells']} "
        f"({d['candidate_usable_cells']/d['possible_cells']:.1%})",
        f"  added by candidate          {d['cells_added_by_candidate']}",
        f"  dropped by candidate        {d['cells_dropped_by_candidate']}",
        "",
        f"  cells both committed on     {d['cells_in_both']}",
        f"  disagreements there         {d['disagreements_where_both_committed']} "
        f"({d['disagreement_rate']:.1%})",
        "",
        f"  excluded gold               {surface['excluded']['gold']}",
        f"  excluded weak-v2 holdout    {surface['excluded']['weak_v2_holdout']}",
        "",
        "  Same studies/order for every arm; no expert gold enters gradients.",
        "  Hybrid raw confidence is diagnostic only; definite states use fixed 0.90.",
    ]
    if surface.get("fill_ablation"):
        f = surface["fill_ablation"]
        lines.extend(
            [
                "",
                f"  B6 cells preserved          {f['b6_cells_preserved']}",
                f"  hybrid-only cells added     {f['hybrid_only_cells_added']}",
                f"  B6 cells overridden         {f['b6_cells_overridden']}",
            ]
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "B25X exploratory matched training with ChatGPT hybrid supervision"
        )
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--hybrid-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--report-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _read_config(args.config)
    config["data_root"] = args.data_root

    candidate_kind = "fill" if args.arm == "fill" else "hybrid"
    surface = build_hybrid_surface(
        config,
        b6_root=args.b6_root,
        hybrid_root=args.hybrid_root,
        weak_holdout_root=args.weak_holdout_root,
        candidate_kind=candidate_kind,
    )

    print("=" * 72)
    print("B25X CHATGPT HYBRID SUPERVISION")
    print("=" * 72)
    print(format_hybrid_surface(surface))
    print()
    print("Source provenance : MIXED/UNKNOWN original LLM provenance")
    print("Formal B23        : INCOMPATIBLE")
    print("Formal B24        : INELIGIBLE")
    print("Gold acceptance   : PROHIBITED")
    print("Promotion         : PROHIBITED")

    print()
    print("Per-target surface")
    print("-" * 72)
    for target, values in surface["diagnostics"]["per_target"].items():
        print(
            f"{target:18s} "
            f"B6={values['control_cells']:4d}  "
            f"Cand={values['candidate_cells']:4d}  "
            f"add={values['added_by_candidate']:4d}  "
            f"drop={values['dropped_by_candidate']:4d}  "
            f"disagree={values['disagreements']:4d}"
        )

    if args.dry_run:
        print()
        print("DRY RUN COMPLETE -- no GPU training performed.")
        return

    mode = MODE_CONTROL if args.arm == "control" else MODE_CANDIDATE
    exploratory_gate = {
        "passed": False,
        "reasons": [
            "ChatGPT hybrid source has mixed/unknown original LLM provenance",
            "experiment is exploratory and is not formal B23/B24",
        ],
        "exploratory_override": True,
        "formal_b24_eligible": False,
        "gold_acceptance_allowed": False,
        "experiment": EXPERIMENT,
    }

    old_gate = bt.require_passed_labeller_gate
    old_surface = bt.build_matched_surface
    old_identity = bt.mode_identity
    old_formatter = bt.format_surface

    def recorded_exploratory_gate(_):
        return exploratory_gate

    def surface_adapter(
        config,
        *,
        b6_root,
        b23_root,
        weak_holdout_root=None,
        b23_holdout_root=None,
    ):
        return build_hybrid_surface(
            config,
            b6_root=b6_root,
            hybrid_root=b23_root,
            weak_holdout_root=weak_holdout_root,
            candidate_kind=candidate_kind,
        )

    def identity_adapter(selected_mode):
        if selected_mode == MODE_CONTROL:
            return (
                "b25x_b6_control_v1",
                "B6 control on B25X matched hybrid surface",
            )
        if candidate_kind == "hybrid":
            return (
                "b25x_chatgpt_hybrid_only_v1",
                "ChatGPT hybrid weak supervision",
            )
        return (
            "b25x_b6_plus_chatgpt_hybrid_fill_v1",
            "B6 preserved + ChatGPT hybrid supervision on B6-silent cells",
        )

    bt.require_passed_labeller_gate = recorded_exploratory_gate
    bt.build_matched_surface = surface_adapter
    bt.mode_identity = identity_adapter
    bt.format_surface = format_hybrid_surface

    try:
        checkpoint = bt.train_b24(
            config,
            mode=mode,
            b6_root=args.b6_root,
            b23_root=args.hybrid_root,
            b23_holdout_root="__B25X_NO_B23_HOLDOUT__",
            weak_holdout_root=args.weak_holdout_root,
            series_policy_path=args.series_policy,
            report_ssl_checkpoint=args.report_ssl_checkpoint,
            out_root=args.out_root,
        )
    finally:
        bt.require_passed_labeller_gate = old_gate
        bt.build_matched_surface = old_surface
        bt.mode_identity = old_identity
        bt.format_surface = old_formatter

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["trainer_mode"] = payload.get("mode")
    payload["mode"] = f"b25x_{args.arm}_exploratory"
    payload["experiment"] = EXPERIMENT
    payload["source_experiment"] = HYBRID_EXPERIMENT
    payload["arm"] = args.arm
    payload["exploratory"] = True
    payload["formal_b23_compatible"] = False
    payload["formal_b24_eligible"] = False
    payload["gold_acceptance_allowed"] = False
    payload["hybrid_cache_file_sha256"] = surface["hybrid_cache_file_sha256"]
    payload["hybrid_source_provenance"] = surface["hybrid_source_provenance"]
    payload["evaluation_policy"] = (
        "Frozen weak-v2 development surface only. No expert-gold evaluation or promotion."
    )
    if surface.get("fill_ablation"):
        payload["fill_ablation"] = surface["fill_ablation"]

    final_checkpoint = Path(args.out_root) / f"b25x_{args.arm}_model.pt"
    torch.save(payload, final_checkpoint)
    old_checkpoint = Path(checkpoint)
    if old_checkpoint != final_checkpoint and old_checkpoint.exists():
        old_checkpoint.unlink()

    history_path = Path(args.out_root) / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history.update(
        {
            "mode": payload["mode"],
            "experiment": EXPERIMENT,
            "source_experiment": HYBRID_EXPERIMENT,
            "arm": args.arm,
            "exploratory": True,
            "formal_b23_compatible": False,
            "formal_b24_eligible": False,
            "gold_acceptance_allowed": False,
            "hybrid_cache_file_sha256": surface["hybrid_cache_file_sha256"],
            "hybrid_source_provenance": surface["hybrid_source_provenance"],
        }
    )
    if surface.get("fill_ablation"):
        history["fill_ablation"] = surface["fill_ablation"]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("B25X TRAINING COMPLETE")
    print("=" * 72)
    print("arm        :", args.arm)
    print("checkpoint :", final_checkpoint)
    print("studies    :", len(payload["study_uids"]))
    print("gold       : prohibited")
    print("promotion  : prohibited")


if __name__ == "__main__":
    main()
