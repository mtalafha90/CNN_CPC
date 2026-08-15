#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rsna_knee.b24_training as bt

from run_b24x import build_pilot_matched_surface

from rsna_knee.b7_weak_supervision import _read_config
from rsna_knee.b23_labeller_audit import (
    load_labeller_audit,
    gate_status,
)
from rsna_knee.b24_protocol import MODE_CANDIDATE
from rsna_knee.b24_supervision import surface_diagnostics


EXPERIMENT = "B24X_density_ablation_v1"


def build_density_surface(
    config,
    *,
    b6_root,
    b23_root,
    weak_holdout_root=None,
    b23_holdout_root=None,
):
    """
    Density-only ablation:

    - same matched studies as B24X
    - every existing B6 committed cell is preserved exactly
    - B23 contributes ONLY where B6 had zero supervision
    - no B23 override of a B6 decision
    - no B6 cell is dropped
    """

    base = build_pilot_matched_surface(
        config,
        b6_root=b6_root,
        b23_root=b23_root,
        weak_holdout_root=weak_holdout_root,
        b23_holdout_root=None,
    )

    y_b6 = np.asarray(
        base["control"]["targets"]
    ).copy()

    w_b6 = np.asarray(
        base["control"]["weights"]
    ).copy()

    y_b23 = np.asarray(
        base["candidate"]["targets"]
    ).copy()

    w_b23 = np.asarray(
        base["candidate"]["weights"]
    ).copy()

    # --------------------------------------------------------
    # Add B23 ONLY where B6 is silent.
    # --------------------------------------------------------
    added = (
        (w_b6 <= 0)
        &
        (w_b23 > 0)
    )

    y_density = y_b6.copy()
    w_density = w_b6.copy()

    y_density[added] = y_b23[added]
    w_density[added] = w_b23[added]

    # --------------------------------------------------------
    # Scientific invariants
    # --------------------------------------------------------

    b6_used = w_b6 > 0

    # Every B6-supervised cell must remain exactly unchanged.
    if not np.array_equal(
        y_density[b6_used],
        y_b6[b6_used],
    ):
        raise RuntimeError(
            "Density arm changed a B6 target"
        )

    if not np.array_equal(
        w_density[b6_used],
        w_b6[b6_used],
    ):
        raise RuntimeError(
            "Density arm changed a B6 weight"
        )

    diagnostics = surface_diagnostics(
        base["study_uids"],
        y_b6,
        w_b6,
        y_density,
        w_density,
    )

    # --------------------------------------------------------
    # Freeze THIS pilot's exact ablation surface.
    # If anything changes, refuse rather than silently run a
    # different experiment.
    # --------------------------------------------------------
    expected = {
        "studies": 692,
        "possible_cells": 8304,
        "control_usable_cells": 3045,
        "candidate_usable_cells": 5889,
        "cells_added_by_candidate": 2844,
        "cells_dropped_by_candidate": 0,
        "cells_in_both": 3045,
        "disagreements_where_both_committed": 0,
    }

    for key, value in expected.items():
        actual = int(diagnostics[key])

        if actual != value:
            raise RuntimeError(
                f"B24X-Density surface drift: "
                f"{key}={actual}, expected {value}"
            )

    result = dict(base)

    result["candidate"] = {
        "targets": y_density,
        "weights": w_density,
    }

    result["diagnostics"] = diagnostics

    result["density_ablation"] = {
        "policy": (
            "preserve every B6 committed cell exactly; "
            "add B23 only where B6 has zero supervision"
        ),
        "b6_cells_preserved": int(
            b6_used.sum()
        ),
        "b23_only_cells_added": int(
            added.sum()
        ),
        "b6_cells_dropped": 0,
        "b6_cells_overridden": 0,
        "final_usable_cells": int(
            (w_density > 0).sum()
        ),
    }

    return result


def format_density_surface(surface):
    d = surface["diagnostics"]

    lines = [
        "B24X-Density matched training surface",
        f"  shared studies              {d['studies']}",
        f"  possible cells              {d['possible_cells']}",
        "",
        f"  B6 usable cells             "
        f"{d['control_usable_cells']} "
        f"({d['control_usable_cells']/d['possible_cells']:.1%})",
        f"  Density usable cells        "
        f"{d['candidate_usable_cells']} "
        f"({d['candidate_usable_cells']/d['possible_cells']:.1%})",
        f"  B23-only cells added        "
        f"{d['cells_added_by_candidate']}",
        f"  B6 cells dropped            "
        f"{d['cells_dropped_by_candidate']}",
        "",
        f"  B6 cells preserved          "
        f"{d['cells_in_both']}",
        f"  B6 labels overridden        "
        f"{d['disagreements_where_both_committed']}",
        "",
        "  Density arm = B6 + B23-only missing cells.",
        "  Existing B6 decisions are never replaced.",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="B24X density-only supervision ablation"
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--b6-root",
        required=True,
    )

    parser.add_argument(
        "--b23-root",
        required=True,
    )

    parser.add_argument(
        "--weak-holdout-root",
        required=True,
    )

    parser.add_argument(
        "--labeller-audit",
        required=True,
    )

    parser.add_argument(
        "--series-policy",
        required=True,
    )

    parser.add_argument(
        "--report-ssl-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        default="runs/b24x_density/density",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Preserve failed B23 formal gate.
    # --------------------------------------------------------
    audit = load_labeller_audit(
        args.labeller_audit
    )

    gate = gate_status(audit)

    if gate["passed"]:
        raise RuntimeError(
            "B23 formal gate now passes; "
            "do not reinterpret this exploratory protocol."
        )

    exploratory_gate = {
        **gate,
        "passed": False,
        "exploratory_override": True,
        "formal_b24_eligible": False,
        "gold_acceptance_allowed": False,
        "experiment": EXPERIMENT,
    }

    config = _read_config(
        args.config
    )
    config["data_root"] = (
        args.data_root
    )

    # --------------------------------------------------------
    # Surface preflight BEFORE any GPU training.
    # --------------------------------------------------------
    surface = build_density_surface(
        config,
        b6_root=args.b6_root,
        b23_root=args.b23_root,
        weak_holdout_root=args.weak_holdout_root,
        b23_holdout_root=None,
    )

    print("=" * 72)
    print("B24X-DENSITY ABLATION")
    print("=" * 72)

    print(format_density_surface(surface))

    print()
    print("Formal B23 gate : FAILED")
    for reason in gate["reasons"]:
        print("  -", reason)

    print()
    print("Gold acceptance : PROHIBITED")
    print("Promotion        : PROHIBITED")

    print()
    print("Per-target density cells")
    print("-" * 60)

    for target, values in (
        surface["diagnostics"]["per_target"].items()
    ):
        print(
            f"{target:18s} "
            f"B6={values['control_cells']:4d}  "
            f"Density={values['candidate_cells']:4d}  "
            f"added={values['added_by_candidate']:4d}  "
            f"dropped={values['dropped_by_candidate']:3d}  "
            f"override={values['disagreements']:3d}"
        )

    if args.dry_run:
        print()
        print("DRY RUN COMPLETE -- no GPU training performed.")
        return

    # --------------------------------------------------------
    # Patch only inside this Python process.
    # --------------------------------------------------------
    old_gate = (
        bt.require_passed_labeller_gate
    )
    old_surface = (
        bt.build_matched_surface
    )
    old_identity = (
        bt.mode_identity
    )
    old_formatter = (
        bt.format_surface
    )

    def recorded_failed_gate(_):
        return exploratory_gate

    def density_identity(mode):
        if mode != MODE_CANDIDATE:
            return old_identity(mode)

        return (
            "b24x_density_ablation_v1",
            "B6 preserved + B23-only missing-cell supervision",
        )

    bt.require_passed_labeller_gate = (
        recorded_failed_gate
    )

    bt.build_matched_surface = (
        build_density_surface
    )

    bt.mode_identity = (
        density_identity
    )

    bt.format_surface = (
        format_density_surface
    )

    try:
        checkpoint = bt.train_b24(
            config,
            mode=MODE_CANDIDATE,
            b6_root=args.b6_root,
            b23_root=args.b23_root,

            b23_holdout_root=(
                "__B24X_DENSITY_NO_B23_HOLDOUT__"
            ),

            weak_holdout_root=(
                args.weak_holdout_root
            ),

            series_policy_path=(
                args.series_policy
            ),

            report_ssl_checkpoint=(
                args.report_ssl_checkpoint
            ),

            out_root=args.out_root,
        )

    finally:
        bt.require_passed_labeller_gate = old_gate
        bt.build_matched_surface = old_surface
        bt.mode_identity = old_identity
        bt.format_surface = old_formatter

    # --------------------------------------------------------
    # Retag so this can never be confused with full B23 B24X.
    # --------------------------------------------------------
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    payload["trainer_mode"] = (
        payload.get("mode")
    )

    payload["mode"] = (
        "density_exploratory"
    )

    payload["experiment"] = (
        EXPERIMENT
    )

    payload["variant"] = (
        "b24x_density_ablation_v1"
    )

    payload["supervision"] = (
        "all B6 committed cells preserved; "
        "B23 contributes only B6-silent cells"
    )

    payload["exploratory"] = True
    payload["formal_b24_eligible"] = False
    payload["gold_acceptance_allowed"] = False

    payload["formal_b23_gate"] = (
        exploratory_gate
    )

    payload["density_ablation"] = (
        surface["density_ablation"]
    )

    payload["evaluation_policy"] = (
        "Frozen weak-v2 development surface only. "
        "No expert-gold evaluation or promotion."
    )

    final_checkpoint = (
        Path(args.out_root)
        / "b24x_density_model.pt"
    )

    torch.save(
        payload,
        final_checkpoint,
    )

    old_checkpoint = Path(
        checkpoint
    )

    if (
        old_checkpoint != final_checkpoint
        and old_checkpoint.exists()
    ):
        old_checkpoint.unlink()

    # --------------------------------------------------------
    # Add governance metadata to history.
    # --------------------------------------------------------
    history_path = (
        Path(args.out_root)
        / "history.json"
    )

    history = json.loads(
        history_path.read_text(
            encoding="utf-8"
        )
    )

    history["mode"] = (
        "density_exploratory"
    )

    history["experiment"] = (
        EXPERIMENT
    )

    history["exploratory"] = True
    history["formal_b24_eligible"] = False
    history["gold_acceptance_allowed"] = False

    history["formal_b23_gate"] = (
        exploratory_gate
    )

    history["density_ablation"] = (
        surface["density_ablation"]
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("B24X-DENSITY COMPLETE")
    print("=" * 72)

    print(
        "checkpoint :",
        final_checkpoint,
    )

    print(
        "studies    :",
        len(payload["study_uids"]),
    )

    print(
        "usable cells:",
        surface["diagnostics"][
            "candidate_usable_cells"
        ],
    )

    print(
        "gold acceptance allowed:",
        payload["gold_acceptance_allowed"],
    )


if __name__ == "__main__":
    main()
