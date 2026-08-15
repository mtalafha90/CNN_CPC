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
from rsna_knee.b23_llm_labels import load_frozen_b23_export
from rsna_knee.b23_labeller_audit import (
    load_labeller_audit,
    gate_status,
)
from rsna_knee.b15_ssl import load_frozen_v2_manifest
from rsna_knee.b24_protocol import (
    MODE_CONTROL,
    MODE_CANDIDATE,
)
from rsna_knee.b24_supervision import (
    surface_diagnostics,
    format_surface,
)
from rsna_knee.data import load_train_csv


EXPERIMENT = "B24X_exploratory_pilot"


def build_pilot_matched_surface(
    config,
    *,
    b6_root,
    b23_root,
    weak_holdout_root=None,
    b23_holdout_root=None,
):
    """
    Pilot-aware version of the formal B24 matched surface.

    B6 is a full-corpus export.
    B23 may be a declared pilot containing only a subset of non-gold studies.

    Both arms are ultimately restricted to the exact same study UIDs.
    """

    root = Path(config["data_root"])

    train = load_train_csv(
        root / config.get("train_csv", "train.csv")
    )
    train["StudyInstanceUID"] = (
        train["StudyInstanceUID"].astype(str)
    )

    # --------------------------------------------------------
    # B6 full-corpus supervision
    # --------------------------------------------------------
    b6_frame, _, _ = load_frozen_b6_export(b6_root)

    b6_uids, b6_y, b6_w, b6_summary = (
        prepare_b7_supervision(train, b6_frame)
    )

    b6_index = {
        str(uid): i
        for i, uid in enumerate(b6_uids)
    }

    b6_active = {
        str(uid)
        for uid, i in b6_index.items()
        if b6_w[i].sum() > 0
    }

    # --------------------------------------------------------
    # B23 declared pilot
    # --------------------------------------------------------
    b23_frame, _, b23_audit = (
        load_frozen_b23_export(b23_root)
    )

    b23_frame["StudyInstanceUID"] = (
        b23_frame["StudyInstanceUID"].astype(str)
    )

    pilot_uids = set(
        b23_frame["StudyInstanceUID"].astype(str)
    )

    train_pilot = train[
        train["StudyInstanceUID"].isin(pilot_uids)
    ].copy()

    if len(train_pilot) != len(b23_frame):
        raise RuntimeError(
            "B23 pilot/train mismatch: "
            f"train subset={len(train_pilot)}, "
            f"B23 frame={len(b23_frame)}"
        )

    b23_uids, b23_y, b23_w, b23_summary = (
        prepare_b7_supervision(
            train_pilot,
            b23_frame,
        )
    )

    b23_index = {
        str(uid): i
        for i, uid in enumerate(b23_uids)
    }

    b23_active = {
        str(uid)
        for uid, i in b23_index.items()
        if b23_w[i].sum() > 0
    }

    # --------------------------------------------------------
    # Frozen B6 weak-v2 holdout stays outside gradients
    # --------------------------------------------------------
    weak_holdout = set()

    if weak_holdout_root is not None:
        _, weak_manifest = load_frozen_v2_manifest(
            weak_holdout_root
        )

        weak_holdout = set(
            weak_manifest.loc[
                weak_manifest["split"] == "holdout",
                "StudyInstanceUID",
            ].astype(str)
        )

    # No B23 holdout exists because B23's formal gate failed.
    # B24X is explicitly exploratory and does not manufacture one.

    shared = sorted(
        (b6_active & b23_active) - weak_holdout
    )

    if len(shared) < 2:
        raise RuntimeError(
            f"B24X matched surface has only "
            f"{len(shared)} studies"
        )

    # --------------------------------------------------------
    # Same study ordering for BOTH arms
    # --------------------------------------------------------
    y_control = np.stack([
        b6_y[b6_index[uid]]
        for uid in shared
    ])
    w_control = np.stack([
        b6_w[b6_index[uid]]
        for uid in shared
    ])

    y_candidate = np.stack([
        b23_y[b23_index[uid]]
        for uid in shared
    ])
    w_candidate = np.stack([
        b23_w[b23_index[uid]]
        for uid in shared
    ])

    diagnostics = surface_diagnostics(
        shared,
        y_control,
        w_control,
        y_candidate,
        w_candidate,
    )

    surface = {
        "study_uids": shared,

        "control": {
            "targets": y_control,
            "weights": w_control,
        },

        "candidate": {
            "targets": y_candidate,
            "weights": w_candidate,
        },

        "diagnostics": diagnostics,

        "excluded": {
            "gold": 58,
            "weak_v2_holdout": len(
                b23_active & weak_holdout
            ),
            "b23_holdout": 0,
        },

        "b6_active_studies": len(b6_active),
        "b23_active_studies": len(b23_active),

        "b23_cell_coverage": float(
            b23_audit.get(
                "cell_coverage",
                float("nan")
            )
        ),

        "b6_supervision": b6_summary,
        "b23_supervision": b23_summary,
    }

    return surface


def main():
    parser = argparse.ArgumentParser(
        description=(
            "B24X exploratory matched pilot: "
            "B6 vs B23 supervision"
        )
    )

    parser.add_argument(
        "--arm",
        required=True,
        choices=["control", "candidate"],
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)

    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--b23-root", required=True)

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
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Record, DO NOT hide, the failed formal gate
    # --------------------------------------------------------
    audit = load_labeller_audit(
        args.labeller_audit
    )

    gate = gate_status(audit)

    if gate["passed"]:
        raise RuntimeError(
            "B23 formal gate passed. "
            "Use formal B24 instead of B24X."
        )

    print("=" * 72)
    print("B24X EXPLORATORY PILOT")
    print("=" * 72)

    print(
        "Formal B23 gate : FAILED "
        "(recorded, not overridden as a pass)"
    )

    for reason in gate["reasons"]:
        print("  -", reason)

    print()
    print("Gold acceptance : PROHIBITED")
    print("Promotion        : PROHIBITED")
    print()

    config = _read_config(args.config)
    config["data_root"] = args.data_root

    mode = (
        MODE_CONTROL
        if args.arm == "control"
        else MODE_CANDIDATE
    )

    exploratory_gate = {
        **gate,
        "passed": False,
        "exploratory_override": True,
        "formal_b24_eligible": False,
        "gold_acceptance_allowed": False,
        "experiment": EXPERIMENT,
    }

    # --------------------------------------------------------
    # Patch only inside THIS process.
    # Formal repo code remains unchanged.
    # --------------------------------------------------------
    original_gate_fn = (
        bt.require_passed_labeller_gate
    )
    original_surface_fn = (
        bt.build_matched_surface
    )

    def recorded_failed_gate(_):
        return exploratory_gate

    bt.require_passed_labeller_gate = (
        recorded_failed_gate
    )

    bt.build_matched_surface = (
        build_pilot_matched_surface
    )

    try:
        checkpoint = bt.train_b24(
            config,
            mode=mode,
            b6_root=args.b6_root,
            b23_root=args.b23_root,

            # Formal trainer expects a parameter,
            # but B24X surface deliberately ignores it.
            b23_holdout_root=(
                "__B24X_NO_B23_HOLDOUT__"
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
        bt.require_passed_labeller_gate = (
            original_gate_fn
        )
        bt.build_matched_surface = (
            original_surface_fn
        )

    # --------------------------------------------------------
    # Retag checkpoint so formal B24 tools cannot silently
    # mistake it for a confirmatory B24 checkpoint.
    # --------------------------------------------------------
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    formal_mode = payload["mode"]

    payload["formal_mode"] = formal_mode

    payload["mode"] = (
        formal_mode + "_exploratory"
    )

    payload["experiment"] = EXPERIMENT
    payload["exploratory"] = True

    payload["formal_b24_eligible"] = False
    payload["gold_acceptance_allowed"] = False

    payload["formal_b23_gate"] = (
        exploratory_gate
    )

    payload["evaluation_policy"] = (
        "Frozen weak-v2 development surface only. "
        "No expert-gold acceptance and no promotion."
    )

    torch.save(
        payload,
        checkpoint,
    )

    # Add the same governance metadata to history.json.
    history_path = (
        Path(args.out_root) / "history.json"
    )

    history = json.loads(
        history_path.read_text(
            encoding="utf-8"
        )
    )

    history["experiment"] = EXPERIMENT
    history["exploratory"] = True

    history["formal_b24_eligible"] = False
    history["gold_acceptance_allowed"] = False

    history["formal_b23_gate"] = (
        exploratory_gate
    )

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("B24X COMPLETE")
    print("=" * 72)

    print("checkpoint :", checkpoint)
    print("formal arm :", formal_mode)
    print("stored arm :", payload["mode"])

    print(
        "studies    :",
        len(payload["study_uids"]),
    )

    print(
        "gold acceptance allowed :",
        payload["gold_acceptance_allowed"],
    )


if __name__ == "__main__":
    main()
