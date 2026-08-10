"""B10 label-free physical-scale audit and policy freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
from .data import (
    backfill_series_metadata,
    build_series_index,
    load_series_csv,
    load_train_csv,
)
from .physical_scale import (
    B10_MIN_GEOMETRY_COVERAGE,
    B10_PHYSICAL_POLICY,
    audit_selected_series_geometry,
    derive_policy_from_geometry,
    selected_series_signature,
)

B10_AUDIT_EXPERIMENT = "B10_physical_scale_audit"


def _read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def run_b10_spacing_audit(
    config: dict,
    *,
    b6_root: str | Path,
    out_root: str | Path = "runs/b10_physical_scale/audit",
) -> dict:
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    b6_frame, _, b6_audit = load_frozen_b6_export(b6_root)
    uids, _, _, supervision = prepare_b7_supervision(train, b6_frame)

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    index = build_series_index(series, uids, mode="dual")
    signature = selected_series_signature(index, uids)

    geometry, audit = audit_selected_series_geometry(
        data_root=root,
        split="train",
        studies=uids,
        series_index=index,
    )
    policy = derive_policy_from_geometry(
        geometry,
        source_study_count=len(uids),
        selected_series_signature_value=signature,
        min_geometry_coverage=float(
            config.get("b10_min_geometry_coverage", B10_MIN_GEOMETRY_COVERAGE)
        ),
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(out / "series_geometry.csv", index=False)

    audit_payload = {
        "experiment": B10_AUDIT_EXPERIMENT,
        "status": "label-free physical-scale policy frozen before B10 training/gold evaluation",
        "physical_policy_name": B10_PHYSICAL_POLICY,
        "uses_gold_labels": False,
        "b6_version": b6_audit.get("b6_version"),
        "active_weak_training_studies": int(len(uids)),
        "training_usable_cells": int(supervision.get("usable_cells", 0)),
        "routing_mode": "historical B7.1 dual routing",
        "selected_series_signature": signature,
        "metadata_repair": metadata_stats,
        "geometry_audit": audit,
        "canonical_geometry": {
            plane: {
                "target_spacing_mm": policy["planes"][plane]["target_spacing_mm"],
                "target_fov_mm": policy["planes"][plane]["target_fov_mm"],
                "geometry_coverage": policy["planes"][plane]["geometry_coverage"],
            }
            for plane in ("Sagittal", "Coronal", "Axial")
        },
        "policy_sha256": policy["policy_sha256"],
    }
    (out / "spacing_audit.json").write_text(
        json.dumps(audit_payload, indent=2), encoding="utf-8"
    )
    (out / "physical_scale_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )

    print(json.dumps(audit_payload, indent=2))
    print(out / "series_geometry.csv")
    print(out / "spacing_audit.json")
    print(out / "physical_scale_policy.json")
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b10-audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--out-root", default="runs/b10_physical_scale/audit")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    run_b10_spacing_audit(config, b6_root=args.b6_root, out_root=args.out_root)


if __name__ == "__main__":
    main()
