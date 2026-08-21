"""B37 fixed-E2 high-resolution training on the full LLM-fill surface.

This wrapper deliberately reuses the completed Phase-9/B34 trainer rather than
forking its optimization logic.  It changes only the dataset implementation and
the permitted input resolution:

- same 4,349 report-only studies;
- same 24,035 eligible series;
- same full LLM-fill supervision export;
- same B34 architecture and report-aligned B16 initialization;
- same seed and fixed two-epoch trajectory;
- same one-tail-stage encoder fine-tuning at 0.05x head LR;
- B37 full-volume-normalize -> native 90% crop -> one resize to 288.

The historical B20 contract is checked on a shadow config with image_size=224,
so every frozen setting except the declared B37 resolution intervention remains
audited by the original contract code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .b20_crop_focus import require_b20_contract as _require_historical_b20_contract
from .b37_highres import (
    B37_VERSION,
    B37HighResVariableSeriesKneeDataset,
    b37_preprocessing_state,
    require_b37_preprocessing_contract,
)
from .b7_weak_supervision import _read_config
from . import phase9_matched_supervision_training as _phase9

B37_ENCODER_TRAINABLE_STAGES = 1
B37_ENCODER_LR_SCALE = 0.05
B37_FIXED_EPOCHS = 2


def require_b37_training_contract(config: dict) -> dict:
    """Validate historical B20/B34 settings plus B37's one declared exception."""
    policy = require_b37_preprocessing_contract(config)
    shadow = dict(config)
    shadow["b7_image_size"] = 224
    _require_historical_b20_contract(shadow)
    return policy


def _annotate_checkpoint(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("arm") != "llm_fill":
        raise ValueError("B37 must use the complete llm_fill supervision arm")
    if int(payload.get("completed_epochs", -1)) != B37_FIXED_EPOCHS:
        raise ValueError("B37 checkpoint is not complete fixed-E2")
    if int(payload.get("encoder_trainable_stages", -1)) != B37_ENCODER_TRAINABLE_STAGES:
        raise ValueError("B37 encoder fine-tuning contract changed")
    if abs(float(payload.get("encoder_lr_scale", -1.0)) - B37_ENCODER_LR_SCALE) > 1e-12:
        raise ValueError("B37 encoder LR scale changed")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B37 unexpectedly used expert labels")

    payload["b37_version"] = B37_VERSION
    payload["b37_preprocessing"] = b37_preprocessing_state()
    payload["b37_protocol"] = {
        "scientific_question": (
            "whether preserving more native in-plane information at 288x288, while "
            "retaining full-volume normalization and removing the historical second "
            "deterministic resize, improves expert pathology ranking"
        ),
        "historical_context": (
            "B21/B22 already tested crop-before-resize at 224 under B6/frozen-encoder "
            "conditions and did not improve expert acceptance; B37 therefore targets "
            "the untested higher-resolution + representation-adaptation regime"
        ),
        "comparison_reference": (
            "matched full-LLM-fill B34 fixed-E2 run with one encoder stage trainable"
        ),
        "encoder_trainable_stages": B37_ENCODER_TRAINABLE_STAGES,
        "encoder_lr_scale": B37_ENCODER_LR_SCALE,
        "fixed_epochs": B37_FIXED_EPOCHS,
        "gold_labels_used_in_gradient": 0,
        "expert58_role": "reused development diagnostic only",
    }
    payload["governance"] = (
        "B37 is a prospective preprocessing/representation test on the reused expert-58 "
        "diagnostic. Do not tune crop fraction, resolution, target subsets or endpoint "
        "after looking at expert-58. Independent hidden competition evidence remains "
        "required for promotion."
    )
    torch.save(payload, path)

    audit = {k: v for k, v in payload.items() if k not in {"model_state", "config"}}
    (path.parent / "training_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def train_b37(
    config: dict,
    *,
    data_root: str | Path,
    latin_script_labels: str | Path,
    all_script_labels: str | Path,
    llm_filled_labels: str | Path,
    series_policy: str | Path,
    encoder_checkpoint: str | Path,
    out_root: str | Path = "runs/b37_highres_288",
) -> Path:
    config = dict(config)
    config["data_root"] = str(Path(data_root).resolve())
    require_b37_training_contract(config)

    out = Path(out_root)
    if not out.name or out.name in (".", ".."):
        raise ValueError("B37 out-root must end in a run directory name")

    # Patch only the two globals that Phase-9 resolves when building its
    # dataset/contract. The optimizer, model, supervision and audit path remain
    # the original validated implementation.
    original_dataset = _phase9.CropFocusedVariableSeriesKneeDataset
    original_contract = _phase9.require_b20_contract
    _phase9.CropFocusedVariableSeriesKneeDataset = B37HighResVariableSeriesKneeDataset
    _phase9.require_b20_contract = require_b37_training_contract
    try:
        checkpoint = _phase9.train_phase9_arm(
            config,
            arm="llm_fill",
            b6_root=latin_script_labels,
            phase8_root=all_script_labels,
            llm_fill_root=llm_filled_labels,
            series_policy_path=series_policy,
            report_ssl_checkpoint=encoder_checkpoint,
            out_root=out.parent,
            out_dirname=out.name,
            encoder_source="report-aligned",
            encoder_trainable_stages=B37_ENCODER_TRAINABLE_STAGES,
            encoder_lr_scale=B37_ENCODER_LR_SCALE,
        )
    finally:
        _phase9.CropFocusedVariableSeriesKneeDataset = original_dataset
        _phase9.require_b20_contract = original_contract

    checkpoint = Path(checkpoint)
    _annotate_checkpoint(checkpoint)
    print(f"[B37] checkpoint={checkpoint}")
    return checkpoint


def main() -> None:
    ap = argparse.ArgumentParser("Train B37 high-resolution full-fill B34")
    ap.add_argument("--config", default="config/b37_highres_288.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--latin-script-labels", required=True)
    ap.add_argument("--all-script-labels", required=True)
    ap.add_argument("--llm-filled-labels", required=True)
    ap.add_argument("--series-policy", required=True)
    ap.add_argument("--encoder-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b37_highres_288")
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    train_b37(
        config,
        data_root=args.data_root,
        latin_script_labels=args.latin_script_labels,
        all_script_labels=args.all_script_labels,
        llm_filled_labels=args.llm_filled_labels,
        series_policy=args.series_policy,
        encoder_checkpoint=args.encoder_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
