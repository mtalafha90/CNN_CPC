"""B3 pathology-aware low-capacity SSL fine-tuning candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from . import training
from .constants import DUAL_STREAMS
from .pathology_model import PathologyAwareMILNet, default_target_stream_priors


def _b3_model_spec(config: dict) -> dict:
    return {
        "architecture": "pathology_aware_stream_mil_v1",
        "n_streams": len(DUAL_STREAMS),
        "n_slices": int(config.get("n_slices", 16)),
        "in_channels": 3,
        "image_size": int(config.get("image_size", 224)),
        "triplet_gap": int(config.get("triplet_gap", 1)),
        "stream_mode": "dual",
        "dropout": float(config.get("dropout", 0.25)),
        "normalize_input": bool(config.get("normalize_input", False)),
        "encoder_batch_size": int(config.get("encoder_batch_size", 24)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "prior_strength": float(config.get("b3_prior_strength", 1.0)),
        "prior_residual_scale": float(config.get("b3_prior_residual_scale", 0.50)),
    }


def _b3_build_model(spec: dict, config: dict, device: torch.device):
    model = PathologyAwareMILNet(
        int(spec["n_streams"]),
        int(spec["n_slices"]),
        in_channels=3,
        pretrained_weights=bool(config.get("pretrained", False)),
        normalize_input=bool(spec["normalize_input"]),
        dropout=float(spec["dropout"]),
        encoder_batch_size=int(spec["encoder_batch_size"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        prior_strength=float(spec["prior_strength"]),
        prior_residual_scale=float(spec["prior_residual_scale"]),
    )
    ssl_path = config.get("ssl_encoder_checkpoint")
    if ssl_path:
        payload = torch.load(Path(ssl_path), map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(payload.get("encoder", payload), strict=True)
    return model.to(device)


def _policy_payload(config: dict) -> dict:
    priors = default_target_stream_priors()
    return {
        "candidate": "B3_pathology_aware_stream_mil",
        "architecture": "pathology_aware_stream_mil_v1",
        "global_mri_transformer": False,
        "pathology_interaction_transformer": False,
        "hard_stream_masks": False,
        "soft_target_stream_priors": True,
        "prior_strength": float(config.get("b3_prior_strength", 1.0)),
        "prior_residual_scale": float(config.get("b3_prior_residual_scale", 0.50)),
        "stream_order": list(DUAL_STREAMS),
        "target_stream_priors": priors.tolist(),
        "ssl_encoder_checkpoint": config.get("ssl_encoder_checkpoint"),
        "ssl_checkpoint_source": config.get("ssl_checkpoint_source"),
        "supervised_lr": float(config.get("lr", 1e-4)),
        "pretrained": bool(config.get("pretrained", False)),
    }


def train_pathology_fold(config: dict, fold: int) -> Path:
    ssl_checkpoint = config.get("ssl_encoder_checkpoint")
    if not ssl_checkpoint:
        raise ValueError("B3 requires ssl_encoder_checkpoint")
    if not Path(ssl_checkpoint).is_file():
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_checkpoint}")
    if config.get("encoder_lr") is not None:
        raise ValueError("B3 uses the standard supervised optimizer; remove encoder_lr from the config")

    previous_spec = training._model_spec
    previous_build = training._build_model
    training._model_spec = _b3_model_spec
    training._build_model = _b3_build_model
    try:
        checkpoint = training.train_fold(config, fold)
    finally:
        training._model_spec = previous_spec
        training._build_model = previous_build

    fold_dir = Path(config.get("output_dir", "runs/model")) / f"fold{fold}"
    (fold_dir / "architecture_policy.json").write_text(
        json.dumps(_policy_payload(config), indent=2), encoding="utf-8"
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a YAML mapping: {args.config}")
    print(train_pathology_fold(config, args.fold))


if __name__ == "__main__":
    main()
