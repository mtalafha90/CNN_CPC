"""B2 discriminative fine-tuning for an in-domain SSL encoder.

This candidate deliberately changes only the supervised optimizer: the
competition-data SSL encoder receives a smaller learning rate than the randomly
initialized Transformer/pathology layers.  All data splits, weak supervision,
losses, augmentation, TTA, early stopping and retraining remain delegated to
``training.train_fold``.

The implementation patches the optimizer factory only for the duration of one
call and restores it in ``finally``.  The production B0/B1 training path is
therefore unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import training
from .runtime import make_scaler

_BASE_OPTIMIZER_BUNDLE = training._optimizer_bundle


def _discriminative_optimizer_bundle(model, config: dict, epochs: int, runtime):
    head_lr = float(config.get("lr", 1e-4))
    encoder_lr = float(config.get("encoder_lr", head_lr))
    min_lr = float(config.get("min_lr", 1e-6))
    weight_decay = float(config.get("weight_decay", 1e-4))

    if head_lr <= 0 or encoder_lr <= 0:
        raise ValueError("lr and encoder_lr must be positive")
    if min_lr < 0 or min_lr >= min(head_lr, encoder_lr):
        raise ValueError("min_lr must be non-negative and smaller than both learning rates")
    if encoder_lr > head_lr:
        raise ValueError("B2 encoder_lr must not exceed the head learning rate")

    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_params}
    head_params = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    if not encoder_params or not head_params:
        raise ValueError("discriminative fine-tuning requires encoder and non-encoder parameters")

    # Keep the head group first so the historical ``history.csv`` column named
    # ``lr`` continues to report the primary/head LR. The encoder LR is recorded
    # in finetune_policy.json and follows the same cosine scheduler.
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "name": "heads"},
            {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=min_lr,
    )
    return optimizer, scheduler, make_scaler(runtime)


def _policy_payload(config: dict) -> dict:
    head_lr = float(config.get("lr", 1e-4))
    encoder_lr = float(config.get("encoder_lr", head_lr))
    return {
        "candidate": "B2_discriminative_ssl_finetune",
        "head_lr": head_lr,
        "encoder_lr": encoder_lr,
        "encoder_to_head_lr_ratio": encoder_lr / head_lr,
        "min_lr": float(config.get("min_lr", 1e-6)),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
        "scheduler": "CosineAnnealingLR",
        "encoder_frozen_epochs": 0,
        "ssl_encoder_checkpoint": config.get("ssl_encoder_checkpoint"),
        "ssl_checkpoint_source": config.get("ssl_checkpoint_source"),
        "only_optimizer_differs_from_standard_train_fold": True,
    }


def train_discriminative_fold(config: dict, fold: int) -> Path:
    ssl_checkpoint = config.get("ssl_encoder_checkpoint")
    if not ssl_checkpoint:
        raise ValueError("B2 requires ssl_encoder_checkpoint")
    if not Path(ssl_checkpoint).exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_checkpoint}")
    if float(config.get("encoder_lr", config.get("lr", 1e-4))) >= float(config.get("lr", 1e-4)):
        raise ValueError("B2 requires encoder_lr < lr")

    previous = training._optimizer_bundle
    training._optimizer_bundle = _discriminative_optimizer_bundle
    try:
        checkpoint = training.train_fold(config, fold)
    finally:
        training._optimizer_bundle = previous

    fold_dir = Path(config.get("output_dir", "runs/model")) / f"fold{fold}"
    (fold_dir / "finetune_policy.json").write_text(
        json.dumps(_policy_payload(config), indent=2), encoding="utf-8"
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    config = training.yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if hasattr(training, "yaml") else None
    if config is None:
        import yaml

        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a YAML mapping: {args.config}")
    print(train_discriminative_fold(config, args.fold))


if __name__ == "__main__":
    main()
