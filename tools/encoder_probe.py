"""Ask whether a frozen encoder's features carry the signal at all.

The full training run answers "does this encoder work under our recipe". That
takes about ninety minutes and confounds two things: whether the features are
useful, and whether the head had time to learn to read them.

This separates them. The encoder is frozen, so it produces the same features
every epoch. Run it once, keep the features, then fit the simplest possible
model on top -- a per-target logistic regression. That model has no trouble
learning; if it still cannot beat chance, the signal is not in the features and
no amount of extra training will find it.

    strong probe score   the features are fine; the head just needed longer
    chance probe score   the features do not support the task

Two stages, so the expensive one is paid once:

    encode   read the MRI, save one vector per study
    probe    fit and score, in seconds, as often as you like
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model._implementation import (
    attach_dinov3,
    autocast,
    collate_studies_fn,
    ensure_developments_source,
    expert_loader,
    freeze_encoder,
    macro_auc,
    read_config,
    report_label_supervision,
    resolve_runtime,
    series_index,
    study_dataset,
    training_dataset_config,
)
from data.dataset import read_series, read_studies

ENCODERS = ("report-aligned", "dinov3")


def _build_encoder(config: dict, *, source: str, checkpoint, variant: str, device: str):
    """Build the frozen encoder on its own, without the study-level head."""
    ensure_developments_source()
    from rsna_knee.b16_report_ssl import load_b16_report_encoder
    from rsna_knee.model import ConvNeXtSliceEncoder

    encoder = ConvNeXtSliceEncoder(
        int(config.get("in_channels", 3)),
        pretrained_weights=False,
        normalize_input=True,
    )
    if source == "report-aligned":
        if not checkpoint:
            raise ValueError("report-aligned needs --encoder-checkpoint")
        encoder.load_state_dict(load_b16_report_encoder(checkpoint)["encoder"], strict=True)
        described = {"source": source, "checkpoint": str(checkpoint)}
    else:
        holder = torch.nn.Module()
        holder.encoder = encoder
        replacement = attach_dinov3(holder, variant=variant, pretrained_weights=True)
        encoder = replacement
        described = {"source": source, **replacement.describe()}

    holder = torch.nn.Module()
    holder.encoder = encoder
    freeze_encoder(holder)
    return encoder.to(device).eval(), described


@torch.no_grad()
def _encode(encoder, loader, runtime, *, label: str) -> tuple[list[str], np.ndarray]:
    """Average every slice of every series into one vector per study."""
    uids: list[str] = []
    vectors: list[np.ndarray] = []
    started = time.monotonic()

    for index, batch in enumerate(loader):
        volumes = batch["volumes"]
        present = batch["present"]
        if volumes.ndim == 7:  # a test-time-augmented loader adds a view axis
            volumes = volumes[:, 0]

        b, k, s = volumes.shape[0], volumes.shape[1], volumes.shape[2]
        flat = volumes.reshape(b * k * s, *volumes.shape[3:]).to(runtime.device)
        with autocast(runtime):
            features = encoder(flat).float()
        features = features.reshape(b, k, s, -1).mean(dim=2)

        mask = present.to(features.device).float().unsqueeze(-1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        vectors.append(((features * mask).sum(dim=1) / counts).cpu().numpy())
        uids.extend(str(uid) for uid in batch["study_uid"])

        if index % 25 == 0:
            done = index + 1
            rate = (time.monotonic() - started) / done
            print(
                f"[{label}] batch {done}/{len(loader)} "
                f"~{rate * (len(loader) - done) / 60:.1f} min left",
                flush=True,
            )

    return uids, np.concatenate(vectors, axis=0)


def _fit_and_score(train_x, train_y, train_w, test_x, test_y, targets) -> dict:
    """Fit one logistic regression per target and score on the expert studies."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    train_s, test_s = scaler.transform(train_x), scaler.transform(test_x)

    predictions = np.zeros((test_s.shape[0], len(targets)), dtype=np.float64)
    skipped = []
    for i, target in enumerate(targets):
        usable = train_w[:, i] > 0
        labels = (train_y[usable, i] > 0.5).astype(int)
        if usable.sum() < 20 or len(np.unique(labels)) < 2:
            predictions[:, i] = 0.5
            skipped.append(target)
            continue
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(train_s[usable], labels, sample_weight=train_w[usable, i])
        predictions[:, i] = model.predict_proba(test_s)[:, 1]

    macro, per_target = macro_auc(test_y, predictions)
    return {
        "macro_auc": float(macro),
        "per_target_auc": {t: float(v) for t, v in zip(targets, per_target)},
        "targets_without_both_classes": skipped,
        "train_studies": int(train_x.shape[0]),
        "expert_studies": int(test_x.shape[0]),
        "feature_width": int(train_x.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a frozen encoder's features")
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--dinov3-variant", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--latin-script-labels", required=True)
    parser.add_argument("--all-script-labels", required=True)
    parser.add_argument("--supervision", choices=("latin-script", "all-script"),
                        default="all-script")
    parser.add_argument("--studies", type=int, default=600,
                        help="how many training studies to fit on; 600 is plenty")
    parser.add_argument("--out", default="runs/encoder_probe")
    args = parser.parse_args()

    if args.encoder == "report-aligned" and not args.encoder_checkpoint:
        parser.error("--encoder report-aligned requires --encoder-checkpoint")

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())
    root = Path(config["data_root"])
    runtime = resolve_runtime(config)
    print(runtime.describe())

    ensure_developments_source()
    from rsna_knee.b20_crop_focus import require_b20_contract
    from rsna_knee.constants import TARGETS

    crop_policy = require_b20_contract(config)
    encoder, described = _build_encoder(
        config,
        source=args.encoder,
        checkpoint=args.encoder_checkpoint,
        variant=args.dinov3_variant,
        device=runtime.device,
    )
    print(f"[probe] encoder: {json.dumps(described)}")

    studies = read_studies(root, config, split="train")
    series, _ = read_series(root, config, split="train")

    uids, targets_array, weights, _, _ = report_label_supervision(
        studies,
        surface=args.supervision,
        latin_root=args.latin_script_labels,
        all_root=args.all_script_labels,
    )
    keep = min(int(args.studies), len(uids))
    uids, targets_array, weights = uids[:keep], targets_array[:keep], weights[:keep]
    print(f"[probe] fitting on {keep} training studies")

    collate = collate_studies_fn()
    train_loader = DataLoader(
        study_dataset(
            uids,
            series_index(series, uids),
            training_dataset_config(config, root, train=False),
            train=False,
            policy=crop_policy,
        ),
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 77_000_000),
    )
    train_uids, train_x = _encode(encoder, train_loader, runtime, label="train")
    if train_uids != uids:
        raise RuntimeError("training study order changed while encoding")

    expert = expert_loader(config, root, studies, series, runtime, crop_policy)
    _, test_x = _encode(encoder, expert["loader"], runtime, label="expert")

    result = _fit_and_score(
        train_x, targets_array, weights, test_x, expert["truth"], list(TARGETS)
    )
    result["encoder"] = described
    result["supervision"] = args.supervision
    result["reading"] = (
        "macro_auc near 0.50 means the features do not carry the signal and more "
        "training epochs will not help; a strong score means they do, and the "
        "head simply needed longer"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.encoder if args.encoder == "report-aligned" else f"dinov3-{args.dinov3_variant}"
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(path)


if __name__ == "__main__":
    main()
