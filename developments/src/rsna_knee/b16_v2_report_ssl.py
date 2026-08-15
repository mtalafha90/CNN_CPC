"""Weak-v2-safe B16 full-report alignment for B21 development.

Historical B16 used all 4,349 non-gold MRI/report pairs, including the 623
weak-v2 holdout studies. That encoder is therefore not eligible for weak-v2
model ranking. This module repeats the same B16 report-alignment objective while
excluding every frozen weak-v2 holdout StudyInstanceUID. It starts from the
existing B15 MRI-SSL encoder, which already excluded weak-v2 holdout and gold.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config, seed_everything
from .b12_variable_series import VariableSeriesKneeDataset, build_variable_series_index, collate_variable_series
from .b13_training import B13_INITIALIZATION, B13_INPUT_NORMALIZATION
from .b15_ssl import (
    B15_SSL_OBJECTIVE,
    B15_SSL_VARIANT,
    WEAK_V2_MANIFEST_SHA256,
    WEAK_V2_SURFACE,
    load_b15_ssl_encoder,
    load_frozen_v2_manifest,
)
from .b16_report_ssl import (
    B16ReportRepresentationLearner,
    B16_REPORT_SSL_OBJECTIVE,
    _require_b16_report_contract,
    _variable_report_examples,
)
from .budget import RuntimeBudget
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .policy import validate_competition_config
from .report_ssl import (
    _aggregate_study_features,
    _report_alignment_losses,
    _update_report_queue,
    fit_report_semantics,
)
from .runtime import autocast, make_scaler, resolve_runtime

B16_V2_REPORT_VARIANT = "b16_b15_full_report_alignment_weak_v2_safe_v1"
B16_V2_REPORT_EXPERIMENT = "B16_full_report_alignment_weak_v2_safe"
B16_V2_REPORT_OBJECTIVE = B16_REPORT_SSL_OBJECTIVE
B16_V2_EXPECTED_STUDIES = 3726


def b16_v2_safe_pool(train_df, manifest):
    work = train_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    gold_uids = set(work.loc[gold_mask(work), "StudyInstanceUID"].astype(str))
    holdout_uids = set(
        manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str)
    )
    if len(gold_uids) != 58 or len(holdout_uids) != 623:
        raise ValueError("B16-v2-safe requires 58 gold and 623 weak-v2 holdout studies")
    frame = work.loc[
        ~work["StudyInstanceUID"].isin(gold_uids | holdout_uids),
        ["StudyInstanceUID", "Report"],
    ].copy()
    if len(frame) != B16_V2_EXPECTED_STUDIES:
        raise ValueError(
            f"B16-v2-safe expected {B16_V2_EXPECTED_STUDIES} representation studies, got {len(frame)}"
        )
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B16-v2-safe pool contains duplicate UIDs")
    uids = frame["StudyInstanceUID"].astype(str).tolist()
    if gold_uids.intersection(uids) or holdout_uids.intersection(uids):
        raise RuntimeError("B16-v2-safe representation pool leaks gold or weak-v2 holdout")
    return uids, frame, {
        "competition_studies": int(len(work)),
        "excluded_gold_studies": int(len(gold_uids)),
        "excluded_weak_v2_holdout_studies": int(len(holdout_uids)),
        "report_alignment_studies": int(len(uids)),
    }


def load_b16_v2_report_encoder(checkpoint: str | Path) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B16_V2_REPORT_VARIANT:
        raise ValueError("not a weak-v2-safe B16 report encoder")
    if payload.get("objective") != B16_V2_REPORT_OBJECTIVE:
        raise ValueError("B16-v2-safe objective mismatch")
    if payload.get("initialization") != B15_SSL_VARIANT:
        raise ValueError("B16-v2-safe must initialize from B15 MRI SSL")
    if payload.get("weak_holdout_manifest_sha256") != WEAK_V2_MANIFEST_SHA256:
        raise ValueError("B16-v2-safe weak-v2 manifest SHA mismatch")
    if int(payload.get("weak_v2_holdout_studies_used", -1)) != 0:
        raise ValueError("B16-v2-safe checkpoint does not certify holdout exclusion")
    if int(payload.get("gold_studies_used", -1)) != 0:
        raise ValueError("B16-v2-safe checkpoint does not certify gold exclusion")
    if int(payload.get("report_alignment_studies", -1)) != B16_V2_EXPECTED_STUDIES:
        raise ValueError("B16-v2-safe representation-study count mismatch")
    history = payload.get("history", [])
    if int(payload.get("completed_epochs", -1)) != 4 or len(history) != 4:
        raise ValueError("B16-v2-safe requires four completed epochs")
    if not all(bool(row.get("full_coverage")) and not bool(row.get("budget_limited")) for row in history):
        raise ValueError("B16-v2-safe lacks four complete unbudgeted passes")
    if not isinstance(payload.get("encoder"), dict):
        raise ValueError("B16-v2-safe checkpoint missing encoder state")
    return payload


def pretrain_b16_v2_report_alignment(
    config: dict,
    *,
    b15_ssl_checkpoint: str | Path,
    weak_holdout_root: str | Path,
    out_root: str | Path = "runs/b16_v2_safe_report/report_ssl",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b16_report_contract(config)
    b15_payload = load_b15_ssl_encoder(b15_ssl_checkpoint)
    weak_payload, manifest = load_frozen_v2_manifest(weak_holdout_root)

    seed = int(config.get("seed", 2026))
    seed_everything(seed + 18_000_000)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    uids, frame, pool_stats = b16_v2_safe_pool(train, manifest)
    frame = frame.set_index("StudyInstanceUID").loc[uids].reset_index()

    report_vectors, report_groups, vectorizer, svd, text_stats = fit_report_semantics(
        frame["Report"].fillna("").tolist(),
        requested_dim=int(config.get("b16_report_dim", 256)),
        max_features=int(config.get("b16_tfidf_max_features", 20000)),
        min_df=int(config.get("b16_tfidf_min_df", 2)),
        seed=seed,
    )
    report_dim = int(report_vectors.shape[1])
    uid_to_report = {uid: i for i, uid in enumerate(uids)}

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    variable_index = build_variable_series_index(series, uids)
    if any(not variable_index.get(uid) for uid in uids):
        raise ValueError("B16-v2-safe pool contains a study with zero eligible series")
    expected_series = int(sum(len(variable_index[uid]) for uid in uids))

    ds_config = make_b7_dataset_config(config, root, train=True)
    ds_config.n_slices = int(config.get("b16_report_n_slices", 5))
    ds_config.slice_dropout = 0.0
    ds = VariableSeriesKneeDataset(uids, variable_index, ds_config, train=True)
    batch_size = int(config.get("b16_report_batch_size", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed + 18_100_000),
    )
    expected_batches = int(math.ceil(len(ds) / batch_size))

    model = B16ReportRepresentationLearner(report_dim=report_dim)
    model.encoder.load_state_dict(b15_payload["encoder"], strict=True)
    model = model.to(runtime.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": float(config.get("b16_report_encoder_lr", 5e-5))},
            {"params": list(model.report_projector.parameters()), "lr": float(config.get("b16_report_head_lr", 2e-4))},
        ],
        weight_decay=float(config.get("b16_report_weight_decay", 1e-4)),
    )
    epochs = int(config.get("b16_report_epochs", 4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(config.get("b16_report_min_lr", 1e-6)),
    )
    scaler = make_scaler(runtime)
    temperature = float(config.get("b16_report_temperature", 0.10))
    cosine_weight = float(config.get("b16_report_cosine_weight", 0.25))
    clip = float(config.get("b16_report_grad_clip", 1.0))
    positions_per_series = int(config.get("b16_report_positions_per_series", 2))
    queue_capacity = int(config.get("b16_report_queue_size", 256))

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "b16_v2_report_encoder.pt"
    joblib.dump(vectorizer, out / "report_vectorizer.joblib")
    joblib.dump(svd, out / "report_svd.joblib")
    np.savez_compressed(
        out / "report_semantics.npz",
        study_uids=np.asarray(uids),
        vectors=report_vectors,
        group_ids=report_groups,
    )
    policy = {
        "variant": B16_V2_REPORT_VARIANT,
        "experiment": B16_V2_REPORT_EXPERIMENT,
        "objective": B16_V2_REPORT_OBJECTIVE,
        "initialization": B15_SSL_VARIANT,
        "initialization_detail": f"{B13_INITIALIZATION} -> {B15_SSL_OBJECTIVE} -> weak-v2-safe full-report alignment",
        "input_normalization": B13_INPUT_NORMALIZATION,
        "weak_holdout_surface": WEAK_V2_SURFACE,
        "weak_holdout_manifest_sha256": WEAK_V2_MANIFEST_SHA256,
        "representation_data_policy": "all non-gold competition MRI/report pairs except the 623 frozen weak-v2 holdout studies",
        "uses_gold_labels": False,
        "uses_b6_labels": False,
        "uses_full_reports": True,
        "weak_v2_holdout_studies_used": 0,
        "pool": pool_stats,
        "eligible_real_series": expected_series,
        "text_semantics": text_stats,
        "metadata_repair": metadata_stats,
        "b15_ssl_checkpoint": str(Path(b15_ssl_checkpoint).resolve()),
        "weak_holdout_metadata": weak_payload,
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    history = []
    epoch_times = []
    queue_z = None
    queue_groups = None
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            break
        start = time.monotonic()
        model.train()
        loss_sum = nce_sum = cosine_sum = 0.0
        steps = study_draws = series_seen = active_examples = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            batch_uids = [str(uid) for uid in batch["study_uid"]]
            x, study_ids = _variable_report_examples(
                volumes, present, positions_per_series=positions_per_series
            )
            if len(x) == 0:
                raise RuntimeError("B16-v2-safe produced no active MRI examples")
            report_idx = np.asarray([uid_to_report[uid] for uid in batch_uids], dtype=np.int64)
            report_target_all = torch.from_numpy(report_vectors[report_idx]).to(runtime.device, non_blocking=True)
            report_group_all = torch.from_numpy(report_groups[report_idx]).to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                feat = model.encode_examples(x)
                pooled, valid = _aggregate_study_features(
                    feat, study_ids, batch_size=int(volumes.shape[0])
                )
                if not bool(valid.any()):
                    raise RuntimeError("B16-v2-safe has no valid study features")
                image_z = model.project_report_space(pooled[valid])
                report_target = report_target_all[valid]
                report_group = report_group_all[valid]
                nce, cosine = _report_alignment_losses(
                    image_z,
                    report_target,
                    report_group,
                    queue_z=queue_z,
                    queue_groups=queue_groups,
                    temperature=temperature,
                )
                loss = nce + cosine_weight * cosine
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            queue_z, queue_groups = _update_report_queue(
                queue_z,
                queue_groups,
                report_target,
                report_group,
                capacity=queue_capacity,
            )
            loss_sum += float(loss.item())
            nce_sum += float(nce.item())
            cosine_sum += float(cosine.item())
            steps += 1
            study_draws += int(volumes.shape[0])
            series_seen += int((present > 0).sum().item())
            active_examples += int(x.shape[0])

        seconds = time.monotonic() - start
        epoch_times.append(seconds)
        if steps == 0:
            raise RuntimeError("B16-v2-safe completed no training batches")
        scheduler.step()
        full_coverage = (
            steps == expected_batches
            and study_draws == len(ds)
            and series_seen == expected_series
        )
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / steps,
            "report_nce": nce_sum / steps,
            "report_cosine": cosine_sum / steps,
            "encoder_lr": float(optimizer.param_groups[0]["lr"]),
            "report_head_lr": float(optimizer.param_groups[1]["lr"]),
            "epoch_seconds": float(seconds),
            "batches": int(steps),
            "expected_batches": int(expected_batches),
            "study_draws": int(study_draws),
            "expected_studies": int(len(ds)),
            "series_instances_seen": int(series_seen),
            "expected_series_instances": int(expected_series),
            "active_2p5d_examples": int(active_examples),
            "full_coverage": bool(full_coverage),
            "budget_limited": bool(budget_exhausted),
        }
        history.append(row)
        print(row)
        torch.save(
            {
                **policy,
                "encoder": model.encoder.state_dict(),
                "config": config,
                "completed_epochs": len(history),
                "history": history,
                "report_alignment_studies": int(len(ds)),
                "gold_studies_used": 0,
                "gold_labels_used": False,
                "b6_labels_used": False,
                "weak_v2_holdout_studies_used": 0,
                "budget": budget.to_dict(),
            },
            checkpoint_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != 4 or not all(
        bool(row.get("full_coverage")) and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise RuntimeError("B16-v2-safe did not complete four exact full passes")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b16-v2-report-ssl")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b15-ssl-checkpoint", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--out-root", default="runs/b16_v2_safe_report/report_ssl")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = pretrain_b16_v2_report_alignment(
        config,
        b15_ssl_checkpoint=args.b15_ssl_checkpoint,
        weak_holdout_root=args.weak_holdout_root,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
