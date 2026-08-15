"""B16 full-report semantic alignment after the completed B15 knee-MRI SSL encoder.

B16 uses the full radiology report as representation supervision instead of
compressing reports to B6 target states.  All 58 gold studies are excluded.
Unlike B15, weak-v2 is not a validation gate for B16, so all 4,349 non-gold
competition MRI/report pairs are eligible for representation learning.

Text semantics are fitted only from competition reports with the established
B5 TF-IDF -> TruncatedSVD pipeline.  The image encoder starts from the completed
B15 knee-MRI SSL checkpoint.  The report branch is training-only; downstream
inference remains MRI-only.
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
from .b12_variable_series import (
    VariableSeriesKneeDataset,
    build_variable_series_index,
    collate_variable_series,
)
from .b13_training import B13_INITIALIZATION, B13_INPUT_NORMALIZATION
from .b15_ssl import B15_SSL_OBJECTIVE, B15_SSL_VARIANT, load_b15_ssl_encoder
from .budget import RuntimeBudget
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .model import ConvNeXtSliceEncoder
from .policy import validate_competition_config
from .report_ssl import (
    _aggregate_study_features,
    _report_alignment_losses,
    _update_report_queue,
    fit_report_semantics,
)
from .runtime import autocast, make_scaler, resolve_runtime
from .ssl import ssl_position_indices

B16_REPORT_SSL_VARIANT = "b16_b15_full_report_alignment_v1"
B16_REPORT_SSL_EXPERIMENT = "B16_full_report_semantic_alignment"
B16_REPORT_SSL_OBJECTIVE = "study_image_to_full_report_tfidf_svd_contrastive_v1"


def b16_report_study_pool(train_df) -> tuple[list[str], dict]:
    work = train_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    gold_uids = set(work.loc[gold_mask(work), "StudyInstanceUID"].astype(str))
    if len(gold_uids) != 58:
        raise ValueError(f"B16 expected 58 gold studies, found {len(gold_uids)}")
    frame = work.loc[~gold_mask(work), ["StudyInstanceUID", "Report"]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if len(frame) != 4349:
        raise ValueError(f"B16 requires all 4,349 non-gold studies, found {len(frame)}")
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B16 non-gold study pool contains duplicate UIDs")
    uids = frame["StudyInstanceUID"].tolist()
    if gold_uids.intersection(uids):
        raise ValueError("B16 full-report pool leaks gold studies")
    return uids, {
        "competition_studies": int(len(work)),
        "excluded_gold_studies": int(len(gold_uids)),
        "report_alignment_studies": int(len(uids)),
        "uses_all_non_gold_reports": True,
    }


class B16ReportRepresentationLearner(nn.Module):
    def __init__(self, *, report_dim: int) -> None:
        super().__init__()
        if report_dim < 2:
            raise ValueError("report_dim must be >=2")
        self.encoder = ConvNeXtSliceEncoder(
            3, pretrained_weights=False, normalize_input=True
        )
        d = self.encoder.out_dim
        self.report_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, int(report_dim)),
        )

    def encode_examples(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project_report_space(self, study_feat: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.report_projector(study_feat), dim=-1)


def _variable_report_examples(
    volumes: torch.Tensor,
    present: torch.Tensor,
    *,
    positions_per_series: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if volumes.ndim != 6:
        raise ValueError(f"expected [B,K,S,C,H,W], got {tuple(volumes.shape)}")
    b, k, s, c, h, w = volumes.shape
    if present.shape != (b, k):
        raise ValueError("present mask does not match variable-series batch")
    positions = ssl_position_indices(s, int(positions_per_series)).to(volumes.device)
    p = int(positions.numel())
    selected = volumes.index_select(2, positions)
    active = present.to(dtype=torch.bool).unsqueeze(-1).expand(b, k, p).reshape(-1)
    x = selected.reshape(b * k * p, c, h, w)[active]
    study_ids = (
        torch.arange(b, device=volumes.device)
        .view(b, 1, 1)
        .expand(b, k, p)
        .reshape(-1)[active]
    )
    return x, study_ids


def _require_b16_report_contract(config: dict) -> None:
    integer_contract = {
        "seed": 2026,
        "requested_gpus": 1,
        "b16_report_n_slices": 5,
        "b16_report_positions_per_series": 2,
        "b16_report_batch_size": 2,
        "b16_report_epochs": 4,
        "b16_report_dim": 256,
        "b16_tfidf_max_features": 20000,
        "b16_tfidf_min_df": 2,
        "b16_report_queue_size": 256,
    }
    for key, expected in integer_contract.items():
        if int(config.get(key, expected)) != expected:
            raise ValueError(f"B16 report alignment freezes {key}={expected}")
    float_contract = {
        "b16_report_encoder_lr": 5e-5,
        "b16_report_head_lr": 2e-4,
        "b16_report_min_lr": 1e-6,
        "b16_report_weight_decay": 1e-4,
        "b16_report_temperature": 0.10,
        "b16_report_cosine_weight": 0.25,
        "b16_report_grad_clip": 1.0,
    }
    for key, expected in float_contract.items():
        value = float(config.get(key, expected))
        if not np.isclose(value, expected, atol=1e-12, rtol=0):
            raise ValueError(f"B16 report alignment freezes {key}={expected}; got {value}")
    if tuple(int(x) for x in config.get("b7_train_gap_choices", [1, 2])) != (1, 2):
        raise ValueError("B16 report alignment freezes train gaps [1,2]")
    if int(config.get("b7_center_jitter", 2)) != 2:
        raise ValueError("B16 report alignment freezes center jitter +/-2")
    if not bool(config.get("allow_external_pretrained", False)):
        raise ValueError("B16 retains the B15/ImageNet external-pretrained protocol")
    if not bool(config.get("pretrained", False)):
        raise ValueError("B16 retains pretrained=true")


def load_b16_report_encoder(checkpoint: str | Path) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != B16_REPORT_SSL_VARIANT:
        raise ValueError("not a frozen B16 full-report alignment checkpoint")
    if payload.get("objective") != B16_REPORT_SSL_OBJECTIVE:
        raise ValueError("B16 report objective mismatch")
    if payload.get("initialization") != B15_SSL_VARIANT:
        raise ValueError("B16 report checkpoint does not certify B15 MRI-SSL initialization")
    if payload.get("input_normalization") != B13_INPUT_NORMALIZATION:
        raise ValueError("B16 report checkpoint normalization mismatch")
    if int(payload.get("gold_studies_used", -1)) != 0:
        raise ValueError("B16 report checkpoint does not certify gold exclusion")
    if int(payload.get("report_alignment_studies", -1)) != 4349:
        raise ValueError("B16 report checkpoint must use all 4,349 non-gold studies")
    history = payload.get("history", [])
    if int(payload.get("completed_epochs", -1)) != 4 or len(history) != 4:
        raise ValueError("B16 report alignment requires four completed epochs")
    if not all(
        bool(row.get("full_coverage")) and not bool(row.get("budget_limited"))
        for row in history
    ):
        raise ValueError("B16 report checkpoint lacks four complete unbudgeted passes")
    if not isinstance(payload.get("encoder"), dict):
        raise ValueError("B16 report checkpoint missing encoder state")
    return payload


def pretrain_b16_report_alignment(
    config: dict,
    *,
    b15_ssl_checkpoint: str | Path,
    out_root: str | Path = "runs/b16_full_report/report_ssl",
) -> Path:
    validate_competition_config(config, purpose="train")
    _require_b16_report_contract(config)
    b15_payload = load_b15_ssl_encoder(b15_ssl_checkpoint)

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
    uids, pool_stats = b16_report_study_pool(train)
    non_gold = train.loc[~gold_mask(train), ["StudyInstanceUID", "Report"]].copy()
    non_gold["StudyInstanceUID"] = non_gold["StudyInstanceUID"].astype(str)
    non_gold = non_gold.set_index("StudyInstanceUID").loc[uids].reset_index()

    report_vectors, report_groups, vectorizer, svd, text_stats = fit_report_semantics(
        non_gold["Report"].fillna("").tolist(),
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
    zero = [uid for uid in uids if not variable_index.get(uid)]
    if zero:
        raise ValueError(f"B16 found {len(zero)} non-gold studies with zero eligible real series")
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
    checkpoint_path = out / "b16_report_encoder.pt"
    joblib.dump(vectorizer, out / "report_vectorizer.joblib")
    joblib.dump(svd, out / "report_svd.joblib")
    np.savez_compressed(
        out / "report_semantics.npz",
        study_uids=np.asarray(uids),
        vectors=report_vectors,
        group_ids=report_groups,
    )
    policy = {
        "variant": B16_REPORT_SSL_VARIANT,
        "experiment": B16_REPORT_SSL_EXPERIMENT,
        "objective": B16_REPORT_SSL_OBJECTIVE,
        "initialization": B15_SSL_VARIANT,
        "initialization_detail": f"{B13_INITIALIZATION} -> {B15_SSL_OBJECTIVE} -> full competition-report alignment",
        "input_normalization": B13_INPUT_NORMALIZATION,
        "representation_data_policy": (
            "all 4,349 non-gold competition MRI/report pairs; all eligible repaired real MRI series; "
            "58 gold studies excluded; weak-v2 is not used as a B16 validation gate"
        ),
        "uses_gold_labels": False,
        "uses_b6_labels": False,
        "uses_full_reports": True,
        "external_text_model": False,
        "pool": pool_stats,
        "eligible_real_series": expected_series,
        "text_semantics": text_stats,
        "metadata_repair": metadata_stats,
        "b15_ssl_checkpoint": str(Path(b15_ssl_checkpoint).resolve()),
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    history: list[dict] = []
    epoch_times: list[float] = []
    queue_z = None
    queue_groups = None
    budget_exhausted = False
    for epoch in range(epochs):
        if epoch_times and not budget.can_start(float(np.median(epoch_times)) * 1.20):
            print("[budget] stopping B16 report alignment before next epoch")
            break
        start = time.monotonic()
        model.train()
        loss_sum = nce_sum = cosine_sum = 0.0
        steps = study_draws = series_seen = active_examples = 0
        for batch in loader:
            if not budget.can_start(120.0):
                budget_exhausted = True
                print("[budget] stopping B16 report-alignment batches before wall-clock reserve")
                break
            volumes = batch["volumes"].to(runtime.device, non_blocking=True)
            present = batch["present"].to(runtime.device, non_blocking=True)
            batch_uids = [str(uid) for uid in batch["study_uid"]]
            x, study_ids = _variable_report_examples(volumes, present, positions_per_series=positions_per_series)
            if len(x) == 0:
                raise RuntimeError("B16 report alignment produced no active MRI examples")
            report_idx = np.asarray([uid_to_report[uid] for uid in batch_uids], dtype=np.int64)
            report_target_all = torch.from_numpy(report_vectors[report_idx]).to(runtime.device, non_blocking=True)
            report_group_all = torch.from_numpy(report_groups[report_idx]).to(runtime.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime):
                feat = model.encode_examples(x)
                pooled, valid = _aggregate_study_features(feat, study_ids, batch_size=int(volumes.shape[0]))
                if not bool(valid.any()):
                    raise RuntimeError("B16 report alignment has no valid study features")
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
            raise RuntimeError("B16 report alignment completed no training batches")
        scheduler.step()
        full_coverage = steps == expected_batches and study_draws == len(ds) and series_seen == expected_series
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
                "weak_v2_used_for_selection": False,
                "budget": budget.to_dict(),
            },
            checkpoint_path,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if budget_exhausted:
            break

    if len(history) != 4 or not all(bool(row.get("full_coverage")) and not bool(row.get("budget_limited")) for row in history):
        print("[warning] B16 report alignment did not complete four exact full passes; do not start B16 downstream training")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b16-report-ssl")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b15-ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b16_full_report/report_ssl")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = pretrain_b16_report_alignment(
        config,
        b15_ssl_checkpoint=args.b15_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(path)


if __name__ == "__main__":
    main()
