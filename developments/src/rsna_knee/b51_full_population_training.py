"""B51 — train the adapted hierarchy on the full report-only population.

B50 established, on a matched pair, that letting the frozen B34 study hierarchy
adapt improves ranking: `+0.011221` on 548 unseen-scanner studies, all twelve
targets improved, with a discordant ceiling of `0.030652` giving the measurement
2.7x the headroom it needed.

B50 trained on 1,447 studies, because its fresh gate had to exclude every row
B48 and B49 validation had spent. B51 runs the same mechanism on all 4,349, so
that it differs from B42 -- the endpoint behind the 0.714 hidden score -- by
exactly one thing.

There is no control arm and no validation split. B50 ran the comparison; B51 is
a production run whose control is B42's existing hidden score, and whose
endpoint was fixed before it started. Nothing here selects anything.

The population is the only difference from B50's trainer. `_report_only_surface`
needs a table naming every report-only study, and uses nothing from it but the
UID column, so B51 supplies one that labels all 4,349 as training. The surface
check that every study is accounted for still runs, which is why it is expressed
this way rather than by bypassing the shared path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .b7_weak_supervision import (
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b13_training import B13_SERIES_SIGNATURE
from .b35_training import _require_base_checkpoint, sha256_file
from .b37_highres_sparse_training import (
    B37_GRAD_CLIP,
    B37_HEAD_LR,
    B37_WEIGHT_DECAY,
    _format_memory_state,
    _memory_state,
    _save_recovery,
    _trim_host_memory,
)
from .b17_training import encoder_state_sha256
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    collate_b42,
)
from .b42_constant_area_aspect_sparse_training import (
    _batch_scales,
    _losses,
    _move_study,
)
from .b48_global_conditioned_sparse_training import (
    _config_sha256,
    _indices_for_split,
    _report_only_surface,
    _uid_sha256,
    b48_fill_artifacts,
)
from .b50_adapted_hierarchy_mil import (
    B50AdaptedHierarchySparseMILResidual,
    b50_parameter_groups,
    b50_state,
    require_b50_contract,
)
from .b50_adapted_hierarchy_training import (
    B50_CONSTRUCTION_SEED_OFFSET,
    B50_EFFECTIVE_BATCH,
    B50_FIXED_EPOCHS,
    B50_LOADER_SEED_OFFSET,
    B50_SEED,
    _check_arm_wiring,
    _hierarchy_gradient_present,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import make_scaler, resolve_runtime

B51_VERSION = "b51_full_population_adapted_hierarchy_v1"
B51_EXPERIMENT = "B51_FULL_POPULATION_ADAPTED_HIERARCHY"
B51_RUN_ROOT = "runs/085_Experiment_B51_full_population_adapted_hierarchy"
B51_CHECKPOINT_NAME = "b51_full_population_adapted_hierarchy_model.pt"

# The complete report-only population, as B42 trained on and as Phase 9 froze.
B51_REPORT_ONLY_STUDIES = 4_349


def _read_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def full_population_rows(root: Path, settings: dict) -> pd.DataFrame:
    """Every report-only study, labelled as training.

    `_report_only_surface` reads only the UID column but checks that the table
    accounts for all 4,349 studies. Expressing "all of them" this way keeps that
    check running instead of bypassing it, which is the point: a population that
    silently lost studies is exactly the failure the check exists to catch.
    """
    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    report_only = train.loc[~gold_mask(train), ["StudyInstanceUID"]].copy()
    report_only["StudyInstanceUID"] = report_only["StudyInstanceUID"].astype(str)
    if len(report_only) != B51_REPORT_ONLY_STUDIES:
        raise ValueError(
            f"B51 requires all {B51_REPORT_ONLY_STUDIES} report-only studies; "
            f"found {len(report_only)}"
        )
    report_only["split"] = "train"
    return report_only.reset_index(drop=True)


def train_b51(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B51_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train exactly two epochs of the adapted hierarchy on all 4,349 studies."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["b50_arm"] = "adapted_hierarchy_candidate"
    contract = require_b50_contract(settings)
    if not contract["adapt_hierarchy"]:
        raise RuntimeError("B51 exists to train the hierarchy; the contract disagreed")

    seed = int(settings.get("seed", B50_SEED))
    if seed != B50_SEED:
        raise ValueError(f"B51 inherits the frozen seed {B50_SEED}; got {seed}")
    settings["seed"] = seed
    seed_everything(seed + B50_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    population = full_population_rows(root, settings)
    fill_artifacts = b48_fill_artifacts(labels_root)
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=population,
        base_payload=base_payload,
    )
    indices = _indices_for_split(all_uids, population, "train")
    if len(indices) != B51_REPORT_ONLY_STUDIES:
        raise RuntimeError("B51 training population is not the full report-only surface")
    uids = [all_uids[i] for i in indices]
    targets, weights = all_targets[indices], all_weights[indices]
    target_multiplier = target_balance_multipliers(weights)

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B51 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    series_summary, variable_index = audit_variable_series_surface(series, uids)
    expected_series = int(series_summary.get("eligible_recognized_plane_series", -1))
    expected_cells = int((weights > 0).sum())
    if (
        expected_series <= 0
        or expected_cells <= 0
        or series_summary.get("viability_passed") is not True
    ):
        raise RuntimeError("B51 MRI/weak-label surface failed viability")

    dataset_config = make_b7_dataset_config(settings, root, train=False)
    dataset_config.tta_center_offsets = ()
    dataset = B42ConstantAreaAspectDataset(
        uids,
        variable_index,
        dataset_config,
        crop_focus_policy=contract["crop_policy"],
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=B50_EFFECTIVE_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=seed + B50_LOADER_SEED_OFFSET),
    )

    model = B50AdaptedHierarchySparseMILResidual(
        base_model,
        grid_size=int(settings["b37_grid_size"]),
        top_k=int(settings["b37_top_k"]),
        temperature=float(settings["b37_temperature"]),
        encoder_trainable_stages=int(settings["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(settings["b37_encoder_chunk_size"]),
        adapt_hierarchy=True,
    ).to(runtime.device)
    model.train()

    trainable = model.trainable_parameter_summary()
    print(f"[B51] studies={len(uids)} series={expected_series} cells={expected_cells}")
    print(f"[B51] trainable={trainable}", flush=True)
    if trainable["hierarchy_trainable_parameters"] == 0:
        raise RuntimeError("B51 has no trainable hierarchy parameters")

    head_lr = float(settings.get("b37_head_lr", B37_HEAD_LR))
    encoder_scale = float(settings["b37_encoder_lr_scale"])
    groups = b50_parameter_groups(
        model,
        head_lr=head_lr,
        encoder_lr_scale=encoder_scale,
        hierarchy_lr_scale=float(contract["hierarchy_lr_scale"]),
    )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(settings.get("b37_weight_decay", B37_WEIGHT_DECAY))
    )
    clipped = [p for group in groups for p in group["params"]]
    scaler = make_scaler(runtime)
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(settings["b37_local_aux_weight"])
    clip = float(settings.get("b37_grad_clip", B37_GRAD_CLIP))

    out = Path(out_root)
    if preflight_only:
        items = [dataset[i] for i in range(min(2, len(dataset)))]
        optimizer.zero_grad(set_to_none=True)
        for item, scale in zip(items, _batch_scales(items, multiplier_cpu)):
            tensors = _move_study(item, runtime.device)
            _out, total, _combined, _local = _losses(
                model, runtime, tensors, multiplier_t, aux_weight
            )
            scaler.scale(total * float(scale)).backward()
        _check_arm_wiring(
            model,
            adapt_hierarchy=True,
            hierarchy_saw_gradient=_hierarchy_gradient_present(model),
        )
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[B51 preflight] PASS {_format_memory_state(_memory_state(runtime))}",
            flush=True,
        )
        return None

    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / B51_CHECKPOINT_NAME
    if checkpoint_path.exists():
        raise FileExistsError(f"B51 will not overwrite an existing checkpoint: {checkpoint_path}")

    history: list[dict] = []
    for epoch in range(1, B50_FIXED_EPOCHS + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        model.train()
        total_sum = combined_sum = local_sum = 0.0
        batches = studies_seen = series_seen = cells_seen = 0
        hierarchy_saw_gradient = False

        for step, items in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            scales = _batch_scales(items, multiplier_cpu)
            batch_total = batch_combined = batch_local = 0.0
            for item, scale in zip(items, scales):
                series_seen += int(item["present"].sum().item())
                cells_seen += int((item["weight"] > 0).sum().item())
                tensors = _move_study(item, runtime.device)
                _o, total, combined, local = _losses(
                    model, runtime, tensors, multiplier_t, aux_weight
                )
                scaler.scale(total * float(scale)).backward()
                batch_total += float(total.detach().item()) * float(scale)
                batch_combined += float(combined.detach().item()) * float(scale)
                batch_local += float(local.detach().item()) * float(scale)
                del tensors, total, combined, local

            hierarchy_saw_gradient |= _hierarchy_gradient_present(model)
            _check_arm_wiring(
                model,
                adapt_hierarchy=True,
                hierarchy_saw_gradient=hierarchy_saw_gradient,
            )
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(clipped, clip)
            scaler.step(optimizer)
            scaler.update()

            total_sum += batch_total
            combined_sum += batch_combined
            local_sum += batch_local
            batches += 1
            studies_seen += len(items)
            if step % 50 == 0 or step == len(loader):
                elapsed = (time.monotonic() - started) / 60.0
                remaining = elapsed / step * (len(loader) - step)
                gate = model.head.effective_gate().detach().abs().mean().item()
                print(
                    f"[B51] E{epoch} {step}/{len(loader)} "
                    f"total={total_sum/batches:.4f} combined={combined_sum/batches:.4f} "
                    f"local={local_sum/batches:.4f} gate_abs_mean={gate:.4f} "
                    f"elapsed={elapsed:.1f} min remaining~{remaining:.1f} min "
                    f"{_format_memory_state(_memory_state(runtime))}",
                    flush=True,
                )
                _trim_host_memory()
            del items

        if (
            studies_seen != len(uids)
            or series_seen != expected_series
            or cells_seen != expected_cells
        ):
            raise RuntimeError(
                "B51 epoch surface changed: "
                f"studies={studies_seen}/{len(uids)} series={series_seen}/{expected_series} "
                f"cells={cells_seen}/{expected_cells}"
            )

        row = {
            "epoch": epoch,
            "seed": seed,
            "loss_total": total_sum / batches,
            "loss_combined": combined_sum / batches,
            "loss_local_aux": local_sum / batches,
            "batches": batches,
            "studies": studies_seen,
            "series": series_seen,
            "supervision_cells": cells_seen,
            "hierarchy_gradient_seen": bool(hierarchy_saw_gradient),
            "sparse_mil": model.head.state(),
            "trainable": model.trainable_parameter_summary(),
            "encoder_sha256": encoder_state_sha256(model.base.encoder),
            "epoch_seconds": float(time.monotonic() - started),
            "memory": _memory_state(runtime),
        }
        history.append(row)
        print(
            f"[B51] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(out, epoch=epoch, model=model, history=history, version=B51_VERSION)

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B51 encoder fingerprint did not move")

    target_balance = {t: float(target_multiplier[i]) for i, t in enumerate(TARGETS)}
    payload = {
        "experiment": B51_EXPERIMENT,
        "version": B51_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B50_FIXED_EPOCHS,
        "checkpoint_selection": "none; fixed epoch 2",
        "seed": seed,
        "adapt_hierarchy": True,
        "hierarchy_lr_scale": float(contract["hierarchy_lr_scale"]),
        "hierarchy_lr": head_lr * float(contract["hierarchy_lr_scale"]),
        "trainable": trainable,
        "hypothesis": (
            "the adapted study hierarchy that B50 validated on 1,447 studies "
            "also improves the full 4,349-study endpoint, making B51 differ from "
            "B42 by exactly one change"
        ),
        "b50_evidence": {
            "unseen_scanner_delta": 0.011221,
            "targets_improved": "12/12",
            "discordant_ceiling": 0.030652,
            "expert58_delta": -0.002432,
            "expert58_note": (
                "inconclusive: 58 studies resolve to about +/-0.03, 2 targets "
                "improved, 5 worsened and 5 were exactly unchanged, one-sided "
                "p=0.227 excluding ties. B51 rests on the report-derived result."
            ),
        },
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "base_state": model.base.state_dict(),
        "head_state": model.head.state_dict(),
        "model_state": model.state(),
        "encoder_sha256_initial": encoder_initial_sha,
        "encoder_sha256_final": encoder_final_sha,
        "head_lr": head_lr,
        "encoder_lr": head_lr * encoder_scale,
        "local_aux_weight": aux_weight,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "training_studies": len(uids),
        "training_series": expected_series,
        "training_supervision_cells": expected_cells,
        "training_uids_sha256": _uid_sha256(uids),
        "target_balance_source": "full_report_only_weak_labels",
        "target_balance_multiplier": target_balance,
        "label_confidence": confidence,
        "fill_policy": fill_policy,
        "fill_audit": fill_audit,
        "fill_artifacts": fill_artifacts,
        "supervision": supervision,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "b50": b50_state("adapted_hierarchy_candidate"),
        "config_sha256": _config_sha256(settings),
        "source_sha256": {
            "model": sha256_file(Path(__file__).with_name("b50_adapted_hierarchy_mil.py")),
            "training": sha256_file(Path(__file__)),
        },
        "history": history,
        "governance": (
            "B51 is a production run of the mechanism B50 validated, not an "
            "experiment. Its control is B42's existing hidden score. Do not tune "
            "the hierarchy learning rate, epoch count, seed, geometry or target "
            "subset from its hidden result."
        ),
    }
    torch.save(payload, checkpoint_path)
    audit = {k: v for k, v in payload.items() if k not in {"base_state", "head_state"}}
    (out / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint_path, flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("Train B51 on the full report-only population")
    parser.add_argument("--config", default="config/b51_full_population_adapted_hierarchy.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", default=B51_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train_b51(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B51_CHECKPOINT_NAME",
    "B51_EXPERIMENT",
    "B51_REPORT_ONLY_STUDIES",
    "B51_RUN_ROOT",
    "B51_VERSION",
    "full_population_rows",
    "train_b51",
]
