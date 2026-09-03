"""Train one fixed B50 matched arm on the scanner-grouped split.

B50 asks whether the study hierarchy — 18,952,716 parameters frozen since B34 —
should be allowed to adapt to the 448-pixel features the encoder now produces.
Both arms are identical except for that. See
`developments/docs/B50_REDESIGN_UNFREEZE_STUDY_HIERARCHY.md`.

The data path, geometry, losses and endpoint are B42's, reused unchanged. The
scanner-split plumbing is B48/B49's, reused unchanged. What is new here is small
and deliberate: which parameters receive gradients, the optimiser group that
gives the hierarchy its own reduced rate, and a per-epoch guard that checks the
arms actually differ in the way the protocol claims.

That guard is the part worth reading. B49's trainer raises if any non-encoder
base parameter carries a gradient, which is exactly right for B49 and exactly
wrong for B50's candidate. So the check becomes arm-dependent, and it is
enforced in both directions:

    frozen_hierarchy_control      a hierarchy gradient is a contamination bug
    adapted_hierarchy_candidate   NO hierarchy gradient is a wiring bug

A silent failure of the second kind would produce two identical arms, a null
result, and a wasted training run that looked exactly like B48 and B49. Given
what those two turned out to be, that is the specific failure worth spending
code on.
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
    B50_ARMS,
    B50_EXPERIMENT,
    B50_HIERARCHY_LR_SCALE,
    B50_RUN_ROOT,
    B50_VERSION,
    B50AdaptedHierarchySparseMILResidual,
    b50_parameter_groups,
    b50_state,
    require_b50_contract,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .b17_training import encoder_state_sha256
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import make_scaler, resolve_runtime

B50_FIXED_EPOCHS = 2
B50_EFFECTIVE_BATCH = 2
B50_SEED = 2026
B50_CONSTRUCTION_SEED_OFFSET = 11
B50_LOADER_SEED_OFFSET = 29
B50_CHECKPOINT_TEMPLATE = "b50_{arm}_model.pt"
B50_SUPERVISION = "report-only weak labels, scanner-split train rows only"


def _read_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_b50_selection_gate(path: str | Path) -> tuple[dict, "pd.DataFrame", dict]:
    """Read the fresh B50 gate, not the split B48 and B49 already spent.

    B50 must not be selected on the B48/B49 scanner surface. Those rows have
    been inspected twice, and the project's own governance records that split as
    spent for new architecture selection. The fresh gate is built only from the
    parent's former `train` rows, with every former B48/B49 validation row
    excluded outright.

    The gate stores its assignment under `b50_split` with an
    `excluded_prior_surface` label the parent format has no equivalent for, so
    it is verified here and then renamed to the `split` column the shared index
    helper expects. Requiring this file rather than accepting either format is
    deliberate: a trainer that silently accepted the spent split would let the
    boundary be crossed by a path argument.

    Every one of the 4,349 report-only rows is returned, the spent ones still
    carrying their `excluded_prior_surface` label. They are excluded by never
    being asked for -- only the `train` split is selected -- rather than by being
    deleted here. Deleting them would defeat the shared surface check that every
    report-only study is accounted for by exactly one split, which is a guard
    worth keeping.
    """
    import pandas as pd

    from .b50_ordered_slice_selection_split import (
        B50_SPLIT_EXCLUDED,
        verify_b50_selection_split,
    )

    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    payload_path = directory / "b50_selection_split.json"
    rows_path = directory / "b50_selection_split_by_study.csv"
    if not payload_path.exists() or not rows_path.exists():
        raise FileNotFoundError(
            f"B50 requires its fresh selection gate at {directory}. Build it once "
            "with developments/scripts/prepare_b50_ordered_slice_gate.sh; the "
            "B48/B49 domain_split.json is spent and must not be used here."
        )

    payload = json.loads(payload_path.read_text())
    rows = pd.read_csv(rows_path)
    verify_b50_selection_split(rows)

    meta = {
        "path": str(payload_path),
        "sha256": sha256_file(payload_path),
        "rows_sha256": sha256_file(rows_path),
        "version": payload.get("version"),
        "salt": payload.get("salt"),
    }
    rows = rows.copy()
    rows["split"] = rows["b50_split"].astype(str)
    if not (rows["split"] == B50_SPLIT_EXCLUDED).any():
        raise ValueError(
            "B50 gate marks no rows as spent by B48/B49, which cannot be right: "
            "the parent's validation rows must all carry excluded_prior_surface"
        )
    return payload, rows, meta


def _hierarchy_gradient_present(model) -> bool:
    """Did any hierarchy parameter actually receive a non-zero gradient?"""
    for parameter in model.hierarchy_parameters():
        grad = parameter.grad
        if grad is not None and torch.count_nonzero(grad).item() > 0:
            return True
    return False


def _check_arm_wiring(model, *, adapt_hierarchy: bool, hierarchy_saw_gradient: bool) -> None:
    """Both directions, because either failure silently ruins the comparison."""
    if adapt_hierarchy:
        if not hierarchy_saw_gradient:
            raise RuntimeError(
                "B50 candidate trained without a single hierarchy gradient. The two "
                "arms would be identical and the run would report a null that means "
                "nothing. Check that the unfreeze ran before the optimiser was built."
            )
        return

    leaked = [
        name
        for name, parameter in model.base.named_parameters()
        if not name.startswith("encoder.")
        and parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
    ]
    if leaked:
        raise RuntimeError(
            f"B50 control received gradients on {len(leaked)} frozen hierarchy "
            f"parameter(s), so it is not a reproduction of B42: {leaked[:5]}"
        )


def train_b50_domain_arm(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    arm: str,
    seed: int = B50_SEED,
    out_root: str | Path = B50_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train exactly two epochs of one B50 matched arm."""
    arm, seed = str(arm), int(seed)
    if arm not in B50_ARMS:
        raise ValueError(f"B50 arm must be one of {B50_ARMS}; got {arm!r}")
    if seed != B50_SEED:
        raise ValueError(f"B50 freezes seed={B50_SEED}; got {seed}")

    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["b50_arm"] = arm
    contract = require_b50_contract(settings)
    adapt_hierarchy = bool(contract["adapt_hierarchy"])

    domain_payload, domain_rows, domain_meta = load_b50_selection_gate(domain_split)
    settings["seed"] = seed
    seed_everything(seed + B50_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    print(
        f"[B50 {arm}] adapt_hierarchy={adapt_hierarchy} "
        f"domain_split_sha={domain_meta['sha256']}",
        flush=True,
    )

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(
        root / settings.get("train_csv", "train.csv")
    ) != expected_train_sha:
        raise ValueError("B50 domain split source train.csv fingerprint mismatch")

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
        domain_rows=domain_rows,
        base_payload=base_payload,
    )
    train_indices = _indices_for_split(all_uids, domain_rows, "train")
    uids = [all_uids[index] for index in train_indices]
    targets, weights = all_targets[train_indices], all_weights[train_indices]
    target_multiplier = target_balance_multipliers(weights)

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B50 requires the frozen B12/B13 series policy")
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
        raise RuntimeError("B50 scanner-split MRI/weak-label surface failed viability")

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
        adapt_hierarchy=adapt_hierarchy,
    ).to(runtime.device)
    model.train()

    trainable = model.trainable_parameter_summary()
    print(f"[B50 {arm}] trainable={trainable}", flush=True)
    if adapt_hierarchy and trainable["hierarchy_trainable_parameters"] == 0:
        raise RuntimeError("B50 candidate has no trainable hierarchy parameters")
    if not adapt_hierarchy and trainable["hierarchy_trainable_parameters"] != 0:
        raise RuntimeError("B50 control must leave the hierarchy frozen")

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

    arm_root = Path(out_root)
    if preflight_only:
        items = [dataset[index] for index in range(min(2, len(dataset)))]
        optimizer.zero_grad(set_to_none=True)
        for item, scale in zip(items, _batch_scales(items, multiplier_cpu)):
            tensors = _move_study(item, runtime.device)
            _out, total, _combined, _local = _losses(
                model, runtime, tensors, multiplier_t, aux_weight
            )
            scaler.scale(total * float(scale)).backward()
        _check_arm_wiring(
            model,
            adapt_hierarchy=adapt_hierarchy,
            hierarchy_saw_gradient=_hierarchy_gradient_present(model),
        )
        optimizer.zero_grad(set_to_none=True)
        print(
            f"[B50 {arm} preflight] PASS {_format_memory_state(_memory_state(runtime))}",
            flush=True,
        )
        return None

    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_root / B50_CHECKPOINT_TEMPLATE.format(arm=arm)
    if checkpoint_path.exists():
        raise FileExistsError(
            f"B50 will not overwrite an existing checkpoint: {checkpoint_path}"
        )

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
                _out, total, combined, local = _losses(
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
                adapt_hierarchy=adapt_hierarchy,
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
            if step % 20 == 0 or step == len(loader):
                elapsed = (time.monotonic() - started) / 60.0
                remaining = elapsed / step * (len(loader) - step)
                gate = model.head.effective_gate().detach().abs().mean().item()
                print(
                    f"[B50 {arm}] E{epoch} {step}/{len(loader)} "
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
                "B50 epoch surface changed: "
                f"studies={studies_seen}/{len(uids)} series={series_seen}/{expected_series} "
                f"cells={cells_seen}/{expected_cells}"
            )

        row = {
            "epoch": epoch,
            "arm": arm,
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
            f"[B50 {arm}] E{epoch} total={row['loss_total']:.10f} "
            f"combined={row['loss_combined']:.10f} local={row['loss_local_aux']:.10f} "
            f"time={row['epoch_seconds']/60:.1f} min",
            flush=True,
        )
        _save_recovery(
            arm_root, epoch=epoch, model=model, history=history, version=B50_VERSION
        )

    encoder_final_sha = encoder_state_sha256(model.base.encoder)
    if encoder_final_sha == encoder_initial_sha:
        raise RuntimeError("B50 encoder fingerprint did not move")

    target_balance = {
        target: float(target_multiplier[index]) for index, target in enumerate(TARGETS)
    }
    source_sha = {
        "model": sha256_file(Path(__file__).with_name("b50_adapted_hierarchy_mil.py")),
        "training": sha256_file(Path(__file__)),
        "b48_domain_protocol": sha256_file(
            Path(__file__).with_name("b48_global_conditioned_sparse_training.py")
        ),
    }
    matched_pair_identity = {
        "seed": seed,
        "config_sha256": _config_sha256(settings),
        "base_checkpoint_sha256": sha256_file(base_path),
        "training_uids_sha256": _uid_sha256(uids),
        "target_balance_multiplier": target_balance,
        "domain_split_sha256": domain_meta["sha256"],
        "domain_rows_sha256": domain_meta["rows_sha256"],
        "fill_artifacts": fill_artifacts,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "source_sha256": source_sha,
    }
    payload = {
        "experiment": B50_EXPERIMENT,
        "version": B50_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B50_FIXED_EPOCHS,
        "checkpoint_selection": "none; fixed epoch 2",
        "arm": arm,
        "seed": seed,
        "hypothesis": (
            "the B34 study hierarchy, frozen since it was trained at 224 pixels, "
            "improves scanner-held-out weak-label ranking when allowed to adapt at a "
            "reduced rate to the 448-pixel features the encoder now produces"
        ),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "base_payload_experiment": base_payload.get("experiment"),
        "base_state": model.base.state_dict(),
        "head_state": model.head.state_dict(),
        "model_state": model.state(),
        "encoder_sha256_initial": encoder_initial_sha,
        "encoder_sha256_final": encoder_final_sha,
        "head_lr": head_lr,
        "encoder_lr": head_lr * encoder_scale,
        "hierarchy_lr": head_lr * float(contract["hierarchy_lr_scale"]) if adapt_hierarchy else 0.0,
        "hierarchy_lr_scale": float(contract["hierarchy_lr_scale"]),
        "adapt_hierarchy": adapt_hierarchy,
        "trainable": model.trainable_parameter_summary(),
        "local_aux_weight": aux_weight,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "supervision_source": B50_SUPERVISION,
        "training_studies": len(uids),
        "training_series": expected_series,
        "training_supervision_cells": expected_cells,
        "training_uids_sha256": _uid_sha256(uids),
        "target_balance_source": "scanner_split_train_only_weak_labels",
        "target_balance_multiplier": target_balance,
        "domain_split": domain_meta,
        "domain_split_summary": domain_payload.get("summary", {}),
        "label_confidence": confidence,
        "fill_policy": fill_policy,
        "fill_audit": fill_audit,
        "fill_artifacts": fill_artifacts,
        "supervision": supervision,
        "series_policy_signature": B13_SERIES_SIGNATURE,
        "series_surface": series_summary,
        "metadata_repair": metadata_stats,
        "b50": b50_state(arm),
        "config_sha256": matched_pair_identity["config_sha256"],
        "source_sha256": source_sha,
        "matched_pair_identity": matched_pair_identity,
        "history": history,
        "governance": (
            "B50 is a prospective matched scanner-split comparison of a frozen against "
            "an adapted B34 study hierarchy. Do not use official gold labels, "
            "checkpoint selection, or any tuning of the hierarchy learning rate, "
            "epoch count, seed, geometry or endpoint after a B50 result is inspected."
        ),
    }
    torch.save(payload, checkpoint_path)
    audit = {
        key: value
        for key, value in payload.items()
        if key not in {"base_state", "head_state"}
    }
    (arm_root / "training_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (arm_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint_path, flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("Train one fixed B50 adapted-hierarchy arm")
    parser.add_argument("--config", default="config/b50_adapted_hierarchy.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--selection-gate",
        required=True,
        dest="domain_split",
        help="directory holding the fresh b50_selection_split.json",
    )
    parser.add_argument("--arm", choices=B50_ARMS, required=True)
    parser.add_argument("--seed", type=int, default=B50_SEED)
    parser.add_argument("--out-root", default=B50_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train_b50_domain_arm(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        arm=args.arm,
        seed=args.seed,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B50_CHECKPOINT_TEMPLATE",
    "B50_FIXED_EPOCHS",
    "B50_HIERARCHY_LR_SCALE",
    "B50_SEED",
    "train_b50_domain_arm",
]
