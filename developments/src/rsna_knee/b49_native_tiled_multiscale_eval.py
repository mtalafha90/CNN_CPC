"""Evaluate the frozen matched B49 native-tiled scanner-domain pair."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import build_variable_series_index
from .b17_training import encoder_state_sha256
from .b35_training import sha256_file
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b48_global_conditioned_sparse_eval import (
    _leave_one_target_out,
    _paired_auc,
    _weighted_macro_bce,
)
from .b48_global_conditioned_sparse_training import (
    _indices_for_split,
    _report_only_surface,
    _uid_sha256,
    b48_fill_artifacts,
    load_b48_domain_split,
)
from .b49_native_tiled_multiscale_mil import (
    B49_ARM_CONTEXT_SOURCE,
    B49_CONTEXT_DIM,
    B49_EXPERIMENT,
    B49_POST_CROSS_ATTENTION_CANDIDATE,
    B49_STATIC_PRIOR_CONTROL,
    B49_VERSION,
    B49NativeTiledFullFOVDataset,
    B49NativeTiledMultiscaleMILResidual,
    collate_b49,
    require_b49_contract,
)
from .b49_native_tiled_multiscale_training import B49_REPLICATION_SEEDS
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime


B49_EVAL_LOADER_SEED_OFFSET = 58_300_000
B49_EVAL_BOOTSTRAP_SEED_OFFSET = 58_400_000


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_b49_checkpoint(
    path: str | Path,
    *,
    config: dict,
    base_checkpoint: str | Path,
    arm: str,
    domain_sha256: str,
    domain_rows_sha256: str,
    training_uids: list[str],
    fill_artifacts: dict[str, str],
    device,
):
    """Rebuild and verify one B49 fixed-E2 arm before it is scored."""
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B49_EXPERIMENT or payload.get("version") != B49_VERSION:
        raise ValueError(f"{checkpoint} is not a B49 checkpoint")
    if bool(payload.get("fixed_endpoint")) is not True or int(payload.get("completed_epochs", -1)) != 2:
        raise ValueError("B49 evaluation requires a completed fixed-E2 checkpoint")
    if str(payload.get("arm")) != str(arm):
        raise ValueError("B49 checkpoint arm identity changed")
    for name, record in (("b49", payload.get("b49", {})), ("model_state", payload.get("model_state", {}))):
        if not isinstance(record, dict) or str(record.get("arm", "")) != str(arm):
            raise ValueError(f"B49 checkpoint {name} arm identity changed")
        if str(record.get("context_source", "")) != B49_ARM_CONTEXT_SOURCE[str(arm)]:
            raise ValueError(f"B49 checkpoint {name} context source changed")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(payload.get("gold_labels_used", True)):
        raise ValueError("B49 checkpoint used official gold labels")
    if str(payload.get("domain_split", {}).get("sha256", "")) != str(domain_sha256):
        raise ValueError("B49 checkpoint domain-split fingerprint mismatch")
    if str(payload.get("domain_split", {}).get("rows_sha256", "")) != str(domain_rows_sha256):
        raise ValueError("B49 checkpoint domain-split row fingerprint mismatch")
    if str(payload.get("training_uids_sha256", "")) != _uid_sha256(training_uids):
        raise ValueError("B49 checkpoint training UID surface changed")
    if payload.get("fill_artifacts") != fill_artifacts:
        raise ValueError("B49 fill-only label artifact fingerprint mismatch")
    source = payload.get("source_sha256", {})
    expected_source = {
        "model": sha256_file(Path(__file__).with_name("b49_native_tiled_multiscale_mil.py")),
        "training": sha256_file(Path(__file__).with_name("b49_native_tiled_multiscale_training.py")),
        "b48_domain_protocol": sha256_file(
            Path(__file__).with_name("b48_global_conditioned_sparse_training.py")
        ),
    }
    if source != expected_source:
        raise ValueError("B49 source fingerprint differs from the trained checkpoint")

    pair = payload.get("matched_pair_identity")
    if not isinstance(pair, dict):
        raise ValueError("B49 checkpoint is missing matched-pair identity")
    expected_pair = {
        "seed": int(payload.get("seed", -1)),
        "config_sha256": str(payload.get("config_sha256", "")),
        "base_checkpoint_sha256": str(payload.get("base_checkpoint_sha256", "")),
        "training_uids_sha256": str(payload.get("training_uids_sha256", "")),
        "target_balance_multiplier": payload.get("target_balance_multiplier"),
        "domain_split_sha256": str(domain_sha256),
        "domain_rows_sha256": str(domain_rows_sha256),
        "fill_artifacts": fill_artifacts,
        "series_policy_signature": payload.get("series_policy_signature"),
        "source_sha256": source,
        "b49_representation": payload.get("preprocessing"),
    }
    for key, value in expected_pair.items():
        if pair.get(key) != value:
            raise ValueError(f"B49 checkpoint matched-pair identity disagrees on {key}")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B49 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")
    model = B49NativeTiledMultiscaleMILResidual(
        base,
        encoder_trainable_stages=int(config["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b37_encoder_chunk_size"]),
        tile_encoder_chunk_size=int(config["b49_tile_encoder_chunk_size"]),
        arm=str(arm),
        context_dim=int(config.get("b49_context_dim", B49_CONTEXT_DIM)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    if encoder_state_sha256(model.base.encoder) != str(payload.get("encoder_sha256_final", "")):
        raise RuntimeError("B49 reconstructed encoder fingerprint changed")
    return model, payload


@torch.no_grad()
def _score_split(
    *,
    model,
    config: dict,
    root: Path,
    runtime,
    uids: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    series_index: dict,
    label: str,
) -> dict:
    cfg = make_b7_dataset_config(config, root, train=False)
    cfg.tta_center_offsets = ()
    dataset = B49NativeTiledFullFOVDataset(
        uids,
        series_index,
        cfg,
        center_offsets=B37_EVAL_OFFSETS,
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b49,
        **runtime.loader_kwargs(seed=int(config["seed"]) + B49_EVAL_LOADER_SEED_OFFSET),
    )
    predicted, global_predicted, scored = [], [], []
    context_abs_sum = np.zeros(len(TARGETS), dtype=np.float64)
    context_overlap_sum = np.zeros(len(TARGETS), dtype=np.float64)
    context_count = 0
    tile_sum = token_sum = 0
    for batch_index, items in enumerate(loader, start=1):
        if len(items) != 1:
            raise RuntimeError("B49 evaluation requires exactly one ragged study per batch")
        item = items[0]
        scored.append(str(item["study_uid"]))
        present = item["present"].to(runtime.device, non_blocking=True)
        meta = item["series_meta"].to(runtime.device, non_blocking=True)
        combined_views, global_views = [], []
        if len(item["views"]) != len(B37_EVAL_OFFSETS):
            raise RuntimeError("B49 TTA view count changed")
        for view in item["views"]:
            context = [volume.to(runtime.device, non_blocking=True) for volume in view["context_volumes"]]
            position = view["slice_position"].to(runtime.device, non_blocking=True)
            with autocast(runtime):
                out = model(
                    context,
                    view["local_sources"],
                    present,
                    meta,
                    position,
                    audit_context=True,
                )
            combined_views.append(torch.sigmoid(out.logits.float()))
            global_views.append(torch.sigmoid(out.base_logits.float()))
            context_abs_sum += out.context_abs_mean.float().sum(dim=0).cpu().numpy()
            if out.topk_overlap_with_static is None:
                raise RuntimeError("B49 evaluation did not return context top-k audit")
            context_overlap_sum += out.topk_overlap_with_static.float().sum(dim=0).cpu().numpy()
            context_count += int(out.context_abs_mean.shape[0])
            tile_sum += int(out.native_tile_count)
            token_sum += int(out.native_valid_token_count)
            del context, position, out
        predicted.append(torch.stack(combined_views).mean(dim=0).cpu().numpy()[0])
        global_predicted.append(torch.stack(global_views).mean(dim=0).cpu().numpy()[0])
        del item, items, present, meta, combined_views, global_views
        _release()
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"[B49 {label}] {batch_index}/{len(loader)}", flush=True)
    if scored != uids:
        raise RuntimeError(f"B49 {label} UID order changed")
    prediction = np.asarray(predicted, dtype=np.float64)
    if prediction.shape != targets.shape or not np.isfinite(prediction).all():
        raise RuntimeError(f"B49 {label} prediction surface is invalid")
    return {
        "uids": uids,
        "target": targets,
        "weight": weights,
        "prediction": prediction,
        "global_prediction": np.asarray(global_predicted, dtype=np.float64),
        "context": {
            "mean_abs_context_score": (context_abs_sum / max(context_count, 1)).tolist(),
            "mean_topk_overlap_with_static": (context_overlap_sum / max(context_count, 1)).tolist(),
            "mean_topk_locations_changed_fraction": (1.0 - context_overlap_sum / max(context_count, 1)).tolist(),
        },
        "native_tile_audit": {
            "total_tiles": int(tile_sum),
            "total_valid_tokens": int(token_sum),
            "mean_tiles_per_study_per_tta_view": float(tile_sum / max(len(uids) * len(B37_EVAL_OFFSETS), 1)),
            "mean_valid_tokens_per_study_per_tta_view": float(token_sum / max(len(uids) * len(B37_EVAL_OFFSETS), 1)),
        },
    }


def _pair_identity(payload: dict) -> dict:
    return {
        key: payload.get(key)
        for key in (
            "arm",
            "seed",
            "config_sha256",
            "training_uids_sha256",
            "base_checkpoint_sha256",
            "matched_pair_identity",
        )
    }


def _matched_seed(control: dict, candidate: dict) -> int:
    if control.get("matched_pair_identity") != candidate.get("matched_pair_identity"):
        raise ValueError("B49 control/candidate checkpoints do not share one frozen pair identity")
    seed = int(control["matched_pair_identity"].get("seed", -1))
    if seed not in B49_REPLICATION_SEEDS:
        raise ValueError(f"B49 pair seed must be one of {B49_REPLICATION_SEEDS}; got {seed}")
    return seed


def _write_predictions(path: Path, score: dict) -> None:
    frame = pd.DataFrame(score["prediction"], columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", score["uids"])
    frame.to_csv(path, index=False)


def evaluate_b49_domain_pair(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    """Evaluate B49 static/candidate arms on the existing frozen scanner split."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    require_b49_contract(settings, arm=B49_STATIC_PRIOR_CONTROL)
    require_b49_contract(settings, arm=B49_POST_CROSS_ATTENTION_CANDIDATE)
    domain_payload, domain_rows, domain_meta = load_b48_domain_split(domain_split)
    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(root / settings.get("train_csv", "train.csv")) != expected_train_sha:
        raise ValueError("B49 domain split source train.csv fingerprint mismatch")
    fill_artifacts = b48_fill_artifacts(labels_root)
    base_path = Path(base_checkpoint).resolve()
    base_probe, base_payload = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")
    del base_probe
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        _confidence,
        _fill_policy,
        _fill_audit,
        _supervision,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
    )
    train_indices = _indices_for_split(all_uids, domain_rows, "train")
    train_uids = [all_uids[index] for index in train_indices]
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    all_index = build_variable_series_index(series, all_uids)

    specs = (
        (B49_STATIC_PRIOR_CONTROL, Path(control_checkpoint)),
        (B49_POST_CROSS_ATTENTION_CANDIDATE, Path(candidate_checkpoint)),
    )
    scores: dict[str, dict[str, dict]] = {}
    records: dict[str, dict] = {}
    pair_seed: int | None = None
    for arm, checkpoint in specs:
        model, payload = load_b49_checkpoint(
            checkpoint,
            config=settings,
            base_checkpoint=base_path,
            arm=arm,
            domain_sha256=domain_meta["sha256"],
            domain_rows_sha256=domain_meta["rows_sha256"],
            training_uids=train_uids,
            fill_artifacts=fill_artifacts,
            device=runtime.device,
        )
        record = _pair_identity(payload)
        if arm == B49_STATIC_PRIOR_CONTROL:
            settings["seed"] = int(payload["seed"])
        else:
            pair_seed = _matched_seed(records[B49_STATIC_PRIOR_CONTROL], record)
        records[arm] = record
        arm_scores: dict[str, dict] = {}
        for split in ("validation_seen_scanners", "holdout_unseen_scanners"):
            indices = _indices_for_split(all_uids, domain_rows, split)
            uids = [all_uids[index] for index in indices]
            if set(uids).intersection(train_uids) or any(uid not in all_index for uid in uids):
                raise RuntimeError(f"B49 {split} UID/MRI surface changed")
            arm_scores[split] = _score_split(
                model=model,
                config=settings,
                root=root,
                runtime=runtime,
                uids=uids,
                targets=all_targets[indices],
                weights=all_weights[indices],
                series_index={uid: all_index[uid] for uid in uids},
                label=f"{arm}:{split}",
            )
        scores[arm] = arm_scores
        del model, payload
        _release()
    if pair_seed is None:
        raise RuntimeError("B49 evaluation did not load a complete matched checkpoint pair")

    selected = [str(value) for value in domain_payload.get("summary", {}).get("comparable_targets", [])]
    if not selected or [target for target in selected if target in TARGETS] != selected:
        raise RuntimeError("B49 frozen domain split comparable-target list changed")
    comparisons = {}
    control, candidate = scores[B49_STATIC_PRIOR_CONTROL], scores[B49_POST_CROSS_ATTENTION_CANDIDATE]
    for offset, split in enumerate(("validation_seen_scanners", "holdout_unseen_scanners")):
        a, b = control[split], candidate[split]
        if a["uids"] != b["uids"] or not np.array_equal(a["target"], b["target"]) or not np.array_equal(a["weight"], b["weight"]):
            raise RuntimeError(f"B49 {split} matched arms are not aligned")
        paired = _paired_auc(
            a["target"],
            a["weight"],
            a["prediction"],
            b["prediction"],
            selected=selected,
            n_bootstrap=int(n_bootstrap),
            seed=int(pair_seed) + B49_EVAL_BOOTSTRAP_SEED_OFFSET + offset,
        )
        paired["control_macro_weighted_bce"] = _weighted_macro_bce(a["target"], a["prediction"], a["weight"], selected)
        paired["candidate_macro_weighted_bce"] = _weighted_macro_bce(a["target"], b["prediction"], a["weight"], selected)
        paired["candidate_minus_control_macro_weighted_bce"] = float(
            paired["candidate_macro_weighted_bce"] - paired["control_macro_weighted_bce"]
        )
        paired["leave_one_target_out_candidate_minus_control"] = _leave_one_target_out(
            paired["control_per_target_auc"], paired["candidate_per_target_auc"], selected
        )
        paired["targets_improved"] = int(sum(
            paired["candidate_per_target_auc"][target] > paired["control_per_target_auc"][target]
            for target in selected
        ))
        comparisons[split] = paired
    unseen, seen = comparisons["holdout_unseen_scanners"], comparisons["validation_seen_scanners"]
    control_gap = float(seen["control_macro_auc"] - unseen["control_macro_auc"])
    candidate_gap = float(seen["candidate_macro_auc"] - unseen["candidate_macro_auc"])
    all_targets = len(selected) == len(TARGETS)
    loto = unseen["leave_one_target_out_candidate_minus_control"]
    supported = bool(
        all_targets
        and unseen["candidate_minus_control"] >= 0.010
        and unseen["ci_lower"] > 0.0
        and unseen["probability_candidate_better"] >= 0.95
        and unseen["targets_improved"] >= 7
        and min(loto.values()) > 0.0
        and seen["candidate_minus_control"] >= -0.005
        and candidate_gap - control_gap <= 0.005
    )
    if supported:
        verdict = "support_for_global_conditioning_within_native_tiled_representation"
    elif not all_targets:
        verdict = "inconclusive_insufficient_comparable_targets"
    elif unseen["candidate_minus_control"] < 0.005 or unseen["ci_lower"] <= 0.0:
        verdict = "no_support_for_global_conditioning_within_native_tiled_representation"
    else:
        verdict = "inconclusive"
    output = {
        "evaluation_role": "prospective matched weak-label scanner-domain comparison; not official-gold or hidden-test evidence",
        "experiment": B49_EXPERIMENT,
        "version": B49_VERSION,
        "domain_split": domain_meta,
        "comparable_targets": selected,
        "all_12_targets_comparable": all_targets,
        "checkpoints": {
            arm: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "seed": int(records[arm]["seed"]),
            }
            for arm, path in specs
        },
        "pair_seed": int(pair_seed),
        "matched_pair_identity": records[B49_STATIC_PRIOR_CONTROL]["matched_pair_identity"],
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "evaluation_dependency_sha256": {
            "b48_metric_helpers": sha256_file(
                Path(__file__).with_name("b48_global_conditioned_sparse_eval.py")
            )
        },
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "comparison": comparisons,
        "domain_gap": {
            "control_seen_minus_unseen": control_gap,
            "candidate_seen_minus_unseen": candidate_gap,
            "candidate_minus_control_gap": float(candidate_gap - control_gap),
        },
        "context_audit": {
            arm: {split: scores[arm][split]["context"] for split in ("validation_seen_scanners", "holdout_unseen_scanners")}
            for arm in (B49_STATIC_PRIOR_CONTROL, B49_POST_CROSS_ATTENTION_CANDIDATE)
        },
        "native_tile_audit": {
            arm: {split: scores[arm][split]["native_tile_audit"] for split in ("validation_seen_scanners", "holdout_unseen_scanners")}
            for arm in (B49_STATIC_PRIOR_CONTROL, B49_POST_CROSS_ATTENTION_CANDIDATE)
        },
        "metadata_repair": metadata_stats,
        "predeclared_decision_rule": {
            "support": "all 12 targets; unseen delta>=+0.010; paired lower CI>0; P>0.95; >=7 targets improve; all LOTO deltas>0; seen delta>=-0.005; domain-gap increase<=+0.005",
            "no_support": "unseen delta<+0.005 OR paired lower CI<=0",
            "otherwise": "inconclusive",
        },
        "verdict": verdict,
        "governance": "Do not tune B49 tile geometry, overlap, query source, rank, seeds, losses, endpoint or blend after this result. A B49-vs-B48 representation comparison is descriptive unless separately predeclared.",
    }
    root_out = Path(out_root)
    root_out.mkdir(parents=True, exist_ok=True)
    for arm in (B49_STATIC_PRIOR_CONTROL, B49_POST_CROSS_ATTENTION_CANDIDATE):
        for split in ("validation_seen_scanners", "holdout_unseen_scanners"):
            _write_predictions(root_out / f"{arm}_{split}_predictions.csv", scores[arm][split])
    (root_out / "b49_domain_evaluation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    print("B49 MATCHED DOMAIN EVALUATION: PASS", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate the matched B49 native-tiled scanner-domain arms")
    parser.add_argument("--config", default="config/b49_native_tiled_multiscale.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    evaluate_b49_domain_pair(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        control_checkpoint=args.control_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()


__all__ = ["evaluate_b49_domain_pair", "load_b49_checkpoint"]
