"""Evaluate the matched B48 arms on the frozen scanner-domain surface.

This is intentionally a paired weak-label comparison, not another adaptive
Expert-58 selection.  The evaluator refuses a checkpoint whose training UID
fingerprint or domain-split fingerprint differs from its matched arm.
"""
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
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    collate_b42,
)
from .b48_global_conditioned_sparse_mil import (
    B48_ARM_CONTEXT_SOURCE,
    B48_CONTEXT_DIM,
    B48_EXPERIMENT,
    B48_POST_CROSS_ATTENTION_CANDIDATE,
    B48_STATIC_PRIOR_CONTROL,
    B48_VERSION,
    B48GlobalConditionedSparseMILResidual,
    require_b48_contract,
)
from .b48_global_conditioned_sparse_training import (
    B48_REPLICATION_SEEDS,
    _indices_for_split,
    _report_only_surface,
    _uid_sha256,
    b48_fill_artifacts,
    load_b48_domain_split,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .evaluation import fast_auc
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime
from .b35_training import sha256_file

B48_EVAL_LOADER_SEED_OFFSET = 57_300_000
B48_EVAL_BOOTSTRAP_SEED_OFFSET = 57_400_000


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_b48_checkpoint(
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
    """Reconstruct one fixed B48 arm and verify its paired-run identity."""
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != B48_EXPERIMENT or payload.get("version") != B48_VERSION:
        raise ValueError(f"{checkpoint} is not a B48 checkpoint")
    if bool(payload.get("fixed_endpoint")) is not True or int(payload.get("completed_epochs", -1)) != 2:
        raise ValueError("B48 evaluation requires a completed fixed-E2 checkpoint")
    if str(payload.get("arm")) != str(arm):
        raise ValueError("B48 checkpoint arm identity changed")
    for record_name, record in (
        ("b48", payload.get("b48", {})),
        ("model_state", payload.get("model_state", {})),
    ):
        if not isinstance(record, dict) or str(record.get("arm", "")) != str(arm):
            raise ValueError(f"B48 checkpoint {record_name} arm identity changed")
        if str(record.get("context_source", "")) != B48_ARM_CONTEXT_SOURCE[str(arm)]:
            raise ValueError(f"B48 checkpoint {record_name} context source changed")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0 or bool(
        payload.get("gold_labels_used", True)
    ):
        raise ValueError("B48 checkpoint used official gold labels")
    if str(payload.get("domain_split", {}).get("sha256", "")) != str(domain_sha256):
        raise ValueError("B48 checkpoint domain-split fingerprint mismatch")
    if str(payload.get("domain_split", {}).get("rows_sha256", "")) != str(domain_rows_sha256):
        raise ValueError("B48 checkpoint domain-split CSV fingerprint mismatch")
    if str(payload.get("training_uids_sha256", "")) != _uid_sha256(training_uids):
        raise ValueError("B48 checkpoint training UID surface changed")
    if payload.get("fill_artifacts") != fill_artifacts:
        raise ValueError("B48 fill-only label artifact fingerprint mismatch")
    source_sha = payload.get("source_sha256", {})
    pair_identity = payload.get("matched_pair_identity", {})
    if not isinstance(pair_identity, dict):
        raise ValueError("B48 checkpoint is missing its matched-pair identity")
    expected_identity = {
        "seed": int(payload.get("seed", -1)),
        "config_sha256": str(payload.get("config_sha256", "")),
        "base_checkpoint_sha256": str(payload.get("base_checkpoint_sha256", "")),
        "training_uids_sha256": str(payload.get("training_uids_sha256", "")),
        "domain_split_sha256": str(domain_sha256),
        "domain_rows_sha256": str(domain_rows_sha256),
        "fill_artifacts": fill_artifacts,
        "target_balance_multiplier": payload.get("target_balance_multiplier"),
        "series_policy_signature": payload.get("series_policy_signature"),
        "source_sha256": source_sha,
    }
    for key, expected in expected_identity.items():
        if pair_identity.get(key) != expected:
            raise ValueError(f"B48 checkpoint matched-pair identity disagrees on {key}")
    current_model_sha = sha256_file(
        Path(__file__).with_name("b48_global_conditioned_sparse_mil.py")
    )
    if str(source_sha.get("model", "")) != current_model_sha:
        raise ValueError("B48 model source fingerprint differs from the trained checkpoint")

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B48 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")
    model = B48GlobalConditionedSparseMILResidual(
        base,
        grid_size=int(config["b37_grid_size"]),
        top_k=int(config["b37_top_k"]),
        temperature=float(config["b37_temperature"]),
        encoder_trainable_stages=int(config["b37_encoder_trainable_stages"]),
        encoder_chunk_size=int(config["b37_encoder_chunk_size"]),
        arm=str(arm),
        context_dim=int(config.get("b48_context_dim", B48_CONTEXT_DIM)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    if encoder_state_sha256(model.base.encoder) != str(payload.get("encoder_sha256_final", "")):
        raise RuntimeError("B48 reconstructed encoder fingerprint changed")
    return model, payload


def _matched_pair_seed(control_payload: dict, candidate_payload: dict) -> int:
    """Return the seed after proving the checkpoints form one frozen pair."""
    control_identity = control_payload.get("matched_pair_identity")
    candidate_identity = candidate_payload.get("matched_pair_identity")
    if not isinstance(control_identity, dict) or not isinstance(candidate_identity, dict):
        raise ValueError("B48 checkpoints are missing matched-pair identity records")
    if control_identity != candidate_identity:
        raise ValueError("B48 control/candidate checkpoints do not share one frozen pair identity")
    seed = int(control_identity.get("seed", -1))
    if seed not in B48_REPLICATION_SEEDS:
        raise ValueError(f"B48 pair seed must be one of {B48_REPLICATION_SEEDS}; got {seed}")
    for payload, arm in (
        (control_payload, B48_STATIC_PRIOR_CONTROL),
        (candidate_payload, B48_POST_CROSS_ATTENTION_CANDIDATE),
    ):
        if str(payload.get("arm", "")) != arm:
            raise ValueError("B48 pair arm identity changed")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError("B48 checkpoint seed differs from matched-pair identity")
        if str(payload.get("config_sha256", "")) != str(control_identity.get("config_sha256", "")):
            raise ValueError("B48 checkpoint config fingerprint differs from matched-pair identity")
        if str(payload.get("training_uids_sha256", "")) != str(
            control_identity.get("training_uids_sha256", "")
        ):
            raise ValueError("B48 checkpoint training UID fingerprint differs from matched-pair identity")
        if str(payload.get("base_checkpoint_sha256", "")) != str(
            control_identity.get("base_checkpoint_sha256", "")
        ):
            raise ValueError("B48 checkpoint base fingerprint differs from matched-pair identity")
    return seed


def _pair_checkpoint_record(payload: dict) -> dict:
    """Retain only pairing metadata after a large checkpoint is reconstructed."""
    keys = (
        "arm",
        "seed",
        "config_sha256",
        "training_uids_sha256",
        "base_checkpoint_sha256",
        "matched_pair_identity",
    )
    return {key: payload.get(key) for key in keys}


def _masked_target_matrix(target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return binary report states, retaining NaN only for unsupervised cells.

    B48 trains against the frozen soft weak targets (normally 0.85 and 0.05),
    but ROC-AUC is a ranking metric over positive/negative states.  It must
    therefore use the same fixed 0.5 state boundary as the scanner-split
    construction rather than looking for literal target values of 1 and 0.
    """
    value = (np.asarray(target, dtype=np.float64) > 0.5).astype(np.float64)
    value[np.asarray(weight, dtype=np.float64) <= 0] = np.nan
    return value


def _per_target_auc(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    masked = _masked_target_matrix(target, weight)
    return {
        name: float(fast_auc(masked[:, index], prediction[:, index]))
        for index, name in enumerate(TARGETS)
    }


def _macro_from_selected(per_target: dict[str, float], selected: list[str]) -> float:
    values = np.asarray([per_target[name] for name in selected], dtype=np.float64)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else float("nan")


def _weighted_macro_bce(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray, selected: list[str]) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    weight = np.asarray(weight, dtype=np.float64)
    values = []
    for index, name in enumerate(TARGETS):
        if name not in selected:
            continue
        valid = weight[:, index] > 0
        if not valid.any():
            return float("nan")
        y = target[valid, index]
        p = prediction[valid, index]
        values.append(float((weight[valid, index] * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))).sum() / weight[valid, index].sum()))
    return float(np.mean(values)) if values else float("nan")


def _paired_auc(
    target: np.ndarray,
    weight: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    selected: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Paired study bootstrap that rejects a replicate missing any selected class."""
    if n_bootstrap < 1:
        raise ValueError("B48 bootstrap count must be positive")
    control_per_target = _per_target_auc(target, control, weight)
    candidate_per_target = _per_target_auc(target, candidate, weight)
    control_macro = _macro_from_selected(control_per_target, selected)
    candidate_macro = _macro_from_selected(candidate_per_target, selected)
    rng = np.random.default_rng(int(seed))
    differences = []
    for _ in range(int(n_bootstrap)):
        index = rng.integers(0, len(target), size=len(target))
        a = _macro_from_selected(
            _per_target_auc(target[index], control[index], weight[index]), selected
        )
        b = _macro_from_selected(
            _per_target_auc(target[index], candidate[index], weight[index]), selected
        )
        if np.isfinite(a) and np.isfinite(b):
            differences.append(float(b - a))
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        lower = upper = probability = median = float("nan")
    else:
        lower, upper = (float(value) for value in np.percentile(values, [2.5, 97.5]))
        probability = float((values > 0).mean())
        median = float(np.median(values))
    return {
        "control_macro_auc": control_macro,
        "candidate_macro_auc": candidate_macro,
        "candidate_minus_control": float(candidate_macro - control_macro),
        "median_difference": median,
        "ci_lower": lower,
        "ci_upper": upper,
        "probability_candidate_better": probability,
        "n_bootstrap": int(n_bootstrap),
        "n_valid_replicates": int(len(values)),
        "valid_replicate_fraction": float(len(values) / float(n_bootstrap)),
        "control_per_target_auc": control_per_target,
        "candidate_per_target_auc": candidate_per_target,
    }


def _leave_one_target_out(
    control_per_target: dict[str, float], candidate_per_target: dict[str, float], selected: list[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for omitted in selected:
        kept = [target for target in selected if target != omitted]
        result[omitted] = float(
            _macro_from_selected(candidate_per_target, kept)
            - _macro_from_selected(control_per_target, kept)
        )
    return result


@torch.no_grad()
def _score_split(
    *,
    model,
    config: dict,
    root: Path,
    runtime,
    crop_policy: dict,
    uids: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    series_index: dict,
    label: str,
) -> dict:
    cfg = make_b7_dataset_config(config, root, train=False)
    cfg.tta_center_offsets = ()
    dataset = B42ConstantAreaAspectDataset(
        uids,
        series_index,
        cfg,
        crop_focus_policy=crop_policy,
        center_offsets=B37_EVAL_OFFSETS,
        targets=targets,
        weights=weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=int(config["seed"]) + B48_EVAL_LOADER_SEED_OFFSET),
    )
    predicted, global_predicted, scored = [], [], []
    context_abs_sum = np.zeros(len(TARGETS), dtype=np.float64)
    context_overlap_sum = np.zeros(len(TARGETS), dtype=np.float64)
    context_count = 0
    for batch_index, items in enumerate(loader, start=1):
        if len(items) != 1:
            raise RuntimeError("B48 domain evaluation requires one ragged study per batch")
        item = items[0]
        uid = str(item["study_uid"])
        scored.append(uid)
        present = item["present"].to(runtime.device, non_blocking=True)
        meta = item["series_meta"].to(runtime.device, non_blocking=True)
        position_all = item["slice_position"].to(runtime.device, non_blocking=True)
        if position_all.ndim != 3 or int(position_all.shape[1]) != len(B37_EVAL_OFFSETS):
            raise RuntimeError("B48 TTA slice-position shape changed")
        combined_views, global_views = [], []
        for view in range(len(B37_EVAL_OFFSETS)):
            volumes = [series_tensor[view].to(runtime.device, non_blocking=True) for series_tensor in item["volumes"]]
            position = position_all[:, view]
            with autocast(runtime):
                out = model(volumes, present, meta, position, audit_context=True)
            combined_views.append(torch.sigmoid(out.logits.float()))
            global_views.append(torch.sigmoid(out.base_logits.float()))
            context_abs_sum += out.context_abs_mean.float().sum(dim=0).cpu().numpy()
            if out.topk_overlap_with_static is None:
                raise RuntimeError("B48 evaluation did not return context top-k audit")
            context_overlap_sum += out.topk_overlap_with_static.float().sum(dim=0).cpu().numpy()
            context_count += int(out.context_abs_mean.shape[0])
            del volumes, position, out
        predicted.append(torch.stack(combined_views).mean(dim=0).cpu().numpy()[0])
        global_predicted.append(torch.stack(global_views).mean(dim=0).cpu().numpy()[0])
        del item, items, present, meta, position_all, combined_views, global_views
        _release()
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"[B48 {label}] {batch_index}/{len(loader)}", flush=True)
    if scored != uids:
        raise RuntimeError(f"B48 {label} UID order changed")
    prediction = np.asarray(predicted, dtype=np.float64)
    global_prediction = np.asarray(global_predicted, dtype=np.float64)
    if prediction.shape != targets.shape or not np.isfinite(prediction).all():
        raise RuntimeError(f"B48 {label} prediction surface is invalid")
    return {
        "uids": uids,
        "target": targets,
        "weight": weights,
        "prediction": prediction,
        "global_prediction": global_prediction,
        "context": {
            "mean_abs_context_score": (context_abs_sum / max(context_count, 1)).tolist(),
            "mean_topk_overlap_with_static": (context_overlap_sum / max(context_count, 1)).tolist(),
            "mean_topk_locations_changed_fraction": (
                1.0 - context_overlap_sum / max(context_count, 1)
            ).tolist(),
        },
    }


def _write_predictions(path: Path, score: dict) -> None:
    frame = pd.DataFrame(score["prediction"], columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", score["uids"])
    frame.to_csv(path, index=False)


def evaluate_b48_domain_pair(
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
    """Score the two fixed B48 arms on seen and unseen scanner rows."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    control_contract = require_b48_contract(settings, arm=B48_STATIC_PRIOR_CONTROL)
    candidate_contract = require_b48_contract(settings, arm=B48_POST_CROSS_ATTENTION_CANDIDATE)
    if control_contract["crop_policy"] != candidate_contract["crop_policy"]:
        raise RuntimeError("B48 matched arms resolved different B42 crop policies")

    domain_payload, domain_rows, domain_meta = load_b48_domain_split(domain_split)
    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha:
        raise ValueError("B48 domain split does not pin source train.csv")
    if sha256_file(root / settings.get("train_csv", "train.csv")) != expected_train_sha:
        raise ValueError("B48 domain split source train.csv fingerprint mismatch")
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

    arm_specs = (
        (B48_STATIC_PRIOR_CONTROL, Path(control_checkpoint)),
        (B48_POST_CROSS_ATTENTION_CANDIDATE, Path(candidate_checkpoint)),
    )
    scores: dict[str, dict[str, dict]] = {}
    checkpoints: dict[str, dict] = {}
    control_pair_record: dict | None = None
    pair_seed: int | None = None
    for arm, checkpoint in arm_specs:
        model, payload = load_b48_checkpoint(
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
        if arm == B48_STATIC_PRIOR_CONTROL:
            control_pair_record = _pair_checkpoint_record(payload)
            seed = int(payload.get("seed", -1))
            if seed not in B48_REPLICATION_SEEDS:
                raise ValueError(f"B48 control seed must be one of {B48_REPLICATION_SEEDS}; got {seed}")
            # The default YAML seed is only the first replication.  Evaluation
            # provenance must follow the actual paired checkpoints.
            settings["seed"] = seed
        else:
            if control_pair_record is None:
                raise RuntimeError("B48 candidate loaded before static control")
            pair_seed = _matched_pair_seed(control_pair_record, _pair_checkpoint_record(payload))
        checkpoints[arm] = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "seed": int(payload.get("seed", -1)),
            "encoder_sha256_final": payload.get("encoder_sha256_final"),
            "model_context": payload.get("b48", {}),
        }
        arm_scores: dict[str, dict] = {}
        for split in ("validation_seen_scanners", "holdout_unseen_scanners"):
            indices = _indices_for_split(all_uids, domain_rows, split)
            uids = [all_uids[index] for index in indices]
            if set(uids).intersection(train_uids):
                raise RuntimeError(f"B48 {split} contains a training UID")
            if any(uid not in all_index for uid in uids):
                raise RuntimeError(f"B48 {split} has a study without eligible MRI series")
            arm_scores[split] = _score_split(
                model=model,
                config=settings,
                root=root,
                runtime=runtime,
                crop_policy=control_contract["crop_policy"],
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
        raise RuntimeError("B48 evaluation did not load a complete matched checkpoint pair")

    control = scores[B48_STATIC_PRIOR_CONTROL]
    candidate = scores[B48_POST_CROSS_ATTENTION_CANDIDATE]
    selected = [str(value) for value in domain_payload.get("summary", {}).get("comparable_targets", [])]
    if not selected:
        raise RuntimeError("B48 domain split defines no comparable weak-label targets")
    selected_known = [target for target in selected if target in TARGETS]
    if selected_known != selected:
        raise RuntimeError("B48 domain split comparable target list changed")

    comparisons = {}
    for offset, split in enumerate(("validation_seen_scanners", "holdout_unseen_scanners")):
        a, b = control[split], candidate[split]
        if a["uids"] != b["uids"]:
            raise RuntimeError(f"B48 {split} arms are not UID-aligned")
        if not np.array_equal(a["target"], b["target"]) or not np.array_equal(a["weight"], b["weight"]):
            raise RuntimeError(f"B48 {split} arms are not label/weight-aligned")
        paired = _paired_auc(
            a["target"],
            a["weight"],
            a["prediction"],
            b["prediction"],
            selected=selected,
            n_bootstrap=int(n_bootstrap),
            seed=int(pair_seed) + B48_EVAL_BOOTSTRAP_SEED_OFFSET + offset,
        )
        paired["control_macro_weighted_bce"] = _weighted_macro_bce(
            a["target"], a["prediction"], a["weight"], selected
        )
        paired["candidate_macro_weighted_bce"] = _weighted_macro_bce(
            a["target"], b["prediction"], a["weight"], selected
        )
        paired["candidate_minus_control_macro_weighted_bce"] = float(
            paired["candidate_macro_weighted_bce"] - paired["control_macro_weighted_bce"]
        )
        paired["leave_one_target_out_candidate_minus_control"] = _leave_one_target_out(
            paired["control_per_target_auc"], paired["candidate_per_target_auc"], selected
        )
        paired["targets_improved"] = int(
            sum(
                paired["candidate_per_target_auc"][target]
                > paired["control_per_target_auc"][target]
                for target in selected
            )
        )
        comparisons[split] = paired

    unseen = comparisons["holdout_unseen_scanners"]
    seen = comparisons["validation_seen_scanners"]
    control_domain_gap = float(seen["control_macro_auc"] - unseen["control_macro_auc"])
    candidate_domain_gap = float(seen["candidate_macro_auc"] - unseen["candidate_macro_auc"])
    full_12_target_surface = len(selected) == len(TARGETS)
    loto = unseen["leave_one_target_out_candidate_minus_control"]
    support = bool(
        full_12_target_surface
        and unseen["candidate_minus_control"] >= 0.010
        and unseen["ci_lower"] > 0.0
        and unseen["probability_candidate_better"] >= 0.95
        and unseen["targets_improved"] >= 7
        and min(loto.values()) > 0.0
        and seen["candidate_minus_control"] >= -0.005
        and candidate_domain_gap - control_domain_gap <= 0.005
    )
    if support:
        verdict = "support_for_study_dependent_global_conditioning"
    elif not full_12_target_surface:
        verdict = "inconclusive_insufficient_comparable_targets"
    elif unseen["candidate_minus_control"] < 0.005 or unseen["ci_lower"] <= 0.0:
        verdict = "no_support_for_global_conditioning"
    else:
        verdict = "inconclusive"

    output = {
        "evaluation_role": (
            "prospective matched weak-label scanner-domain comparison. The unseen-scanner "
            "surface is the primary B48 mechanism gate; it is not official-gold or hidden "
            "competition evidence."
        ),
        "experiment": B48_EXPERIMENT,
        "version": B48_VERSION,
        "domain_split": domain_meta,
        "comparable_targets": selected,
        "all_12_targets_comparable": full_12_target_surface,
        "checkpoints": checkpoints,
        "pair_seed": int(pair_seed),
        "matched_pair_identity": control_pair_record["matched_pair_identity"],
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "comparison": comparisons,
        "domain_gap": {
            "control_seen_minus_unseen": control_domain_gap,
            "candidate_seen_minus_unseen": candidate_domain_gap,
            "candidate_minus_control_gap": float(candidate_domain_gap - control_domain_gap),
        },
        "context_audit": {
            arm: {
                split: scores[arm][split]["context"]
                for split in ("validation_seen_scanners", "holdout_unseen_scanners")
            }
            for arm in (B48_STATIC_PRIOR_CONTROL, B48_POST_CROSS_ATTENTION_CANDIDATE)
        },
        "metadata_repair": metadata_stats,
        "predeclared_decision_rule": {
            "support": (
                "all 12 targets comparable; unseen delta>=+0.010; paired lower CI>0; "
                "P(candidate>control)>=0.95; >=7/12 targets improve; every leave-one-target-out "
                "delta>0; seen delta>=-0.005; and candidate domain-gap increase<=0.005"
            ),
            "no_support": "unseen delta<+0.005 OR paired lower CI<=0",
            "otherwise": "inconclusive",
        },
        "verdict": verdict,
        "governance": (
            "Do not retune B48 rank, context source, gates, sparse settings, seeds, scanner split, "
            "losses, optimizer, endpoints, targets, or blends from this result. Expert-58 is not a "
            "B48 selection surface. No hidden submission follows this two-arm weak-label comparison."
        ),
    }
    root_out = Path(out_root)
    root_out.mkdir(parents=True, exist_ok=True)
    for arm in (B48_STATIC_PRIOR_CONTROL, B48_POST_CROSS_ATTENTION_CANDIDATE):
        for split in ("validation_seen_scanners", "holdout_unseen_scanners"):
            _write_predictions(root_out / f"{arm}_{split}_predictions.csv", scores[arm][split])
    (root_out / "b48_domain_evaluation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)
    print("B48 MATCHED DOMAIN EVALUATION: PASS", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate the matched B48 scanner-domain arms")
    parser.add_argument("--config", default="config/b48_global_conditioned_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    evaluate_b48_domain_pair(
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


__all__ = ["evaluate_b48_domain_pair", "load_b48_checkpoint"]
