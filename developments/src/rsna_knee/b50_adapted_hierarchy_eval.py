"""Score the B50 matched pair on its fresh scanner-grouped gate.

B50 asks whether the study hierarchy, frozen since B34, should adapt to the
448-pixel features the encoder now produces. Unlike B48 and B49, whose
mechanisms lived inside a local branch admitted to the score through a gate
measured at |tanh(g)| ~ 0.022, B50 changes the base logits directly. So the
combined prediction -- the thing a submission would actually use -- is the
meaningful endpoint here, and the local path is reported beside it rather than
in place of it.

Three departures from the B48/B49 evaluators, each for a reason those
experiments taught:

**Every path is persisted, not just the combined one.** B49 computed its base
predictions and discarded them, which made it impossible afterwards to separate
what the base contributed from what the local branch did without re-running
inference. All three surfaces are written here.

**The discordant ceiling is computed before any verdict.** An ROC AUC moves only
on study pairs the two arms order differently, so that fraction bounds how far
their AUCs can differ. B48 and B49 were both filed as `no_support` against a
+0.010 threshold their measurements could not reach -- 0.0015 and 0.0024. When
the ceiling falls below the threshold the outcome recorded here is
`endpoint_underpowered`, which is a statement about the measurement;
`no_support` is reserved for a measurement that could have passed.

**The gate is reported for both arms.** During training the candidate's gate
settled at roughly half the control's, which is indirect evidence that the
adapted hierarchy produced better base logits and the model leaned on the local
correction less. Whether that survives to held-out ranking is exactly what this
evaluator decides.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import audit_variable_series_surface
from .b35_training import sha256_file
from .b37_highres_sparse_eval import B37_EVAL_OFFSETS
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    collate_b42,
)
from .b48_global_conditioned_sparse_eval import (
    _leave_one_target_out,
    _paired_auc,
    _weighted_macro_bce,
)
from .b48_global_conditioned_sparse_training import (
    _indices_for_split,
    _report_only_surface,
    _uid_sha256,
)
from .b50_adapted_hierarchy_mil import (
    B50_ARMS,
    B50_EXPERIMENT,
    B50_VERSION,
    B50AdaptedHierarchySparseMILResidual,
)
from .b50_adapted_hierarchy_training import (
    B50_CHECKPOINT_TEMPLATE,
    load_b50_selection_gate,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast, resolve_runtime

B50_EVAL_LOADER_SEED_OFFSET = 59_300_000
B50_EVAL_BOOTSTRAP_SEED_OFFSET = 59_400_000
B50_PRIMARY_SPLIT = "validation_unseen_scanners"
B50_COMPARATOR_SPLIT = "validation_seen_scanners"

# Predeclared in developments/docs/B50_REDESIGN_UNFREEZE_STUDY_HIERARCHY.md,
# before any B50 result was seen.
B50_SUPPORT_DELTA = 0.010
B50_SUPPORT_PROBABILITY = 0.95
B50_SUPPORT_MIN_TARGETS = 7
B50_SEEN_TOLERANCE = -0.005

B50_SURFACES = ("combined", "base", "local")


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_b50_checkpoint(path: str | Path, *, base_checkpoint: str | Path, device):
    """Rebuild one trained arm exactly as it was saved."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("experiment") != B50_EXPERIMENT:
        raise ValueError(f"{path} is not a B50 checkpoint")
    arm = str(payload.get("arm"))
    if arm not in B50_ARMS:
        raise ValueError(f"{path} records an unknown B50 arm: {arm!r}")

    base_model, _base_payload = load_phase9_checkpoint(
        Path(base_checkpoint).resolve(), expected_arm="llm_fill", device="cpu"
    )
    model = B50AdaptedHierarchySparseMILResidual(
        base_model,
        grid_size=int(payload["model_state"]["grid_size"]),
        top_k=int(payload["model_state"]["top_k"]),
        temperature=float(payload["model_state"]["temperature"]),
        encoder_trainable_stages=int(
            payload["model_state"].get("encoder_trainable_stages", 1)
        ),
        adapt_hierarchy=bool(payload.get("adapt_hierarchy")),
    )
    model.base.load_state_dict(payload["base_state"])
    model.head.load_state_dict(payload["head_state"])
    model.eval().to(device)
    return model, payload


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
    crop_policy: dict,
    label: str,
) -> dict:
    """Return combined, base and local probabilities for every study."""
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
        **runtime.loader_kwargs(
            seed=int(config["seed"]) + B50_EVAL_LOADER_SEED_OFFSET
        ),
    )

    combined_rows, base_rows, local_rows, scored = [], [], [], []
    with torch.inference_mode():
        for batch_index, items in enumerate(loader, start=1):
            if len(items) != 1:
                raise RuntimeError("B50 evaluation requires one ragged study per batch")
            item = items[0]
            scored.append(str(item["study_uid"]))

            present = item["present"].to(runtime.device, non_blocking=True)
            meta = item["series_meta"].to(runtime.device, non_blocking=True)
            position_all = item["slice_position"].to(runtime.device, non_blocking=True)
            if position_all.ndim != 3 or int(position_all.shape[1]) != len(
                B37_EVAL_OFFSETS
            ):
                raise RuntimeError("B50 TTA slice-position shape changed")

            combined_views, base_views, local_views = [], [], []
            for view in range(len(B37_EVAL_OFFSETS)):
                volumes = [
                    series_tensor[view].to(runtime.device, non_blocking=True)
                    for series_tensor in item["volumes"]
                ]
                with autocast(runtime):
                    out = model(volumes, present, meta, position_all[:, view])
                combined_views.append(torch.sigmoid(out.logits.float()))
                base_views.append(torch.sigmoid(out.base_logits.float()))
                local_views.append(torch.sigmoid(out.local_logits.float()))
                del volumes, out

            combined_rows.append(torch.stack(combined_views).mean(dim=0).cpu().numpy()[0])
            base_rows.append(torch.stack(base_views).mean(dim=0).cpu().numpy()[0])
            local_rows.append(torch.stack(local_views).mean(dim=0).cpu().numpy()[0])
            del item, items, present, meta, position_all
            _release()
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(f"[B50 {label}] {batch_index}/{len(loader)}", flush=True)

    if scored != uids:
        raise RuntimeError(f"B50 {label} UID order changed")

    surfaces = {
        "combined": np.asarray(combined_rows, dtype=np.float64),
        "base": np.asarray(base_rows, dtype=np.float64),
        "local": np.asarray(local_rows, dtype=np.float64),
    }
    for name, values in surfaces.items():
        if values.shape != targets.shape or not np.isfinite(values).all():
            raise RuntimeError(f"B50 {label} {name} prediction surface is invalid")
    return {"uids": uids, "target": targets, "weight": weights, **surfaces}


def discordant_pair_fraction(control: np.ndarray, candidate: np.ndarray) -> float:
    """The share of study pairs the two arms order differently, averaged.

    An ROC AUC is the share of positive/negative pairs a model orders correctly,
    so two arms' AUCs differ only on pairs they order differently. This bounds
    how far the measured delta could possibly have been -- the number B48 and
    B49 needed and did not have.
    """
    fractions = []
    for column in range(control.shape[1]):
        left, right = control[:, column], candidate[:, column]
        n = len(left)
        if n < 2:
            fractions.append(0.0)
            continue
        upper = np.triu_indices(n, k=1)
        ls = np.sign(left[:, None] - left[None, :])[upper]
        rs = np.sign(right[:, None] - right[None, :])[upper]
        fractions.append(float(((ls * rs) < 0).sum() / len(ls)))
    return float(np.mean(fractions))


def _surface_scores(
    control: dict,
    candidate: dict,
    *,
    surface: str,
    selected: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict:
    target, weight = control["target"], control["weight"]
    control_pred, candidate_pred = control[surface], candidate[surface]

    # `_paired_auc` already returns the point macro and per-target AUCs it
    # bootstraps around. Recomputing them separately would create two numbers
    # that are meant to be identical and could quietly stop being so.
    paired = _paired_auc(
        target,
        weight,
        control_pred,
        candidate_pred,
        selected=selected,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )
    control_macro = float(paired["control_macro_auc"])
    candidate_macro = float(paired["candidate_macro_auc"])
    control_per = dict(paired["control_per_target_auc"])
    candidate_per = dict(paired["candidate_per_target_auc"])

    ceiling = discordant_pair_fraction(control_pred, candidate_pred)
    improved = [
        name
        for name in selected
        if np.isfinite(control_per.get(name, np.nan))
        and np.isfinite(candidate_per.get(name, np.nan))
        and candidate_per[name] > control_per[name]
    ]
    return {
        "surface": surface,
        "control_macro_auc": control_macro,
        "candidate_macro_auc": candidate_macro,
        "delta": float(candidate_macro - control_macro),
        "control_per_target": {k: float(v) for k, v in control_per.items()},
        "candidate_per_target": {k: float(v) for k, v in candidate_per.items()},
        "targets_improved": improved,
        "targets_improved_count": len(improved),
        "paired_bootstrap": paired,
        "discordant_pair_fraction": ceiling,
        "max_possible_abs_delta": ceiling,
        "endpoint_could_reach_threshold": bool(ceiling >= B50_SUPPORT_DELTA),
        "control_weighted_macro_bce": _weighted_macro_bce(
            target, control_pred, weight, selected
        ),
        "candidate_weighted_macro_bce": _weighted_macro_bce(
            target, candidate_pred, weight, selected
        ),
    }


def decide(primary: dict, seen: dict) -> dict:
    """Apply the rule frozen before any B50 result was inspected."""
    ceiling = float(primary["discordant_pair_fraction"])
    if ceiling < B50_SUPPORT_DELTA:
        return {
            "outcome": "endpoint_underpowered",
            "reason": (
                f"the two arms order only {ceiling:.6f} of study pairs differently, "
                f"so no measurement here could have reached the frozen "
                f"{B50_SUPPORT_DELTA:+.3f} threshold. This is a statement about the "
                f"endpoint, not about the adapted hierarchy."
            ),
        }

    delta = float(primary["delta"])
    paired = primary["paired_bootstrap"]
    lower = float(paired.get("ci_lower", float("nan")))
    probability = float(paired.get("probability_candidate_better", float("nan")))
    loto = primary.get("leave_one_target_out_candidate_minus_control", {})
    checks = {
        "delta_at_least_threshold": delta >= B50_SUPPORT_DELTA,
        "ci_lower_above_zero": lower > 0.0,
        "probability_at_least": probability >= B50_SUPPORT_PROBABILITY,
        "targets_improved_at_least": (
            primary["targets_improved_count"] >= B50_SUPPORT_MIN_TARGETS
        ),
        "every_leave_one_target_out_positive": bool(loto)
        and all(float(value) > 0.0 for value in loto.values()),
        "seen_scanner_delta_within_tolerance": (
            float(seen["delta"]) >= B50_SEEN_TOLERANCE
        ),
    }
    if all(checks.values()):
        outcome = "supported"
    elif delta < 0.005 or not np.isfinite(lower) or lower <= 0.0:
        outcome = "no_support"
    else:
        outcome = "inconclusive"
    return {"outcome": outcome, "checks": checks}


def evaluate_b50_pair(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    base_checkpoint: str | Path,
    selection_gate: str | Path,
    run_root: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    root = Path(settings["data_root"])
    settings.setdefault("seed", 2026)

    from .b50_adapted_hierarchy_mil import require_b50_contract

    crop_policy = require_b50_contract(
        {**settings, "b50_arm": "adapted_hierarchy_candidate"}
    )["crop_policy"]

    gate_payload, gate_rows, gate_meta = load_b50_selection_gate(selection_gate)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)

    base_path = Path(base_checkpoint).resolve()
    _base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
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
        domain_rows=gate_rows,
        base_payload=base_payload,
    )

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, _stats = backfill_series_metadata(series, root, split="train")

    checkpoints = {
        arm: Path(run_root) / arm / B50_CHECKPOINT_TEMPLATE.format(arm=arm)
        for arm in B50_ARMS
    }
    for arm, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"B50 {arm} checkpoint is missing: {path}")

    scores: dict[str, dict[str, dict]] = {arm: {} for arm in B50_ARMS}
    arm_payloads: dict[str, dict] = {}
    for arm, path in checkpoints.items():
        model, payload = load_b50_checkpoint(
            path, base_checkpoint=base_path, device=runtime.device
        )
        arm_payloads[arm] = payload
        for split in (B50_PRIMARY_SPLIT, B50_COMPARATOR_SPLIT):
            indices = _indices_for_split(all_uids, gate_rows, split)
            uids = [all_uids[i] for i in indices]
            _summary, series_index = audit_variable_series_surface(series, uids)
            scores[arm][split] = _score_split(
                model=model,
                config=settings,
                root=root,
                runtime=runtime,
                uids=uids,
                targets=all_targets[indices],
                weights=all_weights[indices],
                series_index=series_index,
                crop_policy=crop_policy,
                label=f"{arm}/{split}",
            )
        del model
        _release()

    control_arm, candidate_arm = B50_ARMS
    selected = list(TARGETS)
    results: dict[str, dict] = {}
    for split in (B50_PRIMARY_SPLIT, B50_COMPARATOR_SPLIT):
        results[split] = {}
        for surface in B50_SURFACES:
            entry = _surface_scores(
                scores[control_arm][split],
                scores[candidate_arm][split],
                surface=surface,
                selected=selected,
                n_bootstrap=n_bootstrap,
                seed=int(settings["seed"]) + B50_EVAL_BOOTSTRAP_SEED_OFFSET,
            )
            entry["leave_one_target_out_candidate_minus_control"] = _leave_one_target_out(
                entry["control_per_target"], entry["candidate_per_target"], selected
            )
            results[split][surface] = entry

    primary = results[B50_PRIMARY_SPLIT]["combined"]
    seen = results[B50_COMPARATOR_SPLIT]["combined"]
    verdict = decide(primary, seen)

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    for arm in B50_ARMS:
        for split, score in scores[arm].items():
            for surface in B50_SURFACES:
                frame = pd.DataFrame(score[surface], columns=TARGETS)
                frame.insert(0, "StudyInstanceUID", score["uids"])
                frame.to_csv(out / f"{arm}_{split}_{surface}_predictions.csv", index=False)

    payload = {
        "experiment": B50_EXPERIMENT,
        "version": B50_VERSION,
        "primary_split": B50_PRIMARY_SPLIT,
        "primary_surface": "combined",
        "primary_surface_rationale": (
            "B50 changes the base logits directly, unlike B48 and B49 whose "
            "mechanisms sat behind a near-closed gate, so the combined prediction "
            "is what the experiment is about. Base and local are reported beside it."
        ),
        "selection_gate": gate_meta,
        "selection_gate_summary": gate_payload.get("summary", {}),
        "base_checkpoint_sha256": sha256_file(base_path),
        "checkpoints": {
            arm: {
                "path": str(path),
                "sha256": sha256_file(path),
                "adapt_hierarchy": bool(arm_payloads[arm].get("adapt_hierarchy")),
                "trainable": arm_payloads[arm].get("trainable"),
                "gate": arm_payloads[arm]["history"][-1].get("sparse_mil", {}),
                "final_loss_combined": arm_payloads[arm]["history"][-1].get(
                    "loss_combined"
                ),
            }
            for arm, path in checkpoints.items()
        },
        "uids_sha256": {
            split: _uid_sha256(scores[control_arm][split]["uids"])
            for split in (B50_PRIMARY_SPLIT, B50_COMPARATOR_SPLIT)
        },
        "results": results,
        "decision_rule": {
            "delta_threshold": B50_SUPPORT_DELTA,
            "probability_threshold": B50_SUPPORT_PROBABILITY,
            "minimum_targets_improved": B50_SUPPORT_MIN_TARGETS,
            "seen_scanner_tolerance": B50_SEEN_TOLERANCE,
            "frozen_in": "developments/docs/B50_REDESIGN_UNFREEZE_STUDY_HIERARCHY.md",
        },
        "verdict": verdict,
        "governance": (
            "B50 is a prospective matched comparison of a frozen against an adapted "
            "B34 study hierarchy on a fresh scanner-grouped gate. Do not tune the "
            "hierarchy learning rate, epoch count, seed, geometry, target subset or "
            "endpoint from this result, and do not submit from it without a "
            "separately declared protocol."
        ),
    }
    (out / "b50_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"[B50] primary surface: combined on {B50_PRIMARY_SPLIT}")
    print(
        f"[B50] control {primary['control_macro_auc']:.6f}   "
        f"candidate {primary['candidate_macro_auc']:.6f}   "
        f"delta {primary['delta']:+.6f}"
    )
    print(
        f"[B50] discordant pairs {primary['discordant_pair_fraction']:.6f} "
        f"-> max possible |delta| {primary['max_possible_abs_delta']:.6f} "
        f"against a {B50_SUPPORT_DELTA:+.3f} threshold"
    )
    print(f"[B50] targets improved {primary['targets_improved_count']}/12")
    for surface in B50_SURFACES:
        entry = results[B50_PRIMARY_SPLIT][surface]
        print(
            f"[B50]   {surface:<9} control {entry['control_macro_auc']:.6f} "
            f"candidate {entry['candidate_macro_auc']:.6f} "
            f"delta {entry['delta']:+.6f}"
        )
    print(f"[B50] VERDICT: {verdict['outcome']}")
    if "reason" in verdict:
        print(f"[B50] {verdict['reason']}")
    print(out / "b50_evaluation.json", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate the B50 matched pair")
    parser.add_argument("--config", default="config/b50_adapted_hierarchy.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--selection-gate", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    evaluate_b50_pair(
        dict(_read_config(args.config)),
        data_root=args.data_root,
        labels_root=args.labels_root,
        base_checkpoint=args.base_checkpoint,
        selection_gate=args.selection_gate,
        run_root=args.run_root,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B50_PRIMARY_SPLIT",
    "B50_SUPPORT_DELTA",
    "B50_SURFACES",
    "decide",
    "discordant_pair_fraction",
    "evaluate_b50_pair",
    "load_b50_checkpoint",
]
