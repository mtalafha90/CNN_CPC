"""Reused Expert-58 diagnostic for B41 native-aspect sparse MIL.

This script evaluates the one fixed B41 endpoint against the historical 224
base.  It does not alter B41 after seeing Expert-58: the diagnostic is retained
for mechanism review only, while hidden competition evidence decides promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .b7_weak_supervision import _read_config
from .b12_variable_series import build_variable_series_index
from .b12_1_gold_eval import predict_b12_1
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b35_training import sha256_file
from .b37_highres_sparse_eval import (
    B37_BASE_TOLERANCE,
    B37_EVAL_OFFSETS,
    B37_EXPECTED_BASE_MACRO,
    FOCAL_SIX,
    _base_loader,
    _candidate_loader,
    _release,
    load_b37_checkpoint,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import compare_runs, macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import autocast
from .b41_highres_aspect_sparse_mil import (
    B41_EXPERIMENT,
    B41_EXPERT58_ROOT,
    B41_VERSION,
    B41HighResAspectSparseDataset,
    b41_preprocessing_state,
    require_b41_aspect_contract,
)

# Use a distinct fixed bootstrap stream for B41's recorded diagnostic only.
B41_EVAL_BOOTSTRAP_SEED_OFFSET = 51_400_000
# Use a distinct fixed loader stream while retaining B37's three symmetric offsets.
B41_EVAL_LOADER_SEED_OFFSET = 51_300_000


def _per_target(truth, base, global448, combined):
    """Compute macro and target-level AUCs with B41-specific output labels."""
    # Calculate the historical baseline AUC once across all twelve targets.
    base_macro, base_auc = macro_auc_from_arrays(truth, base)
    # Calculate the B41 global branch AUC before the learned sparse residual.
    global_macro, global_auc = macro_auc_from_arrays(truth, global448)
    # Calculate B41's complete sparse-MIL prediction AUC.
    combined_macro, combined_auc = macro_auc_from_arrays(truth, combined)
    # Preserve each target's comparison values in an explicit auditable mapping.
    rows = {
        target: {
            "base_224_auc": float(a),
            "b41_global_448_auc": float(b),
            "b41_combined_auc": float(c),
            "global_minus_base": float(b - a),
            "combined_minus_base": float(c - a),
            "sparse_residual_increment": float(c - b),
        }
        for target, a, b, c in zip(TARGETS, base_auc, global_auc, combined_auc)
    }
    # Return macro values alongside the full target mapping.
    return float(base_macro), float(global_macro), float(combined_macro), rows


@torch.no_grad()
def evaluate_b41(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B41_EXPERT58_ROOT,
    n_bootstrap: int = 5000,
) -> dict:
    """Compare the fixed B41 aspect-preserving endpoint against historical B34."""
    # Make a private configuration copy before adding the resolved local data path.
    settings = dict(config)
    # Use absolute paths in saved artifacts so the evaluated inputs remain traceable.
    settings["data_root"] = str(Path(data_root).resolve())
    # Reject any model or resize control that is not the prospective B41 contract.
    require_b41_aspect_contract(settings)
    # Resolve the local competition-data root once for every loader.
    root = Path(settings["data_root"])

    # Read the study table containing the fixed Expert-58 truth values.
    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    # Select only complete expert-labelled studies in their native CSV order.
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    # Normalize study identifiers to strings for exact loader-order checks.
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    # Refuse a changed, incomplete, or label-missing development surface.
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B41 requires the complete reused 58-study expert surface")
    # Store the exact scoring order and binary truth matrix.
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    # Read all series metadata and repair only the repository's supported fields.
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    # Build the variable-series MRI index for the fixed expert study order.
    index = build_variable_series_index(series, uids)
    # Confirm the historical 336-series Expert-58 surface before scoring it.
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B41 expert MRI series surface changed")

    # Build the untouched 224 historical baseline loader and replay its base checkpoint.
    base_loader, base_runtime = _base_loader(settings, root, uids, index, truth)
    print(base_runtime.describe(), flush=True)
    base_model, _ = load_phase9_checkpoint(
        Path(base_checkpoint).resolve(),
        expected_arm="llm_fill",
        device=base_runtime.device,
    )
    # Make the historical base model deterministic for inference.
    base_model.eval()
    # Score the baseline before constructing B41's high-resolution dataset.
    base_uids, base_prediction = predict_b12_1(base_model, base_loader, base_runtime)
    if [str(uid) for uid in base_uids] != uids:
        raise RuntimeError("B41 historical base study order changed")
    # Convert to a durable CPU float array for comparison and output.
    base_prediction = np.asarray(base_prediction, dtype=np.float32)
    # Free the baseline graph, model, and caches before loading 448 inputs.
    del base_model, base_loader
    _release()

    # Build B41's three-offset 448 loader with aspect-preserving preprocessing.
    candidate_loader, runtime = _candidate_loader(
        settings,
        root,
        uids,
        index,
        truth,
        dataset_class=B41HighResAspectSparseDataset,
        contract_validator=require_b41_aspect_contract,
        loader_seed_offset=B41_EVAL_LOADER_SEED_OFFSET,
    )
    print(runtime.describe(), flush=True)
    # Load the immutable B41 checkpoint through the shared sparse-MIL architecture loader.
    model, payload = load_b37_checkpoint(
        checkpoint,
        base_checkpoint=base_checkpoint,
        device=runtime.device,
        expected_version=B41_VERSION,
        expected_experiment=B41_EXPERIMENT,
        checkpoint_label="B41",
    )

    # Collect TTA-averaged global and sparse-MIL predictions in CSV order.
    global_blocks: list[np.ndarray] = []
    combined_blocks: list[np.ndarray] = []
    scored_uids: list[str] = []
    # Track sparse evidence statistics without using them for model selection.
    top1_sum = np.zeros(len(TARGETS), dtype=np.float64)
    topk_sum = np.zeros(len(TARGETS), dtype=np.float64)
    gap_sum = np.zeros(len(TARGETS), dtype=np.float64)
    selected_count = 0
    unique_sum = 0
    unique_n = 0

    # Evaluate one variable-size study batch at a time.
    for batch in candidate_loader:
        # Move the complete B41 image, presence, metadata, and positions to the device.
        volumes = batch["volumes"].to(runtime.device, non_blocking=True)
        position = batch["slice_position"].to(runtime.device, non_blocking=True)
        present = batch["present"].to(runtime.device, non_blocking=True)
        meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        # Ensure the dataset retained [batch, TTA-view, series, triplet, channel, H, W].
        if volumes.ndim != 7 or int(volumes.shape[1]) != len(B37_EVAL_OFFSETS):
            raise RuntimeError("B41 evaluation TTA view shape changed")
        # Create one probability tensor per symmetric centre offset.
        global_views: list[torch.Tensor] = []
        combined_views: list[torch.Tensor] = []
        for view in range(volumes.shape[1]):
            # Run the B41 model without gradient storage and with the resolved precision policy.
            with autocast(runtime):
                output = model(
                    volumes[:, view],
                    present,
                    meta,
                    position[:, view],
                )
            # Convert logits to raw probabilities before averaging the predeclared TTA views.
            global_views.append(torch.sigmoid(output.base_logits.float()))
            combined_views.append(torch.sigmoid(output.logits.float()))
            # Inspect selected evidence only for interpretation, never tuning.
            values = output.top_values.float()
            top1_sum += values[:, :, 0].sum(dim=0).cpu().numpy()
            topk_sum += values.mean(dim=-1).sum(dim=0).cpu().numpy()
            gap_sum += (values[:, :, 0] - values[:, :, -1]).sum(dim=0).cpu().numpy()
            selected_count += int(values.shape[0])
            # Count the diversity of all target-specific selected spatial locations.
            indices = output.top_indices.detach().cpu()
            for batch_index in range(indices.shape[0]):
                unique_sum += int(torch.unique(indices[batch_index]).numel())
                unique_n += 1
        # Average raw probabilities over the frozen offsets and move them to CPU.
        global_blocks.append(torch.stack(global_views).mean(dim=0).cpu().numpy())
        combined_blocks.append(torch.stack(combined_views).mean(dim=0).cpu().numpy())
        # Record the exact scoring order before releasing each batch.
        scored_uids.extend(str(uid) for uid in batch["study_uid"])
        # Release high-resolution tensors before materializing the next DICOM study.
        del batch, volumes, position, present, meta, output, global_views, combined_views
        _release()

    # Refuse an output whose order does not exactly match the fixed truth order.
    if scored_uids != uids:
        raise RuntimeError("B41 candidate study order changed")
    # Join all one-study probability blocks into complete Expert-58 matrices.
    global_prediction = np.concatenate(global_blocks, axis=0)
    combined_prediction = np.concatenate(combined_blocks, axis=0)
    # Refuse unusable probability values rather than writing a misleading metric.
    if not np.isfinite(global_prediction).all() or not np.isfinite(combined_prediction).all():
        raise RuntimeError("B41 produced non-finite predictions")

    # Calculate macro and target-level comparisons after all scoring has finished.
    base_macro, global_macro, combined_macro, per_target = _per_target(
        truth,
        base_prediction,
        global_prediction,
        combined_prediction,
    )
    # Confirm the historical base replay has not changed before interpreting B41.
    if abs(base_macro - B37_EXPECTED_BASE_MACRO) > B37_BASE_TOLERANCE:
        raise RuntimeError(
            "historical base replay changed: "
            f"expected ~{B37_EXPECTED_BASE_MACRO:.10f}, got {base_macro:.10f}"
        )
    # Calculate the predeclared six-target descriptive subset.
    focal_base = float(np.mean([per_target[target]["base_224_auc"] for target in FOCAL_SIX]))
    focal_global = float(
        np.mean([per_target[target]["b41_global_448_auc"] for target in FOCAL_SIX])
    )
    focal_combined = float(
        np.mean([per_target[target]["b41_combined_auc"] for target in FOCAL_SIX])
    )
    # Compute descriptive deltas without using them to change the fixed B41 endpoint.
    macro_delta = float(combined_macro - base_macro)
    focal_delta = float(focal_combined - focal_base)
    # Bootstrap paired study differences with B41's fixed diagnostic random seed.
    paired = compare_runs(
        truth,
        base_prediction,
        combined_prediction,
        n_bootstrap=int(n_bootstrap),
        seed=int(settings.get("seed", 2026)) + B41_EVAL_BOOTSTRAP_SEED_OFFSET,
    )
    # Record sparse-MIL selection behavior for scientific interpretation only.
    mil = {
        "top_k": int(model.head.top_k),
        "grid_size": int(model.head.grid_size),
        "regions_per_slice": int(model.head.n_regions),
        "mean_unique_selected_locations_across_12_targets": float(
            unique_sum / max(unique_n, 1)
        ),
        "max_possible_selected_locations_across_12_targets": int(
            len(TARGETS) * model.head.top_k
        ),
        "per_target": {
            target: {
                "mean_top1_evidence": float(top1_sum[index] / max(selected_count, 1)),
                "mean_topk_evidence": float(topk_sum[index] / max(selected_count, 1)),
                "mean_top1_minus_kth": float(gap_sum[index] / max(selected_count, 1)),
            }
            for index, target in enumerate(TARGETS)
        },
    }

    # Build the audit record before writing any evaluation artifact.
    result = {
        "evaluation_role": (
            "reused Expert-58 development diagnostic for the fixed B41 endpoint; "
            "not independent test evidence and not a tuning or promotion criterion"
        ),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint).resolve()),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "n_studies": len(uids),
        "n_series": int(sum(counts)),
        "tta_offsets": list(B37_EVAL_OFFSETS),
        "base_224_macro_auc": base_macro,
        "b41_global_448_macro_auc": global_macro,
        "b41_combined_macro_auc": combined_macro,
        "global_448_minus_base_224": float(global_macro - base_macro),
        "sparse_residual_macro_increment": float(combined_macro - global_macro),
        "macro_delta_primary": macro_delta,
        "focal_six": list(FOCAL_SIX),
        "focal_six_base_224": focal_base,
        "focal_six_global_448": focal_global,
        "focal_six_combined": focal_combined,
        "focal_six_delta_primary": focal_delta,
        "per_target": per_target,
        "mil": mil,
        "head": model.head.state(),
        "paired_macro_auc_bootstrap": paired,
        "preprocessing": b41_preprocessing_state(),
        "training_contract": {
            "completed_epochs": int(payload.get("completed_epochs", -1)),
            "training_studies": int(payload.get("training_studies", -1)),
            "training_series": int(payload.get("training_series", -1)),
            "training_supervision_cells": int(
                payload.get("training_supervision_cells", -1)
            ),
            "encoder_sha256_initial": payload.get("encoder_sha256_initial"),
            "encoder_sha256_final": payload.get("encoder_sha256_final"),
        },
        "metadata_repair": metadata_stats,
        "governance": (
            "B41 is fixed at full-native normalization, a native 90% centre crop, "
            "one aspect-preserving antialiased resize-to-fit, zero padding to 448, "
            "32 centres, 6x6, top-k=8, B37 supervision, and two epochs. Do not use "
            "this reused Expert-58 result to tune it; hidden competition evidence is "
            "required for promotion."
        ),
    }

    # Create the permanent B41 Expert-58 output folder only after successful scoring.
    output_root = Path(out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Save all probability tables in the fixed study order for later plots or audits.
    for name, prediction in (
        ("base_224", base_prediction),
        ("b41_global_448", global_prediction),
        ("b41_combined", combined_prediction),
    ):
        frame = pd.DataFrame(prediction, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", uids)
        frame.to_csv(output_root / f"{name}_predictions.csv", index=False)
    # Save the machine-readable diagnostic record beside the probability tables.
    (output_root / "expert58.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    # Print the complete result for terminal logs and return it to library callers.
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    """Parse the standalone local B41 Expert-58 command-line interface."""
    # Describe the endpoint as B41 rather than the shared B37 architecture loader.
    parser = argparse.ArgumentParser(
        "Evaluate B41 native-aspect-preserving high-resolution sparse MIL"
    )
    # Use the frozen B41 config by default.
    parser.add_argument("--config", default="config/b41_highres_aspect_sparse_448.yaml")
    # Require the local competition train CSV and DICOM root.
    parser.add_argument("--data-root", required=True)
    # Require B41's completed two-epoch checkpoint.
    parser.add_argument("--checkpoint", required=True)
    # Require the immutable B67 full-fill base checkpoint for historical replay.
    parser.add_argument("--base-checkpoint", required=True)
    # Keep reports below B41's permanent run root by default.
    parser.add_argument("--out-root", default=B41_EXPERT58_ROOT)
    # Keep the paired bootstrap count explicit and reproducible.
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    # Parse arguments once.
    args = parser.parse_args()
    # Load YAML through the repository's established config reader.
    config = dict(_read_config(args.config))
    # Evaluate the fixed checkpoint and write its audit artifacts.
    evaluate_b41(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
