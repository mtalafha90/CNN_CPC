#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.b7_weak_supervision import (
    _read_config,
    load_frozen_b6_export,
    make_b7_dataset_config,
)
from rsna_knee.b12_variable_series import (
    build_variable_series_index,
    collate_variable_series,
)
from rsna_knee.b12_1_gold_eval import predict_b12_1
from rsna_knee.b12_1_hierarchical import build_b12_1_model
from rsna_knee.b15_ssl import (
    WEAK_V2_MANIFEST_SHA256,
    load_frozen_v2_manifest,
)
from rsna_knee.b15_weak_eval import _holdout_supervision
from rsna_knee.b21_dataset import make_matched_crop_dataset
from rsna_knee.constants import TARGETS
from rsna_knee.data import (
    backfill_series_metadata,
    load_series_csv,
    load_train_csv,
)
from rsna_knee.runtime import resolve_runtime
from rsna_knee.weak_validation import (
    compare_on_weak_surface,
    evaluate_on_weak_surface,
)


def load_checkpoint(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)

    parser.add_argument(
        "--density-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--control-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--b23-checkpoint",
        required=True,
    )

    parser.add_argument("--b6-root", required=True)
    parser.add_argument(
        "--weak-holdout-root",
        required=True,
    )

    parser.add_argument(
        "--existing-predictions",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        default="runs/b24x_density/weak_v2_eval",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    config = _read_config(args.config)
    config["data_root"] = args.data_root

    # --------------------------------------------------------
    # Checkpoint integrity
    # --------------------------------------------------------
    d = load_checkpoint(args.density_checkpoint)
    c = load_checkpoint(args.control_checkpoint)
    q = load_checkpoint(args.b23_checkpoint)

    assert d["experiment"] == "B24X_density_ablation_v1"
    assert d["mode"] == "density_exploratory"
    assert d["completed_epochs"] == 2
    assert d["fixed_endpoint"] is True
    assert d["exploratory"] is True
    assert d["formal_b24_eligible"] is False
    assert d["gold_acceptance_allowed"] is False

    ab = d["density_ablation"]

    assert ab["b6_cells_preserved"] == 3045
    assert ab["b23_only_cells_added"] == 2844
    assert ab["b6_cells_dropped"] == 0
    assert ab["b6_cells_overridden"] == 0
    assert ab["final_usable_cells"] == 5889

    # Same 692 knees and same frozen encoder.
    assert d["study_uids"] == c["study_uids"]
    assert d["study_uids"] == q["study_uids"]

    assert (
        d["encoder_sha256_initial"]
        == c["encoder_sha256_initial"]
        == q["encoder_sha256_initial"]
    )

    assert (
        float(d["crop_fraction"])
        == float(c["crop_fraction"])
        == float(q["crop_fraction"])
    )

    print("=" * 72)
    print("B24X-DENSITY CHECKPOINT VERIFICATION")
    print("=" * 72)
    print("training studies      :", len(d["study_uids"]))
    print("B6 cells preserved    :", ab["b6_cells_preserved"])
    print("B23-only cells added  :", ab["b23_only_cells_added"])
    print("final usable cells    :", ab["final_usable_cells"])
    print("B6 overrides          :", ab["b6_cells_overridden"])
    print("same encoder          : True")
    print("gold allowed          :", d["gold_acceptance_allowed"])

    # --------------------------------------------------------
    # Frozen weak-v2 surface
    # --------------------------------------------------------
    weak_payload, manifest = load_frozen_v2_manifest(
        args.weak_holdout_root
    )

    root = Path(config["data_root"])

    train = load_train_csv(
        root / config.get("train_csv", "train.csv")
    )

    b6_frame, _, _ = load_frozen_b6_export(
        args.b6_root
    )

    (
        uids,
        weak_targets,
        weak_weights,
        _,
    ) = _holdout_supervision(
        train,
        b6_frame,
        manifest,
    )

    assert len(uids) == 623

    train_set = set(map(str, d["study_uids"]))
    holdout_set = set(map(str, uids))

    overlap = train_set & holdout_set

    print()
    print("Frozen weak-v2")
    print("  studies :", len(uids))
    print("  overlap :", len(overlap))

    if overlap:
        raise RuntimeError(
            "weak-v2 leakage detected"
        )

    # --------------------------------------------------------
    # Load PREVIOUSLY SAVED B6/B23 predictions.
    # Do not recompute them.
    # --------------------------------------------------------
    previous = pd.read_csv(
        args.existing_predictions
    )

    previous["StudyInstanceUID"] = (
        previous["StudyInstanceUID"].astype(str)
    )

    expected_uids = [str(x) for x in uids]

    if (
        previous["StudyInstanceUID"].tolist()
        != expected_uids
    ):
        raise RuntimeError(
            "saved B24X predictions do not match "
            "the frozen weak-v2 order"
        )

    pred_b6 = np.column_stack([
        previous[f"{target}__control"].to_numpy(
            dtype=float
        )
        for target in TARGETS
    ])

    pred_b23 = np.column_stack([
        previous[f"{target}__candidate"].to_numpy(
            dtype=float
        )
        for target in TARGETS
    ])

    # --------------------------------------------------------
    # Build Density model
    # --------------------------------------------------------
    runtime = resolve_runtime(config)
    print()
    print(runtime.describe())

    model = build_b12_1_model(
        d["model_spec"],
        pretrained_weights=False,
    )

    model.load_state_dict(
        d["model_state"],
        strict=True,
    )

    model = model.to(runtime.device).eval()

    # --------------------------------------------------------
    # MRI series
    # --------------------------------------------------------
    series = load_series_csv(
        root
        / config.get(
            "train_series_csv",
            "train_series.csv",
        )
    )

    series, metadata_stats = backfill_series_metadata(
        series,
        root,
        split="train",
    )

    variable_index = build_variable_series_index(
        series,
        uids,
    )

    counts = [
        len(variable_index[str(uid)])
        for uid in uids
    ]

    if any(x == 0 for x in counts):
        raise RuntimeError(
            "weak-v2 study has zero eligible series"
        )

    offsets = tuple(
        int(x)
        for x in config.get(
            "b7_eval_tta_offsets",
            [-1, 0, 1],
        )
    )

    if offsets != (-1, 0, 1):
        raise RuntimeError(
            "TTA must remain [-1,0,1]"
        )

    dataset_config = make_b7_dataset_config(
        config,
        root,
        train=False,
        tta_offsets=offsets,
    )

    ds = make_matched_crop_dataset(
        "control",
        uids,
        variable_index,
        dataset_config,
        crop_fraction=float(d["crop_fraction"]),
        train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=int(
            config.get("b7_eval_batch_size", 2)
        ),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(
            seed=int(config.get("seed", 2026))
            + 24_600_000
        ),
    )

    print()
    print(
        "[B24X-Density] predicting density model "
        "on frozen weak-v2..."
    )

    pred_uids, pred_density = predict_b12_1(
        model,
        loader,
        runtime,
    )

    if [str(x) for x in pred_uids] != expected_uids:
        raise RuntimeError(
            "Density prediction order changed"
        )

    # --------------------------------------------------------
    # Individual Density score
    # --------------------------------------------------------
    n_bootstrap = int(args.n_bootstrap)
    seed = int(config.get("seed", 2026))

    density_eval = evaluate_on_weak_surface(
        weak_targets,
        pred_density,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 251,
    )

    # --------------------------------------------------------
    # Three pairwise comparisons
    # --------------------------------------------------------

    # Density minus B6
    b6_vs_density = compare_on_weak_surface(
        weak_targets,
        pred_b6,
        pred_density,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 252,
    )

    # Full B23 minus Density
    density_vs_b23 = compare_on_weak_surface(
        weak_targets,
        pred_density,
        pred_b23,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 253,
    )

    # Reproduce original Full-B23 minus B6 comparison
    b6_vs_b23 = compare_on_weak_surface(
        weak_targets,
        pred_b6,
        pred_b23,
        weak_weights,
        n_bootstrap=n_bootstrap,
        seed=seed + 243,
    )

    b6_auc = float(
        b6_vs_density["macro_auc_a"]
    )

    density_auc = float(
        b6_vs_density["macro_auc_b"]
    )

    b23_auc = float(
        b6_vs_b23["macro_auc_b"]
    )

    total_gain = b23_auc - b6_auc
    density_gain = density_auc - b6_auc

    fraction = (
        density_gain / total_gain
        if abs(total_gain) > 1e-12
        else float("nan")
    )

    # --------------------------------------------------------
    # Per-target decomposition
    # --------------------------------------------------------
    b6_eval = evaluate_on_weak_surface(
        weak_targets,
        pred_b6,
        weak_weights,
        n_bootstrap=100,
        seed=seed + 254,
    )

    b23_eval = evaluate_on_weak_surface(
        weak_targets,
        pred_b23,
        weak_weights,
        n_bootstrap=100,
        seed=seed + 255,
    )

    per_target = {}

    for target in TARGETS:
        a = float(
            b6_eval["per_target_auc"][target]
        )
        m = float(
            density_eval["per_target_auc"][target]
        )
        b = float(
            b23_eval["per_target_auc"][target]
        )

        per_target[target] = {
            "b6": a,
            "density": m,
            "full_b23": b,
            "density_minus_b6": m - a,
            "full_b23_minus_density": b - m,
            "full_b23_minus_b6": b - a,
        }

    result = {
        "experiment": "B24X_density_ablation_v1",
        "surface": "frozen_weak_b6_holdout_v2",
        "gold_used": False,
        "promotion_allowed": False,

        "b6_macro_auc": b6_auc,
        "density_macro_auc": density_auc,
        "full_b23_macro_auc": b23_auc,

        "density_gain_over_b6":
            density_gain,

        "full_b23_gain_over_b6":
            total_gain,

        "fraction_of_full_b23_gain_captured_by_density":
            fraction,

        "b6_vs_density":
            b6_vs_density,

        "density_vs_full_b23":
            density_vs_b23,

        "b6_vs_full_b23":
            b6_vs_b23,

        "density_evaluation":
            density_eval,

        "per_target":
            per_target,

        "n_holdout_studies":
            len(uids),

        "holdout_usable_cells":
            int((weak_weights > 0).sum()),

        "weak_holdout_manifest_sha256":
            WEAK_V2_MANIFEST_SHA256,

        "weak_holdout_metadata":
            weak_payload,

        "metadata_repair":
            metadata_stats,

        "interpretation": (
            "Exploratory density ablation. "
            "Density preserves all B6 committed labels "
            "and adds B23 only on B6-silent cells. "
            "Weak-v2 measures agreement with B6, not expert truth. "
            "No gold evaluation or promotion."
        ),
    }

    out = Path(args.out_root)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save all three predictions.
    frame = pd.DataFrame(
        {"StudyInstanceUID": expected_uids}
    )

    for j, target in enumerate(TARGETS):
        frame[f"{target}__b6"] = pred_b6[:, j]
        frame[f"{target}__density"] = pred_density[:, j]
        frame[f"{target}__b23"] = pred_b23[:, j]

    frame.to_csv(
        out / "three_arm_predictions.csv",
        index=False,
    )

    (
        out / "density_comparison.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("B24X-DENSITY FROZEN WEAK-V2 RESULT")
    print("=" * 72)

    print(f"B6 control : {b6_auc:.10f}")
    print(f"Density    : {density_auc:.10f}")
    print(f"Full B23   : {b23_auc:.10f}")

    print()
    print(
        f"Density - B6      : "
        f"{density_gain:+.10f}"
    )

    print(
        f"Full B23 - B6     : "
        f"{total_gain:+.10f}"
    )

    print(
        f"Full B23 - Density: "
        f"{b23_auc-density_auc:+.10f}"
    )

    print()
    print(
        f"Density captured  : "
        f"{100*fraction:.1f}% "
        f"of full B23 point-estimate gain"
    )

    print()
    print("Paired B6 -> Density")
    print(
        f"  median : "
        f"{b6_vs_density['median_difference']:+.10f}"
    )
    print(
        f"  95% CI : "
        f"[{b6_vs_density['ci_lower']:+.10f}, "
        f"{b6_vs_density['ci_upper']:+.10f}]"
    )
    print(
        f"  P(Density > B6): "
        f"{b6_vs_density['probability_b_better']:.4f}"
    )

    print()
    print("Paired Density -> Full B23")
    print(
        f"  median : "
        f"{density_vs_b23['median_difference']:+.10f}"
    )
    print(
        f"  95% CI : "
        f"[{density_vs_b23['ci_lower']:+.10f}, "
        f"{density_vs_b23['ci_upper']:+.10f}]"
    )
    print(
        f"  P(B23 > Density): "
        f"{density_vs_b23['probability_b_better']:.4f}"
    )

    print()
    print("Per-target")
    print("-" * 76)
    print(
        f"{'Target':20s} "
        f"{'B6':>8s} "
        f"{'Density':>8s} "
        f"{'B23':>8s} "
        f"{'D-B6':>9s} "
        f"{'B23-D':>9s}"
    )
    print("-" * 76)

    for target in TARGETS:
        x = per_target[target]

        print(
            f"{target:20s} "
            f"{x['b6']:8.4f} "
            f"{x['density']:8.4f} "
            f"{x['full_b23']:8.4f} "
            f"{x['density_minus_b6']:+9.4f} "
            f"{x['full_b23_minus_density']:+9.4f}"
        )

    print()
    print("NO GOLD USED.")
    print("NO PROMOTION DECISION.")

    print()
    print(
        "saved:",
        out / "three_arm_predictions.csv",
    )
    print(
        "saved:",
        out / "density_comparison.json",
    )


if __name__ == "__main__":
    main()
