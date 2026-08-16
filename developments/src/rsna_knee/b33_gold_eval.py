"""Four-way reused-expert diagnostic for B20, B29, B31 and fixed-E2 B33.

This is descriptive/post-hoc only. The 58-study expert surface is heavily reused;
B20 was historically selected on it and B29/B31 have already been inspected.
B33 is frozen before this outcome is inspected and cannot be promoted from this
comparison alone.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from .b7_weak_supervision import _read_config
from .b12_1_gold_eval import predict_b12_1
from .b20_crop_focus import _expert_loader, load_b20_checkpoint, require_b20_contract
from .b29_training import load_b29_checkpoint
from .b31_training import load_b31_checkpoint
from .b33_training import B33_EXPERIMENT, B33_VARIANT, load_b33_checkpoint
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .runtime import resolve_runtime

B33_GOLD_EVAL_VERSION = "1.0.0"


def _historical_shared_spec(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    for key in list(out):
        if key in {"architecture", "aggregation"} or key.startswith("b29_") or key.startswith("b31_") or key.startswith("b33_"):
            out.pop(key, None)
    return out


def evaluate_b33_reused_gold(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    b29_checkpoint: str | Path,
    b31_checkpoint: str | Path,
    b33_checkpoint: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    b20_model, b20_payload = load_b20_checkpoint(b20_checkpoint, device=runtime.device)
    b29_model, b29_payload = load_b29_checkpoint(b29_checkpoint, device=runtime.device)
    b31_model, b31_payload = load_b31_checkpoint(b31_checkpoint, device=runtime.device)
    b33_model, b33_payload = load_b33_checkpoint(b33_checkpoint, device=runtime.device)
    for model in (b20_model, b29_model, b31_model, b33_model):
        model.eval()

    payloads = (b20_payload, b29_payload, b31_payload, b33_payload)
    shas = {str(p.get("encoder_sha256_initial", "")) for p in payloads}
    if len(shas) != 1 or "" in shas:
        raise RuntimeError("B20/B29/B31/B33 encoder fingerprints differ")
    crops = [p.get("crop_focus_policy") for p in payloads]
    if not all(crop == crops[0] for crop in crops[1:]):
        raise RuntimeError("B20/B29/B31/B33 crop policies differ")
    shared_specs = [_historical_shared_spec(p.get("model_spec") or {}) for p in payloads]
    if not all(spec == shared_specs[0] for spec in shared_specs[1:]):
        raise RuntimeError("B20/B29/B31/B33 historical shared model specs differ")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)
    expected_uids = [str(x) for x in expert["uids"]]

    predictions = {}
    for name, model in (
        ("B20", b20_model),
        ("B29", b29_model),
        ("B31", b31_model),
        ("B33", b33_model),
    ):
        print(f"[B33 gold] predicting {name}")
        uids, pred = predict_b12_1(model, expert["loader"], runtime)
        if [str(x) for x in uids] != expected_uids:
            raise RuntimeError(f"{name} expert prediction order changed")
        predictions[name] = pred

    truth = expert["truth"]
    seed = int(config.get("seed", 2026))
    evals = {
        "B20": bootstrap_macro_auc(truth, predictions["B20"], n_bootstrap=n_bootstrap, seed=seed + 33_001),
        "B29": bootstrap_macro_auc(truth, predictions["B29"], n_bootstrap=n_bootstrap, seed=seed + 33_002),
        "B31": bootstrap_macro_auc(truth, predictions["B31"], n_bootstrap=n_bootstrap, seed=seed + 33_003),
        "B33": bootstrap_macro_auc(truth, predictions["B33"], n_bootstrap=n_bootstrap, seed=seed + 33_004),
    }
    paired_b20_b33 = compare_runs(
        truth, predictions["B20"], predictions["B33"], n_bootstrap=n_bootstrap, seed=seed + 33_005
    )
    paired_b29_b33 = compare_runs(
        truth, predictions["B29"], predictions["B33"], n_bootstrap=n_bootstrap, seed=seed + 33_006
    )
    paired_b31_b33 = compare_runs(
        truth, predictions["B31"], predictions["B33"], n_bootstrap=n_bootstrap, seed=seed + 33_007
    )

    per_target = {}
    for target in TARGETS:
        a = float(evals["B20"].per_target[target])
        b = float(evals["B29"].per_target[target])
        c = float(evals["B31"].per_target[target])
        d = float(evals["B33"].per_target[target])
        per_target[target] = {
            "b20": a,
            "b29": b,
            "b31": c,
            "b33": d,
            "b33_minus_b20": d - a,
            "b33_minus_b29": d - b,
            "b33_minus_b31": d - c,
        }

    result = {
        "evaluation_version": B33_GOLD_EVAL_VERSION,
        "experiment": B33_EXPERIMENT,
        "variant": B33_VARIANT,
        "surface": "58-study reused expert development surface",
        "n_studies": len(expected_uids),
        "independent_validation": False,
        "weak_v2_used": False,
        "b20_control_was_selected_on_this_expert_surface": True,
        "b29_was_previously_inspected_on_this_expert_surface": True,
        "b31_was_previously_inspected_on_this_expert_surface": True,
        "b33_checkpoint_selection_used_expert_surface": False,
        "promotion_allowed_from_this_result_alone": False,
        "b20": evals["B20"].to_dict(),
        "b29": evals["B29"].to_dict(),
        "b31": evals["B31"].to_dict(),
        "b33": evals["B33"].to_dict(),
        "raw_b33_minus_b20": float(evals["B33"].macro_auc - evals["B20"].macro_auc),
        "raw_b33_minus_b29": float(evals["B33"].macro_auc - evals["B29"].macro_auc),
        "raw_b33_minus_b31": float(evals["B33"].macro_auc - evals["B31"].macro_auc),
        "paired_b33_minus_b20": paired_b20_b33,
        "paired_b33_minus_b29": paired_b29_b33,
        "paired_b33_minus_b31": paired_b31_b33,
        "per_target": per_target,
        "uniform_gate_final": b33_payload.get("uniform_gate_final"),
        "uniform_audit_history": b33_payload.get("uniform_audit_history"),
        "encoder_sha256": str(b33_payload["encoder_sha256_initial"]),
        "metadata_repair": metadata_stats,
        "interpretation_guardrail": (
            "Reused/post-hoc development evidence only. Do not tune B33 gate, mean definition, endpoint, "
            "target-specific behavior, or B29/B31/B33 blending from this result."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        for name in ("B20", "B29", "B31", "B33"):
            frame[f"{target}__{name}"] = predictions[name][:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("B33 REUSED EXPERT DEVELOPMENT RESULT -- NOT INDEPENDENT VALIDATION")
    print("=" * 88)
    for name in ("B20", "B29", "B31", "B33"):
        print(f"{name} macro AUC : {evals[name].macro_auc:.10f}")
    print(f"B33-B20 raw   : {evals['B33'].macro_auc - evals['B20'].macro_auc:+.10f}")
    print(f"B33-B29 raw   : {evals['B33'].macro_auc - evals['B29'].macro_auc:+.10f}")
    print(f"B33-B31 raw   : {evals['B33'].macro_auc - evals['B31'].macro_auc:+.10f}")
    for label, paired in (
        ("B33-B20", paired_b20_b33),
        ("B33-B29", paired_b29_b33),
        ("B33-B31", paired_b31_b33),
    ):
        print(
            f"{label} CI    : [{paired['ci_lower']:+.10f}, {paired['ci_upper']:+.10f}] "
            f"P={paired['probability_b_better']:.4f}"
        )

    print("\nPer-target")
    print("-" * 86)
    print(
        f"{'Target':18s} {'B20':>8s} {'B29':>8s} {'B31':>8s} {'B33':>8s} "
        f"{'33-20':>8s} {'33-29':>8s} {'33-31':>8s}"
    )
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['b20']:8.4f} {row['b29']:8.4f} {row['b31']:8.4f} {row['b33']:8.4f} "
            f"{row['b33_minus_b20']:+8.4f} {row['b33_minus_b29']:+8.4f} {row['b33_minus_b31']:+8.4f}"
        )
    print("\nNO AUTOMATIC PROMOTION FROM REUSED GOLD.")
    print("saved:", out / "paired_predictions.csv")
    print("saved:", out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("B33 four-way reused-expert development evaluation")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b20-checkpoint", required=True)
    ap.add_argument("--b29-checkpoint", required=True)
    ap.add_argument("--b31-checkpoint", required=True)
    ap.add_argument("--b33-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b33_uniform_complementary_mean/reused_gold_eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_b33_reused_gold(
        config,
        b20_checkpoint=args.b20_checkpoint,
        b29_checkpoint=args.b29_checkpoint,
        b31_checkpoint=args.b31_checkpoint,
        b33_checkpoint=args.b33_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
