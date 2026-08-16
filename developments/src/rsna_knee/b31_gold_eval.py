"""Three-way reused-expert diagnostic for B20, frozen B29 and fixed-E2 B31.

This is descriptive/post-hoc only. The 58-study expert surface is heavily reused
and historically selected B20. B31 is frozen before this outcome is inspected.
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
from .b31_training import B31_EXPERIMENT, B31_VARIANT, load_b31_checkpoint
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .runtime import resolve_runtime

B31_GOLD_EVAL_VERSION = "1.0.0"


def _historical_shared_spec(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    for key in list(out):
        if key in {"architecture", "aggregation"} or key.startswith("b29_") or key.startswith("b31_"):
            out.pop(key, None)
    return out


def evaluate_b31_reused_gold(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    b29_checkpoint: str | Path,
    b31_checkpoint: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    b20_model, b20_payload = load_b20_checkpoint(b20_checkpoint, device=runtime.device)
    b29_model, b29_payload = load_b29_checkpoint(b29_checkpoint, device=runtime.device)
    b31_model, b31_payload = load_b31_checkpoint(b31_checkpoint, device=runtime.device)
    b20_model.eval()
    b29_model.eval()
    b31_model.eval()

    shas = {
        str(b20_payload.get("encoder_sha256_initial", "")),
        str(b29_payload.get("encoder_sha256_initial", "")),
        str(b31_payload.get("encoder_sha256_initial", "")),
    }
    if len(shas) != 1 or "" in shas:
        raise RuntimeError("B20/B29/B31 encoder fingerprints differ")
    if not (
        b20_payload.get("crop_focus_policy")
        == b29_payload.get("crop_focus_policy")
        == b31_payload.get("crop_focus_policy")
    ):
        raise RuntimeError("B20/B29/B31 crop policies differ")
    shared_specs = [
        _historical_shared_spec(p.get("model_spec") or {})
        for p in (b20_payload, b29_payload, b31_payload)
    ]
    if not (shared_specs[0] == shared_specs[1] == shared_specs[2]):
        raise RuntimeError("B20/B29/B31 historical shared model specs differ")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)
    expected_uids = [str(x) for x in expert["uids"]]

    print("[B31 gold] predicting B20")
    b20_uids, b20_pred = predict_b12_1(b20_model, expert["loader"], runtime)
    print("[B31 gold] predicting frozen B29")
    b29_uids, b29_pred = predict_b12_1(b29_model, expert["loader"], runtime)
    print("[B31 gold] predicting fixed-E2 B31")
    b31_uids, b31_pred = predict_b12_1(b31_model, expert["loader"], runtime)
    for name, uids in (("B20", b20_uids), ("B29", b29_uids), ("B31", b31_uids)):
        if [str(x) for x in uids] != expected_uids:
            raise RuntimeError(f"{name} expert prediction order changed")

    truth = expert["truth"]
    seed = int(config.get("seed", 2026))
    b20_eval = bootstrap_macro_auc(truth, b20_pred, n_bootstrap=n_bootstrap, seed=seed + 31_001)
    b29_eval = bootstrap_macro_auc(truth, b29_pred, n_bootstrap=n_bootstrap, seed=seed + 31_002)
    b31_eval = bootstrap_macro_auc(truth, b31_pred, n_bootstrap=n_bootstrap, seed=seed + 31_003)
    paired_b20_b31 = compare_runs(truth, b20_pred, b31_pred, n_bootstrap=n_bootstrap, seed=seed + 31_004)
    paired_b29_b31 = compare_runs(truth, b29_pred, b31_pred, n_bootstrap=n_bootstrap, seed=seed + 31_005)

    per_target = {}
    for target in TARGETS:
        a = float(b20_eval.per_target[target])
        b = float(b29_eval.per_target[target])
        c = float(b31_eval.per_target[target])
        per_target[target] = {
            "b20": a,
            "b29": b,
            "b31": c,
            "b31_minus_b20": c - a,
            "b31_minus_b29": c - b,
        }

    result = {
        "evaluation_version": B31_GOLD_EVAL_VERSION,
        "experiment": B31_EXPERIMENT,
        "variant": B31_VARIANT,
        "surface": "58-study reused expert development surface",
        "n_studies": len(expected_uids),
        "independent_validation": False,
        "weak_v2_used": False,
        "b20_control_was_selected_on_this_expert_surface": True,
        "b29_was_previously_inspected_on_this_expert_surface": True,
        "b31_checkpoint_selection_used_expert_surface": False,
        "promotion_allowed_from_this_result_alone": False,
        "b20": b20_eval.to_dict(),
        "b29": b29_eval.to_dict(),
        "b31": b31_eval.to_dict(),
        "raw_b31_minus_b20": float(b31_eval.macro_auc - b20_eval.macro_auc),
        "raw_b31_minus_b29": float(b31_eval.macro_auc - b29_eval.macro_auc),
        "paired_b31_minus_b20": paired_b20_b31,
        "paired_b31_minus_b29": paired_b29_b31,
        "per_target": per_target,
        "complementary_pool_final": b31_payload.get("complementary_pool_final"),
        "local_context_final": b31_payload.get("local_context_final"),
        "attention_audit_history": b31_payload.get("attention_audit_history"),
        "encoder_sha256": str(b31_payload["encoder_sha256_initial"]),
        "metadata_repair": metadata_stats,
        "interpretation_guardrail": (
            "Reused/post-hoc development evidence only. Do not tune B31 local-context kernel, "
            "context strength, gate, endpoint, target-specific behavior, or B29/B31 blending from this result."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        frame[f"{target}__B20"] = b20_pred[:, j]
        frame[f"{target}__B29"] = b29_pred[:, j]
        frame[f"{target}__B31"] = b31_pred[:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("B31 REUSED EXPERT DEVELOPMENT RESULT -- NOT INDEPENDENT VALIDATION")
    print("=" * 80)
    print(f"B20 macro AUC : {b20_eval.macro_auc:.10f}")
    print(f"B29 macro AUC : {b29_eval.macro_auc:.10f}")
    print(f"B31 macro AUC : {b31_eval.macro_auc:.10f}")
    print(f"B31-B20 raw   : {b31_eval.macro_auc - b20_eval.macro_auc:+.10f}")
    print(f"B31-B29 raw   : {b31_eval.macro_auc - b29_eval.macro_auc:+.10f}")
    print(
        "B31-B20 CI    : "
        f"[{paired_b20_b31['ci_lower']:+.10f}, {paired_b20_b31['ci_upper']:+.10f}] "
        f"P={paired_b20_b31['probability_b_better']:.4f}"
    )
    print(
        "B31-B29 CI    : "
        f"[{paired_b29_b31['ci_lower']:+.10f}, {paired_b29_b31['ci_upper']:+.10f}] "
        f"P={paired_b29_b31['probability_b_better']:.4f}"
    )
    print("\nPer-target")
    print("-" * 76)
    print(f"{'Target':18s} {'B20':>9s} {'B29':>9s} {'B31':>9s} {'31-20':>9s} {'31-29':>9s}")
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['b20']:9.4f} {row['b29']:9.4f} {row['b31']:9.4f} "
            f"{row['b31_minus_b20']:+9.4f} {row['b31_minus_b29']:+9.4f}"
        )
    print("\nNO AUTOMATIC PROMOTION FROM REUSED GOLD.")
    print("saved:", out / "paired_predictions.csv")
    print("saved:", out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("B31 three-way reused-expert development evaluation")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b20-checkpoint", required=True)
    ap.add_argument("--b29-checkpoint", required=True)
    ap.add_argument("--b31-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b31_local_context_complementary_pool/reused_gold_eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_b31_reused_gold(
        config,
        b20_checkpoint=args.b20_checkpoint,
        b29_checkpoint=args.b29_checkpoint,
        b31_checkpoint=args.b31_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
