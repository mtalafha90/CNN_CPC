"""Paired reused-expert diagnostic for historical B20 versus fixed-E2 B30.

B30 trains on the full 3,120-study historical B20 gradient surface, so weak-v2
is not a holdout. The 58-study expert surface is heavily reused development data
and historically selected B20 checkpoints. This comparison is descriptive/post-
hoc and cannot independently promote B30.
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
from .b30_training import B30_EXPERIMENT, B30_VARIANT, load_b30_checkpoint
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .runtime import resolve_runtime

B30_GOLD_EVAL_VERSION = "1.0.0"


def _shared_spec(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    out.pop("architecture", None)
    out.pop("aggregation", None)
    out.pop("b30_residual_version", None)
    out.pop("b30_new_parameter_count", None)
    out.pop("b30_complementary_attention", None)
    out.pop("b30_shared_projection_gradient", None)
    out.pop("b30_gate_constraint", None)
    out.pop("b30_stochastic_path", None)
    return out


def evaluate_b30_reused_gold(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    b30_checkpoint: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    b20_model, b20_payload = load_b20_checkpoint(b20_checkpoint, device=runtime.device)
    candidate_model, candidate_payload = load_b30_checkpoint(
        b30_checkpoint, device=runtime.device
    )
    b20_model.eval()
    candidate_model.eval()

    if str(b20_payload.get("encoder_sha256_initial", "")) != str(
        candidate_payload.get("encoder_sha256_initial", "")
    ):
        raise RuntimeError("B20 and B30 do not share the same frozen encoder fingerprint")
    if b20_payload.get("crop_focus_policy") != candidate_payload.get("crop_focus_policy"):
        raise RuntimeError("B20 and B30 crop policies differ")
    if _shared_spec(b20_payload.get("model_spec") or {}) != _shared_spec(
        candidate_payload.get("model_spec") or {}
    ):
        raise RuntimeError("B20 and B30 shared model specifications differ")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)
    expected_uids = [str(x) for x in expert["uids"]]

    print("[B30 gold] predicting historical B20 on reused 58-study expert surface")
    b20_uids, b20_pred = predict_b12_1(b20_model, expert["loader"], runtime)
    print("[B30 gold] predicting fixed-E2 B30 on the same surface")
    cand_uids, cand_pred = predict_b12_1(candidate_model, expert["loader"], runtime)
    if [str(x) for x in b20_uids] != expected_uids:
        raise RuntimeError("B20 expert prediction order changed")
    if [str(x) for x in cand_uids] != expected_uids:
        raise RuntimeError("B30 expert prediction order changed")

    truth = expert["truth"]
    seed = int(config.get("seed", 2026))
    b20_eval = bootstrap_macro_auc(
        truth, b20_pred, n_bootstrap=n_bootstrap, seed=seed + 30_001
    )
    cand_eval = bootstrap_macro_auc(
        truth, cand_pred, n_bootstrap=n_bootstrap, seed=seed + 30_002
    )
    paired = compare_runs(
        truth,
        b20_pred,
        cand_pred,
        n_bootstrap=n_bootstrap,
        seed=seed + 30_003,
    )
    raw_delta = float(cand_eval.macro_auc - b20_eval.macro_auc)

    per_target = {}
    for target in TARGETS:
        a = float(b20_eval.per_target[target])
        b = float(cand_eval.per_target[target])
        per_target[target] = {
            "b20": a,
            "b30": b,
            "b30_minus_b20": b - a,
        }

    result = {
        "evaluation_version": B30_GOLD_EVAL_VERSION,
        "experiment": B30_EXPERIMENT,
        "variant": B30_VARIANT,
        "surface": "58-study reused expert development surface",
        "n_studies": len(expected_uids),
        "independent_validation": False,
        "weak_v2_used": False,
        "weak_v2_reason": (
            "B30 trains on all 3,120 historical B20 weak-supervision studies, including "
            "the 623 UIDs used as weak-v2 only in partial-surface experiments."
        ),
        "b20_control_was_selected_on_this_expert_surface": True,
        "b30_checkpoint_selection_used_expert_surface": False,
        "promotion_allowed_from_this_result_alone": False,
        "b20": b20_eval.to_dict(),
        "b30": cand_eval.to_dict(),
        "raw_b30_minus_b20": raw_delta,
        "paired_b30_minus_b20": paired,
        "per_target": per_target,
        "complementary_pool_final": candidate_payload.get("complementary_pool_final"),
        "attention_audit_history": candidate_payload.get("attention_audit_history"),
        "encoder_sha256": str(candidate_payload["encoder_sha256_initial"]),
        "metadata_repair": metadata_stats,
        "interpretation_guardrail": (
            "This is reused/post-hoc development evidence. Do not tune B30 query, gate, "
            "endpoint, shared-projection policy, or target-specific behavior from this result. "
            "Hidden competition evaluation remains the independent performance signal."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        frame[f"{target}__B20"] = b20_pred[:, j]
        frame[f"{target}__B30"] = cand_pred[:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("B30 REUSED EXPERT DEVELOPMENT RESULT -- NOT INDEPENDENT VALIDATION")
    print("=" * 76)
    print(f"B20   macro AUC : {b20_eval.macro_auc:.10f}")
    print(f"B30   macro AUC : {cand_eval.macro_auc:.10f}")
    print(f"raw delta        : {raw_delta:+.10f}")
    print(
        "paired 95% CI   : "
        f"[{paired['ci_lower']:+.10f}, {paired['ci_upper']:+.10f}]"
    )
    print(f"P(B30 > B20)     : {paired['probability_b_better']:.4f}")
    print("\nPer-target")
    print("-" * 64)
    print(f"{'Target':18s} {'B20':>10s} {'B30':>10s} {'Delta':>10s}")
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['b20']:10.4f} {row['b30']:10.4f} "
            f"{row['b30_minus_b20']:+10.4f}"
        )
    print("\nNO WEAK-V2: it is not a holdout for this full-surface checkpoint.")
    print("NO AUTOMATIC PROMOTION FROM REUSED GOLD.")
    print("saved:", out / "paired_predictions.csv")
    print("saved:", out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("B30 paired reused-expert development evaluation")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b20-checkpoint", required=True)
    ap.add_argument("--b30-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b30_projected_complementary_series_pool/reused_gold_eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_b30_reused_gold(
        config,
        b20_checkpoint=args.b20_checkpoint,
        b30_checkpoint=args.b30_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
