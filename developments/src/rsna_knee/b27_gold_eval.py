"""Paired reused-expert diagnostic for historical B20 versus fixed-E2 B27.

B27 is trained on the full historical 3,120-study B20 gradient surface, so the
623-study weak-v2 partition is not a holdout for it.  The 58-study expert set is
also already reused development data and selected the historical B20 epoch.
This comparison is therefore descriptive/post-hoc and cannot independently
promote B27.
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
from .b27_training import B27_EXPERIMENT, B27_VARIANT, load_b27_checkpoint
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .runtime import resolve_runtime

B27_GOLD_EVAL_VERSION = "1.0.0"


def _shared_spec(spec: dict) -> dict:
    """Remove only B27 identity fields before checking the shared B20 contract."""
    out = copy.deepcopy(spec)
    out.pop("architecture", None)
    out.pop("aggregation", None)
    out.pop("b27_routing_version", None)
    out.pop("b27_route_parameter_count", None)
    out.pop("b27_routing_unknown_metadata_bias", None)
    return out


def evaluate_b27_reused_gold(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    b27_checkpoint: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    b20_model, b20_payload = load_b20_checkpoint(b20_checkpoint, device=runtime.device)
    b27_model, b27_payload = load_b27_checkpoint(b27_checkpoint, device=runtime.device)
    b20_model.eval()
    b27_model.eval()

    if str(b20_payload.get("encoder_sha256_initial", "")) != str(
        b27_payload.get("encoder_sha256_initial", "")
    ):
        raise RuntimeError("B20 and B27 do not share the same frozen encoder fingerprint")
    if b20_payload.get("crop_focus_policy") != b27_payload.get("crop_focus_policy"):
        raise RuntimeError("B20 and B27 crop policies differ")
    if _shared_spec(b20_payload.get("model_spec") or {}) != _shared_spec(
        b27_payload.get("model_spec") or {}
    ):
        raise RuntimeError("B20 and B27 shared model specifications differ")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)
    expected_uids = [str(x) for x in expert["uids"]]

    print("[B27 gold] predicting historical B20 on reused 58-study expert surface")
    b20_uids, b20_pred = predict_b12_1(b20_model, expert["loader"], runtime)
    print("[B27 gold] predicting fixed-E2 B27 on the same surface")
    b27_uids, b27_pred = predict_b12_1(b27_model, expert["loader"], runtime)
    if [str(x) for x in b20_uids] != expected_uids:
        raise RuntimeError("B20 expert prediction order changed")
    if [str(x) for x in b27_uids] != expected_uids:
        raise RuntimeError("B27 expert prediction order changed")

    truth = expert["truth"]
    seed = int(config.get("seed", 2026))
    b20_eval = bootstrap_macro_auc(
        truth, b20_pred, n_bootstrap=n_bootstrap, seed=seed + 27_201
    )
    b27_eval = bootstrap_macro_auc(
        truth, b27_pred, n_bootstrap=n_bootstrap, seed=seed + 27_202
    )
    paired = compare_runs(
        truth,
        b20_pred,
        b27_pred,
        n_bootstrap=n_bootstrap,
        seed=seed + 27_203,
    )
    raw_delta = float(b27_eval.macro_auc - b20_eval.macro_auc)

    per_target = {}
    for target in TARGETS:
        a = float(b20_eval.per_target[target])
        b = float(b27_eval.per_target[target])
        per_target[target] = {
            "b20": a,
            "b27": b,
            "b27_minus_b20": b - a,
        }

    result = {
        "evaluation_version": B27_GOLD_EVAL_VERSION,
        "experiment": B27_EXPERIMENT,
        "variant": B27_VARIANT,
        "surface": "58-study reused expert development surface",
        "n_studies": len(expected_uids),
        "independent_validation": False,
        "weak_v2_used": False,
        "weak_v2_reason": (
            "B27 trains on all 3,120 historical B20 weak-supervision studies, including "
            "the 623 UIDs used as weak-v2 only in leakage-safe partial-surface experiments."
        ),
        "b20_control_was_selected_on_this_expert_surface": True,
        "b27_checkpoint_selection_used_expert_surface": False,
        "promotion_allowed_from_this_result_alone": False,
        "b20": b20_eval.to_dict(),
        "b27": b27_eval.to_dict(),
        "raw_b27_minus_b20": raw_delta,
        "paired_b27_minus_b20": paired,
        "per_target": per_target,
        "routing_final": b27_payload.get("routing_final"),
        "encoder_sha256": str(b27_payload["encoder_sha256_initial"]),
        "metadata_repair": metadata_stats,
        "interpretation_guardrail": (
            "This is reused/post-hoc development evidence. Hidden competition evaluation "
            "remains the independent performance signal. Do not tune B27 route biases, "
            "metadata categories or endpoint from this 58-study result."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        frame[f"{target}__B20"] = b20_pred[:, j]
        frame[f"{target}__B27"] = b27_pred[:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("B27 REUSED EXPERT DEVELOPMENT RESULT -- NOT INDEPENDENT VALIDATION")
    print("=" * 76)
    print(f"B20   macro AUC : {b20_eval.macro_auc:.10f}")
    print(f"B27   macro AUC : {b27_eval.macro_auc:.10f}")
    print(f"raw delta        : {raw_delta:+.10f}")
    print(
        "paired 95% CI   : "
        f"[{paired['ci_lower']:+.10f}, {paired['ci_upper']:+.10f}]"
    )
    print(f"P(B27 > B20)     : {paired['probability_b_better']:.4f}")
    print("\nPer-target")
    print("-" * 64)
    print(f"{'Target':18s} {'B20':>10s} {'B27':>10s} {'Delta':>10s}")
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['b20']:10.4f} {row['b27']:10.4f} "
            f"{row['b27_minus_b20']:+10.4f}"
        )
    print("\nNO WEAK-V2: it is not a holdout for this full-surface checkpoint.")
    print("NO AUTOMATIC PROMOTION FROM REUSED GOLD.")
    print("saved:", out / "paired_predictions.csv")
    print("saved:", out / "comparison.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser("B27 paired reused-expert development evaluation")
    ap.add_argument("--config", default="config/current_model.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b20-checkpoint", required=True)
    ap.add_argument("--b27-checkpoint", required=True)
    ap.add_argument("--out-root", default="runs/b27_pathology_routing/reused_gold_eval")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_b27_reused_gold(
        config,
        b20_checkpoint=args.b20_checkpoint,
        b27_checkpoint=args.b27_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
