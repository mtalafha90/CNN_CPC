"""Paired post-hoc reused-gold evaluation for B20 versus B26.2.

B26.2 was trained on the exact historical 3,120-study B20 gradient surface.
Therefore the 623-study weak-v2 surface is NOT a holdout for this checkpoint
(the B25X leakage-safe protocol used only the complementary 2,497 studies).

The only immediately available labelled performance surface is the 58-study
expert set. That set was already consumed repeatedly during development and
was used to select the historical B20 epoch, so this module treats it strictly
as a reused/post-hoc development diagnostic. It cannot independently promote
B26.2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .b7_weak_supervision import _read_config
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import build_b12_1_model
from .b17_training import encoder_state_sha256, freeze_encoder
from .b20_crop_focus import _expert_loader, load_b20_checkpoint, require_b20_contract
from .b26_2_training import (
    B26_2_FIXED_EPOCHS,
    B26_2_TRAIN_EXPERIMENT,
    B26_2_TRAIN_VARIANT,
    EXPECTED_ACCEPTED_TOTAL,
    EXPECTED_FINAL_CELLS,
    EXPECTED_FINAL_SYN_NEG,
    EXPECTED_FINAL_SYN_POS,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs
from .runtime import resolve_runtime

B26_2_GOLD_EVAL_VERSION = "1.0.0"


def load_b26_2_checkpoint(path: str | Path, *, device: torch.device | str = "cpu"):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if payload.get("experiment") != B26_2_TRAIN_EXPERIMENT:
        raise ValueError("not a B26.2 training checkpoint")
    if payload.get("variant") != B26_2_TRAIN_VARIANT:
        raise ValueError("unexpected B26.2 training variant")
    if payload.get("fixed_endpoint") is not True:
        raise ValueError("B26.2 checkpoint must certify a fixed endpoint")
    if int(payload.get("selected_epoch", -1)) != B26_2_FIXED_EPOCHS:
        raise ValueError("B26.2 checkpoint is not fixed E2")
    if int(payload.get("completed_epochs", -1)) != B26_2_FIXED_EPOCHS:
        raise ValueError("B26.2 checkpoint did not complete E2")
    if payload.get("encoder_frozen") is not True:
        raise ValueError("B26.2 checkpoint does not certify a frozen encoder")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B26.2 checkpoint unexpectedly used expert-gold studies in gradients")
    if payload.get("expert_checkpoint_selection") is not False:
        raise ValueError("B26.2 checkpoint must not use expert checkpoint selection")

    supervision = payload.get("supervision") or {}
    if int(supervision.get("accepted_total", -1)) != EXPECTED_ACCEPTED_TOTAL:
        raise ValueError("B26.2 supervision count changed")
    if int(supervision.get("final_usable_cells", -1)) != EXPECTED_FINAL_CELLS:
        raise ValueError("B26.2 usable-cell count changed")
    if int(supervision.get("final_synovitis_positive", -1)) != EXPECTED_FINAL_SYN_POS:
        raise ValueError("B26.2 Synovitis-positive count changed")
    if int(supervision.get("final_synovitis_negative", -1)) != EXPECTED_FINAL_SYN_NEG:
        raise ValueError("B26.2 Synovitis-negative count changed")
    if int(supervision.get("base_cells_dropped", -1)) != 0:
        raise ValueError("B26.2 dropped B6 supervision")
    if int(supervision.get("base_cells_overridden", -1)) != 0:
        raise ValueError("B26.2 overrode B6 supervision")

    initial_sha = str(payload.get("encoder_sha256_initial", ""))
    final_sha = str(payload.get("encoder_sha256_final", ""))
    if not initial_sha or initial_sha != final_sha:
        raise ValueError("B26.2 encoder fingerprint changed")

    spec = payload.get("model_spec")
    state = payload.get("model_state")
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("B26.2 checkpoint missing model specification/state")
    model = build_b12_1_model(spec, pretrained_weights=False)
    model.load_state_dict(state, strict=True)
    freeze_encoder(model)
    if encoder_state_sha256(model.encoder) != initial_sha:
        raise ValueError("B26.2 reconstructed encoder fingerprint mismatch")
    return model.to(device).eval(), payload


def evaluate_b26_2_reused_gold(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    b26_2_checkpoint: str | Path,
    out_root: str | Path,
    n_bootstrap: int = 5000,
) -> dict:
    crop_policy = require_b20_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())

    control_model, control_payload = load_b20_checkpoint(
        b20_checkpoint, device=runtime.device
    )
    candidate_model, candidate_payload = load_b26_2_checkpoint(
        b26_2_checkpoint, device=runtime.device
    )

    if str(control_payload.get("encoder_sha256_initial", "")) != str(
        candidate_payload.get("encoder_sha256_initial", "")
    ):
        raise RuntimeError("B20 and B26.2 do not share the same frozen encoder fingerprint")
    if control_payload.get("crop_focus_policy") != candidate_payload.get("crop_focus_policy"):
        raise RuntimeError("B20 and B26.2 crop policies differ")
    if control_payload.get("model_spec") != candidate_payload.get("model_spec"):
        raise RuntimeError("B20 and B26.2 model specifications differ")

    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    expert = _expert_loader(config, root, train, series, runtime, crop_policy)

    print("[B26.2 gold] predicting historical B20 control on reused 58-study expert surface")
    control_uids, control_pred = predict_b12_1(control_model, expert["loader"], runtime)
    print("[B26.2 gold] predicting fixed-E2 B26.2 candidate on the same surface")
    candidate_uids, candidate_pred = predict_b12_1(candidate_model, expert["loader"], runtime)
    expected_uids = [str(x) for x in expert["uids"]]
    if [str(x) for x in control_uids] != expected_uids:
        raise RuntimeError("B20 expert prediction order changed")
    if [str(x) for x in candidate_uids] != expected_uids:
        raise RuntimeError("B26.2 expert prediction order changed")

    truth = expert["truth"]
    seed = int(config.get("seed", 2026))
    control_eval = bootstrap_macro_auc(
        truth, control_pred, n_bootstrap=n_bootstrap, seed=seed + 26_201
    )
    candidate_eval = bootstrap_macro_auc(
        truth, candidate_pred, n_bootstrap=n_bootstrap, seed=seed + 26_202
    )
    paired = compare_runs(
        truth,
        control_pred,
        candidate_pred,
        n_bootstrap=n_bootstrap,
        seed=seed + 26_203,
    )
    raw_delta = float(candidate_eval.macro_auc - control_eval.macro_auc)

    per_target = {}
    for target in TARGETS:
        a = float(control_eval.per_target[target])
        b = float(candidate_eval.per_target[target])
        per_target[target] = {
            "b20": a,
            "b26_2": b,
            "b26_2_minus_b20": b - a,
        }

    result = {
        "evaluation_version": B26_2_GOLD_EVAL_VERSION,
        "experiment": B26_2_TRAIN_EXPERIMENT,
        "surface": "58-study reused expert development surface",
        "n_studies": len(expected_uids),
        "independent_validation": False,
        "weak_v2_used": False,
        "weak_v2_reason": (
            "B26.2 trained on the full historical 3,120-study B20 surface, which includes "
            "the 623 studies reserved as weak-v2 only in leakage-safe B25X experiments."
        ),
        "b20_control_was_selected_on_this_expert_surface": True,
        "b26_2_checkpoint_selection_used_expert_surface": False,
        "promotion_allowed_from_this_result_alone": False,
        "b20": control_eval.to_dict(),
        "b26_2": candidate_eval.to_dict(),
        "raw_b26_2_minus_b20": raw_delta,
        "paired_b26_2_minus_b20": paired,
        "per_target": per_target,
        "encoder_sha256": str(candidate_payload["encoder_sha256_initial"]),
        "metadata_repair": metadata_stats,
        "interpretation_guardrail": (
            "This is a reused/post-hoc development diagnostic. The historical B20 control is "
            "favored by having been selected on this same expert surface. Hidden competition "
            "evaluation remains the independent predictive-performance signal."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"StudyInstanceUID": expected_uids})
    for j, target in enumerate(TARGETS):
        frame[f"{target}__B20"] = control_pred[:, j]
        frame[f"{target}__B26_2"] = candidate_pred[:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("B26.2 REUSED EXPERT DEVELOPMENT RESULT -- NOT INDEPENDENT VALIDATION")
    print("=" * 76)
    print(f"B20    macro AUC : {control_eval.macro_auc:.10f}")
    print(f"B26.2  macro AUC : {candidate_eval.macro_auc:.10f}")
    print(f"raw delta         : {raw_delta:+.10f}")
    print(
        "paired 95% CI    : "
        f"[{paired['ci_lower']:+.10f}, {paired['ci_upper']:+.10f}]"
    )
    print(f"P(B26.2 > B20)    : {paired['probability_b_better']:.4f}")
    print("\nPer-target")
    print("-" * 64)
    print(f"{'Target':18s} {'B20':>10s} {'B26.2':>10s} {'Delta':>10s}")
    for target, row in per_target.items():
        print(
            f"{target:18s} {row['b20']:10.4f} {row['b26_2']:10.4f} "
            f"{row['b26_2_minus_b20']:+10.4f}"
        )
    print("\nNO WEAK-V2: it is not a holdout for this 3,120-study checkpoint.")
    print("NO AUTOMATIC PROMOTION FROM REUSED GOLD.")
    print("saved:", out / "paired_predictions.csv")
    print("saved:", out / "comparison.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser("B26.2 paired reused-gold development evaluation")
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b20-checkpoint", required=True)
    parser.add_argument("--b26-2-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b26_2_training/reused_gold_eval")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    evaluate_b26_2_reused_gold(
        config,
        b20_checkpoint=args.b20_checkpoint,
        b26_2_checkpoint=args.b26_2_checkpoint,
        out_root=args.out_root,
        n_bootstrap=int(args.n_bootstrap),
    )


if __name__ == "__main__":
    main()
