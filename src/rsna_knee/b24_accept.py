"""The single predeclared gold look for B24.

This consumes B24's one and only expert-surface comparison. Everything about it
is fixed before it runs: the comparator (canonical B20), the statistic (paired
study bootstrap of the 12-target macro AUC), and the rule (paired median > 0 and
P(B24 > B20) >= 0.95).

The threshold is deliberately not a bare point-estimate win. B22 measured a
0.0439 swing across a single run's epochs on this surface, and the reported
bootstrap intervals imply a macro standard error near 0.0250, so a small
positive difference is not evidence of anything. Requiring 0.95 probability of
superiority is what stops B24 being promoted by noise.

Running this a second time, or after seeing the result and adjusting anything,
would destroy the only property that makes it worth running at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import build_b12_1_model
from .b13_training import B13_INPUT_NORMALIZATION
from .b21_dataset import make_matched_crop_dataset
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b24_protocol import (
    B20_REPLAY_TOLERANCE,
    B24_CROP_FRACTION,
    MODE_CANDIDATE,
    gold_promotion_decision,
)
from .constants import TARGETS
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs, macro_auc_from_arrays
from .runtime import resolve_runtime

CANONICAL_B20_GOLD_MACRO_AUC = 0.667159355531343


def _predict_gold(model, uids, config, root, runtime):
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, _ = backfill_series_metadata(series, root, split="train")
    _summary, index = audit_variable_series_surface(series, uids)
    dataset_config = make_b7_dataset_config(
        config,
        normalization=B13_INPUT_NORMALIZATION,
        offsets=tuple(config.get("b7_eval_tta_offsets", (-1, 0, 1))),
    )
    ds = make_matched_crop_dataset(
        "control",
        uids,
        {uid: index[uid] for uid in uids},
        dataset_config,
        crop_fraction=B24_CROP_FRACTION,
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("b7_eval_batch_size", 2)),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026))),
    )
    pred_uids, predictions = predict_b12_1(model, loader, runtime)
    if [str(u) for u in pred_uids] != [str(u) for u in uids]:
        raise RuntimeError("gold prediction order changed")
    return predictions


def _load(path, expected_mode=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_mode is not None and payload.get("mode") != expected_mode:
        raise ValueError(f"expected mode {expected_mode!r}, got {payload.get('mode')!r}")
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def accept_b24(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    b24_checkpoint: str | Path,
    out_root: str | Path = "runs/b24_supervision/gold_acceptance",
    n_bootstrap: int = 5000,
) -> dict:
    out = Path(out_root)
    if (out / "acceptance.json").is_file():
        raise RuntimeError(
            f"{out / 'acceptance.json'} already exists. B24 gets exactly one gold "
            "look; re-running it would invalidate the only property that makes "
            "the result meaningful."
        )

    runtime = resolve_runtime(config)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train)].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    uids = gold["StudyInstanceUID"].tolist()
    if len(uids) != 58:
        raise ValueError(f"expected the 58 expert studies, found {len(uids)}")
    truth = gold[TARGETS].to_numpy(dtype=np.float64)

    b20_model, _b20_payload = _load(b20_checkpoint)
    b24_model, b24_payload = _load(b24_checkpoint, MODE_CANDIDATE)

    pred_b20 = _predict_gold(b20_model.to(runtime.device).eval(), uids, config, root, runtime)
    pred_b24 = _predict_gold(b24_model.to(runtime.device).eval(), uids, config, root, runtime)

    replay_macro, _ = macro_auc_from_arrays(truth, pred_b20)
    drift = abs(float(replay_macro) - CANONICAL_B20_GOLD_MACRO_AUC)
    if drift > B20_REPLAY_TOLERANCE:
        raise RuntimeError(
            f"B20 replay drifted by {drift:.6f} (tolerance {B20_REPLAY_TOLERANCE}); "
            "the evaluation path does not reproduce the canonical result, so the "
            "comparison would not be trustworthy"
        )

    candidate_macro, candidate_per_target = macro_auc_from_arrays(truth, pred_b24)
    candidate_ci = bootstrap_macro_auc(truth, pred_b24, n_bootstrap=n_bootstrap)
    paired = compare_runs(truth, pred_b20, pred_b24, n_bootstrap=n_bootstrap)
    decision = gold_promotion_decision(
        paired_median=float(paired["median_difference"]),
        probability_candidate_better=float(paired["probability_b_better"]),
    )

    payload = {
        "canonical_b20_macro_auc": CANONICAL_B20_GOLD_MACRO_AUC,
        "replayed_b20_macro_auc": float(replay_macro),
        "replay_drift": drift,
        "replay_tolerance": B20_REPLAY_TOLERANCE,
        "b24_macro_auc": float(candidate_macro),
        "b24_ci": {"low": candidate_ci.ci_low, "high": candidate_ci.ci_high},
        "b24_minus_b20_raw": float(candidate_macro) - float(replay_macro),
        "paired": paired,
        "decision": decision,
        "per_target_auc": {
            t: float(v) for t, v in zip(TARGETS, candidate_per_target)
        },
        "per_target_note": (
            "descriptive only. At n=58 the per-target standard error is about "
            "0.080, so a 95% interval spans roughly +/-0.157. These must not be "
            "used to build a target-wise B20/B24 mixture."
        ),
        "supervision": b24_payload.get("supervision"),
        "surface_diagnostics": b24_payload.get("surface_diagnostics"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd_frame = {"StudyInstanceUID": uids}
    for j, target in enumerate(TARGETS):
        pd_frame[f"{target}__b20"] = pred_b20[:, j]
        pd_frame[f"{target}__b24"] = pred_b24[:, j]
    import pandas as pd

    pd.DataFrame(pd_frame).to_csv(out / "gold_predictions.csv", index=False)
    return payload


def format_acceptance(payload: dict) -> str:
    d = payload["decision"]
    return "\n".join(
        [
            "B24 gold acceptance (ONE predeclared look)",
            "",
            f"  B20 canonical            {payload['canonical_b20_macro_auc']:.10f}",
            f"  B20 replayed             {payload['replayed_b20_macro_auc']:.10f}"
            f"  (drift {payload['replay_drift']:.6f})",
            f"  B24 candidate            {payload['b24_macro_auc']:.10f}",
            f"  raw difference           {payload['b24_minus_b20_raw']:+.10f}",
            "",
            f"  paired median            {d['paired_median']:+.10f}",
            f"  P(B24 > B20)             {d['probability_candidate_better']:.4f}"
            f"  (need >= {d['required_probability']})",
            "",
            f"  PROMOTED: {d['promoted']}",
            f"  rule: {d['rule']}",
            "",
            f"  {d['interpretation']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="B24 single gold acceptance look")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b20-checkpoint", required=True)
    parser.add_argument("--b24-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b24_supervision/gold_acceptance")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    payload = accept_b24(
        config,
        b20_checkpoint=args.b20_checkpoint,
        b24_checkpoint=args.b24_checkpoint,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )
    print(format_acceptance(payload))


if __name__ == "__main__":  # pragma: no cover
    main()
