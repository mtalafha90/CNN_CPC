"""Score both B24 arms on both weak surfaces, and read the asymmetry.

Each surface favours the arm trained by its own labeller, so a single-surface
result proves nothing. What is informative is whether the B23-supervised arm
also wins on the surface built from B6 labels: that would mean B23 supervision
produced a model which reproduces the B6 teacher better than training on B6
labels did, which no self-fulfilling mechanism explains.

Neither surface is expert truth. B15 and B21 both improved on a weak surface and
then failed on gold. This evaluator therefore produces evidence, not a decision;
the decision comes from the single predeclared gold look in `b24_accept`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config, make_b7_dataset_config, prepare_b7_supervision
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import build_b12_1_model
from .b15_ssl import load_frozen_v2_manifest
from .b21_dataset import make_matched_crop_dataset
from .b23_llm_labels import load_frozen_b23_export
from .b23_validation_split import load_frozen_b23_holdout
from .b24_protocol import B24_CROP_FRACTION, MODE_CANDIDATE, MODE_CONTROL, cross_labeller_verdict
from .b7_weak_supervision import load_frozen_b6_export
from .data import backfill_series_metadata, load_series_csv, load_train_csv
from .runtime import resolve_runtime
from .weak_validation import compare_on_weak_surface, evaluate_on_weak_surface


def _load_arm(path: str | Path, mode: str, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("mode") != mode:
        raise ValueError(f"expected a {mode!r} checkpoint, got {payload.get('mode')!r}")
    if not bool(payload.get("fixed_endpoint", False)):
        raise ValueError("B24 checkpoints must be the fixed epoch-2 endpoint")
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload


def _predict(model, uids, config, root, series_policy_path, runtime):
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, _ = backfill_series_metadata(series, root, split="train")
    _summary, index = audit_variable_series_surface(series, uids)
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", (-1, 0, 1)))
    dataset_config = make_b7_dataset_config(
        config,
        root,
        train=False,
        tta_offsets=offsets,
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
        raise RuntimeError("prediction order changed during B24 evaluation")
    return predictions


def _surface(train, frame, holdout_uids):
    """Targets/weights restricted to a holdout, in a fixed order."""
    uids, y, w, _summary = prepare_b7_supervision(train, frame)
    keep = [i for i, uid in enumerate(uids) if str(uid) in holdout_uids]
    ordered = [str(uids[i]) for i in keep]
    return ordered, np.asarray(y)[keep], np.asarray(w)[keep]


def evaluate_b24(
    config: dict,
    *,
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    b6_root: str | Path,
    b23_root: str | Path,
    weak_holdout_root: str | Path,
    b23_holdout_root: str | Path,
    series_policy_path: str | Path,
    out_root: str | Path = "runs/b24_supervision/eval",
    n_bootstrap: int = 5000,
) -> dict:
    runtime = resolve_runtime(config)
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))

    control, control_payload = _load_arm(control_checkpoint, MODE_CONTROL, runtime.device)
    candidate, candidate_payload = _load_arm(candidate_checkpoint, MODE_CANDIDATE, runtime.device)
    if control_payload.get("encoder_sha256_initial") != candidate_payload.get("encoder_sha256_initial"):
        raise RuntimeError("B24 arms did not start from the same encoder")
    if list(control_payload.get("study_uids", [])) != list(candidate_payload.get("study_uids", [])):
        raise RuntimeError("B24 arms did not train on the same studies")

    b6_frame, _p, _a = load_frozen_b6_export(b6_root)
    b23_frame, _p2, _a2 = load_frozen_b23_export(b23_root)
    _weak_payload, weak_manifest = load_frozen_v2_manifest(weak_holdout_root)
    _b23_payload, b23_manifest = load_frozen_b23_holdout(b23_holdout_root)

    results: dict = {"surfaces": {}}
    for name, frame, manifest in (
        ("weak_v2_b6", b6_frame, weak_manifest),
        ("b23_holdout", b23_frame, b23_manifest),
    ):
        holdout_uids = set(
            manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str)
        )
        uids, y, w = _surface(train, frame, holdout_uids)
        if not uids:
            raise ValueError(f"surface {name} produced no scoreable studies")
        pred_control = _predict(control, uids, config, root, series_policy_path, runtime)
        pred_candidate = _predict(candidate, uids, config, root, series_policy_path, runtime)

        control_eval = evaluate_on_weak_surface(y, pred_control, w, n_bootstrap=n_bootstrap)
        candidate_eval = evaluate_on_weak_surface(y, pred_candidate, w, n_bootstrap=n_bootstrap)
        paired = compare_on_weak_surface(y, pred_control, pred_candidate, w, n_bootstrap=n_bootstrap)
        results["surfaces"][name] = {
            "studies": len(uids),
            "labelled_by": "B6 v1.2.1" if name == "weak_v2_b6" else "B23",
            "favours_by_construction": MODE_CONTROL if name == "weak_v2_b6" else MODE_CANDIDATE,
            "control_macro_auc": control_eval.get("macro_auc"),
            "candidate_macro_auc": candidate_eval.get("macro_auc"),
            "paired": paired,
        }

    weak = results["surfaces"]["weak_v2_b6"]
    own = results["surfaces"]["b23_holdout"]
    results["cross_labeller"] = cross_labeller_verdict(
        candidate_on_b23=float(own["candidate_macro_auc"]),
        control_on_b23=float(own["control_macro_auc"]),
        candidate_on_weak_v2=float(weak["candidate_macro_auc"]),
        control_on_weak_v2=float(weak["control_macro_auc"]),
    )
    results["decision"] = (
        "evidence only; the promotion decision comes from the single predeclared "
        "gold look in rsna-knee-b24-accept"
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_labeller_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def format_eval(results: dict) -> str:
    lines = ["B24 cross-labeller evaluation", ""]
    for name, block in results["surfaces"].items():
        lines.append(f"  {name}  ({block['studies']} studies, labels from {block['labelled_by']})")
        lines.append(f"    favours by construction  {block['favours_by_construction']}")
        lines.append(f"    B6-supervised control    {block['control_macro_auc']:.10f}")
        lines.append(f"    B23-supervised candidate {block['candidate_macro_auc']:.10f}")
        paired = block["paired"]
        lines.append(
            f"    paired median            {paired.get('median_difference', float('nan')):+.10f}"
        )
        lines.append("")
    verdict = results["cross_labeller"]
    lines.append(f"  verdict: {verdict['strength'].upper()}")
    lines.append(f"  {verdict['reading']}")
    lines.append("")
    lines.append(f"  {verdict['note']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="B24 cross-labeller evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--b23-root", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--b23-holdout-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--out-root", default="runs/b24_supervision/eval")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    results = evaluate_b24(
        config,
        control_checkpoint=args.control_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        b6_root=args.b6_root,
        b23_root=args.b23_root,
        weak_holdout_root=args.weak_holdout_root,
        b23_holdout_root=args.b23_holdout_root,
        series_policy_path=args.series_policy,
        out_root=args.out_root,
        n_bootstrap=args.n_bootstrap,
    )
    print(format_eval(results))


if __name__ == "__main__":  # pragma: no cover
    main()
