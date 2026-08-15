"""Hidden-test inference for the final all-data production checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import _read_config
from .b12_variable_series import VariableSeriesKneeDataset, build_variable_series_index, collate_variable_series
from .b17_submission import _test_dataset_config, _validate_sample_submission, _validate_submission
from .budget import RuntimeBudget
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv
from .final_all_data_training import FINAL_EXPERIMENT, FINAL_VARIANT, load_final_checkpoint, require_final_contract
from .runtime import autocast, resolve_runtime


@torch.no_grad()
def generate_final_submission(config: dict, *, checkpoint: str | Path, out_path: str | Path = "submission.csv") -> Path:
    require_final_contract(config)
    runtime = resolve_runtime(config)
    print(runtime.describe())
    model, payload = load_final_checkpoint(checkpoint, device=runtime.device)
    if payload.get("variant") != FINAL_VARIANT or payload.get("experiment") != FINAL_EXPERIMENT:
        raise ValueError("checkpoint is not the final all-data production model")

    root = Path(config["data_root"])
    test = load_test_csv(root / config.get("test_csv", "test.csv"))
    uids = test["StudyInstanceUID"].astype(str).tolist()
    if not uids:
        raise ValueError("test.csv contains no studies")
    series = load_series_csv(root / config.get("test_series_csv", "test_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="test")
    variable_index = build_variable_series_index(series, uids)
    counts = [len(variable_index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("final submission found a test study with zero eligible series")

    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("final submission freezes TTA at [-1,0,1]")
    ds = VariableSeriesKneeDataset(uids, variable_index, _test_dataset_config(config, root, offsets), train=False)
    loader = DataLoader(
        ds,
        batch_size=max(1, int(config.get("b7_eval_batch_size", 2))),
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=int(config.get("seed", 2026)) + 23_100_000),
    )

    budget = RuntimeBudget(
        max_hours=float(config.get("runtime_budget_hours", 8.5)),
        reserve_minutes=float(config.get("runtime_reserve_minutes", 10.0)),
    )
    model.eval()
    probability_rows, uid_rows, batch_times = [], [], []
    for batch_index, batch in enumerate(loader):
        if batch_times:
            projected = float(np.mean(batch_times[-5:])) * max(1, len(loader) - batch_index) * 1.35
            budget.require(projected, label="remaining final submission inference")
        else:
            budget.require(180.0, label="first final submission batch")
        start = time.monotonic()
        present = batch["present"].to(runtime.device, non_blocking=True)
        series_meta = batch["series_meta"].to(runtime.device, non_blocking=True)
        volumes = batch["volumes"]
        if volumes.ndim != 7 or int(volumes.shape[1]) != 3:
            raise RuntimeError(f"final submission expects 3-view [B,V,K,S,C,H,W], got {tuple(volumes.shape)}")
        view_probs = []
        for view in range(volumes.shape[1]):
            with autocast(runtime):
                logits = model(volumes[:, view].to(runtime.device, non_blocking=True), present, series_meta)
            view_probs.append(torch.sigmoid(logits.float()))
        probability_rows.append(torch.stack(view_probs, dim=0).mean(dim=0).cpu().numpy())
        uid_rows.extend([str(uid) for uid in batch["study_uid"]])
        batch_times.append(time.monotonic() - start)

    if uid_rows != uids:
        raise RuntimeError("final test inference changed StudyInstanceUID order")
    probabilities = np.concatenate(probability_rows, axis=0)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    _validate_submission(frame, uids)
    sample_validation = _validate_sample_submission(root, frame)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest = {
        "experiment": "FINAL_all_data_hidden_test_submission",
        "variant": FINAL_VARIANT,
        "checkpoint": str(Path(checkpoint).resolve()),
        "completed_epochs": int(payload.get("completed_epochs", -1)),
        "gold_studies_used_in_gradient": int(payload.get("gold_studies_used_in_gradient", -1)),
        "gold_evaluation_after_training_permitted": False,
        "encoder_frozen": bool(payload.get("encoder_frozen")),
        "encoder_sha256": payload.get("encoder_sha256_final"),
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(counts)),
        "test_series_min": int(min(counts)),
        "test_series_median": float(np.median(counts)),
        "test_series_max": int(max(counts)),
        "tta_center_offsets": list(offsets),
        "metadata_repair": metadata_stats,
        "runtime_elapsed_hours": float(budget.elapsed_seconds / 3600.0),
        "runtime_budget_hours": float(budget.max_hours),
        **sample_validation,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)
    print(manifest_path)
    print(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-final-submit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    generate_final_submission(config, checkpoint=args.checkpoint, out_path=args.out)


if __name__ == "__main__":
    main()
