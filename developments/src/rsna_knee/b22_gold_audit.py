from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .b7_weak_supervision import _read_config
from .b17_training import encoder_state_sha256
from .b20_crop_focus import B20_EXPERIMENT, B20_VARIANT, load_b20_checkpoint
from .b21_acceptance_protocol import B20_CANONICAL_EPOCH
from .b21_gold_setup import gold_surface, make_gold_datasets, predict_gold
from .b22_duration_contract import require_b22_duration_contract
from .b22_duration_protocol import B22_EPOCHS, require_failed_b21_acceptance
from .b22_gold_metrics import build_b22_gold_trajectory
from .b22_gold_setup import load_b22_epoch
from .constants import TARGET_SLUGS
from .runtime import resolve_runtime


def run_b22_gold_audit(
    config: dict,
    *,
    b20_checkpoint: str | Path,
    candidate_root: str | Path,
    b21_acceptance: str | Path,
    out_root: str | Path,
) -> dict:
    require_b22_duration_contract(config)
    require_failed_b21_acceptance(b21_acceptance)
    out = Path(out_root)
    if out.exists():
        raise RuntimeError("B22 gold trajectory output already exists; exploratory audit is single-record")

    runtime = resolve_runtime(config)
    model20, payload20 = load_b20_checkpoint(b20_checkpoint, device=runtime.device)
    if (
        payload20.get("variant") != B20_VARIANT
        or payload20.get("experiment") != B20_EXPERIMENT
        or int(payload20.get("selected_epoch", -1)) != B20_CANONICAL_EPOCH
    ):
        raise ValueError("invalid historical B20 control")
    sha20 = encoder_state_sha256(model20.encoder)

    models = {}
    payloads = {}
    candidate_root = Path(candidate_root)
    for epoch in range(1, B22_EPOCHS + 1):
        model, payload = load_b22_epoch(candidate_root / f"epoch_{epoch}.pt", epoch, runtime.device)
        if encoder_state_sha256(model.encoder) != sha20:
            raise ValueError(f"B22 epoch {epoch} encoder SHA differs from B20")
        if payload.get("model_spec") != payload20.get("model_spec"):
            raise ValueError(f"B22 epoch {epoch} model spec differs from B20")
        models[epoch] = model
        payloads[epoch] = payload

    root, uids, truth, index, counts, metadata = gold_surface(config)
    offsets, ds20, ds22 = make_gold_datasets(config, root, uids, index)
    seed = int(config.get("seed", 2026))
    print("[B22 duration audit] replaying B20 once")
    pred20 = predict_gold(model20, ds20, uids, runtime, seed + 27_100_000)

    epoch_predictions = {}
    for epoch in range(1, B22_EPOCHS + 1):
        print(f"[B22 duration audit] predicting pre-resize epoch {epoch}")
        epoch_predictions[epoch] = predict_gold(
            models[epoch], ds22, uids, runtime, seed + 27_100_000
        )

    result = build_b22_gold_trajectory(
        truth,
        pred20,
        epoch_predictions,
        b21_acceptance_path=b21_acceptance,
        n_bootstrap=int(config.get("b7_n_bootstrap", 5000)),
        seed=seed,
    )
    result.update({
        "b20_checkpoint": str(Path(b20_checkpoint)),
        "candidate_root": str(candidate_root),
        "n_gold_studies": len(uids),
        "gold_series_total": int(sum(counts)),
        "tta_center_offsets": list(offsets),
        "encoder_sha256": sha20,
        "metadata_repair": metadata,
        "training_history": {
            str(epoch): payloads[epoch]["history"][-1] for epoch in range(1, B22_EPOCHS + 1)
        },
    })

    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for j, slug in enumerate(TARGET_SLUGS):
        frame[f"b20_{slug}"] = pred20[:, j]
        for epoch in range(1, B22_EPOCHS + 1):
            frame[f"e{epoch}_{slug}"] = epoch_predictions[epoch][:, j]
    frame.to_csv(out / "trajectory_predictions.csv", index=False)
    (out / "trajectory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out / "trajectory.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b22-gold-audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b20-checkpoint", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--b21-acceptance", required=True)
    parser.add_argument("--out-root", default="runs/b22_duration_audit/gold_trajectory")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    run_b22_gold_audit(
        config,
        b20_checkpoint=args.b20_checkpoint,
        candidate_root=args.candidate_root,
        b21_acceptance=args.b21_acceptance,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
