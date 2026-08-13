from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .b7_weak_supervision import _read_config
from .b17_training import encoder_state_sha256
from .b20_crop_focus import B20_EXPERIMENT, B20_VARIANT, load_b20_checkpoint
from .b21_acceptance_protocol import B20_CANONICAL_EPOCH
from .b21_contract import require_b21_contract
from .b21_gold_decision import gold_acceptance_metrics
from .b21_gold_setup import gold_surface, load_b21_full, make_gold_datasets, predict_gold
from .constants import TARGET_SLUGS
from .runtime import resolve_runtime


def run(config, b20_checkpoint, b21_checkpoint, out_root):
    require_b21_contract(config)
    out = Path(out_root)
    if out.exists():
        raise RuntimeError("acceptance output already exists; one recorded acceptance look only")
    runtime = resolve_runtime(config)
    m20, p20 = load_b20_checkpoint(b20_checkpoint, device=runtime.device)
    if p20.get("variant") != B20_VARIANT or p20.get("experiment") != B20_EXPERIMENT or int(p20.get("selected_epoch", -1)) != B20_CANONICAL_EPOCH:
        raise ValueError("invalid historical B20 control")
    m21, p21 = load_b21_full(b21_checkpoint, runtime.device)
    if p20.get("model_spec") != p21.get("model_spec"):
        raise ValueError("B20/B21 model specs differ")
    sha20, sha21 = encoder_state_sha256(m20.encoder), encoder_state_sha256(m21.encoder)
    if sha20 != sha21:
        raise ValueError("B20/B21 frozen encoder SHA differs")
    root, uids, truth, index, counts, metadata = gold_surface(config)
    offsets, ds20, ds21 = make_gold_datasets(config, root, uids, index)
    seed = int(config.get("seed", 2026))
    print("[B21 acceptance] replaying historical B20")
    y20 = predict_gold(m20, ds20, uids, runtime, seed + 27_100_000)
    print("[B21 acceptance] evaluating frozen full-data B21 once")
    y21 = predict_gold(m21, ds21, uids, runtime, seed + 27_100_000)
    result = gold_acceptance_metrics(truth, y20, y21, b20_checkpoint=b20_checkpoint, b21_checkpoint=b21_checkpoint, encoder_sha256=sha21, n_bootstrap=int(config.get("b7_n_bootstrap", 5000)), seed=seed)
    result.update({"n_gold_studies": len(uids), "gold_series_total": int(sum(counts)), "tta_center_offsets": list(offsets), "metadata_repair": metadata})
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame({"StudyInstanceUID": uids})
    for j, slug in enumerate(TARGET_SLUGS):
        frame[f"b20_{slug}"] = y20[:, j]; frame[f"b21_{slug}"] = y21[:, j]
    frame.to_csv(out / "paired_predictions.csv", index=False)
    (out / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(out / "acceptance.json")
    return result


def main():
    p = argparse.ArgumentParser("rsna-knee-b21-accept")
    p.add_argument("--config", required=True); p.add_argument("--data-root", default=None)
    p.add_argument("--b20-checkpoint", required=True); p.add_argument("--b21-checkpoint", required=True)
    p.add_argument("--out-root", default="runs/b21_full_acceptance/gold_acceptance")
    a = p.parse_args(); c = _read_config(a.config)
    if a.data_root: c = dict(c); c["data_root"] = a.data_root
    run(c, a.b20_checkpoint, a.b21_checkpoint, a.out_root)

if __name__ == "__main__": main()
