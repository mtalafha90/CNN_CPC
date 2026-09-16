"""Portable comparison with the completed B57 DINOv2 arm on another machine."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from . import VERSION
from .protocol import load, rsna_identity
from ..b57_protocol import ARMS, digest, load_protocol, sha256_file, write_json
from ..b57_evaluation import align_pair, bootstrap_delta, finite_json, read_predictions
from ..b52_competition_training import macro_auc


def candidate_predictions(root, split, p):
    root = Path(root)
    path = root/"finetune"/f"{split}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    complete = json.loads((root/"finetune/complete.json").read_text())
    identity = meta["contract"]
    if (meta["version"] != VERSION or meta["split"] != split or complete["version"] != VERSION
            or complete["contract"] != identity or identity["rsna_identity"] != p["rsna_identity"]
            or identity["protocol_sha256"] != sha256_file(root/"protocol.json")
            or complete["completed_epochs"] != identity["epochs"] or identity["gold_studies_in_gradient"] != 0):
        raise ValueError("B58 prediction identity mismatch")
    checkpoint = sha256_file(root/"finetune/final.pt")
    if checkpoint != meta["checkpoint_sha256"] or checkpoint != complete["checkpoint_sha256"]:
        raise ValueError("B58 prediction/checkpoint mismatch")
    if sha256_file(path) != meta["npz_sha256"]:
        raise ValueError("B58 prediction file changed")
    with np.load(path, allow_pickle=False) as f:
        values = {k: f[k].copy() for k in ("uids", "target", "weight", "prediction")}
    if digest(values["uids"].tolist()) != meta["uid_sha256"]:
        raise ValueError("B58 prediction UIDs changed")
    return values, identity


def verify_reference_boundary(candidate_root, reference_root, p):
    # A copied B57 protocol points to the 5090's old paths and source digest.
    # Verify its exported artifacts/hashes here, not unrelated local source files.
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    reference = load_protocol(reference_root/"protocol", verify_sources=False)
    candidate = load_protocol(candidate_root/"rsna_protocol")
    if rsna_identity(reference_root/"protocol", reference) != p["rsna_identity"]:
        raise ValueError("B57 and B58 differ in RSNA data, labels, masks, split or supervised recipe")
    return reference, candidate


def evaluate(root, reference_root):
    root, reference_root = Path(root), Path(reference_root)
    p = load(root)
    reference, rsna = verify_reference_boundary(root, reference_root, p)
    result = {"version": VERSION, "rsna_identity": p["rsna_identity"],
              "reference": "B57 dinov2_slice_candidate", "candidate": "B58 full-data native-label adaptation",
              "automatic_promotion_or_submission": False,
              "interpretation": "reused development comparison; extra SSL, native-label supervision and external images change together; confirm independently"}
    for split in ("validation", "expert"):
        a, ca = read_predictions(reference_root, ARMS[1], split, sha256_file(reference_root/"protocol/protocol.json"))
        b, cb = candidate_predictions(root, split, p)
        for key in ("epochs", "training_uids_sha256", "validation_uids_sha256"):
            if ca[key] != cb[key]:
                raise ValueError(f"B57/B58 comparison mismatch: {key}")
        if ca["config_sha256"] != cb["finetune_config_sha256"]:
            raise ValueError("B57/B58 supervised recipe mismatch")
        uids = rsna["splits"]["validation"] if split == "validation" else rsna["expert_uids"]
        a, b = align_pair(a, b, uids)
        with np.load(root/"rsna_protocol/labels.npz", allow_pickle=False) as f:
            if split == "validation":
                lookup = {u: i for i, u in enumerate(f["uids"].tolist())}
                ix = [lookup[u] for u in uids]
                y, w = f["target"][ix], f["weight"][ix]
            else:
                lookup = {u: i for i, u in enumerate(f["expert_uids"].tolist())}
                raw = f["expert_target"][[lookup[u] for u in uids]]
                y, w = np.nan_to_num(raw, nan=.5), np.isfinite(raw).astype(np.float32)
        if not np.array_equal(a["target"], y) or not np.array_equal(a["weight"], w):
            raise ValueError("B57/B58 predictions do not match frozen label/mask artifacts")
        sa, sb = macro_auc(y, w, a["prediction"]), macro_auc(y, w, b["prediction"])
        if split == "validation" and (sa["targets_defined"] != 12 or sb["targets_defined"] != 12):
            raise ValueError("B58 primary comparison must measure all 12 targets")
        groups = [rsna["scanner_profiles"][u] for u in uids] if split == "validation" else uids
        interval = bootstrap_delta(y, w, a["prediction"], b["prediction"], groups=groups,
                                   replicates=p["config"]["bootstrap_replicates"], seed=p["config"]["seed"])
        delta = sb["macro_auc"]-sa["macro_auc"]
        result[split] = {"reference": sa, "candidate": sb, "delta": delta, "paired_interval": interval,
            "development_support": bool(split == "validation" and delta >= p["config"]["candidate_min_delta"]
                                        and interval["ci95"] is not None and interval["ci95"][0] > 0),
            "runtime_differences": {k: [ca.get(k), cb.get(k)] for k in
                                    ("torch", "torchvision", "timm", "numpy", "precision") if ca.get(k) != cb.get(k)}}
    result = finite_json(result)
    write_json(root/"comparison_with_b57.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
