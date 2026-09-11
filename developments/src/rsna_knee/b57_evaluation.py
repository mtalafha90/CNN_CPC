"""Paired B57 AUCs, coverage, scanner bootstrap and one fixed ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b52_competition_training import macro_auc
from .b57_protocol import ARMS, VERSION, coverage, digest, load_protocol, sha256_file, write_json
from .constants import TARGETS


def read_predictions(root, arm, split, protocol_sha):
    root = Path(root)
    path = root / arm / f"{split}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    complete = json.loads((root / arm / "complete.json").read_text())
    if meta.get("version") != VERSION or meta.get("split") != split:
        raise ValueError("B57 prediction identity mismatch")
    contract = meta["contract"]
    if (contract != complete["contract"] or contract["arm"] != arm
            or contract["protocol_sha256"] != protocol_sha
            or contract["competition_supervised_ancestor"] is not False
            or contract["gold_studies_in_gradient"] != 0):
        raise ValueError("B57 prediction provenance mismatch")
    if complete["completed_epochs"] != contract["epochs"]:
        raise ValueError("B57 comparison requires both complete fixed endpoints")
    checkpoint_sha = sha256_file(root / arm / "final.pt")
    if meta["checkpoint_sha256"] != checkpoint_sha or complete["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("B57 prediction/checkpoint mismatch")
    if meta["npz_sha256"] != sha256_file(path):
        raise ValueError("B57 prediction file changed")
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in ("uids", "target", "weight", "prediction")}
    if digest(data["uids"].tolist()) != meta["uid_sha256"]:
        raise ValueError("B57 prediction UID hash mismatch")
    return data, contract


def align_pair(reference, candidate, expected_uids):
    """Align by UID, then demand identical full targets AND confidence masks."""
    out = []
    for data in (reference, candidate):
        uids = data["uids"].tolist()
        if len(uids) != len(set(uids)) or set(uids) != set(expected_uids):
            raise ValueError("B57 predictions must contain exactly the frozen UIDs")
        lookup = {uid: i for i, uid in enumerate(uids)}
        ix = [lookup[uid] for uid in expected_uids]
        aligned = {k: v[ix] for k, v in data.items()}
        for k in ("prediction", "target", "weight"):
            a = aligned[k]
            if a.shape != (len(expected_uids), len(TARGETS)) or not np.isfinite(a).all():
                raise ValueError(f"invalid B57 {k} shape/values")
        if ((aligned["prediction"] < 0) | (aligned["prediction"] > 1)).any():
            raise ValueError("B57 predictions must be probabilities")
        if (aligned["weight"] < 0).any():
            raise ValueError("B57 weights must be nonnegative")
        out.append(aligned)
    for k in ("target", "weight"):
        if not np.array_equal(out[0][k], out[1][k]):
            raise ValueError(f"B57 cannot compare different {k} surfaces")
    return out


def bootstrap_delta(target, weight, reference, candidate, *, groups, replicates=2000, seed=2026):
    """Resample the same scanner clusters for both models, preserving pairing.

    Replicates missing a class for any full-surface measurable target are
    counted and skipped. Do not silently change which targets define macro AUC.
    The interval is descriptive on this reused development surface, not a
    confidence interval for an unobserved Kaggle score.
    """
    groups = np.asarray(groups)
    if len(groups) != len(target) or replicates < 1:
        raise ValueError("invalid B57 bootstrap groups/replicates")
    unique = sorted(set(groups.tolist()))
    if len(unique) < 2:
        return {"ci95": None, "valid_replicates": 0, "skipped_replicates": replicates,
                "clusters": len(unique)}
    rows = [np.flatnonzero(groups == g) for g in unique]
    defined = macro_auc(target, weight, reference)["targets_defined"]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        ix = np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        a = macro_auc(target[ix], weight[ix], reference[ix])
        b = macro_auc(target[ix], weight[ix], candidate[ix])
        if a["targets_defined"] == defined and b["targets_defined"] == defined:
            delta = b["macro_auc"] - a["macro_auc"]
            if np.isfinite(delta):
                values.append(delta)
    return {"ci95": np.quantile(values, [.025, .975]).tolist() if len(values) >= replicates * .5 else None,
            "valid_replicates": len(values), "skipped_replicates": replicates - len(values),
            "clusters": len(unique)}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def compare(reference, candidate, *, profiles, config, expert=False):
    y, w, a, b = reference["target"], reference["weight"], reference["prediction"], candidate["prediction"]
    # Only this one blend is tested. No per-target weights, threshold search,
    # temperature search or use of Expert-58 to select the mixture.
    mixture = .5 * a + .5 * b
    scored = {"reference": macro_auc(y, w, a), "candidate": macro_auc(y, w, b),
              "equal_probability_ensemble": macro_auc(y, w, mixture)}
    primary = scored["reference"]["macro_auc"]
    deltas = {name: scored[name]["macro_auc"] - primary
              for name in ("candidate", "equal_probability_ensemble")}
    intervals = {name: bootstrap_delta(y, w, a, prob, groups=profiles,
                    replicates=config["bootstrap_replicates"], seed=config["seed"])
                 for name, prob in (("candidate", b), ("equal_probability_ensemble", mixture))}
    lower_positive = lambda name: intervals[name]["ci95"] is not None and intervals[name]["ci95"][0] > 0
    candidate_supported = not expert and deltas["candidate"] >= config["candidate_min_delta"] and lower_positive("candidate")
    ensemble_supported = (not expert and deltas["candidate"] >= -config["ensemble_candidate_tolerance"]
                          and deltas["equal_probability_ensemble"] >= config["ensemble_min_delta"]
                          and lower_positive("equal_probability_ensemble"))
    return finite_json({
        "scores": scored, "delta_vs_reference": deltas, "paired_intervals": intervals,
        "coverage": coverage(y, w),
        "per_target_candidate_delta": {name: scored["candidate"]["per_target_auc"][name]
                                       - scored["reference"]["per_target_auc"][name] for name in TARGETS},
        "mean_absolute_probability_difference": float(np.abs(a - b).mean()),
        "candidate_meets_development_gate": bool(candidate_supported),
        "ensemble_meets_development_gate": bool(ensemble_supported),
        "automatic_promotion_or_submission": False,
        "interpretation": ("Expert-58 diagnostic only; never used to select a model or blend" if expert else
                           "reused development evidence; confirm on another group-disjoint fold before promotion"),
    })


def evaluate(run_root):
    root = Path(run_root)
    p = load_protocol(root / "protocol")
    protocol_sha = sha256_file(root / "protocol" / "protocol.json")
    result = {"version": VERSION, "protocol_sha256": protocol_sha, "counts": p["counts"]}
    for split in ("validation", "expert"):
        a, ca = read_predictions(root, ARMS[0], split, protocol_sha)
        b, cb = read_predictions(root, ARMS[1], split, protocol_sha)
        for key in ("config_sha256", "source_digest", "training_uids_sha256",
                    "validation_uids_sha256", "epochs", "precision", "torch", "torchvision", "timm", "numpy"):
            if ca[key] != cb[key]:
                raise ValueError(f"B57 arm contract mismatch: {key}")
        uids = p["splits"]["validation"] if split == "validation" else p["expert_uids"]
        a, b = align_pair(a, b, uids)
        # Pair agreement alone is insufficient if both exports accidentally
        # used another teacher. Compare back to the frozen protocol itself.
        with np.load(root / "protocol" / "labels.npz", allow_pickle=False) as f:
            if split == "validation":
                lookup = {u: i for i, u in enumerate(f["uids"].tolist())}
                ix = [lookup[u] for u in uids]
                expected_y, expected_w = f["target"][ix], f["weight"][ix]
            else:
                lookup = {u: i for i, u in enumerate(f["expert_uids"].tolist())}
                raw = f["expert_target"][[lookup[u] for u in uids]]
                expected_y = np.nan_to_num(raw, nan=.5)
                expected_w = np.isfinite(raw).astype(np.float32)
        if not np.array_equal(a["target"], expected_y) or not np.array_equal(a["weight"], expected_w):
            raise ValueError("B57 predictions disagree with the frozen label/mask artifact")
        if split == "validation" and macro_auc(a["target"], a["weight"], a["prediction"])["targets_defined"] != 12:
            raise ValueError("B57 primary surface must define all 12 AUCs")
        profiles = [p["scanner_profiles"][u] for u in uids] if split == "validation" else uids
        result[split] = compare(a, b, profiles=profiles, config=p["config"], expert=split == "expert")
    write_json(root / "comparison.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/093_Experiment_B57_clean_backbone_comparison")
    args = parser.parse_args()
    evaluate(args.run_root)


if __name__ == "__main__":
    main()
