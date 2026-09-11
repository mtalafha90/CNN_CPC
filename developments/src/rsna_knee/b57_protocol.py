"""B57 data and provenance boundary; never load a competition-trained parent."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGETS

VERSION = "b57_v1"
ARMS = ("clean_b52_reference", "dinov2_slice_candidate")
VALID = "validation_unseen_scanners"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_digest():
    # Include transitive historical dependencies, not just the new trainer.
    root = Path(__file__).parent
    return digest({p.name: sha256_file(p) for p in sorted(root.glob("*.py"))})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def coverage(target, weight):
    result = {}
    for j, name in enumerate(TARGETS):
        active = weight[:, j] > 0
        result[name] = {
            "positive": int((active & (target[:, j] > .5)).sum()),
            "negative": int((active & (target[:, j] < .5)).sum()),
            "unlabelled": int((~active).sum()),
        }
    return result


def partition(uids, rows, expert_uids):
    """Keep the old development UIDs; exclude their profiles from ALL training.

    B50's original check covered train/seen but not excluded_prior_surface.
    Adding that pool back must not undo scanner separation. The excluded rows
    here are neither new validation labels nor training examples.
    """
    if len(uids) != len(set(uids)) or set(uids) & set(expert_uids):
        raise ValueError("duplicate report UIDs or expert overlap")
    needed = ["StudyInstanceUID", "scanner_profile", "b50_split"]
    if not set(needed) <= set(rows) or rows[needed].isna().any().any():
        raise ValueError("split has missing UIDs, profiles or assignments")
    rows = rows.copy()
    rows["StudyInstanceUID"] = rows.StudyInstanceUID.astype(str)
    if rows.StudyInstanceUID.duplicated().any() or set(rows.StudyInstanceUID) != set(uids):
        raise ValueError("split/report UID population mismatch")
    rows = rows.set_index("StudyInstanceUID").loc[uids]
    profiles = rows.scanner_profile.astype(str).to_numpy()
    if any(not x.strip() for x in profiles):
        raise ValueError("blank scanner profile")
    validation = rows.b50_split.eq(VALID).to_numpy()
    held_profiles = set(profiles[validation])
    train = ~validation & ~np.isin(profiles, list(held_profiles))
    excluded = ~validation & ~train
    if not train.any() or not validation.any():
        raise ValueError("B57 needs nonempty train and validation")
    return {"train": np.flatnonzero(train), "validation": np.flatnonzero(validation),
            "excluded_profile_overlap": np.flatnonzero(excluded)}, profiles


def prepare_protocol(*, data_root, labels_root, domain_split, series_policy,
                     b42_config, b57_config, out_root):
    """Freeze once; repeated identical preparation verifies and returns it."""
    import yaml
    from .b12_training import _load_series_policy
    from .b12_variable_series import audit_variable_series_surface
    from .b13_training import B13_SERIES_SIGNATURE
    from .b42_constant_area_aspect_sparse_mil import require_b42_contract
    from .b50_ordered_slice_selection_split import verify_b50_selection_split
    from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
    from .phase9_supervision import load_fill_merged_export, prepare_all_report_only_supervision

    root, labels, gate, out = map(Path, (data_root, labels_root, domain_split, out_root))
    if gate.is_file():
        gate = gate.parent
    inputs = {
        "train_csv": root / "train.csv", "train_series_csv": root / "train_series.csv",
        "training_targets": labels / "training_targets.csv", "label_policy": labels / "policy.json",
        "label_audit": labels / "audit.json", "series_policy": Path(series_policy),
        "gate_json": gate / "b50_selection_split.json",
        "gate_rows": gate / "b50_selection_split_by_study.csv",
        "gate_hash": gate / "b50_selection_split.sha256",
        "b42_config": Path(b42_config), "b57_config": Path(b57_config),
    }
    fingerprints = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for name, path in inputs.items()}
    if (out / "protocol.json").exists():
        saved = load_protocol(out)
        if saved["inputs"] != fingerprints:
            raise ValueError("B57 inputs changed; keep the frozen run and its original inputs")
        return saved
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"incomplete/nonempty protocol directory: {out}; inspect it before retrying")

    config = json.loads(inputs["b57_config"].read_text())
    if config["version"] != VERSION or config["arms"] != list(ARMS):
        raise ValueError("B57 config identity mismatch")
    if config["augmentation"] or config["selection"] != "fixed_final_epoch":
        raise ValueError("B57 v1 uses deterministic pixels and a fixed final endpoint")
    for key, value in {"candidate_slice_layers": 1, "candidate_dim": 384,
                       "candidate_slice_position_embedding": False,
                       "candidate_aux_weight": 0.0}.items():
        if config[key] != value:
            raise ValueError(f"B57 v1 implements {key}={value}; config must describe the actual model")
    for key in ("epochs", "batch_size", "encoder_chunk_size", "encoder_lr", "head_lr",
                "grad_clip", "bootstrap_replicates"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"B57 requires positive finite {key}")
    settings = yaml.safe_load(inputs["b42_config"].read_text())
    require_b42_contract(settings)
    gate_payload = json.loads(inputs["gate_json"].read_text())
    if inputs["gate_hash"].read_text().strip().split()[0] != fingerprints["gate_json"]["sha256"]:
        raise ValueError("B50 gate JSON/hash mismatch")
    for key, source in (("source_train_csv_sha256", "train_csv"),
                        ("source_training_targets_sha256", "training_targets")):
        if gate_payload.get(key) != fingerprints[source]["sha256"]:
            raise ValueError(f"B57 requires the original gate/teacher: {key} mismatch")
    rows = pd.read_csv(inputs["gate_rows"], dtype={"StudyInstanceUID": str})
    verify_b50_selection_split(rows)
    actual_profiles = set(rows.loc[rows.b50_split.eq(VALID), "scanner_profile"].astype(str))
    if actual_profiles != set(gate_payload["unseen_scanner_profiles"]):
        raise ValueError("gate JSON and CSV disagree about validation profiles")
    train = load_train_csv(inputs["train_csv"])
    frame, _, _ = load_fill_merged_export(labels)
    uids, target, weight, summary = prepare_all_report_only_supervision(train, frame)
    if summary["usable_cells"] != 34010:
        raise ValueError("B57 v1 requires the original 34,010-cell teacher, not a regrade")
    if not np.isfinite(target).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("invalid supervision")
    experts = train.loc[gold_mask(train)].copy()
    expert_uids = experts.StudyInstanceUID.tolist()
    indices, profiles = partition(uids, rows, expert_uids)
    audits = {name: coverage(target[ix], weight[ix]) for name, ix in indices.items()}
    for name in ("train", "validation"):
        if any(min(v["positive"], v["negative"]) == 0 for v in audits[name].values()):
            raise ValueError(f"B57 {name} cannot measure/train all 12 targets")
    policy = _load_series_policy(inputs["series_policy"])
    if policy.get("series_summary", {}).get("series_signature_sha256") != B13_SERIES_SIGNATURE:
        raise ValueError("B57 requires the B12/B13 series policy")
    series = load_series_csv(inputs["train_series_csv"])
    series, repair = backfill_series_metadata(series, root, split="train")
    _, index = audit_variable_series_surface(series, uids + expert_uids)
    if any(not index.get(uid) for uid in uids + expert_uids):
        raise ValueError("study with no eligible MRI series")
    split_uids = {name: [uids[int(i)] for i in ix] for name, ix in indices.items()}
    # Actual source manifests certify byte identity and ancestry through our
    # training path, not an independent audit of public pretraining datasets.
    protocol = {
        "version": VERSION, "config": config, "b42_config": settings,
        "inputs": fingerprints, "source_digest": source_digest(),
        "data_root": str(root.resolve()), "splits": split_uids,
        "split_uid_hashes": {k: digest(v) for k, v in split_uids.items()},
        "scanner_profiles": dict(zip(uids, profiles.tolist())),
        "counts": {k: len(v) for k, v in split_uids.items()}, "coverage": audits,
        "expert_uids": expert_uids, "metadata_repair": repair,
        "validation_role": "reused development surface; not an untouched test",
        "grouping": "scanner profile; patient identities are unavailable in this contract",
        "initialization": "public encoder weights only; fresh competition heads",
        "competition_supervised_ancestor": False, "gold_studies_in_gradient": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "series_index.json", index)
    protocol["series_index_sha256"] = sha256_file(out / "series_index.json")
    np.savez_compressed(out / "labels.npz", uids=np.asarray(uids), target=target, weight=weight,
                        expert_uids=np.asarray(expert_uids),
                        expert_target=experts[TARGETS].to_numpy(np.float32))
    protocol["labels_sha256"] = sha256_file(out / "labels.npz")
    write_json(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(sha256_file(out / "protocol.json") + "\n")
    return protocol


def load_protocol(root, *, verify_sources=True):
    root = Path(root)
    if sha256_file(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("B57 protocol/hash mismatch")
    p = json.loads((root / "protocol.json").read_text())
    if p["version"] != VERSION or p["competition_supervised_ancestor"] is not False:
        raise ValueError("B57 requires its own clean protocol")
    for name, key in (("labels.npz", "labels_sha256"), ("series_index.json", "series_index_sha256")):
        if sha256_file(root / name) != p[key]:
            raise ValueError(f"B57 frozen artifact changed: {name}")
    if verify_sources:
        if source_digest() != p["source_digest"]:
            raise ValueError("B57 source changed; resume using the code that froze this protocol")
        for name, item in p["inputs"].items():
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"B57 source input changed: {name}")
    return p


def require_same_run(saved, expected):
    if saved != expected:
        keys = sorted(k for k in set(saved) | set(expected) if saved.get(k) != expected.get(k))
        raise ValueError(f"B57 resume/comparison contract mismatch: {keys}")
