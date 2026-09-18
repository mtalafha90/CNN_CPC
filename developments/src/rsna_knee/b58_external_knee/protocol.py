"""B58's separate immutable run boundary and portable RSNA comparison identity."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from . import VERSION
from .data import (MAX_UNREADABLE_FRACTION, SOURCES, build_cache, index_fastmri, index_mrnet, index_oai,
                   index_rsna, select_records)
from ..b57_models import prepare_public_weights
from ..b57_protocol import (digest, load_protocol as load_rsna, prepare_protocol,
                           sha256_file, source_digest, write_json)


def implementation_digest():
    root = Path(__file__).parent
    return digest({"shared": source_digest(), "b58": {
        p.name: sha256_file(p) for p in sorted(root.glob("*.py"))}})


def rsna_identity(root, p):
    """Compare input bytes and semantic labels, allowing different mount paths."""
    with np.load(Path(root) / "labels.npz", allow_pickle=False) as f:
        labels = {k: np.nan_to_num(f[k], nan=-1).tolist() if f[k].dtype.kind == "f"
                  else f[k].tolist() for k in f.files}
    return digest({"input_hashes": {k: v["sha256"] for k, v in p["inputs"].items()},
                   "splits": p["split_uid_hashes"], "expert_uids": p["expert_uids"],
                   "labels": labels, "series_index": p["series_index_sha256"],
                   "profiles": p["scanner_profiles"], "config": p["config"],
                   "b42_config": p["b42_config"]})


def validate_config(c):
    if c.get("version") != VERSION or c.get("sources") != list(SOURCES):
        raise ValueError("B58 config identity/source set mismatch")
    if c.get("source_cycle") != ["rsna", "mrnet", "rsna", "fastmri", "rsna", "oai"]:
        raise ValueError("B58 v1 fixes source sampling at 1/2 RSNA, 1/6 each external source")
    for key in ("max_groups_per_source", "max_series_per_group", "cache_centres", "ssl_side",
                "ssl_steps", "ssl_batch_size", "ssl_trainable_blocks", "ssl_prototypes", "ssl_save_every",
                "bootstrap_replicates"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    if c["ssl_batch_size"] < 6 or c["ssl_side"] % 14:
        raise ValueError("B58 SSL batch must cover all sources, and image size must fit patch-14")
    for key in ("ssl_encoder_lr", "ssl_head_lr", "ssl_weight_decay", "ssl_student_temperature",
                "ssl_teacher_temperature", "candidate_min_delta"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"positive finite value required: {key}")
    for key in ("ssl_center_momentum", "ssl_teacher_momentum", "ssl_crop_min"):
        if not 0 < c[key] < 1:
            raise ValueError(f"{key} must be inside (0,1)")


def prepare(*, run_root, data_root, labels_root, domain_split, series_policy,
            mrnet_root, fastmri_root, oai_root,
            config_path="config/b58_external_knee_pretraining.json"):
    root = Path(run_root)
    c = json.loads(Path(config_path).read_text())
    validate_config(c)
    roots = {s: str(Path(p).resolve()) for s, p in
             (("mrnet", mrnet_root), ("fastmri", fastmri_root), ("oai", oai_root))}
    request = {"roots": roots, "rsna_paths": {k: str(Path(v).resolve()) for k, v in {
                   "data_root": data_root, "labels_root": labels_root,
                   "domain_split": domain_split, "series_policy": series_policy}.items()},
               "config_path": str(Path(config_path).resolve()),
               "config_sha256": sha256_file(config_path), "implementation": implementation_digest()}
    if (root / "protocol.json").exists():
        saved = load(root, verify_cache=True)
        if saved["request"] != request:
            raise ValueError("B58 frozen preparation request changed")
        return saved
    # Reusing B57's preparation code creates a NEW local RSNA boundary here.
    # It never opens or rewrites the active B57 run's protocol.
    p = prepare_protocol(data_root=data_root, labels_root=labels_root, domain_split=domain_split,
        series_policy=series_policy, b42_config="config/b42_constant_area_aspect_sparse.yaml",
        b57_config="config/b57_clean_backbone_comparison.json", out_root=root / "rsna_protocol")
    prepare_public_weights(root / "public_init")
    index = json.loads((root / "rsna_protocol" / "series_index.json").read_text())
    forbidden_series = {str(r["series_uid"]) for records in index.values() for r in records}
    rows = index_rsna(p, root / "rsna_protocol")
    rows += index_mrnet(roots["mrnet"])
    rows += index_fastmri(roots["fastmri"])
    rows += index_oai(roots["oai"], forbidden_studies=index, forbidden_series=forbidden_series)
    selected, counts = select_records(rows, c)
    selection = {"request": request, "rsna_identity": rsna_identity(root / "rsna_protocol", p),
                 "counts": counts, "records": selected}
    selection_path = root / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("B58 inputs/selection changed during preparation; preserve the partial run")
    else:
        write_json(selection_path, selection)
    estimate = len(selected) * c["cache_centres"] * 3 * c["ssl_side"]**2 * 2
    root.mkdir(parents=True, exist_ok=True)
    existing_bytes = sum(p.stat().st_size for p in (root / "cache").glob("*.npy"))
    if shutil.disk_usage(root).free < max(0, estimate - existing_bytes) + 2 * 2**30:
        raise OSError(f"B58 needs approximately {estimate/2**30:.1f} GiB of cache plus 2 GiB headroom")
    print(f"[B58] source counts: {json.dumps(counts)}", flush=True)
    print(f"[B58] bounded float16 cache estimate: {estimate/2**30:.1f} GiB", flush=True)
    # A series that would not decode is skipped rather than ending a multi-hour
    # prepare -- and the skip is frozen, not merely logged. The file is written
    # even when nothing failed, so its hash always exists and a later deletion
    # is a mismatch rather than an absence nobody notices.
    quarantine_path = root / "quarantine.json"
    cached = build_cache(selected, root / "cache", c, quarantine_path=quarantine_path)
    if not quarantine_path.exists():
        write_json(quarantine_path, {"unreadable": [], "by_source": {},
                                     "max_fraction": MAX_UNREADABLE_FRACTION})
    write_json(root / "cache_manifest.json", cached)

    # `counts` is what selection CHOSE. After quarantine it can overstate what
    # was actually cached, and the selected figure is the one that used to be
    # frozen alone -- a record saying 2,048 OAI series over 1,966 that existed.
    cached_counts = {}
    for row in cached:
        entry = cached_counts.setdefault(row["source"], {"cached_series": 0, "cached_groups": set()})
        entry["cached_series"] += 1
        entry["cached_groups"].add(row["group"])
    cached_counts = {source: {"cached_series": entry["cached_series"],
                              "cached_groups": len(entry["cached_groups"])}
                     for source, entry in cached_counts.items()}
    protocol = {"version": VERSION, "request": request, "config": c,
                "rsna_identity": selection["rsna_identity"], "rsna_counts": p["counts"],
                "rsna_protocol_sha256": sha256_file(root / "rsna_protocol" / "protocol.json"),
                "selection_sha256": sha256_file(selection_path), "source_counts": counts,
                "cached_counts": cached_counts,
                "quarantine_sha256": sha256_file(quarantine_path),
                "cache_manifest_sha256": sha256_file(root / "cache_manifest.json"),
                "external_labels_used": False, "ssl_uses_rsna_training_images": True,
                "validation_images_in_gradient": 0, "gold_studies_in_gradient": 0,
                "comparison_role": "extra mixed-source adaptation recipe versus B57 DINOv2; not an external-data-only ablation",
                "overlap_audit": "RSNA UIDs excluded from OAI; cached duplicates refused. Cross-dataset patient identity is not certified."}
    write_json(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sha256_file(root / "protocol.json") + "\n")
    print("[B58] preparation: COMPLETE", flush=True)
    return protocol


def load(root, *, verify_cache=False):
    root = Path(root)
    if sha256_file(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("B58 protocol hash mismatch")
    p = json.loads((root / "protocol.json").read_text())
    if p["version"] != VERSION or p["request"]["implementation"] != implementation_digest():
        raise ValueError("B58 source changed; use the implementation that prepared this run")
    if sha256_file(p["request"]["config_path"]) != p["request"]["config_sha256"]:
        raise ValueError("B58 config file changed")
    validate_config(p["config"])
    for name, key in (("selection.json", "selection_sha256"), ("cache_manifest.json", "cache_manifest_sha256"),
                      ("quarantine.json", "quarantine_sha256"),
                      ("rsna_protocol/protocol.json", "rsna_protocol_sha256")):
        # Only what this protocol actually recorded. `quarantine_sha256` was
        # added after the first runs were frozen, and demanding it here would
        # make every earlier protocol unloadable -- punishing an old run for a
        # guard it predates. Once a protocol carries the key it is enforced.
        if key not in p:
            continue
        if sha256_file(root / name) != p[key]:
            raise ValueError(f"B58 frozen artifact changed: {name}")
    rsna = load_rsna(root / "rsna_protocol")
    if rsna_identity(root / "rsna_protocol", rsna) != p["rsna_identity"]:
        raise ValueError("B58 RSNA identity mismatch")
    if verify_cache:
        for row in json.loads((root / "cache_manifest.json").read_text()):
            if sha256_file(row["cache"]) != row["cache_sha256"]:
                raise ValueError(f"B58 cached pixels changed: {row['cache']}")
    return p
