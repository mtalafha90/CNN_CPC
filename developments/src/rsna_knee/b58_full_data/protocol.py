"""Write-once full-data B58 protocol, with independent preservation of v1/B57."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import VERSION
from .data import complete_inventory, freeze_pixels, build_cases, label_coverage
from .labels import binary_task, mrnet_labels, fastmri_labels, oai_labels
from ..b58_external_knee.data import SOURCES, index_rsna, index_mrnet, index_fastmri, index_oai
from ..b58_external_knee.protocol import implementation_digest as inherited_digest, rsna_identity
from ..b57_protocol import digest, sha256_file, write_json, prepare_protocol, load_protocol
from ..b57_models import prepare_public_weights
from ..constants import TARGETS


def implementation_digest():
    return digest({"inherited": inherited_digest(), "full_data": {
        p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}})


def validate_config(c):
    if c.get("version") != VERSION or c.get("sources") != list(SOURCES):
        raise ValueError("full-data B58 config identity mismatch")
    if any(k in c for k in ("max_groups_per_source", "max_series_per_group", "ssl_steps")):
        raise ValueError("full-data training cannot use subset/step caps")
    for key in ("ssl_epochs", "ssl_batch_size", "ssl_side", "ssl_prototypes", "save_every",
                "supervised_epochs", "slices_per_series", "supervised_side", "encoder_chunk_size", "mrnet_expected_exams"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    if c["ssl_side"] % 14 or c["supervised_side"] % 14 or c["ssl_trainable_blocks"] != 12:
        raise ValueError("full run trains all twelve DINOv2 blocks on patch-14 compatible images")
    for key in ("ssl_encoder_lr", "ssl_head_lr", "ssl_weight_decay", "encoder_lr", "head_lr", "weight_decay",
                "ssl_student_temperature", "ssl_teacher_temperature", "candidate_min_delta"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"positive finite value required: {key}")
    for key in ("ssl_center_momentum", "ssl_teacher_momentum", "ssl_crop_min"):
        if not 0 < c[key] < 1:
            raise ValueError(f"{key} must be inside (0,1)")


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen or partially prepared input changed: {path}")
    else:
        write_json(path, value)


def prepare(*, run_root, data_root, labels_root, domain_split, series_policy,
            mrnet_root, fastmri_root, oai_root, fastmri_annotations, fastmri_reviewed,
            oai_series, oai_labels_csv, oai_tasks,
            config_path="config/b58_full_data.json"):
    root = Path(run_root)
    paths = {k: str(Path(v).resolve()) for k, v in locals().copy().items()
             if k not in ("root", "run_root")}
    c = json.loads(Path(config_path).read_text())
    validate_config(c)
    request = {"paths": paths, "implementation": implementation_digest(), "config_sha256": sha256_file(config_path)}
    if (root / "protocol.json").exists():
        saved = load(root)
        if saved["request"] != request:
            raise ValueError("full-data B58 preparation request changed")
        return saved
    # Fail on missing external label files before scanning thousands of DICOMs.
    for path in (fastmri_annotations, fastmri_reviewed, oai_series, oai_labels_csv, oai_tasks):
        if not Path(path).is_file():
            raise FileNotFoundError(f"full-data supervised run requires: {path}")
    write_once(root / "preparation_request.json", request)
    rsna = prepare_protocol(data_root=data_root, labels_root=labels_root, domain_split=domain_split,
        series_policy=series_policy, b42_config="config/b42_constant_area_aspect_sparse.yaml",
        b57_config="config/b57_clean_backbone_comparison.json", out_root=root / "rsna_protocol")
    index = json.loads((root / "rsna_protocol/series_index.json").read_text())
    rows = index_rsna(rsna, root / "rsna_protocol")
    mrnet = index_mrnet(mrnet_root)
    fastmri = index_fastmri(fastmri_root)
    oai = index_oai(oai_root, forbidden_studies=index,
                    forbidden_series={r["series_uid"] for entries in index.values() for r in entries})
    tasks, tables, reports, inputs = {}, {}, {}, []
    tasks["mrnet"], tables["mrnet"], files, reports["mrnet"] = mrnet_labels(mrnet_root, mrnet, c["mrnet_expected_exams"])
    inputs += files
    tasks["fastmri"], tables["fastmri"], files, reports["fastmri"] = fastmri_labels(fastmri_annotations, fastmri_reviewed, fastmri)
    inputs += files
    tasks["oai"], tables["oai"], oai, files, reports["oai"] = oai_labels(oai_series, oai_labels_csv, oai_tasks, oai)
    inputs += files
    tasks["rsna"] = [binary_task(name) for name in TARGETS]
    with np.load(root / "rsna_protocol/labels.npz", allow_pickle=False) as f:
        lookup = {uid: i for i, uid in enumerate(f["uids"].tolist())}
        tables["rsna"] = {uid: {"values": f["target"][lookup[uid]].tolist(),
                               "weights": f["weight"][lookup[uid]].tolist()} for uid in rsna["splits"]["train"]}
    rows, counts = complete_inventory(rows + mrnet + fastmri + oai)
    if {r["group"] for r in rows if r["source"] == "rsna"} != set(rsna["splits"]["train"]):
        raise ValueError("some RSNA training studies have no eligible MRI series")
    manifest = {"records": rows, "tasks": tasks, "cases": build_cases(rows, tables, tasks),
                "label_inputs": {str(path.resolve()): sha256_file(path) for path in inputs}}
    write_once(root / "selection.json", manifest)
    manifest["records"] = freeze_pixels(rows, root / "volume_audit")
    write_once(root / "manifest.json", manifest)
    coverage = label_coverage(manifest["cases"], tasks)
    prepare_public_weights(root / "public_init")
    protocol = {"version": VERSION, "request": request, "config": c,
        "rsna_identity": rsna_identity(root / "rsna_protocol", rsna), "rsna_counts": rsna["counts"],
        "rsna_protocol_sha256": sha256_file(root / "rsna_protocol/protocol.json"),
        "selection_sha256": sha256_file(root / "selection.json"), "manifest_sha256": sha256_file(root / "manifest.json"),
        "counts": counts, "label_reports": reports, "label_coverage": coverage,
        "external_labels_used": True, "validation_images_in_gradient": 0, "gold_studies_in_gradient": 0,
        "ssl_steps": len(rows) * c["ssl_epochs"],
        "supervised_steps": sum(sum(case["weights"]) > 0 for case in manifest["cases"]) * c["supervised_epochs"],
        "scope": "all eligible series in provided training inventories; sampled slice centres, not every voxel",
        "overlap_audit": "OAI participant partitions and RSNA UID guards; exact pixel duplicates refused; cross-source patient identity not certified"}
    write_once(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sha256_file(root / "protocol.json") + "\n")
    print(json.dumps({"counts": counts, "labels": coverage, "ssl_steps": protocol["ssl_steps"],
                      "supervised_steps": protocol["supervised_steps"]}, indent=2), flush=True)
    print("[B58 full] preparation COMPLETE", flush=True)
    return protocol


def load(root):
    root = Path(root)
    if sha256_file(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("full-data protocol hash mismatch")
    p = json.loads((root / "protocol.json").read_text())
    if p["version"] != VERSION or p["request"]["implementation"] != implementation_digest():
        raise ValueError("full-data B58 source identity changed")
    validate_config(p["config"])
    if sha256_file(p["request"]["paths"]["config_path"]) != p["request"]["config_sha256"]:
        raise ValueError("full-data B58 config file changed")
    for file, key in (("selection.json", "selection_sha256"), ("manifest.json", "manifest_sha256"),
                      ("rsna_protocol/protocol.json", "rsna_protocol_sha256")):
        if sha256_file(root / file) != p[key]:
            raise ValueError(f"full-data frozen artifact changed: {file}")
    manifest = json.loads((root / "manifest.json").read_text())
    for path, fingerprint in manifest["label_inputs"].items():
        if sha256_file(path) != fingerprint:
            raise ValueError(f"native label input changed: {path}")
    rsna = load_protocol(root / "rsna_protocol")
    if rsna_identity(root / "rsna_protocol", rsna) != p["rsna_identity"]:
        raise ValueError("full-data RSNA boundary changed")
    return p
