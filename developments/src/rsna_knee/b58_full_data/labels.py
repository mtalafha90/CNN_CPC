"""Explicit native targets; an absent annotation is never silently a negative."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np


def csv_rows(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
            raise ValueError(f"{path}: required CSV columns {required}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate column names")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV rows")
    return [{k: v.strip() for k, v in row.items()} for row in rows]


def binary_task(name):
    return {"name": name, "kind": "binary"}


def mrnet_labels(root, records, expected_exams=1130):
    root = Path(root)
    if (root / "MRNet-v1.0").is_dir():
        root /= "MRNet-v1.0"
    if root.name == "train":
        root = root.parent
    tasks = [binary_task(n) for n in ("abnormal", "acl", "meniscus")]
    tables, inputs, repairs = [], [], []
    for task in tasks:
        path = root / f"train-{task['name']}.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
        if not rows or any(len(row) != 2 for row in rows):
            raise ValueError(f"{path}: expected two columns: exam ID, binary label")
        # Redivis can promote the first headerless observation into column names.
        # Recover ONLY when both names encode an exact valid observation.
        first = [x.strip() for x in rows[0]]
        if re.fullmatch(r"_\d{4}", first[0]) and re.fullmatch(r"_[01]", first[1]):
            rows[0] = [first[0][1:], first[1][1:]]
            repairs.append({"file": str(path), "recovered_first_row": rows[0]})
        elif [v.lower() for v in first] in (["exam_id", "label"], ["id", "label"]):
            rows = rows[1:]
        table = {}
        for raw_uid, raw_value in rows:
            if not re.fullmatch(r"\d{1,4}", raw_uid.strip()) or raw_value.strip() not in ("0", "1"):
                raise ValueError(f"{path}: invalid MRNet exam/label {raw_uid!r}, {raw_value!r}")
            uid = raw_uid.strip().zfill(4)
            if uid in table:
                raise ValueError(f"{path}: duplicate MRNet exam {uid}")
            table[uid] = int(raw_value)
        tables.append(table)
        inputs.append(path)
    groups = defaultdict(set)
    for row in records:
        groups[row["group"]].add(row["series"].split("/")[-1])
    expected = {str(i).zfill(4) for i in range(expected_exams)}
    if set(groups) != expected or any(set(t) != expected for t in tables):
        raise ValueError(f"full MRNet run requires all {expected_exams} training exams and all three label tables; "
                         f"images={len(groups)}, labels={[len(t) for t in tables]}")
    if any(planes != {"axial", "coronal", "sagittal"} for planes in groups.values()):
        raise ValueError("MRNet training exams must have all three planes")
    labels = {uid: {"values": [t[uid] for t in tables], "weights": [1.] * 3} for uid in sorted(groups)}
    return tasks, labels, inputs, {"recovered_rows": repairs, "labelled_exams": len(labels)}


def fastmri_labels(annotation_csv, reviewed_csv, records):
    annotations = csv_rows(annotation_csv, ("file", "slice", "study_level", "label"))
    with Path(reviewed_csv).open(newline="", encoding="utf-8-sig") as handle:
        reviewed_rows = list(csv.reader(handle))
    if reviewed_rows and reviewed_rows[0] == ["file"]:
        reviewed_rows = reviewed_rows[1:]
    if any(len(row) != 1 or not row[0].strip() for row in reviewed_rows):
        raise ValueError("fastMRI+ reviewed file list must have one filename per row")
    reviewed = {Path(row[0].strip()).stem for row in reviewed_rows}
    if len(reviewed) != len(reviewed_rows):
        raise ValueError("duplicate fastMRI+ reviewed file IDs")
    findings = defaultdict(set)
    for row in annotations:
        uid = Path(row["file"]).stem
        if uid not in reviewed or not row["label"]:
            raise ValueError(f"fastMRI+ annotation lacks review identity or finding: {uid}")
        findings[uid].add(row["label"])
    # Use the release vocabulary, never rename meniscus into a medial/lateral label.
    names = sorted({label for values in findings.values() for label in values})
    if not names:
        raise ValueError("empty fastMRI+ diagnosis vocabulary")
    tasks = [binary_task(name) for name in names]
    labels, counts = {}, defaultdict(int)
    for record in records:
        uid = record["group"]
        present = findings.get(uid, set())
        no_findings = uid in reviewed and not present
        labels[uid] = {"values": [float(n in present) for n in names],
                       "weights": [float(n in present or no_findings) for n in names]}
        counts["positive_review" if present else "reviewed_no_findings" if no_findings else "unreviewed"] += 1
    if not counts["positive_review"] or not counts["reviewed_no_findings"]:
        raise ValueError("full fastMRI+ training needs positive annotations and reviewed no-finding scans")
    return tasks, labels, [Path(annotation_csv), Path(reviewed_csv)], dict(counts)


def validate_tasks(tasks):
    if not tasks or len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("OAI needs uniquely named native tasks")
    for task in tasks:
        if not task["name"].strip() or not task.get("provenance", "").strip():
            raise ValueError("each OAI task needs its release/data-dictionary provenance")
        if task["kind"] == "categorical":
            values = task.get("classes", [])
            if len(values) < 2 or len(set(values)) != len(values) or any(not isinstance(x, str) for x in values):
                raise ValueError("categorical classes must be unique release values as strings")
        elif task["kind"] == "regression":
            lo, hi = task["minimum"], task["maximum"]
            if not np.isfinite([lo, hi]).all() or not lo < hi:
                raise ValueError("continuous tasks need fixed finite data-dictionary bounds")
        elif task["kind"] != "binary":
            raise ValueError(f"unsupported native task kind {task['kind']}")


def parse_value(raw, task):
    if raw in ("", *task.get("missing_values", [])):
        return 0., 0.
    if task["kind"] == "categorical":
        if raw not in task["classes"]:
            raise ValueError(f"unknown native class {raw!r} for {task['name']}")
        return task["classes"].index(raw), 1.
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"nonfinite observed label for {task['name']}")
    if task["kind"] == "binary":
        if value not in (0., 1.):
            raise ValueError(f"binary task {task['name']} requires 0/1 or explicitly missing")
    else:
        lo, hi = task["minimum"], task["maximum"]
        if not lo <= value <= hi:
            raise ValueError(f"out-of-range {task['name']}: {value}")
        value = (value - lo) / (hi - lo)
    return value, 1.


def oai_labels(series_csv, labels_csv, tasks_json, records):
    """Join release labels through a reviewed participant/visit/knee series map.

    No inference of visit or laterality from folder names or acquisition dates.
    All loaded series must be accounted for, including declared held-out ones.
    """
    schema = json.loads(Path(tasks_json).read_text())
    tasks = schema["tasks"]
    validate_tasks(tasks)
    identity = ("patient_id", "visit", "side")
    mapping = csv_rows(series_csv, (*identity, "series_uid", "partition"))
    observed = csv_rows(labels_csv, (*identity, *(t["name"] for t in tasks)))
    if any(set(row) != set(identity) | {t["name"] for t in tasks} for row in observed):
        raise ValueError("OAI label CSV has undeclared columns; declare every target in tasks.json or remove metadata explicitly")
    series, participants, cases = {}, {}, {}
    for row in mapping:
        key = tuple(row[k] for k in identity)
        if not all(key) or key[-1] not in ("L", "R") or row["partition"] not in ("train", "validation", "test"):
            raise ValueError("OAI map needs patient, explicit visit, L/R knee and train/validation/test partition")
        if row["series_uid"] in series:
            raise ValueError("duplicate OAI series mapping")
        if participants.setdefault(key[0], row["partition"]) != row["partition"]:
            raise ValueError("OAI participant crosses training/held-out partitions")
        cases[key] = row["partition"]
        series[row["series_uid"]] = row
    labels = {}
    for row in observed:
        key = tuple(row[k] for k in identity)
        if not all(key) or key[-1] not in ("L", "R") or key in labels:
            raise ValueError("invalid/duplicate OAI participant-visit-knee label row")
        values = [parse_value(row[t["name"]], t) for t in tasks]
        labels[key] = {"values": [v for v, w in values], "weights": [w for v, w in values]}
    kept, case_labels, excluded = [], {}, 0
    used = set()
    for record in records:
        row = series.get(record["series"])
        if row is None or row["patient_id"] != record["group"]:
            raise ValueError(f"OAI series missing/incorrect participant join: {record['series']}")
        key = tuple(row[k] for k in identity)
        used.add(key)
        if record["kind"] == "dicom":
            import pydicom
            ds = pydicom.dcmread(record["paths"][0], stop_before_pixels=True,
                                specific_tags=["ImageLaterality", "Laterality"])
            laterality = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))).strip()
            if laterality and laterality != key[2]:
                raise ValueError(f"OAI knee side conflicts with DICOM: {record['series']}")
        if row["partition"] != "train":
            excluded += 1
            continue
        # Tuple encoded without ambiguous delimiter collisions.
        case_id = json.dumps(key, separators=(",", ":"))
        kept.append(record | {"case_id": case_id, "partition": "explicit_train", "visit": key[1], "side": key[2]})
        case_labels[case_id] = labels.get(key, {"values": [0.] * len(tasks), "weights": [0.] * len(tasks)})
    if not kept or not any(sum(v["weights"]) for v in case_labels.values()):
        raise ValueError("no OAI training images with joined native labels")
    weight = np.asarray([v["weights"] for v in case_labels.values()])
    if np.any(weight.sum(0) == 0):
        raise ValueError("an OAI task has no observed training labels; correct the join/task declaration")
    # A manifest must not imply that missing downloaded MRI series were used.
    missing_series = set(series) - {r["series"] for r in records}
    if missing_series:
        raise ValueError(f"OAI map references {len(missing_series)} unavailable/ineligible MRI series")
    return tasks, case_labels, kept, [Path(p) for p in (series_csv, labels_csv, tasks_json)], {
        "held_out_series": excluded, "training_cases": len(case_labels),
        "unmatched_label_rows": len(set(labels) - used),
        "observed_training_labels_per_task": weight.sum(0).astype(int).tolist()}
