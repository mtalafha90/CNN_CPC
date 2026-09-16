"""Streaming full inventories: no patient/series caps and no permanent pixel cache."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from ..b58_external_knee.data import SOURCES, cache_triplets, raw_fingerprint, read_volume
from ..b57_protocol import digest, write_json


def complete_inventory(records):
    rows = sorted(records, key=lambda r: (r["source"], r["group"], r["series"]))
    keys = [(r["source"], r["series"]) for r in rows]
    if len(keys) != len(set(keys)) or {r["source"] for r in rows} != set(SOURCES):
        raise ValueError("full run needs unique series from all four sources")
    counts = {s: {"series": sum(r["source"] == s for r in rows),
                  "groups": len({r["group"] for r in rows if r["source"] == s})} for s in SOURCES}
    return rows, counts


def freeze_pixels(records, root):
    """Validate every volume once. Resume preparation using per-series byte hashes.

    Only metadata is persisted. Full-volume float32 hashes detect exact duplicate
    reconstructions; this does not certify cross-dataset patient disjointness.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result, seen = [], {}
    for i, row in enumerate(records):
        fingerprint = raw_fingerprint(row["paths"])
        path = root / (digest([row["source"], row["series"]]) + ".json")
        identity = {"record": row, "raw_sha256": fingerprint}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise ValueError("full-data MRI input changed during preparation")
        else:
            import hashlib
            x = read_volume(row)
            pixel_sha = hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()
            saved = {"identity": identity, "shape": list(x.shape), "pixel_sha256": pixel_sha}
            write_json(path, saved)
        key = (tuple(saved["shape"]), saved["pixel_sha256"])
        if key in seen:
            raise ValueError(f"duplicate MRI content: {seen[key]} and {row['source']}/{row['series']}")
        seen[key] = f"{row['source']}/{row['series']}"
        result.append(row | {"raw_sha256": fingerprint, "shape": saved["shape"]})
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(records):
            print(f"[B58 full] validated MRI series {i+1}/{len(records)}", flush=True)
    return result


def triplets(row, centres, side):
    if raw_fingerprint(row["paths"]) != row["raw_sha256"]:
        raise ValueError(f"frozen MRI bytes changed: {row['source']}/{row['series']}")
    return torch.from_numpy(cache_triplets(read_volume(row), centres=centres, side=side).astype(np.float32))


def epoch_order(items, seed, epoch):
    """Each item exactly once; shuffle within source and interleave by progress."""
    rng = np.random.default_rng(seed + 1009 * epoch)
    groups = defaultdict(list)
    for index, row in enumerate(items):
        groups[row["source"]].append(index)
    order = []
    for source, indices in sorted(groups.items()):
        for rank, index in enumerate(rng.permutation(indices)):
            order.append(((rank + .5) / len(indices), source, int(index)))
    return [index for _, _, index in sorted(order)]


def build_cases(records, label_tables, tasks):
    members = defaultdict(list)
    for i, row in enumerate(records):
        members[(row["source"], row.get("case_id", row["group"]))].append(i)
    cases = []
    for (source, key), indices in sorted(members.items()):
        labels = label_tables[source].get(key, {"values": [0.] * len(tasks[source]),
                                                "weights": [0.] * len(tasks[source])})
        cases.append({"source": source, "case_id": key, "record_indices": indices, **labels})
    counts = Counter(r["source"] for r in cases if sum(r["weights"]) > 0)
    if set(counts) != set(SOURCES):
        raise ValueError("all four sources must contribute actual supervised labels")
    return cases


def label_coverage(cases, tasks):
    result = {}
    for source in SOURCES:
        rows = [c for c in cases if c["source"] == source]
        y, w = np.asarray([r["values"] for r in rows]), np.asarray([r["weights"] for r in rows])
        result[source] = {"cases": len(rows), "labelled_cases": int((w.sum(1) > 0).sum()),
            "tasks": [{"name": task["name"], "observed": int((w[:, i] > 0).sum()),
                       "distinct_training_values": sorted(set(y[w[:, i] > 0, i].tolist()))}
                      for i, task in enumerate(tasks[source])]}
    return result
