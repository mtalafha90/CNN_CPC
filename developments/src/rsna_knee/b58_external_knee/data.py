"""Read real MRI volumes, retain source identities, and build a bounded SSL cache."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..b57_protocol import digest, sha256_file, write_json
from ..dicom import _normalise_volume, find_series_dir

SOURCES = ("rsna", "mrnet", "fastmri", "oai")


def record(source, group, series, paths, kind, **extra):
    paths = [str(Path(p).resolve()) for p in paths]
    if not paths or not str(group).strip() or not str(series).strip():
        raise ValueError("empty MRI identity or paths")
    return {"source": source, "group": str(group), "series": str(series),
            "paths": paths, "kind": kind, **extra}


def training_directory(root, names):
    """Require an explicit official training partition; never scan valid/test."""
    root = Path(root).resolve()
    choices = ([root] if root.name in names else []) + [root / n for n in names if (root / n).is_dir()]
    choices = sorted(set(p for p in choices if p.is_dir()))
    if len(choices) != 1:
        raise ValueError(f"expected one training directory {names} under {root}; found {choices}")
    return choices[0]


def index_mrnet(root):
    root = Path(root).resolve()
    if (root / "MRNet-v1.0").is_dir():
        root = root / "MRNet-v1.0"
    train = training_directory(root, ("train",))
    rows = []
    for plane in ("axial", "coronal", "sagittal"):
        for path in sorted((train / plane).glob("*.npy")):
            rows.append(record("mrnet", path.stem, f"{path.stem}/{plane}", [path], "npy",
                               partition="official_train", grouping="exam_id; official split is patient-disjoint"))
    if not rows:
        raise ValueError(f"no MRNet train/{{axial,coronal,sagittal}}/*.npy at {root}")
    return rows


def index_fastmri(root):
    train = training_directory(root, ("knee_multicoil_train", "multicoil_train"))
    rows = [record("fastmri", p.stem, p.stem, [p], "h5",
                   partition="official_knee_multicoil_train", grouping="acquisition_id; patient identity unavailable")
            for p in sorted(train.glob("*.h5"))]
    if not rows:
        raise ValueError(f"no fastMRI training HDF5 files in {train}")
    return rows


def oai_knee_header(ds):
    text = (str(getattr(ds, "BodyPartExamined", "")) + " "
            + str(getattr(ds, "SeriesDescription", ""))).upper()
    if re.search(r"SCOUT|LOCALIZER|THIGH", text):
        return False
    return bool(re.search(r"KNEE|DESS|(?:SAG|COR).*IW|COR.*T1|SAG.*T2", text))


def index_oai(root, forbidden_studies=(), forbidden_series=()):
    import pydicom
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OAI extracted DICOM directory missing: {root}")
    tags = ["Modality", "PatientID", "StudyInstanceUID", "SeriesInstanceUID",
            "SeriesDescription", "BodyPartExamined", "Rows", "Columns", "NumberOfFrames"]
    groups, identities, ignored = defaultdict(list), {}, Counter()
    forbidden_studies, forbidden_series = set(forbidden_studies), set(forbidden_series)
    for number, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file()), 1):
        if number % 10000 == 0:
            print(f"[B58] OAI headers: {number} files, {len(groups)} eligible series", flush=True)
        # OAI DICOM files can be extensionless or have numeric suffixes.
        if path.suffix.lower() in {".zip", ".gz", ".csv", ".txt", ".json", ".pdf", ".xml"}:
            ignored["non_dicom_file"] += 1
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        except pydicom.errors.InvalidDicomError as exc:
            if path.suffix.lower() == ".dcm":
                raise ValueError(f"invalid OAI DICOM: {path}") from exc
            ignored["unrecognised_file"] += 1
            continue
        if getattr(ds, "Modality", "") != "MR" or not oai_knee_header(ds):
            ignored["not_eligible_knee_mri"] += 1
            continue
        patient, study, series = [str(getattr(ds, key, "")).strip()
                                  for key in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID")]
        if not all((patient, study, series)):
            raise ValueError(f"OAI needs participant, study and series identity: {path}")
        if study in forbidden_studies or series in forbidden_series:
            raise ValueError(f"external OAI series overlaps an RSNA identity: {study}/{series}")
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError(f"enhanced multiframe DICOM needs an explicit adapter: {path}")
        if series in identities and identities[series] != (patient, study):
            raise ValueError(f"conflicting OAI identity for series {series}")
        identities[series] = (patient, study)
        groups[series].append(path)
    rows = []
    for series, paths in sorted(groups.items()):
        if len(paths) < 3:
            ignored["series_with_fewer_than_three_slices"] += 1
            continue
        patient, study = identities[series]
        rows.append(record("oai", patient, series, paths, "dicom", study_uid=study,
                           partition="external_unlabelled_pool", grouping="PatientID; visits and knees grouped"))
    print(f"[B58] OAI exclusions: {dict(ignored)}", flush=True)
    if not rows:
        raise ValueError(f"no eligible OAI knee MRI series in {root}; extract the original DICOM archives")
    return rows


def index_rsna(p, protocol_root):
    from ..dicom import _iter_dicom_files
    index = json.loads((Path(protocol_root) / "series_index.json").read_text())
    held = set(p["splits"]["validation"]) | set(p["expert_uids"]) | set(p["splits"]["excluded_profile_overlap"])
    rows, short_series = [], 0
    for uid in p["splits"]["train"]:
        if uid in held:
            raise ValueError("RSNA validation/expert/excluded study reached the SSL index")
        for entry in index[uid]:
            series = str(entry["series_uid"])
            path = find_series_dir(p["data_root"], "train", uid, series)
            if path is None:
                raise FileNotFoundError(f"missing original RSNA series {uid}/{series}")
            paths = list(_iter_dicom_files(path))
            if len(paths) < 3:
                short_series += 1
                continue
            rows.append(record("rsna", uid, series, paths, "dicom",
                               study_uid=uid, partition="frozen_rsna_train", grouping="StudyInstanceUID"))
    print(f"[B58] RSNA SSL exclusions: {short_series} series with fewer than three files", flush=True)
    return rows


def select_records(rows, config):
    """Cap groups and series by seeded hashes, not labels or measured scores."""
    grouped = {s: defaultdict(list) for s in SOURCES}
    seen = set()
    for row in rows:
        key = (row["source"], row["series"])
        if row["source"] not in grouped or key in seen:
            raise ValueError(f"unknown source or duplicate series: {key}")
        seen.add(key)
        grouped[row["source"]][row["group"]].append(row)
    selected, counts = [], {}
    rank = lambda value: digest([config["seed"], value])
    for source in SOURCES:
        pool = grouped[source]
        if not pool:
            raise ValueError(f"B58 requires all four sources; missing {source}")
        groups = sorted(pool, key=lambda g: rank([source, g]))[:config["max_groups_per_source"]]
        chosen = []
        for group in groups:
            chosen.extend(sorted(pool[group], key=lambda r: rank([source, r["series"]]))[:config["max_series_per_group"]])
        counts[source] = {"available_groups": len(pool), "available_series": sum(map(len, pool.values())),
                          "selected_groups": len(groups), "selected_series": len(chosen)}
        selected.extend(chosen)
    return selected, counts


def dicom_volume(paths):
    """Strict pixel decoding and physical slice sorting for any MRI plane."""
    import pydicom
    datasets = [pydicom.dcmread(p) for p in paths]
    if not datasets:
        raise ValueError("empty DICOM series")
    orientation = getattr(datasets[0], "ImageOrientationPatient", None)
    physical = orientation is not None and all(hasattr(ds, "ImagePositionPatient") for ds in datasets)
    if physical:
        orientation = np.asarray(orientation, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if np.linalg.norm(normal) < .99:
            raise ValueError("invalid DICOM orientation")
        for ds in datasets:
            if not np.allclose(getattr(ds, "ImageOrientationPatient", []), orientation, atol=1e-4):
                raise ValueError("inconsistent orientation within DICOM series")
        positions = [float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal)) for ds in datasets]
    else:
        if not all(hasattr(ds, "InstanceNumber") for ds in datasets):
            raise ValueError("DICOM lacks physical ordering and InstanceNumber")
        positions = [int(ds.InstanceNumber) for ds in datasets]
    if len(set(positions)) != len(positions):
        raise ValueError("duplicate DICOM slice positions; separate repeated acquisitions")
    images = []
    for i in np.argsort(positions, kind="stable"):
        ds = datasets[int(i)]
        x = np.asarray(ds.pixel_array, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("B58 expects single-frame greyscale DICOM slices")
        x = x * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            x = x.max() + x.min() - x
        images.append(x)
    if len({x.shape for x in images}) != 1:
        raise ValueError("mixed matrices within a DICOM series need explicit repair")
    return np.stack(images)


def read_volume(row):
    if row["kind"] == "npy":
        volume = np.load(row["paths"][0], allow_pickle=False)
    elif row["kind"] == "h5":
        import h5py
        with h5py.File(row["paths"][0], "r") as f:
            if "reconstruction_rss" not in f:
                raise ValueError("fastMRI requires fully sampled training reconstruction_rss, not raw/test k-space")
            volume = f["reconstruction_rss"][:]
    elif row["kind"] == "dicom":
        volume = dicom_volume(row["paths"])
    else:
        raise ValueError(f"unsupported MRI format: {row['kind']}")
    volume = np.asarray(volume)
    if (volume.ndim != 3 or min(volume.shape) < 3 or volume.dtype.kind not in "fiu"
            or not np.isfinite(volume).all() or float(volume.max()) <= float(volume.min())):
        raise ValueError(f"invalid/nonfinite/constant MRI volume: {row['source']}/{row['series']}")
    return volume.astype(np.float32)


def cache_triplets(volume, *, centres=8, side=224):
    """Full-volume intensity normalisation; 90% crop; isotropic fit and edge pad.

    SSL uses compact square views. Downstream diagnosis retains B42's original
    448-area rectangular preprocessing. The cache never quantises to uint8.
    """
    x = _normalise_volume(volume)
    n, h, w = x.shape
    ch, cw = max(3, round(h * .9)), max(3, round(w * .9))
    x = x[:, (h-ch)//2:(h-ch)//2+ch, (w-cw)//2:(w-cw)//2+cw]
    z = np.linspace(1, n - 2, centres).round().astype(int)
    ix = np.clip(z[:, None] + np.array([-1, 0, 1]), 0, n - 1)
    t = torch.from_numpy(np.ascontiguousarray(x[ix]))
    scale = side / max(ch, cw)
    nh, nw = max(2, round(ch * scale)), max(2, round(cw * scale))
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    dh, dw = side - nh, side - nw
    t = F.pad(t, (dw//2, dw-dw//2, dh//2, dh-dh//2), mode="replicate")
    return t.clamp(0, 1).numpy().astype(np.float16)


def raw_fingerprint(paths):
    return digest([{ "path": str(p), "sha256": sha256_file(p)} for p in paths])


def build_cache(rows, root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result, pixels = [], {}
    for i, row in enumerate(rows, 1):
        key = digest([row["source"], row["group"], row["series"]])
        path, meta_path = root / f"{key}.npy", root / f"{key}.json"
        raw_sha = raw_fingerprint(row["paths"])
        identity = {"record": row, "raw_sha256": raw_sha,
                    "centres": config["cache_centres"], "side": config["ssl_side"]}
        if meta_path.exists():
            saved = json.loads(meta_path.read_text())
            if saved["identity"] != identity or sha256_file(path) != saved["cache_sha256"]:
                raise ValueError(f"existing B58 cache/input changed: {path}")
        else:
            if path.exists():
                raise FileExistsError(f"unmanifested B58 cache: {path}; inspect it before retrying")
            x = cache_triplets(read_volume(row), centres=config["cache_centres"], side=config["ssl_side"])
            temporary = path.with_suffix(".writing")
            with temporary.open("wb") as f:
                np.save(f, x, allow_pickle=False)
            temporary.replace(path)
            saved = {"identity": identity, "cache_sha256": sha256_file(path), "shape": list(x.shape)}
            write_json(meta_path, saved)
        pixel_sha = saved["cache_sha256"]
        if pixel_sha in pixels:
            raise ValueError(f"duplicate cached MRI content: {pixels[pixel_sha]} and {row['source']}/{row['series']}")
        pixels[pixel_sha] = f"{row['source']}/{row['series']}"
        result.append({k: v for k, v in row.items() if k != "paths"} | {
            "cache": str(path.resolve()), "cache_sha256": pixel_sha,
            "raw_sha256": raw_sha, "shape": saved["shape"]})
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"[B58] cached {i}/{len(rows)} MRI series", flush=True)
    return result
