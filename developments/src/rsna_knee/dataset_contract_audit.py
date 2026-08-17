"""Reproducible audit of the official knee-MRI dataset contract.

This module is deliberately descriptive.  It does not train a model, choose a
checkpoint, alter B6 supervision, or use PV1/PV2 outcomes.  It answers the data
questions that must be settled before further architecture development:

* how many studies have any official labels versus all twelve labels;
* per-target official-label availability and prevalence;
* report availability and Unicode-script composition (not language inference);
* series-per-study and supplied sequence-metadata distributions;
* B6 weak-supervision coverage by report script bucket, when a B6 root is given;
* optional on-disk DICOM slice-count distributions.

The official competition description states that reports may be multilingual.
This audit therefore uses conservative Unicode *script* buckets.  A script bucket
must never be reported as a detected language.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .b7_weak_supervision import B7_MIN_CONFIDENCE, load_frozen_b6_export, prepare_b7_supervision
from .constants import TARGETS
from .data import gold_mask, load_series_csv, load_train_csv

DATASET_CONTRACT_AUDIT_VERSION = "official_dataset_contract_audit_v1"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _json_number(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    qs = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)
    out: dict[str, float | int | None] = {"n": int(arr.size), "mean": float(arr.mean())}
    for q in qs:
        label = f"q{int(round(q * 100)):02d}"
        out[label] = float(np.quantile(arr, q))
    return out


def _unicode_script_name(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return "Other"
    if "LATIN" in name:
        return "Latin"
    if "CYRILLIC" in name:
        return "Cyrillic"
    if "ARABIC" in name:
        return "Arabic"
    if "GREEK" in name:
        return "Greek"
    if "HEBREW" in name:
        return "Hebrew"
    if "DEVANAGARI" in name:
        return "Devanagari"
    if "HANGUL" in name:
        return "Hangul"
    if "HIRAGANA" in name:
        return "Hiragana"
    if "KATAKANA" in name:
        return "Katakana"
    if "CJK" in name or "IDEOGRAPH" in name:
        return "CJK"
    return "Other"


def report_script_profile(text: object) -> dict:
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    counts: Counter[str] = Counter()
    for ch in text:
        script = _unicode_script_name(ch)
        if script is not None:
            counts[script] += 1
    total = int(sum(counts.values()))
    if total == 0:
        bucket = "Empty/no-letters"
        dominant_fraction = 0.0
    else:
        ranked = counts.most_common()
        dominant, dominant_n = ranked[0]
        dominant_fraction = float(dominant_n / total)
        if dominant_fraction >= 0.80:
            bucket = dominant
        else:
            second = ranked[1][0] if len(ranked) > 1 else "Other"
            bucket = f"Mixed:{dominant}+{second}"
    return {
        "bucket": bucket,
        "letter_count": total,
        "dominant_fraction": dominant_fraction,
        "script_counts": dict(sorted(counts.items())),
    }


def audit_train_table(train: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = train[TARGETS].apply(pd.to_numeric, errors="coerce")
    available = numeric.notna()
    invalid = available & ~numeric.isin([0.0, 1.0])
    if invalid.any().any():
        bad = {target: int(invalid[target].sum()) for target in TARGETS if int(invalid[target].sum()) > 0}
        raise ValueError(f"official label columns contain non-binary populated values: {bad}")

    any_labeled = available.any(axis=1)
    fully_labeled = available.all(axis=1)
    n_labels = available.sum(axis=1).astype(int)
    repo_gold = gold_mask(train).astype(bool)
    if not repo_gold.equals(any_labeled):
        raise RuntimeError("repository gold_mask policy no longer equals any-populated-official-label policy")

    reports = train["Report"].fillna("").astype(str)
    report_nonempty = reports.str.strip().ne("")
    report_lengths = reports.str.len().astype(int)

    per_target_rows = []
    for target in TARGETS:
        observed = available[target]
        vals = numeric.loc[observed, target].astype(float)
        positives = int(vals.eq(1.0).sum())
        negatives = int(vals.eq(0.0).sum())
        labeled = int(observed.sum())
        per_target_rows.append({
            "target": target,
            "labeled_cells": labeled,
            "positive_cells": positives,
            "negative_cells": negatives,
            "positive_prevalence_among_labeled": float(positives / labeled) if labeled else np.nan,
            "missing_cells": int((~observed).sum()),
        })
    per_target = pd.DataFrame(per_target_rows)

    histogram = (
        n_labels.value_counts().sort_index().rename_axis("official_label_count").rename("studies").reset_index()
    )

    script_rows = []
    profiles = reports.map(report_script_profile)
    script_bucket = profiles.map(lambda x: x["bucket"])
    letter_count = profiles.map(lambda x: int(x["letter_count"]))
    for bucket, idx in script_bucket.groupby(script_bucket).groups.items():
        index = list(idx)
        script_rows.append({
            "script_bucket": str(bucket),
            "studies": int(len(index)),
            "repository_gold_any_label": int(repo_gold.loc[index].sum()),
            "zero_official_label_studies": int((~repo_gold.loc[index]).sum()),
            "fully_labeled_studies": int(fully_labeled.loc[index].sum()),
            "reports_nonempty": int(report_nonempty.loc[index].sum()),
            "report_letters_mean": float(letter_count.loc[index].mean()) if index else 0.0,
        })
    script_summary = pd.DataFrame(script_rows).sort_values("studies", ascending=False).reset_index(drop=True)

    labeled_studies = train.loc[repo_gold, ["StudyInstanceUID", "Report", *TARGETS]].copy()
    labeled_studies.insert(1, "official_label_count", n_labels.loc[repo_gold].to_numpy())
    labeled_studies.insert(2, "fully_labeled_12", fully_labeled.loc[repo_gold].to_numpy())
    labeled_studies.insert(3, "report_script_bucket", script_bucket.loc[repo_gold].to_numpy())
    # Do not write report text into the audit artifact; only report presence/length.
    labeled_studies["report_present"] = labeled_studies["Report"].fillna("").astype(str).str.strip().ne("")
    labeled_studies["report_chars"] = labeled_studies["Report"].fillna("").astype(str).str.len()
    labeled_studies = labeled_studies.drop(columns=["Report"])

    summary = {
        "training_studies": int(len(train)),
        "repository_gold_definition": "any of the 12 official target columns is populated",
        "repository_gold_any_label_studies": int(repo_gold.sum()),
        "fully_labeled_12_studies": int(fully_labeled.sum()),
        "partially_labeled_studies": int((repo_gold & ~fully_labeled).sum()),
        "zero_official_label_studies": int((~repo_gold).sum()),
        "reports_nonempty": int(report_nonempty.sum()),
        "reports_empty": int((~report_nonempty).sum()),
        "report_character_count": _quantiles(report_lengths),
        "official_labels_per_study": _quantiles(n_labels),
    }
    return summary, per_target, histogram, script_summary, labeled_studies


def audit_series_table(series_raw: pd.DataFrame, train: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required = {
        "StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"
    }
    missing = sorted(required.difference(series_raw.columns))
    if missing:
        raise ValueError(f"train_series.csv missing columns: {missing}")

    series = series_raw.copy()
    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    train_uids = set(train["StudyInstanceUID"].astype(str))
    extra_studies = sorted(set(series["StudyInstanceUID"]).difference(train_uids))
    if extra_studies:
        raise ValueError(f"train_series.csv contains {len(extra_studies)} study UID(s) absent from train.csv")

    counts = series.groupby("StudyInstanceUID").size().rename("series_count")
    per_study = train[["StudyInstanceUID"]].copy()
    per_study["StudyInstanceUID"] = per_study["StudyInstanceUID"].astype(str)
    per_study = per_study.merge(counts, left_on="StudyInstanceUID", right_index=True, how="left")
    per_study["series_count"] = per_study["series_count"].fillna(0).astype(int)
    gold_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))
    per_study["repository_gold_any_label"] = per_study["StudyInstanceUID"].isin(gold_uids)

    normalized = load_series_csv_from_frame(series_raw)
    metadata_rows = []
    for field in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"):
        values = normalized[field].astype("string").fillna("<NA>")
        for value, n in values.value_counts(dropna=False).items():
            metadata_rows.append({"field": field, "value": str(value), "series": int(n)})
    combo = (
        normalized.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"], dropna=False)
        .size().reset_index(name="series")
    )
    for _, row in combo.iterrows():
        metadata_rows.append({
            "field": "Plane_x_Fluid_x_FatSuppression",
            "value": f"{row['Anatomical_Plane']}|fluid={row['Fluid_Sensitive']}|fat={row['Fat_Suppression']}",
            "series": int(row["series"]),
        })
    metadata_counts = pd.DataFrame(metadata_rows)

    summary = {
        "series_rows": int(len(series)),
        "unique_series": int(series["SeriesInstanceUID"].nunique()),
        "studies_with_series": int(counts.shape[0]),
        "studies_without_series": int(per_study["series_count"].eq(0).sum()),
        "series_per_study": _quantiles(per_study["series_count"]),
        "raw_missing_or_blank": {
            "Anatomical_Plane": int(series_raw["Anatomical_Plane"].isna().sum() + series_raw["Anatomical_Plane"].fillna("").astype(str).str.strip().eq("").sum()),
            "Fluid_Sensitive": int(series_raw["Fluid_Sensitive"].isna().sum()),
            "Fat_Suppression": int(series_raw["Fat_Suppression"].isna().sum()),
        },
    }
    return summary, per_study, metadata_counts


def load_series_csv_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the repository's series normalization without a temporary file."""
    from .data import coerce_bool, normalise_plane

    out = frame.copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    out["SeriesInstanceUID"] = out["SeriesInstanceUID"].astype(str)
    out["Fluid_Sensitive"] = coerce_bool(out["Fluid_Sensitive"], preserve_unknown=True)
    out["Fat_Suppression"] = coerce_bool(out["Fat_Suppression"], preserve_unknown=True)
    out["Anatomical_Plane"] = normalise_plane(out["Anatomical_Plane"])
    return out


def audit_b6_by_script(train: pd.DataFrame, b6_root: Path) -> tuple[dict, pd.DataFrame]:
    b6_frame, policy, audit = load_frozen_b6_export(b6_root)
    active_uids, targets, weights, supervision = prepare_b7_supervision(train, b6_frame)
    del targets

    non_gold = train.loc[~gold_mask(train), ["StudyInstanceUID", "Report"]].copy()
    non_gold["StudyInstanceUID"] = non_gold["StudyInstanceUID"].astype(str)
    non_gold["script_bucket"] = non_gold["Report"].fillna("").map(lambda x: report_script_profile(x)["bucket"])
    joined = non_gold.merge(b6_frame, on="StudyInstanceUID", how="left", validate="one_to_one")

    usable_by_target: dict[str, np.ndarray] = {}
    positive_by_target: dict[str, np.ndarray] = {}
    negative_by_target: dict[str, np.ndarray] = {}
    for target in TARGETS:
        state = joined[f"{target}__state"].fillna("").astype(str).to_numpy()
        conf = pd.to_numeric(joined[f"{target}__confidence"], errors="coerce").fillna(0.0).to_numpy(float)
        positive = (state == "positive") & (conf >= B7_MIN_CONFIDENCE)
        negative = (state == "negated") & (conf >= B7_MIN_CONFIDENCE)
        positive_by_target[target] = positive
        negative_by_target[target] = negative
        usable_by_target[target] = positive | negative

    usable_matrix = np.column_stack([usable_by_target[t] for t in TARGETS])
    positive_matrix = np.column_stack([positive_by_target[t] for t in TARGETS])
    negative_matrix = np.column_stack([negative_by_target[t] for t in TARGETS])
    row_active = usable_matrix.any(axis=1)

    rows = []
    buckets = joined["script_bucket"].astype(str).to_numpy()
    for bucket in sorted(set(buckets)):
        mask = buckets == bucket
        studies = int(mask.sum())
        rows.append({
            "script_bucket": bucket,
            "report_only_studies": studies,
            "active_b6_studies": int((row_active & mask).sum()),
            "inactive_zero_usable_cells": int((~row_active & mask).sum()),
            "active_study_fraction": float((row_active & mask).sum() / studies) if studies else np.nan,
            "usable_cells": int(usable_matrix[mask].sum()),
            "positive_cells": int(positive_matrix[mask].sum()),
            "negative_cells": int(negative_matrix[mask].sum()),
            "usable_cells_per_study": float(usable_matrix[mask].sum() / studies) if studies else np.nan,
        })
    by_script = pd.DataFrame(rows).sort_values("report_only_studies", ascending=False).reset_index(drop=True)

    summary = {
        "b6_version": str(policy.get("version", audit.get("b6_version", ""))),
        "min_confidence_for_usable_cell": float(B7_MIN_CONFIDENCE),
        "report_only_rows": int(supervision["report_only_rows"]),
        "active_studies": int(supervision["active_studies"]),
        "inactive_studies_zero_usable_cells": int(supervision["inactive_studies_zero_usable_cells"]),
        "usable_cells": int(supervision["usable_cells"]),
        "positive_cells": int(supervision["positive_cells"]),
        "negative_cells": int(supervision["negative_cells"]),
        "active_uid_count_crosscheck": int(len(active_uids)),
        "weight_shape_crosscheck": list(weights.shape),
        "per_target": supervision["targets"],
    }
    return summary, by_script


def audit_slice_counts(data_root: Path, series_raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    base = data_root / "train_series"
    rows = []
    for row in series_raw[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False):
        study_uid = str(row.StudyInstanceUID)
        series_uid = str(row.SeriesInstanceUID)
        folder = base / study_uid / series_uid
        if folder.is_dir():
            count = sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
            exists = True
        else:
            count = 0
            exists = False
        rows.append({
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "directory_exists": exists,
            "dicom_slices": int(count),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "series_scanned": int(len(frame)),
        "missing_series_directories": int((~frame["directory_exists"]).sum()),
        "zero_dicom_series": int(frame["dicom_slices"].eq(0).sum()),
        "dicom_slices_per_series": _quantiles(frame.loc[frame["dicom_slices"] > 0, "dicom_slices"]),
        "series_over_100_slices": int(frame["dicom_slices"].gt(100).sum()),
        "series_over_200_slices": int(frame["dicom_slices"].gt(200).sum()),
    }
    return summary, frame


def run_dataset_contract_audit(
    *,
    data_root: str | Path,
    out_root: str | Path,
    b6_root: str | Path | None = None,
    scan_slices: bool = False,
) -> dict:
    root = Path(data_root).resolve()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.csv"
    series_path = root / "train_series.csv"
    train = load_train_csv(train_path)
    series_raw = pd.read_csv(series_path)

    train_summary, per_target, label_hist, scripts, labeled_studies = audit_train_table(train)
    series_summary, per_study_series, metadata_counts = audit_series_table(series_raw, train)

    per_target.to_csv(out / "official_label_by_target.csv", index=False)
    label_hist.to_csv(out / "official_label_count_histogram.csv", index=False)
    scripts.to_csv(out / "report_script_buckets.csv", index=False)
    labeled_studies.to_csv(out / "officially_labeled_studies.csv", index=False)
    per_study_series.to_csv(out / "series_per_study.csv", index=False)
    metadata_counts.to_csv(out / "series_metadata_counts.csv", index=False)

    summary: dict = {
        "audit_version": DATASET_CONTRACT_AUDIT_VERSION,
        "data_root": str(root),
        "input_fingerprints": {
            "train_csv_sha256": _sha256_file(train_path),
            "train_series_csv_sha256": _sha256_file(series_path),
        },
        "official_train_table": train_summary,
        "series_table": series_summary,
        "report_script_note": (
            "Unicode script buckets are descriptive character-system groups, not detected languages. "
            "Do not interpret Latin/Cyrillic/Arabic/etc. as a specific reporting language or institution."
        ),
    }

    if b6_root is not None:
        b6_summary, b6_by_script = audit_b6_by_script(train, Path(b6_root))
        summary["b6_weak_supervision"] = b6_summary
        b6_by_script.to_csv(out / "b6_coverage_by_report_script.csv", index=False)

    if scan_slices:
        slice_summary, slice_frame = audit_slice_counts(root, series_raw)
        summary["slice_count_scan"] = slice_summary
        slice_frame.to_csv(out / "slice_counts_by_series.csv", index=False)
    else:
        summary["slice_count_scan"] = {
            "performed": False,
            "note": "rerun with --scan-slices to count on-disk DICOM files for every listed training series",
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_number), encoding="utf-8")
    print(json.dumps({
        "audit_version": DATASET_CONTRACT_AUDIT_VERSION,
        "training_studies": train_summary["training_studies"],
        "repository_gold_any_label_studies": train_summary["repository_gold_any_label_studies"],
        "fully_labeled_12_studies": train_summary["fully_labeled_12_studies"],
        "partially_labeled_studies": train_summary["partially_labeled_studies"],
        "zero_official_label_studies": train_summary["zero_official_label_studies"],
        "series_rows": series_summary["series_rows"],
        "b6_audited": b6_root is not None,
        "slice_counts_scanned": bool(scan_slices),
        "summary": str(out / "summary.json"),
    }, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser("Audit the official RSNA knee dataset contract without model selection")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", default=None)
    ap.add_argument("--out-root", default="runs/dataset_contract_audit")
    ap.add_argument("--scan-slices", action="store_true")
    args = ap.parse_args()
    run_dataset_contract_audit(
        data_root=args.data_root,
        out_root=args.out_root,
        b6_root=args.b6_root,
        scan_slices=bool(args.scan_slices),
    )


if __name__ == "__main__":
    main()
