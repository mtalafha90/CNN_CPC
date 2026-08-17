"""Descriptive intersection audit for report supervision and MRI acquisition domain.

This phase follows the tabular/script audit and the DICOM header audit. It asks
whether the frozen B6-active population differs systematically from report-only
studies that receive no usable B6 cells in scanner/acquisition characteristics.

It is intentionally non-predictive: no model outputs, PV1/PV2 outcomes, or target-
wise performance results are read, and it does not alter B6 or define B35.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
from .data import gold_mask, load_train_csv
from .dataset_contract_audit import report_script_profile

DOMAIN_INTERSECTION_AUDIT_VERSION = "official_dataset_domain_intersection_audit_v1"


def manufacturer_family(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    if not text:
        return "Missing"
    if "SIEMENS" in text:
        return "Siemens"
    if text.startswith("GE") or "GE MEDICAL" in text or "GEHC" in text:
        return "GE"
    if "PHILIPS" in text:
        return "Philips"
    if "TOSHIBA" in text or "CANON" in text:
        return "Canon/Toshiba"
    if "FUJIFILM" in text or "HITACHI" in text:
        return "Fujifilm/Hitachi"
    return "Other"


def normalize_acquisition_type(value: object) -> str:
    if value is None or pd.isna(value):
        return "Missing"
    text = str(value).strip().upper()
    if text in {"2D", "3D"}:
        return text
    return "Missing" if text in {"", "NAN", "NONE"} else text


def _dominant(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "Missing"
    counts = values.value_counts()
    maximum = int(counts.max())
    return sorted(counts[counts.eq(maximum)].index.astype(str))[0]


def build_study_domain_table(
    train: pd.DataFrame,
    header: pd.DataFrame,
    *,
    b6_active_uids: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive",
        "Fat_Suppression", "dicom_files", "manufacturer", "manufacturer_model",
        "magnetic_field_strength_t", "mr_acquisition_type",
    }
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"header audit CSV missing columns: {missing}")

    hdr = header.copy()
    hdr["StudyInstanceUID"] = hdr["StudyInstanceUID"].astype(str)
    hdr["SeriesInstanceUID"] = hdr["SeriesInstanceUID"].astype(str)
    if hdr[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().any():
        raise ValueError("header audit contains duplicate study/series rows")

    train_uids = set(train["StudyInstanceUID"].astype(str))
    header_uids = set(hdr["StudyInstanceUID"])
    extra = header_uids.difference(train_uids)
    missing_studies = train_uids.difference(header_uids)
    if extra or missing_studies:
        raise ValueError(
            f"train/header study mismatch: header_extra={len(extra)}, header_missing={len(missing_studies)}"
        )

    hdr["manufacturer_family"] = hdr["manufacturer"].map(manufacturer_family)
    hdr["acquisition_type_norm"] = hdr["mr_acquisition_type"].map(normalize_acquisition_type)
    hdr["is_3d"] = hdr["acquisition_type_norm"].eq("3D")
    hdr["gt78"] = pd.to_numeric(hdr["dicom_files"], errors="coerce").fillna(0).gt(78)
    hdr["gt100"] = pd.to_numeric(hdr["dicom_files"], errors="coerce").fillna(0).gt(100)
    hdr["gt200"] = pd.to_numeric(hdr["dicom_files"], errors="coerce").fillna(0).gt(200)

    grouped = hdr.groupby("StudyInstanceUID", sort=False)
    domain = grouped.agg(
        series_count=("SeriesInstanceUID", "size"),
        n_3d=("is_3d", "sum"),
        n_gt78=("gt78", "sum"),
        n_gt100=("gt100", "sum"),
        n_gt200=("gt200", "sum"),
        dominant_manufacturer_family=("manufacturer_family", _dominant),
    ).reset_index()
    for stem in ("3d", "gt78", "gt100", "gt200"):
        domain[f"any_{stem}"] = domain[f"n_{stem}"].astype(int).gt(0)

    work = train[["StudyInstanceUID", "Report"]].copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    work["repository_gold"] = gold_mask(train).to_numpy(bool)
    work["report_script_bucket"] = work["Report"].fillna("").map(
        lambda text: report_script_profile(text)["bucket"]
    )
    active = set(str(uid) for uid in b6_active_uids)
    work["b6_status"] = np.where(
        work["repository_gold"],
        "gold_not_in_b6",
        np.where(work["StudyInstanceUID"].isin(active), "active", "inactive"),
    )
    work = work.drop(columns=["Report"])
    study = work.merge(domain, on="StudyInstanceUID", how="left", validate="one_to_one")
    if study["series_count"].isna().any():
        raise RuntimeError("domain merge produced studies without header-derived series statistics")

    return study.sort_values("StudyInstanceUID").reset_index(drop=True), hdr


def _cohort_masks(study: pd.DataFrame) -> dict[str, pd.Series]:
    report_only = ~study["repository_gold"].astype(bool)
    masks: dict[str, pd.Series] = {
        "all_studies": pd.Series(True, index=study.index),
        "gold": study["repository_gold"].astype(bool),
        "report_only_all": report_only,
        "report_only_b6_active": report_only & study["b6_status"].eq("active"),
        "report_only_b6_inactive": report_only & study["b6_status"].eq("inactive"),
    }
    for bucket in sorted(study.loc[report_only, "report_script_bucket"].astype(str).unique()):
        key = bucket.replace(" ", "_").replace("/", "-")
        base = report_only & study["report_script_bucket"].astype(str).eq(bucket)
        masks[f"report_only_script_{key}"] = base
        masks[f"report_only_script_{key}_b6_active"] = base & study["b6_status"].eq("active")
        masks[f"report_only_script_{key}_b6_inactive"] = base & study["b6_status"].eq("inactive")
    return masks


def summarize_cohorts(study: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, mask in _cohort_masks(study).items():
        part = study.loc[mask]
        n = int(len(part))
        total_series = int(part["series_count"].sum()) if n else 0
        row = {
            "cohort": cohort,
            "studies": n,
            "series": total_series,
            "mean_series_per_study": float(part["series_count"].mean()) if n else np.nan,
            "median_series_per_study": float(part["series_count"].median()) if n else np.nan,
        }
        for stem in ("3d", "gt78", "gt100", "gt200"):
            studies_n = int(part[f"any_{stem}"].sum()) if n else 0
            series_n = int(part[f"n_{stem}"].sum()) if n else 0
            row[f"studies_any_{stem}"] = studies_n
            row[f"fraction_studies_any_{stem}"] = float(studies_n / n) if n else np.nan
            row[f"series_{stem}"] = series_n
            row[f"fraction_series_{stem}"] = float(series_n / total_series) if total_series else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_series_categories(
    study: pd.DataFrame,
    header: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cohort, mask in _cohort_masks(study).items():
        uids = set(study.loc[mask, "StudyInstanceUID"].astype(str))
        part = header.loc[header["StudyInstanceUID"].isin(uids)].copy()
        fields = {
            "manufacturer_family": part["manufacturer_family"].astype(str),
            "acquisition_type": part["acquisition_type_norm"].astype(str),
            "Anatomical_Plane": part["Anatomical_Plane"].fillna("Missing").astype(str),
            "Fluid_Sensitive": part["Fluid_Sensitive"].astype("string").fillna("Missing"),
            "Fat_Suppression": part["Fat_Suppression"].astype("string").fillna("Missing"),
            "magnetic_field_strength_t": pd.to_numeric(
                part["magnetic_field_strength_t"], errors="coerce"
            ).map(lambda x: "Missing" if pd.isna(x) else f"{float(x):g}"),
        }
        denom = max(int(len(part)), 1)
        for field, values in fields.items():
            counts = values.value_counts(dropna=False)
            for value, count in counts.items():
                rows.append({
                    "cohort": cohort,
                    "field": field,
                    "value": str(value),
                    "series": int(count),
                    "fraction_of_cohort_series": float(count / denom),
                })
    return pd.DataFrame(rows)


def script_b6_crosstab(study: pd.DataFrame) -> pd.DataFrame:
    report = study.loc[~study["repository_gold"].astype(bool)].copy()
    rows = []
    for (script, status), part in report.groupby(["report_script_bucket", "b6_status"], sort=True):
        n = int(len(part))
        rows.append({
            "report_script_bucket": str(script),
            "b6_status": str(status),
            "studies": n,
            "studies_any_3d": int(part["any_3d"].sum()),
            "fraction_studies_any_3d": float(part["any_3d"].mean()) if n else np.nan,
            "studies_any_gt78": int(part["any_gt78"].sum()),
            "fraction_studies_any_gt78": float(part["any_gt78"].mean()) if n else np.nan,
            "studies_any_gt100": int(part["any_gt100"].sum()),
            "studies_any_gt200": int(part["any_gt200"].sum()),
            "mean_series_per_study": float(part["series_count"].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def run_domain_intersection_audit(
    *,
    data_root: str | Path,
    b6_root: str | Path,
    header_csv: str | Path,
    out_root: str | Path,
) -> dict:
    root = Path(data_root).resolve()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    train = load_train_csv(root / "train.csv")
    header = pd.read_csv(header_csv)
    b6_frame, _, _ = load_frozen_b6_export(b6_root)
    active_uids, _, _, supervision = prepare_b7_supervision(train, b6_frame)

    study, normalized_header = build_study_domain_table(
        train, header, b6_active_uids=active_uids
    )
    cohorts = summarize_cohorts(study)
    categories = summarize_series_categories(study, normalized_header)
    cross = script_b6_crosstab(study)

    study.to_csv(out / "study_domain_table.csv", index=False)
    cohorts.to_csv(out / "cohort_summary.csv", index=False)
    categories.to_csv(out / "cohort_series_categories.csv", index=False)
    cross.to_csv(out / "script_b6_domain_crosstab.csv", index=False)

    def cohort_row(name: str) -> dict:
        row = cohorts.loc[cohorts["cohort"].eq(name)]
        if len(row) != 1:
            raise RuntimeError(f"missing cohort summary row {name!r}")
        return row.iloc[0].to_dict()

    active = cohort_row("report_only_b6_active")
    inactive = cohort_row("report_only_b6_inactive")
    result = {
        "audit_version": DOMAIN_INTERSECTION_AUDIT_VERSION,
        "purpose": "descriptive intersection of frozen B6 supervision coverage with MRI acquisition domain",
        "training_studies": int(len(study)),
        "gold_studies": int(study["repository_gold"].sum()),
        "report_only_studies": int((~study["repository_gold"]).sum()),
        "b6_active_report_only_studies": int(study["b6_status"].eq("active").sum()),
        "b6_inactive_report_only_studies": int(study["b6_status"].eq("inactive").sum()),
        "b6_usable_cells_crosscheck": int(supervision["usable_cells"]),
        "b6_active_domain": active,
        "b6_inactive_domain": inactive,
        "governance": (
            "Descriptive only. This audit may identify acquisition-domain selection associated with B6 coverage, "
            "but it does not identify institutions, authorize target-specific changes, modify B6, define B35, "
            "or promote a model."
        ),
    }
    (out / "domain_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "audit_version": DOMAIN_INTERSECTION_AUDIT_VERSION,
        "report_only_b6_active": result["b6_active_report_only_studies"],
        "report_only_b6_inactive": result["b6_inactive_report_only_studies"],
        "active_fraction_any_3d": active["fraction_studies_any_3d"],
        "inactive_fraction_any_3d": inactive["fraction_studies_any_3d"],
        "out": str(out),
    }, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser("Audit B6 supervision coverage against MRI acquisition domain")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--header-csv", required=True)
    ap.add_argument("--out-root", default="runs/dataset_domain_intersection_audit")
    args = ap.parse_args()
    run_domain_intersection_audit(
        data_root=args.data_root,
        b6_root=args.b6_root,
        header_csv=args.header_csv,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
