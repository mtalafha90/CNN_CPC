"""Deterministic local-only audit sample for report-supervision coverage gaps.

This module does not train a model or modify frozen B6 supervision.  It creates
one reproducible sample of actual report text spanning the script/B6 strata that
were identified by the completed dataset-contract audits.

The text export is intentionally a local analysis artifact.  Do not commit its
outputs to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import B7_MIN_CONFIDENCE, load_frozen_b6_export
from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .dataset_contract_audit import report_script_profile

REPORT_GAP_AUDIT_VERSION = "report_supervision_gap_sample_v1"
REPORT_GAP_SALT = "CNN_CPC|report-gap-sample|2026-08-17"
DEFAULT_PER_STRATUM = 12


def _rank(uid: str) -> str:
    return hashlib.sha256((REPORT_GAP_SALT + "\0" + str(uid)).encode("utf-8")).hexdigest()


def _deterministic_take(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    out = frame.copy()
    out["_rank"] = out["StudyInstanceUID"].astype(str).map(_rank)
    out = out.sort_values(["_rank", "StudyInstanceUID"], kind="mergesort").head(int(n))
    return out.drop(columns=["_rank"])


def _b6_cell_summary(row: pd.Series) -> tuple[int, int, int, dict]:
    usable = positive = negative = 0
    per_target = {}
    for target in TARGETS:
        state = str(row.get(f"{target}__state", "") or "")
        try:
            confidence = float(row.get(f"{target}__confidence", 0.0))
        except Exception:
            confidence = 0.0
        is_pos = state == "positive" and confidence >= B7_MIN_CONFIDENCE
        is_neg = state == "negated" and confidence >= B7_MIN_CONFIDENCE
        positive += int(is_pos)
        negative += int(is_neg)
        usable += int(is_pos or is_neg)
        per_target[target] = {"state": state, "confidence": confidence}
    return usable, positive, negative, per_target


def build_gap_sample(
    train: pd.DataFrame,
    b6_frame: pd.DataFrame,
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    domain_study: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict], dict]:
    work = train.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    work["repository_gold"] = gold_mask(work).astype(bool)
    work["report_script_bucket"] = work["Report"].fillna("").map(
        lambda x: report_script_profile(x)["bucket"]
    )
    work["report_chars"] = work["Report"].fillna("").astype(str).str.len().astype(int)

    b6 = b6_frame.copy()
    b6["StudyInstanceUID"] = b6["StudyInstanceUID"].astype(str)
    report_only = work.loc[~work["repository_gold"]].merge(
        b6, on="StudyInstanceUID", how="left", validate="one_to_one", suffixes=("", "__b6")
    )

    b6_summaries = report_only.apply(_b6_cell_summary, axis=1)
    report_only["b6_usable_cells"] = [x[0] for x in b6_summaries]
    report_only["b6_positive_cells"] = [x[1] for x in b6_summaries]
    report_only["b6_negative_cells"] = [x[2] for x in b6_summaries]
    report_only["b6_active"] = report_only["b6_usable_cells"].gt(0)
    report_only["_b6_per_target"] = [x[3] for x in b6_summaries]

    selected_parts: list[pd.DataFrame] = []

    # All six expected non-Latin gold studies are retained; no sampling.
    gold_nonlatin = work.loc[
        work["repository_gold"] & work["report_script_bucket"].isin(["Greek", "Cyrillic"])
    ].copy()
    gold_nonlatin["sample_stratum"] = "gold_nonlatin_all"
    selected_parts.append(gold_nonlatin)

    gold_latin = _deterministic_take(
        work.loc[work["repository_gold"] & work["report_script_bucket"].eq("Latin")],
        per_stratum,
    )
    gold_latin["sample_stratum"] = "gold_latin_control"
    selected_parts.append(gold_latin)

    strata = [
        ("Latin", False, "latin_b6_inactive"),
        ("Greek", False, "greek_b6_inactive"),
        ("Cyrillic", False, "cyrillic_b6_inactive"),
        ("Latin", True, "latin_b6_active_control"),
        ("Greek", True, "greek_b6_active_control"),
        ("Cyrillic", True, "cyrillic_b6_active_all"),
    ]
    for script, active, name in strata:
        candidate = report_only.loc[
            report_only["report_script_bucket"].eq(script)
            & report_only["b6_active"].eq(active)
        ]
        # The one Cyrillic active case is kept in full; other strata are sampled.
        take_n = len(candidate) if name == "cyrillic_b6_active_all" else per_stratum
        part = _deterministic_take(candidate, take_n)
        part["sample_stratum"] = name
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    if selected["StudyInstanceUID"].duplicated().any():
        dup = selected.loc[selected["StudyInstanceUID"].duplicated(), "StudyInstanceUID"].tolist()
        raise RuntimeError(f"report-gap sample contains duplicate studies: {dup[:5]}")

    if domain_study is not None:
        domain = domain_study.copy()
        domain["StudyInstanceUID"] = domain["StudyInstanceUID"].astype(str)
        keep = [
            c for c in [
                "StudyInstanceUID", "dominant_manufacturer_family", "series_count",
                "any_3d", "any_gt78", "any_gt100", "any_gt200"
            ] if c in domain.columns
        ]
        selected = selected.merge(domain[keep], on="StudyInstanceUID", how="left", validate="one_to_one")

    records: list[dict] = []
    manifest_rows = []
    for _, row in selected.iterrows():
        is_gold = bool(row.get("repository_gold", False))
        uid = str(row["StudyInstanceUID"])
        record = {
            "audit_version": REPORT_GAP_AUDIT_VERSION,
            "sample_stratum": str(row["sample_stratum"]),
            "StudyInstanceUID": uid,
            "report_script_bucket": str(row["report_script_bucket"]),
            "report_text": str(row.get("Report", "")),
            "repository_gold": is_gold,
        }
        if is_gold:
            record["official_labels"] = {
                target: int(float(row[target])) for target in TARGETS
            }
            usable = positive = negative = None
        else:
            usable, positive, negative, per_target = _b6_cell_summary(row)
            record["b6"] = {
                "min_confidence_for_usable_cell": float(B7_MIN_CONFIDENCE),
                "usable_cells": int(usable),
                "positive_cells": int(positive),
                "negative_cells": int(negative),
                "targets": per_target,
            }
        for key in [
            "dominant_manufacturer_family", "series_count", "any_3d",
            "any_gt78", "any_gt100", "any_gt200"
        ]:
            if key in row.index and not pd.isna(row[key]):
                value = row[key]
                if isinstance(value, (np.bool_, bool)):
                    value = bool(value)
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                record[key] = value
        records.append(record)
        manifest_rows.append({
            "sample_stratum": str(row["sample_stratum"]),
            "StudyInstanceUID": uid,
            "report_script_bucket": str(row["report_script_bucket"]),
            "repository_gold": is_gold,
            "report_chars": int(row.get("report_chars", len(str(row.get("Report", ""))))),
            "b6_active": None if is_gold else bool(row.get("b6_active", False)),
            "b6_usable_cells": None if is_gold else int(row.get("b6_usable_cells", 0)),
            "b6_positive_cells": None if is_gold else int(row.get("b6_positive_cells", 0)),
            "b6_negative_cells": None if is_gold else int(row.get("b6_negative_cells", 0)),
            "dominant_manufacturer_family": row.get("dominant_manufacturer_family", None),
            "series_count": row.get("series_count", None),
            "any_3d": row.get("any_3d", None),
            "any_gt78": row.get("any_gt78", None),
        })

    manifest = pd.DataFrame(manifest_rows)
    counts = manifest["sample_stratum"].value_counts().sort_index().to_dict()
    summary = {
        "audit_version": REPORT_GAP_AUDIT_VERSION,
        "salt": REPORT_GAP_SALT,
        "per_stratum_requested": int(per_stratum),
        "selected_studies": int(len(manifest)),
        "stratum_counts": {str(k): int(v) for k, v in counts.items()},
        "gold_nonlatin_selected": int((manifest["sample_stratum"] == "gold_nonlatin_all").sum()),
        "raw_text_export_is_local_only": True,
        "governance": (
            "Inspection sample only. Do not commit report_text_sample.jsonl. "
            "Do not use the sample to modify frozen B6 in place or define B35."
        ),
    }
    return manifest, records, summary


def run_report_gap_audit(
    *,
    data_root: str | Path,
    b6_root: str | Path,
    out_root: str | Path,
    domain_study_csv: str | Path | None = None,
    per_stratum: int = DEFAULT_PER_STRATUM,
) -> dict:
    root = Path(data_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    train = load_train_csv(root / "train.csv")
    b6_frame, policy, audit = load_frozen_b6_export(b6_root)
    del policy, audit
    domain = pd.read_csv(domain_study_csv) if domain_study_csv is not None else None
    manifest, records, summary = build_gap_sample(
        train, b6_frame, per_stratum=per_stratum, domain_study=domain
    )
    manifest.to_csv(out / "sample_manifest.csv", index=False)
    with (out / "report_text_sample.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser("Create a deterministic local-only report supervision gap sample")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--b6-root", required=True)
    ap.add_argument("--domain-study-csv", default=None)
    ap.add_argument("--out-root", default="runs/report_supervision_gap_audit")
    ap.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    args = ap.parse_args()
    if args.per_stratum < 1:
        raise ValueError("--per-stratum must be >=1")
    run_report_gap_audit(
        data_root=args.data_root,
        b6_root=args.b6_root,
        out_root=args.out_root,
        domain_study_csv=args.domain_study_csv,
        per_stratum=args.per_stratum,
    )


if __name__ == "__main__":
    main()
