"""Descriptive audit of Phase-7 rescue evidence for Phase-9 v2.

This module does NOT inspect PV2 predictions and does NOT alter training labels.
It summarizes the frozen Phase-7 translation cache and recovered cells, with
special attention to Contusion versus Effusion.  All outcome-linked filtering
is forbidden: every eligible rescued cell remains in the descriptive tables.

Inputs are local-only Phase-7 artifacts:
  * translation_cache.jsonl
  * full_population_rescue_audit.csv
  * recovered_cells.csv
  * train.csv (for original report text)

Outputs are descriptive audit tables suitable for investigating whether the
observed Phase-9 pathology-specific AUC pattern could plausibly reflect rescue
population composition or translation evidence. No statistical claim about
label correctness is made from lexical overlap alone.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .constants import TARGETS

TARGET_TERMS = {
    "ACL": [r"\bacl\b", r"anterior cruciate ligament"],
    "MCL": [r"\bmcl\b", r"medial collateral ligament"],
    "Medial Meniscus": [r"medial meniscus", r"medial meniscal"],
    "Lateral Meniscus": [r"lateral meniscus", r"lateral meniscal"],
    "Medial OA": [r"medial compartment", r"medial osteoarthritis", r"medial joint space"],
    "Lateral OA": [r"lateral compartment", r"lateral osteoarthritis", r"lateral joint space"],
    "PF OA": [r"patellofemoral", r"patellofemoral osteoarthritis"],
    "Effusion": [r"effusion", r"joint fluid", r"fluid collection"],
    "Synovitis": [r"synovitis", r"synovial thickening"],
    "Baker's": [r"baker'?s", r"popliteal cyst"],
    "Contusion": [r"contusion", r"bone bruise", r"bone marrow edema"],
    "Fracture": [r"fracture", r"fractured"],
}


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _hits(text: str, patterns: list[str]) -> int:
    return int(any(re.search(p, text, flags=re.I) for p in patterns))


def _script_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (script, target), g in cells.groupby(["report_script_bucket", "target"], dropna=False):
        rows.append({
            "report_script_bucket": str(script),
            "target": str(target),
            "rescued_cells": int(len(g)),
            "positive_cells": int((g["state"] == "positive").sum()),
            "negative_cells": int((g["state"] == "negated").sum()),
            "positive_fraction": float((g["state"] == "positive").mean()),
            "mean_confidence": float(pd.to_numeric(g["confidence"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def _target_evidence(cells: pd.DataFrame, cache: dict[str, dict], reports: pd.DataFrame) -> pd.DataFrame:
    report_map = reports.set_index("StudyInstanceUID")["Report"].astype(str).to_dict()
    rows = []
    for target in TARGETS:
        g = cells[cells["target"] == target]
        uids = set(g["StudyInstanceUID"].astype(str))
        if not uids:
            continue
        original_hit = translated_hit = 0
        translated_lengths = []
        original_lengths = []
        script_counts: dict[str, int] = {}
        for uid in uids:
            raw = _norm(report_map.get(uid, ""))
            cached = cache.get(uid, {})
            trans = cached.get("translation", "")
            trans_norm = _norm(trans)
            original_hit += _hits(raw, TARGET_TERMS[target])
            translated_hit += _hits(trans_norm, TARGET_TERMS[target])
            original_lengths.append(len(raw))
            translated_lengths.append(len(trans_norm))
            script = str(cached.get("report_script_bucket", "unknown"))
            script_counts[script] = script_counts.get(script, 0) + 1
        rows.append({
            "target": target,
            "rescued_studies": int(len(uids)),
            "rescued_cells": int(len(g)),
            "original_report_target_term_present_studies": int(original_hit),
            "translated_report_target_term_present_studies": int(translated_hit),
            "original_term_fraction": float(original_hit / len(uids)),
            "translated_term_fraction": float(translated_hit / len(uids)),
            "mean_original_chars": float(sum(original_lengths) / len(original_lengths)),
            "mean_translated_chars": float(sum(translated_lengths) / len(translated_lengths)),
            "script_distribution": json.dumps(script_counts, sort_keys=True),
        })
    return pd.DataFrame(rows)


def _contusion_effusion_cases(cells: pd.DataFrame, cache: dict[str, dict], reports: pd.DataFrame) -> pd.DataFrame:
    report_map = reports.set_index("StudyInstanceUID")["Report"].astype(str).to_dict()
    rows = []
    for target in ["Contusion", "Effusion"]:
        g = cells[cells["target"] == target].copy()
        for uid in sorted(g["StudyInstanceUID"].astype(str).unique()):
            cached = cache.get(uid)
            if cached is None:
                continue
            original = str(report_map.get(uid, ""))
            translated = str(cached.get("translation", ""))
            cell = g[g["StudyInstanceUID"].astype(str) == uid].iloc[0]
            rows.append({
                "target": target,
                "StudyInstanceUID": uid,
                "report_script_bucket": str(cached.get("report_script_bucket", "unknown")),
                "state": str(cell["state"]),
                "confidence": float(cell["confidence"]),
                "probability": float(cell["probability"]),
                "original_report_chars": len(original),
                "translated_report_chars": len(translated),
                "original_target_term_present": _hits(_norm(original), TARGET_TERMS[target]),
                "translated_target_term_present": _hits(_norm(translated), TARGET_TERMS[target]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser("Phase-9 v2 rescue mechanism audit")
    ap.add_argument("--phase7-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="runs/phase9_matched_supervision_v2/rescue_mechanism_audit")
    args = ap.parse_args()

    phase7 = Path(args.phase7_root)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    cache_path = phase7 / "translation_cache.jsonl"
    audit_path = phase7 / "full_population_rescue_audit.csv"
    cells_path = phase7 / "recovered_cells.csv"
    train_path = Path(args.data_root) / "train.csv"
    for path in [cache_path, audit_path, cells_path, train_path]:
        if not path.is_file():
            raise FileNotFoundError(f"required Phase-7/local artifact missing: {path}")

    cache: dict[str, dict] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row["StudyInstanceUID"])
        if uid in cache:
            raise RuntimeError(f"duplicate cache UID: {uid}")
        cache[uid] = row

    audit = pd.read_csv(audit_path)
    cells = pd.read_csv(cells_path)
    train = pd.read_csv(train_path, usecols=["StudyInstanceUID", "Report"])
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    cells["StudyInstanceUID"] = cells["StudyInstanceUID"].astype(str)

    if len(cells) != 3901:
        raise RuntimeError(f"expected 3901 recovered cells, got {len(cells)}")
    if int((cells["state"] == "positive").sum()) != 2719 or int((cells["state"] == "negated").sum()) != 1182:
        raise RuntimeError("recovered-cell class totals do not match frozen Phase-8 contract")

    script = _script_summary(cells)
    evidence = _target_evidence(cells, cache, train)
    contrast = _contusion_effusion_cases(cells, cache, train)

    script.to_csv(out / "rescued_by_script_target.csv", index=False)
    evidence.to_csv(out / "target_translation_evidence_summary.csv", index=False)
    contrast.to_csv(out / "contusion_effusion_rescue_cases.csv", index=False)

    audit_summary = {
        "version": "phase9_v2_rescue_mechanism_audit_v1",
        "phase7_recovered_cells": int(len(cells)),
        "phase7_positive_cells": int((cells["state"] == "positive").sum()),
        "phase7_negative_cells": int((cells["state"] == "negated").sum()),
        "cache_entries": int(len(cache)),
        "contusion_cells": int((cells["target"] == "Contusion").sum()),
        "effusion_cells": int((cells["target"] == "Effusion").sum()),
        "uses_pv2_predictions": False,
        "changes_training_labels": False,
        "target_filtering": False,
        "interpretation_note": "Lexical evidence is descriptive only; target-specific rescue retention or model tuning is forbidden.",
    }
    (out / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    main()
