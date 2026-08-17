"""Phase 7: full-population translation -> frozen-B6 rescue audit.

This stage is descriptive supervision generation only. It translates exactly the
1,229 report-only studies that have ZERO usable cells in frozen B6 v1.2.1,
re-applies unchanged B6 to the English translation, and records the cells that
would be recovered. B6-active studies are never translated for rescue and no MRI
model is trained here.

The translator provenance is frozen to the successful Phase-6 pilot. Raw
translations are local-only artifacts. The run is resumable because 1,229 local
LLM translations may take many hours.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .b23_local_llm import ModelProvenance
from .b7_weak_supervision import B7_MIN_CONFIDENCE, load_frozen_b6_export
from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .dataset_contract_audit import report_script_profile
from .report_translation_rescue_pilot import b6_snapshot, make_ollama_translator

PHASE7_VERSION = "translation_to_frozen_b6_full_inactive_audit_v1"
EXPECTED_REPORT_ONLY = 4349
EXPECTED_B6_ACTIVE = 3120
EXPECTED_B6_INACTIVE = 1229
EXPECTED_ORIGINAL_USABLE_CELLS = 14123

# Frozen from the successful Phase-6 pilot result.
PHASE6_MODEL_ID = "qwen3:14b"
PHASE6_OLLAMA_DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
PHASE6_PROMPT_SHA256 = "086e1daae2843c70712a29662a589dee629d32d7f014a9a51613be496a95ee1a"
PHASE6_QUANTISATION = "Q4_K_M"
PHASE6_SEED = 2026
PHASE6_MAX_NEW_TOKENS = 4096


def _is_usable(state: str, confidence: float) -> bool:
    return str(state) in {"positive", "negated"} and float(confidence) >= B7_MIN_CONFIDENCE


def _original_b6_summary(row: pd.Series) -> dict:
    targets = {}
    usable = positive = negative = 0
    for target in TARGETS:
        state = str(row.get(f"{target}__state", "") or "")
        try:
            confidence = float(row.get(f"{target}__confidence", 0.0))
        except Exception:
            confidence = 0.0
        use = _is_usable(state, confidence)
        usable += int(use)
        positive += int(use and state == "positive")
        negative += int(use and state == "negated")
        targets[target] = {"state": state, "confidence": confidence}
    return {
        "usable_cells": int(usable),
        "positive_cells": int(positive),
        "negative_cells": int(negative),
        "targets": targets,
    }


def validate_phase6_provenance(provenance: ModelProvenance) -> None:
    """Abort if the full run is not using the exact Phase-6 translator."""
    failures = []
    if not provenance.reproducible:
        failures.append("provenance is not reproducible")
    if provenance.model_id != PHASE6_MODEL_ID:
        failures.append(f"model_id={provenance.model_id!r}")
    if provenance.revision != PHASE6_OLLAMA_DIGEST:
        failures.append(f"revision={provenance.revision!r}")
    if provenance.prompt_sha256 != PHASE6_PROMPT_SHA256:
        failures.append(f"prompt_sha256={provenance.prompt_sha256!r}")
    if provenance.quantisation != PHASE6_QUANTISATION:
        failures.append(f"quantisation={provenance.quantisation!r}")
    if int(provenance.seed) != PHASE6_SEED:
        failures.append(f"seed={provenance.seed!r}")
    if int(provenance.max_new_tokens) != PHASE6_MAX_NEW_TOKENS:
        failures.append(f"max_new_tokens={provenance.max_new_tokens!r}")
    if failures:
        raise RuntimeError(
            "Phase-7 translator does not match frozen Phase-6 provenance: " + "; ".join(failures)
        )


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache: dict[str, dict] = {}
    for row in rows:
        uid = str(row["StudyInstanceUID"])
        if uid in cache:
            raise RuntimeError(f"duplicate UID in Phase-7 cache: {uid}")
        cache[uid] = row
    return cache


def _append_cache(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _target_rows(uid: str, script: str, translated: dict) -> list[dict]:
    rows = []
    for target in TARGETS:
        cell = translated["targets"][target]
        if not _is_usable(cell["state"], cell["confidence"]):
            continue
        rows.append(
            {
                "StudyInstanceUID": uid,
                "report_script_bucket": script,
                "target": target,
                "state": str(cell["state"]),
                "confidence": float(cell["confidence"]),
                "probability": float(cell["probability"]),
            }
        )
    return rows


def _domain_summary(audit: pd.DataFrame, domain_csv: str | Path | None) -> dict | None:
    if domain_csv is None:
        return None
    domain = pd.read_csv(domain_csv)
    domain["StudyInstanceUID"] = domain["StudyInstanceUID"].astype(str)
    keep = [
        c for c in [
            "StudyInstanceUID", "dominant_manufacturer_family", "any_3d",
            "any_gt78", "any_gt100", "any_gt200"
        ] if c in domain.columns
    ]
    merged = audit.merge(domain[keep], on="StudyInstanceUID", how="left", validate="one_to_one")
    result = {}
    for label, part in [("rescued", merged[merged["rescued_to_active"]]),
                        ("unrecovered", merged[~merged["rescued_to_active"]])]:
        entry = {"studies": int(len(part))}
        for flag in ["any_3d", "any_gt78", "any_gt100", "any_gt200"]:
            if flag in part.columns:
                vals = part[flag].fillna(False).astype(bool)
                entry[flag] = int(vals.sum())
                entry[f"{flag}_fraction"] = float(vals.mean()) if len(vals) else 0.0
        if "dominant_manufacturer_family" in part.columns:
            counts = part["dominant_manufacturer_family"].fillna("Unknown").astype(str).value_counts()
            entry["manufacturer_family_counts"] = {str(k): int(v) for k, v in counts.items()}
        result[label] = entry
    return result


def run_full_translation_rescue(
    *,
    data_root: str | Path,
    b6_root: str | Path,
    out_root: str | Path,
    translate: Callable[[str], str],
    provenance: ModelProvenance,
    domain_study_csv: str | Path | None = None,
) -> dict:
    validate_phase6_provenance(provenance)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "translation_cache.jsonl"

    train = load_train_csv(Path(data_root) / "train.csv")
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    b6, policy, b6_audit = load_frozen_b6_export(b6_root)
    del policy
    b6["StudyInstanceUID"] = b6["StudyInstanceUID"].astype(str)

    report_only = train.loc[~gold_mask(train), ["StudyInstanceUID", "Report"]].copy()
    if len(report_only) != EXPECTED_REPORT_ONLY or len(b6) != EXPECTED_REPORT_ONLY:
        raise RuntimeError(
            f"expected {EXPECTED_REPORT_ONLY} report-only/B6 rows; "
            f"train={len(report_only)}, b6={len(b6)}"
        )
    work = report_only.merge(b6, on="StudyInstanceUID", how="left", validate="one_to_one")
    summaries = work.apply(_original_b6_summary, axis=1)
    work["original_b6_usable_cells"] = [x["usable_cells"] for x in summaries]
    work["original_b6_positive_cells"] = [x["positive_cells"] for x in summaries]
    work["original_b6_negative_cells"] = [x["negative_cells"] for x in summaries]
    work["report_script_bucket"] = work["Report"].fillna("").map(
        lambda x: report_script_profile(x)["bucket"]
    )

    active = work[work["original_b6_usable_cells"].gt(0)]
    inactive = work[work["original_b6_usable_cells"].eq(0)].copy()
    original_cells = int(work["original_b6_usable_cells"].sum())
    if len(active) != EXPECTED_B6_ACTIVE or len(inactive) != EXPECTED_B6_INACTIVE:
        raise RuntimeError(
            f"frozen population mismatch: active={len(active)}, inactive={len(inactive)}"
        )
    if original_cells != EXPECTED_ORIGINAL_USABLE_CELLS:
        raise RuntimeError(
            f"frozen B6 usable-cell mismatch: expected {EXPECTED_ORIGINAL_USABLE_CELLS}, "
            f"got {original_cells}"
        )

    cache = _load_cache(cache_path)
    expected_uids = set(inactive["StudyInstanceUID"])
    extra_cache = set(cache).difference(expected_uids)
    if extra_cache:
        raise RuntimeError(f"Phase-7 cache contains {len(extra_cache)} non-eligible UID(s)")

    failures: list[dict] = []
    ordered = inactive.sort_values("StudyInstanceUID", kind="mergesort").reset_index(drop=True)
    for i, row in ordered.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid in cache:
            continue
        try:
            translated_text = translate(str(row["Report"]))
            translated = b6_snapshot(translated_text)
            cache_row = {
                "version": PHASE7_VERSION,
                "StudyInstanceUID": uid,
                "report_script_bucket": str(row["report_script_bucket"]),
                "translation": translated_text,
                "translated_b6": translated,
            }
            _append_cache(cache_path, cache_row)
            cache[uid] = cache_row
        except Exception as exc:
            failures.append({
                "StudyInstanceUID": uid,
                "report_script_bucket": str(row["report_script_bucket"]),
                "error": f"{type(exc).__name__}: {exc}",
            })
        done = len(cache)
        if done % 25 == 0 or done == len(ordered):
            print(f"[Phase 7] translated {done}/{len(ordered)} eligible reports")

    audit_rows = []
    cell_rows = []
    for _, row in ordered.iterrows():
        uid = str(row["StudyInstanceUID"])
        script = str(row["report_script_bucket"])
        cached = cache.get(uid)
        if cached is None:
            audit_rows.append({
                "StudyInstanceUID": uid,
                "report_script_bucket": script,
                "translation_success": False,
                "translated_b6_usable_cells": 0,
                "added_positive_cells": 0,
                "added_negative_cells": 0,
                "added_usable_cells": 0,
                "rescued_to_active": False,
            })
            continue
        translated = cached["translated_b6"]
        pos = int(translated["positive_cells"])
        neg = int(translated["negative_cells"])
        usable = int(translated["usable_cells"])
        audit_rows.append({
            "StudyInstanceUID": uid,
            "report_script_bucket": script,
            "translation_success": True,
            "translated_b6_usable_cells": usable,
            "added_positive_cells": pos,
            "added_negative_cells": neg,
            "added_usable_cells": usable,
            "rescued_to_active": bool(usable > 0),
        })
        cell_rows.extend(_target_rows(uid, script, translated))

    audit = pd.DataFrame(audit_rows)
    cells = pd.DataFrame(cell_rows, columns=[
        "StudyInstanceUID", "report_script_bucket", "target", "state", "confidence", "probability"
    ])
    audit.to_csv(out / "full_population_rescue_audit.csv", index=False)
    cells.to_csv(out / "recovered_cells.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "translation_failures.csv", index=False)

    script_summary = {}
    for script, part in audit.groupby("report_script_bucket", dropna=False):
        script_summary[str(script)] = {
            "studies": int(len(part)),
            "translations_successful": int(part["translation_success"].sum()),
            "rescued_to_active": int(part["rescued_to_active"].sum()),
            "rescue_rate": float(part["rescued_to_active"].mean()) if len(part) else 0.0,
            "added_usable_cells": int(part["added_usable_cells"].sum()),
            "added_positive_cells": int(part["added_positive_cells"].sum()),
            "added_negative_cells": int(part["added_negative_cells"].sum()),
        }

    target_summary = {}
    if not cells.empty:
        for target in TARGETS:
            part = cells[cells["target"].eq(target)]
            target_summary[target] = {
                "added_usable_cells": int(len(part)),
                "added_positive_cells": int(part["state"].eq("positive").sum()),
                "added_negative_cells": int(part["state"].eq("negated").sum()),
            }
    else:
        target_summary = {
            target: {"added_usable_cells": 0, "added_positive_cells": 0, "added_negative_cells": 0}
            for target in TARGETS
        }

    rescued = int(audit["rescued_to_active"].sum())
    added_cells = int(audit["added_usable_cells"].sum())
    summary = {
        "version": PHASE7_VERSION,
        "translator_provenance": provenance.to_dict(),
        "translator_matches_phase6_exactly": True,
        "frozen_population": {
            "report_only_studies": int(len(work)),
            "original_b6_active_studies": int(len(active)),
            "original_b6_inactive_studies": int(len(inactive)),
            "original_b6_usable_cells": original_cells,
        },
        "translation": {
            "eligible_studies": int(len(inactive)),
            "cached_successful": int(len(cache)),
            "failures_this_run": int(len(failures)),
            "all_eligible_successfully_translated": bool(len(cache) == len(inactive)),
        },
        "rescue": {
            "rescued_to_active": rescued,
            "rescue_rate": float(rescued / len(inactive)),
            "added_usable_cells": added_cells,
            "added_positive_cells": int(audit["added_positive_cells"].sum()),
            "added_negative_cells": int(audit["added_negative_cells"].sum()),
            "candidate_active_studies_if_later_used": int(len(active) + rescued),
            "candidate_usable_cells_if_later_used": int(original_cells + added_cells),
        },
        "by_script": script_summary,
        "by_target": target_summary,
        "domain_recovery": _domain_summary(audit, domain_study_csv),
        "b6_audit_version": str(b6_audit.get("b6_version", "")),
        "governance": {
            "b6_modified": False,
            "only_zero_original_cell_studies_translated_for_rescue": True,
            "partially_silent_b6_active_cells_filled": False,
            "target_specific_rules_added": False,
            "mri_training_authorized": False,
            "b35_defined": False,
            "translation_cache_local_only": True,
            "stage_meaning": (
                "full-population supervision/domain audit only; inspect results before defining "
                "any downstream MRI experiment"
            ),
        },
    }
    (out / "full_population_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser("Phase-7 full B6-inactive translation rescue audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--domain-study-csv", default=None)
    parser.add_argument("--out-root", default="runs/report_translation_rescue_full")
    parser.add_argument("--model", default=PHASE6_MODEL_ID)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=PHASE6_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=PHASE6_SEED)
    args = parser.parse_args()

    translate, provenance = make_ollama_translator(
        model=args.model,
        host=args.ollama_host,
        num_ctx=args.num_ctx,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    validate_phase6_provenance(provenance)
    run_full_translation_rescue(
        data_root=args.data_root,
        b6_root=args.b6_root,
        out_root=args.out_root,
        translate=translate,
        provenance=provenance,
        domain_study_csv=args.domain_study_csv,
    )


if __name__ == "__main__":
    main()
