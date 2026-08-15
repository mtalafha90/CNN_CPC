"""Exploratory supervision built from a user-supplied ChatGPT hybrid cache.

This source is intentionally NOT treated as B23.  The cache may contain labels
assembled from more than one historical LLM run, so its original per-entry
provenance is mixed/unknown.  The file itself is pinned by SHA-256 and can be
used reproducibly as a derived weak-label artifact, but it cannot satisfy the
formal B23 provenance contract.

The exporter matches cache entries back to competition studies through the
same normalized report SHA-1 used by B23.  Raw model confidence is preserved
for diagnostics only.  Downstream supervision follows the frozen B23 state
policy:

    positive   -> probability 0.97, confidence 0.90
    negated    -> probability 0.03, confidence 0.90
    uncertain  -> probability 0.50, confidence 0.00
    unmentioned-> probability 0.50, confidence 0.00

Thus report silence is never converted to a negative label, even when a hybrid
cache row carries a high raw confidence for an ``unmentioned`` state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .b23_llm_labels import (
    B23_DEFINITE_STATE_CONFIDENCE,
    B23_IGNORED_STATE_CONFIDENCE,
    B23_NEGATED_PROBABILITY,
    B23_POSITIVE_PROBABILITY,
    B23_STATES,
    B23_UNCERTAIN_PROBABILITY,
    B23_UNMENTIONED_PROBABILITY,
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
)
from .constants import TARGETS
from .data import gold_mask, load_train_csv, report_hash

HYBRID_VERSION = "1.0.0"
HYBRID_EXPERIMENT = "B25X_chatgpt_hybrid_supervision_v1"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_probability(state: str) -> float:
    if state == STATE_POSITIVE:
        return float(B23_POSITIVE_PROBABILITY)
    if state == STATE_NEGATED:
        return float(B23_NEGATED_PROBABILITY)
    if state == STATE_UNCERTAIN:
        return float(B23_UNCERTAIN_PROBABILITY)
    return float(B23_UNMENTIONED_PROBABILITY)


def _usable_confidence(state: str) -> float:
    if state in (STATE_POSITIVE, STATE_NEGATED):
        return float(B23_DEFINITE_STATE_CONFIDENCE)
    return float(B23_IGNORED_STATE_CONFIDENCE)


def load_hybrid_cache(path: str | Path) -> tuple[dict[str, dict], dict]:
    """Validate the JSONL cache and index it by normalized report SHA-1."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    by_hash: dict[str, dict] = {}
    cache_keys: set[str] = set()
    state_counts = {state: 0 for state in B23_STATES}

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"cache row {line_number} is not an object")
            report_sha1 = str(row.get("report_sha1", "")).strip()
            cache_key = str(row.get("cache_key", "")).strip()
            findings = row.get("findings")
            if len(report_sha1) != 40:
                raise ValueError(f"cache row {line_number} has invalid report_sha1")
            if not cache_key:
                raise ValueError(f"cache row {line_number} is missing cache_key")
            if report_sha1 in by_hash:
                raise ValueError(f"duplicate report_sha1 in hybrid cache: {report_sha1}")
            if cache_key in cache_keys:
                raise ValueError(f"duplicate cache_key in hybrid cache: {cache_key}")
            if not isinstance(findings, dict) or set(findings) != set(TARGETS):
                raise ValueError(
                    f"cache row {line_number} findings must contain exactly the 12 targets"
                )

            clean_findings: dict[str, dict] = {}
            for target in TARGETS:
                cell = findings[target]
                if not isinstance(cell, dict):
                    raise ValueError(f"row {line_number} target {target!r} is not an object")
                state = str(cell.get("state", "")).strip().lower()
                if state not in B23_STATES:
                    raise ValueError(
                        f"row {line_number} target {target!r} has invalid state {state!r}"
                    )
                try:
                    raw_confidence = float(cell.get("confidence", 0.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"row {line_number} target {target!r} has invalid confidence"
                    ) from exc
                if not 0.0 <= raw_confidence <= 1.0:
                    raise ValueError(
                        f"row {line_number} target {target!r} confidence outside [0,1]"
                    )
                evidence = str(cell.get("evidence", ""))
                clean_findings[target] = {
                    "state": state,
                    "confidence": raw_confidence,
                    "evidence": evidence,
                }
                state_counts[state] += 1

            by_hash[report_sha1] = {
                "cache_key": cache_key,
                "report_sha1": report_sha1,
                "findings": clean_findings,
            }
            cache_keys.add(cache_key)

    if not by_hash:
        raise ValueError("hybrid cache contains no usable rows")

    return by_hash, {
        "cache_entries": len(by_hash),
        "unique_report_hashes": len(by_hash),
        "unique_cache_keys": len(cache_keys),
        "state_counts": state_counts,
        "cache_file_sha256": _sha256_file(source),
    }


def build_hybrid_export(
    train_csv: str | Path,
    cache_path: str | Path,
    *,
    out_root: str | Path,
) -> dict:
    """Convert the hybrid cache into a safe B7-compatible exploratory export."""
    train = load_train_csv(train_csv).copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    train["report_sha1"] = train["Report"].fillna("").astype(str).map(report_hash)
    train["is_gold"] = gold_mask(train).astype(bool)

    cache, cache_meta = load_hybrid_cache(cache_path)
    train_hashes = set(train["report_sha1"].astype(str))
    cache_hashes = set(cache)
    matched_cache_hashes = cache_hashes & train_hashes
    orphan_cache_hashes = cache_hashes - train_hashes

    rows: list[dict] = []
    for item in train.itertuples(index=False):
        uid = str(item.StudyInstanceUID)
        sha1 = str(item.report_sha1)
        cached = cache.get(sha1)
        matched = cached is not None
        row: dict = {
            "StudyInstanceUID": uid,
            "report_sha1": sha1,
            "hybrid_cache_match": bool(matched),
            "hybrid_cache_key": str(cached["cache_key"]) if matched else "",
            "is_gold": bool(item.is_gold),
        }
        for target in TARGETS:
            if matched:
                cell = cached["findings"][target]
                state = str(cell["state"])
                raw_confidence = float(cell["confidence"])
                evidence = str(cell["evidence"])
            else:
                state = STATE_UNMENTIONED
                raw_confidence = 0.0
                evidence = ""

            row[target] = _state_probability(state)
            row[f"{target}__confidence"] = _usable_confidence(state)
            row[f"{target}__state"] = state
            row[f"{target}__model_confidence"] = raw_confidence
            row[f"{target}__evidence"] = evidence
        rows.append(row)

    structured = pd.DataFrame(rows)
    non_gold = structured.loc[~structured["is_gold"].astype(bool)].copy()

    training_columns = ["StudyInstanceUID"]
    for target in TARGETS:
        training_columns.extend([target, f"{target}__confidence", f"{target}__state"])

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    structured.to_csv(out / "structured_labels.csv", index=False)
    non_gold[training_columns].to_csv(out / "training_targets.csv", index=False)

    per_target: dict[str, dict] = {}
    usable_total = 0
    positive_total = 0
    negative_total = 0
    active_mask = pd.Series(False, index=non_gold.index)

    for target in TARGETS:
        state = non_gold[f"{target}__state"].astype(str)
        positive = state.eq(STATE_POSITIVE)
        negative = state.eq(STATE_NEGATED)
        usable = positive | negative
        active_mask |= usable
        usable_total += int(usable.sum())
        positive_total += int(positive.sum())
        negative_total += int(negative.sum())
        per_target[target] = {
            "positive_cells": int(positive.sum()),
            "negative_cells": int(negative.sum()),
            "uncertain_cells": int(state.eq(STATE_UNCERTAIN).sum()),
            "unmentioned_cells": int(state.eq(STATE_UNMENTIONED).sum()),
            "usable_cells": int(usable.sum()),
        }

    possible_total = int(len(non_gold) * len(TARGETS))
    matched_studies = int(structured["hybrid_cache_match"].sum())
    matched_non_gold = int(non_gold["hybrid_cache_match"].sum())
    matched_gold = int(
        structured.loc[structured["is_gold"].astype(bool), "hybrid_cache_match"].sum()
    )

    audit = {
        "experiment": HYBRID_EXPERIMENT,
        "version": HYBRID_VERSION,
        "scope": "full_train_index_with_unmatched_rows_silent",
        "exploratory": True,
        "formal_b23_compatible": False,
        "formal_b24_eligible": False,
        "gold_acceptance_allowed": False,
        "source_provenance": "mixed_or_unknown_original_llm_provenance",
        "derived_artifact_reproducible": True,
        "cache_file_sha256": cache_meta["cache_file_sha256"],
        "cache_entries": cache_meta["cache_entries"],
        "cache_unique_report_hashes": cache_meta["unique_report_hashes"],
        "cache_unique_keys": cache_meta["unique_cache_keys"],
        "cache_hashes_matching_train": int(len(matched_cache_hashes)),
        "cache_hashes_not_in_train": int(len(orphan_cache_hashes)),
        "n_studies": int(len(structured)),
        "n_gold_audit_only": int(structured["is_gold"].sum()),
        "n_report_only_training": int(len(non_gold)),
        "matched_studies": matched_studies,
        "unmatched_studies": int(len(structured) - matched_studies),
        "matched_non_gold_studies": matched_non_gold,
        "matched_gold_studies": matched_gold,
        "gold_rows_in_training_targets": 0,
        "active_training_studies": int(active_mask.sum()),
        "inactive_training_studies_zero_usable_cells": int((~active_mask).sum()),
        "usable_cells_total": int(usable_total),
        "positive_cells_total": int(positive_total),
        "negative_cells_total": int(negative_total),
        "possible_cells_total": possible_total,
        "cell_coverage": float(usable_total / possible_total) if possible_total else 0.0,
        "raw_cache_state_counts": cache_meta["state_counts"],
        "targets": per_target,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    policy = {
        "experiment": HYBRID_EXPERIMENT,
        "version": HYBRID_VERSION,
        "purpose": "exploratory MRI weak supervision from a ChatGPT-created hybrid report-label cache",
        "exploratory": True,
        "formal_b23_compatible": False,
        "formal_b24_eligible": False,
        "gold_acceptance_allowed": False,
        "source_provenance": "mixed_or_unknown_original_llm_provenance",
        "cache_file_sha256": cache_meta["cache_file_sha256"],
        "matching": "report_sha1 = SHA1(normalize_report(Report))",
        "states": list(B23_STATES),
        "fixed_soft_labels": {
            STATE_POSITIVE: {
                "probability": B23_POSITIVE_PROBABILITY,
                "confidence": B23_DEFINITE_STATE_CONFIDENCE,
            },
            STATE_NEGATED: {
                "probability": B23_NEGATED_PROBABILITY,
                "confidence": B23_DEFINITE_STATE_CONFIDENCE,
            },
            STATE_UNCERTAIN: {
                "probability": B23_UNCERTAIN_PROBABILITY,
                "confidence": B23_IGNORED_STATE_CONFIDENCE,
            },
            STATE_UNMENTIONED: {
                "probability": B23_UNMENTIONED_PROBABILITY,
                "confidence": B23_IGNORED_STATE_CONFIDENCE,
            },
        },
        "raw_model_confidence_usage": "diagnostic only; never thresholds supervision",
        "unmentioned_is_negative": False,
        "gold_usage": "excluded from training_targets.csv",
        "unmatched_report_policy": "unmentioned with zero usable confidence for all targets",
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return audit


def load_hybrid_export(root: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load a derived hybrid export while preserving its exploratory status."""
    root = Path(root)
    targets_path = root / "training_targets.csv"
    policy_path = root / "policy.json"
    audit_path = root / "audit.json"
    for path in (targets_path, policy_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"hybrid export is missing artifact: {path}")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(policy.get("version")) != HYBRID_VERSION or str(audit.get("version")) != HYBRID_VERSION:
        raise ValueError(f"expected hybrid export v{HYBRID_VERSION}")
    if not bool(policy.get("exploratory", False)) or not bool(audit.get("exploratory", False)):
        raise ValueError("hybrid export must remain explicitly exploratory")
    if bool(policy.get("formal_b23_compatible", True)) or bool(audit.get("formal_b23_compatible", True)):
        raise ValueError("hybrid export must never masquerade as formal B23")
    if int(audit.get("gold_rows_in_training_targets", -1)) != 0:
        raise ValueError("hybrid export does not certify zero gold training rows")
    if bool(policy.get("unmentioned_is_negative", True)):
        raise ValueError("hybrid supervision must not map report silence to negative")

    frame = pd.read_csv(targets_path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError("hybrid training_targets.csv is missing StudyInstanceUID")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("hybrid training_targets.csv contains duplicate StudyInstanceUID values")
    return frame, policy, audit
