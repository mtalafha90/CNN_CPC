"""Phase 6: deterministic translation -> frozen-B6 rescue feasibility pilot.

This is a supervision-only experiment. It never modifies B6 v1.2.1 and never
trains an MRI model. A locally executed, pinned open-weight language model
translates the frozen Phase-5 report sample into English. The unchanged B6
parser is then applied to the translation.

The candidate merge is deliberately narrower than B23/B24X/B25X:
translated-B6 cells are eligible only for report-only studies whose ORIGINAL
B6 export contains zero usable cells. Existing B6-active studies are left
bit-for-bit unchanged by the merge.

Raw report text and translations are local analysis artifacts and must not be
committed to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .b23_local_llm import (
    DEFAULT_LOCAL_MODEL,
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_NUM_CTX,
    ModelProvenance,
    make_ollama_backend,
)
from .b6_report_labels import predict_target_b6
from .b7_weak_supervision import B7_MIN_CONFIDENCE
from .constants import TARGETS

TRANSLATION_RESCUE_VERSION = "translation_to_frozen_b6_rescue_pilot_v1"
EXPECTED_SAMPLE_VERSION = "report_supervision_gap_sample_v1"
INACTIVE_STRATA = (
    "latin_b6_inactive",
    "greek_b6_inactive",
    "cyrillic_b6_inactive",
)
ACTIVE_CONTROL_STRATA = (
    "latin_b6_active_control",
    "greek_b6_active_control",
    "cyrillic_b6_active_all",
)
OVERALL_RESCUE_RATE_GATE = 0.75
PER_SCRIPT_RESCUE_RATE_GATE = 0.50

TRANSLATION_SYSTEM_PROMPT = """You are a medical translator.

Translate the supplied knee MRI radiology report faithfully into English.

Requirements:
- Translate the complete diagnostic content; do not summarize.
- Preserve section structure when practical.
- Preserve every assertion, negation, uncertainty/hedge, grade, compartment,
  anatomical structure, laterality, measurement, and comparison statement.
- Do not infer a diagnosis that is not written.
- Do not remove a diagnosis because another sentence says a structure is intact.
- Keep standard MRI abbreviations and add their English expansion only when the
  source wording itself makes the meaning clear.
- Do not answer questions about the report and do not classify the 12 targets.
- Return only the requested JSON object.

The task is translation, not interpretation."""

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _is_usable(state: str, confidence: float) -> bool:
    return state in {"positive", "negated"} and float(confidence) >= B7_MIN_CONFIDENCE


def b6_snapshot(report: str) -> dict:
    targets = {}
    usable = positive = negative = 0
    for target in TARGETS:
        pred = predict_target_b6(str(report), target)
        targets[target] = {
            "state": str(pred.state),
            "confidence": float(pred.confidence),
            "probability": float(pred.probability),
        }
        if _is_usable(pred.state, pred.confidence):
            usable += 1
            positive += int(pred.state == "positive")
            negative += int(pred.state == "negated")
    return {
        "usable_cells": int(usable),
        "positive_cells": int(positive),
        "negative_cells": int(negative),
        "targets": targets,
    }


def _parse_translation(completion: str) -> str:
    try:
        payload = json.loads(str(completion))
    except json.JSONDecodeError as exc:
        raise ValueError("translation backend did not return valid JSON") from exc
    translation = str(payload.get("translation", "")).strip()
    if not translation:
        raise ValueError("translation backend returned an empty translation")
    return translation


def make_ollama_translator(
    *,
    model: str = DEFAULT_LOCAL_MODEL,
    host: str = OLLAMA_DEFAULT_HOST,
    num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
    max_new_tokens: int = 4096,
    seed: int = 2026,
) -> tuple[Callable[[str], str], ModelProvenance]:
    backend, provenance = make_ollama_backend(
        TRANSLATION_SYSTEM_PROMPT,
        model=model,
        host=host,
        num_ctx=int(num_ctx),
        max_new_tokens=int(max_new_tokens),
        seed=int(seed),
        think=False,
        schema=TRANSLATION_SCHEMA,
    )

    def translate(report: str) -> str:
        user = "<report>\n" + str(report).strip() + "\n</report>"
        return _parse_translation(backend(TRANSLATION_SYSTEM_PROMPT, user))

    return translate, provenance


def _candidate_merge(original: dict, translated: dict, *, eligible: bool) -> dict:
    """Merge without ever overriding an original usable B6 cell.

    The Phase-6 candidate is even stricter: `eligible` is true only for
    report-only studies with ZERO original usable cells.
    """
    if not eligible:
        return json.loads(json.dumps(original))

    out = json.loads(json.dumps(original))
    for target in TARGETS:
        old = original["targets"][target]
        new = translated["targets"][target]
        if _is_usable(old["state"], old["confidence"]):
            continue
        if _is_usable(new["state"], new["confidence"]):
            out["targets"][target] = dict(new)

    states = [out["targets"][target]["state"] for target in TARGETS]
    conf = [float(out["targets"][target]["confidence"]) for target in TARGETS]
    usable_flags = [_is_usable(s, c) for s, c in zip(states, conf)]
    out["usable_cells"] = int(sum(usable_flags))
    out["positive_cells"] = int(
        sum(flag and state == "positive" for flag, state in zip(usable_flags, states))
    )
    out["negative_cells"] = int(
        sum(flag and state == "negated" for flag, state in zip(usable_flags, states))
    )
    return out


def _gold_diagnostic(records: list[dict], translated_snapshots: dict[str, dict]) -> dict:
    """Descriptive only: the gold reports are reused and partially inspected."""
    definite = correct = positive_calls = negative_calls = 0
    positive_correct = negative_correct = 0
    total_cells = 0
    for record in records:
        if not bool(record.get("repository_gold", False)):
            continue
        uid = str(record["StudyInstanceUID"])
        labels = record.get("official_labels") or {}
        snapshot = translated_snapshots.get(uid)
        if snapshot is None:
            continue
        for target in TARGETS:
            total_cells += 1
            cell = snapshot["targets"][target]
            if not _is_usable(cell["state"], cell["confidence"]):
                continue
            definite += 1
            truth = int(labels[target])
            pred = int(cell["state"] == "positive")
            correct += int(pred == truth)
            if pred:
                positive_calls += 1
                positive_correct += int(truth == 1)
            else:
                negative_calls += 1
                negative_correct += int(truth == 0)
    return {
        "role": "reused_gold_diagnostic_only_not_acceptance_or_promotion",
        "gold_studies": int(sum(bool(r.get("repository_gold", False)) for r in records)),
        "official_cells": int(total_cells),
        "translated_b6_definite_calls": int(definite),
        "translated_b6_definite_coverage": float(definite / total_cells) if total_cells else 0.0,
        "translated_b6_definite_accuracy": float(correct / definite) if definite else None,
        "translated_b6_positive_call_precision": (
            float(positive_correct / positive_calls) if positive_calls else None
        ),
        "translated_b6_negative_call_precision": (
            float(negative_correct / negative_calls) if negative_calls else None
        ),
    }


def run_translation_rescue_pilot(
    *,
    sample_jsonl: str | Path,
    out_root: str | Path,
    translate: Callable[[str], str],
    provenance: dict | ModelProvenance | None = None,
) -> dict:
    sample_path = Path(sample_jsonl)
    records = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Phase-6 sample is empty")
    versions = {str(row.get("audit_version", "")) for row in records}
    if versions != {EXPECTED_SAMPLE_VERSION}:
        raise ValueError(f"unexpected Phase-5 sample version(s): {sorted(versions)}")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    local_records: list[dict] = []
    translated_snapshots: dict[str, dict] = {}
    failures: list[dict] = []

    for record in records:
        uid = str(record["StudyInstanceUID"])
        report = str(record.get("report_text", ""))
        stratum = str(record.get("sample_stratum", ""))
        script = str(record.get("report_script_bucket", ""))
        is_gold = bool(record.get("repository_gold", False))

        original = b6_snapshot(report)
        try:
            translated_text = translate(report)
            translated = b6_snapshot(translated_text)
            translated_snapshots[uid] = translated
            translation_ok = True
            error = ""
        except Exception as exc:
            translated_text = ""
            translated = {
                "usable_cells": 0,
                "positive_cells": 0,
                "negative_cells": 0,
                "targets": {},
            }
            translation_ok = False
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"StudyInstanceUID": uid, "sample_stratum": stratum, "error": error})

        eligible = (
            (not is_gold)
            and stratum in INACTIVE_STRATA
            and int(original["usable_cells"]) == 0
            and translation_ok
        )
        candidate = _candidate_merge(original, translated, eligible=eligible) if translation_ok else original
        added = max(0, int(candidate["usable_cells"]) - int(original["usable_cells"]))
        added_pos = max(0, int(candidate["positive_cells"]) - int(original["positive_cells"]))
        added_neg = max(0, int(candidate["negative_cells"]) - int(original["negative_cells"]))

        original_cells_preserved = True
        if stratum in ACTIVE_CONTROL_STRATA:
            original_cells_preserved = candidate == original

        result_rows.append(
            {
                "StudyInstanceUID": uid,
                "sample_stratum": stratum,
                "report_script_bucket": script,
                "repository_gold": is_gold,
                "translation_success": bool(translation_ok),
                "translation_error": error,
                "original_b6_usable_cells": int(original["usable_cells"]),
                "translated_b6_usable_cells": int(translated["usable_cells"]),
                "translated_b6_positive_cells": int(translated["positive_cells"]),
                "translated_b6_negative_cells": int(translated["negative_cells"]),
                "rescue_eligible": bool(eligible),
                "candidate_usable_cells": int(candidate["usable_cells"]),
                "added_usable_cells": int(added),
                "added_positive_cells": int(added_pos),
                "added_negative_cells": int(added_neg),
                "rescued_to_active": bool(eligible and added > 0),
                "original_active_control_preserved": bool(original_cells_preserved),
            }
        )
        local_records.append(
            {
                "version": TRANSLATION_RESCUE_VERSION,
                "StudyInstanceUID": uid,
                "sample_stratum": stratum,
                "report_script_bucket": script,
                "repository_gold": is_gold,
                "translation": translated_text,
                "translated_b6": translated,
            }
        )

    audit = pd.DataFrame(result_rows)
    inactive = audit[audit["sample_stratum"].isin(INACTIVE_STRATA)].copy()
    if len(inactive) != 36:
        raise RuntimeError(
            f"expected 36 frozen inactive-sample studies, found {len(inactive)}"
        )

    strata_summary = {}
    per_script_gate_pass = True
    per_script_balance_pass = True
    for stratum in INACTIVE_STRATA:
        part = inactive[inactive["sample_stratum"].eq(stratum)]
        rescue_rate = float(part["rescued_to_active"].mean()) if len(part) else 0.0
        pos = int(part["added_positive_cells"].sum())
        neg = int(part["added_negative_cells"].sum())
        gate = bool(rescue_rate >= PER_SCRIPT_RESCUE_RATE_GATE)
        balance = bool(pos > 0 and neg > 0)
        per_script_gate_pass &= gate
        per_script_balance_pass &= balance
        strata_summary[stratum] = {
            "studies": int(len(part)),
            "translations_successful": int(part["translation_success"].sum()),
            "rescued_to_b6_active": int(part["rescued_to_active"].sum()),
            "rescue_rate": rescue_rate,
            "added_usable_cells": int(part["added_usable_cells"].sum()),
            "added_positive_cells": pos,
            "added_negative_cells": neg,
            "rescue_rate_gate": float(PER_SCRIPT_RESCUE_RATE_GATE),
            "rescue_rate_gate_passed": gate,
            "both_positive_and_negative_cells_recovered": balance,
        }

    overall_rate = float(inactive["rescued_to_active"].mean())
    active_controls = audit[audit["sample_stratum"].isin(ACTIVE_CONTROL_STRATA)]
    active_preserved = bool(active_controls["original_active_control_preserved"].all())
    no_failures = len(failures) == 0
    overall_gate = bool(overall_rate >= OVERALL_RESCUE_RATE_GATE)

    if isinstance(provenance, ModelProvenance):
        provenance_dict = provenance.to_dict()
        provenance_reproducible = bool(provenance.reproducible)
    elif provenance is None:
        provenance_dict = {"backend": "injected_or_unspecified"}
        provenance_reproducible = None
    else:
        provenance_dict = dict(provenance)
        provenance_reproducible = provenance_dict.get("reproducible")

    feasibility = bool(
        no_failures
        and active_preserved
        and overall_gate
        and per_script_gate_pass
        and per_script_balance_pass
    )

    summary = {
        "version": TRANSLATION_RESCUE_VERSION,
        "input_sample_version": EXPECTED_SAMPLE_VERSION,
        "input_sample_sha256": _sha256_file(sample_path),
        "selected_studies": int(len(audit)),
        "translation_failures": int(len(failures)),
        "inactive_sample_studies": int(len(inactive)),
        "inactive_rescued_to_active": int(inactive["rescued_to_active"].sum()),
        "inactive_overall_rescue_rate": overall_rate,
        "inactive_added_usable_cells": int(inactive["added_usable_cells"].sum()),
        "inactive_added_positive_cells": int(inactive["added_positive_cells"].sum()),
        "inactive_added_negative_cells": int(inactive["added_negative_cells"].sum()),
        "strata": strata_summary,
        "active_control_studies": int(len(active_controls)),
        "active_control_original_b6_cells_preserved": active_preserved,
        "predeclared_feasibility_rules": {
            "zero_translation_failures": True,
            "overall_inactive_rescue_rate_at_least": float(OVERALL_RESCUE_RATE_GATE),
            "each_inactive_script_stratum_rescue_rate_at_least": float(PER_SCRIPT_RESCUE_RATE_GATE),
            "each_inactive_script_stratum_recovers_positive_and_negative_cells": True,
            "all_original_b6_active_controls_preserved": True,
        },
        "feasibility_passed": feasibility,
        "gold_diagnostic": _gold_diagnostic(records, translated_snapshots),
        "translator_provenance": provenance_dict,
        "translator_provenance_reproducible": provenance_reproducible,
        "governance": {
            "b6_version_modified": False,
            "translated_cells_replace_original_b6_cells": False,
            "translated_cells_eligible_only_when_original_study_has_zero_usable_b6_cells": True,
            "mri_model_trained": False,
            "b35_defined": False,
            "raw_translation_output_local_only": True,
            "gold_role": "reused diagnostic safety only",
            "feasibility_meaning": (
                "coverage-mechanism feasibility only; passing does not authorize model promotion "
                "or establish clinical label accuracy"
            ),
        },
    }

    audit.to_csv(out / "pilot_cell_audit.csv", index=False)
    (out / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out / "translation_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in local_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if failures:
        pd.DataFrame(failures).to_csv(out / "translation_failures.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        "Phase-6 deterministic translation -> frozen-B6 rescue feasibility pilot"
    )
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--out-root", default="runs/report_translation_rescue_pilot")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST)
    parser.add_argument("--num-ctx", type=int, default=OLLAMA_DEFAULT_NUM_CTX)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    translate, provenance = make_ollama_translator(
        model=args.model,
        host=args.ollama_host,
        num_ctx=args.num_ctx,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    if not provenance.reproducible:
        raise RuntimeError(
            "Phase-6 translation provenance is not reproducible; use a pinned local "
            "open-weight backend before running the pilot"
        )

    run_translation_rescue_pilot(
        sample_jsonl=args.sample_jsonl,
        out_root=args.out_root,
        translate=translate,
        provenance=provenance,
    )


if __name__ == "__main__":
    main()
