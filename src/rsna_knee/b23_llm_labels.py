"""B23: LLM-based multilingual report labeller.

B6 v1.2.1 extracts the 12 target states with roughly 626 lines of hand-written
regular expressions. On the reused 58-study expert surface it measures:

```text
sensitivity   0.9749
specificity   0.6061
precision     0.6905
coverage      0.3606
```

Only 14,123 of the 52,188 possible report cells survive that parser, and about
a third of the positives it does emit disagree with expert truth. Every model
from B7 onward is trained on those cells, and the frozen B6 state-only ranking
scores 0.7025 on gold while B20 scores 0.6672 -- the downstream models sit at
roughly 95% of their own supervision.

B23 replaces the parser, not the policy. It keeps the same four states, the
same export contract and the same "report silence is not a negative" rule, so
the resulting export is a drop-in alternative supervision source rather than a
new training recipe. B6 v1.2.1 remains frozen for every historical comparison.

The labeller is deliberately backend-injectable: `run_b23_export` accepts any
callable implementing `ReportLabelBackend`, so the extraction contract can be
tested exactly without network access, and a cached run can resume after an
interruption without re-billing completed reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from .constants import TARGETS
from .data import gold_mask, load_train_csv, normalize_report, report_hash

B23_VERSION = "1.0.0"
B23_EXPERIMENT = "B23_llm_report_labels"

STATE_POSITIVE = "positive"
STATE_NEGATED = "negated"
STATE_UNCERTAIN = "uncertain"
STATE_UNMENTIONED = "unmentioned"
B23_STATES = (STATE_POSITIVE, STATE_NEGATED, STATE_UNCERTAIN, STATE_UNMENTIONED)

# Downstream B7-family supervision reads `probability` and `confidence` and keeps
# a cell only when the state is positive/negated AND confidence >= 0.75. B23 uses
# the model's own per-target confidence for those two states so that a hedged
# extraction is downweighted rather than silently promoted, and pins the two
# ignored states to a confidence that can never clear the usable threshold.
B23_POSITIVE_PROBABILITY = 0.97
B23_NEGATED_PROBABILITY = 0.03
B23_UNCERTAIN_PROBABILITY = 0.50
B23_UNMENTIONED_PROBABILITY = 0.50
B23_IGNORED_STATE_CONFIDENCE = 0.0
B23_MIN_REPORTED_CONFIDENCE = 0.0
B23_MAX_REPORTED_CONFIDENCE = 1.0

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0

TARGET_DEFINITIONS: dict[str, str] = {
    "ACL": "Anterior cruciate ligament tear, rupture, sprain or graft failure.",
    "MCL": "Medial collateral ligament tear, sprain or injury.",
    "Medial Meniscus": "Medial meniscus tear, degeneration or maceration.",
    "Lateral Meniscus": "Lateral meniscus tear, degeneration or maceration.",
    "Medial OA": "Osteoarthritis / cartilage loss / degenerative change in the MEDIAL tibiofemoral compartment.",
    "Lateral OA": "Osteoarthritis / cartilage loss / degenerative change in the LATERAL tibiofemoral compartment.",
    "PF OA": "Osteoarthritis / chondromalacia / cartilage loss in the PATELLOFEMORAL compartment.",
    "Effusion": "Joint effusion or increased intra-articular fluid.",
    "Synovitis": "Synovitis, synovial thickening, synovial proliferation or synovial enhancement.",
    "Baker's": "Baker's cyst, popliteal cyst or gastrocnemio-semimembranosus bursal cyst.",
    "Contusion": "Bone contusion, bone bruise or bone marrow oedema of traumatic origin.",
    "Fracture": "Fracture, including occult, insufficiency, avulsion and osteochondral fracture.",
}

SYSTEM_PROMPT = """You are a careful musculoskeletal radiologist extracting structured findings from knee MRI reports.

Reports may be written in ANY language. Read the report in its original language. Do not translate before deciding; decide from the original text.

For each of the 12 target findings, assign exactly one state:

- "positive": the report asserts the finding is present.
- "negated": the report explicitly states the finding is absent (e.g. "no meniscal tear", "intact ACL", "sin derrame articular").
- "uncertain": the report hedges (possible, suspected, cannot exclude, questionable, borderline).
- "unmentioned": the report says nothing about this finding, either way.

Critical rules:

1. "unmentioned" is NOT "negated". If the report is simply silent about a finding, the state is "unmentioned". Never infer absence from silence. This distinction is the single most important part of the task.
2. A generic normality statement ("normal knee MRI", "unremarkable study") DOES negate the findings it plausibly covers. Use "negated" for those, and keep "unmentioned" for findings such a sentence would not conventionally address.
3. Compartment matters. "Medial OA", "Lateral OA" and "PF OA" are separate findings. Degenerative change described only in one compartment must not be assigned to the others. If the report says "tricompartmental osteoarthritis", all three are positive. If it says "osteoarthritis" with no compartment, mark the specific compartments "uncertain", not positive.
4. Laterality matters for menisci: medial and lateral meniscus are separate findings.
5. Bone marrow oedema of clearly degenerative origin is NOT "Contusion". Contusion means traumatic bone bruise.
6. Post-operative or graft findings: a failed/re-torn ACL graft is "positive" for ACL. An intact graft is "negated".

Report a calibrated confidence in [0,1] for each finding: how certain you are that the STATE you assigned is the correct reading of the report. This is confidence about your extraction, not about the patient's disease.

Also give a short verbatim `evidence` span copied from the report (in its original language) that justifies the state. Use an empty string when the state is "unmentioned".

Return ONLY a JSON object, no prose and no code fences, with exactly this shape:

{"findings": {"<target name>": {"state": "...", "confidence": 0.0, "evidence": "..."}, ...}}

The `findings` object must contain all 12 target names exactly as given, spelled exactly as listed."""


def build_user_prompt(report: str) -> str:
    """Render the per-report extraction request."""
    lines = ["The 12 target findings, with definitions:", ""]
    for target in TARGETS:
        lines.append(f"- {target}: {TARGET_DEFINITIONS[target]}")
    lines.extend(["", "Knee MRI report:", "", "<report>", report.strip(), "</report>"])
    return "\n".join(lines)


@dataclass(frozen=True)
class TargetExtraction:
    """One target's extracted state for one report."""

    state: str
    confidence: float
    evidence: str

    def probability(self) -> float:
        if self.state == STATE_POSITIVE:
            return B23_POSITIVE_PROBABILITY
        if self.state == STATE_NEGATED:
            return B23_NEGATED_PROBABILITY
        if self.state == STATE_UNCERTAIN:
            return B23_UNCERTAIN_PROBABILITY
        return B23_UNMENTIONED_PROBABILITY

    def usable_confidence(self) -> float:
        """Confidence as downstream B7 supervision reads it.

        Uncertain/unmentioned are pinned to zero so the frozen 0.75 usable-cell
        threshold cannot admit them, matching the B6 v1.2.1 policy. Only a
        separately versioned successor may relax that.
        """
        if self.state in (STATE_POSITIVE, STATE_NEGATED):
            return float(self.confidence)
        return B23_IGNORED_STATE_CONFIDENCE


class ReportLabelBackend(Protocol):
    """Anything that can turn one report into raw JSON text."""

    def __call__(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


def parse_extraction_response(text: str) -> dict[str, TargetExtraction]:
    """Validate and coerce one backend response into all 12 target extractions.

    Raises ValueError on anything the downstream export cannot trust: missing
    targets, unknown states, non-numeric or out-of-range confidence. A hard
    failure here is deliberate -- a silently defaulted label is far more
    expensive than a retried request.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Tolerate a fenced block even though the prompt forbids one.
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"backend response is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("backend response must be a JSON object")
    findings = payload.get("findings")
    if not isinstance(findings, dict):
        raise ValueError("backend response is missing a 'findings' object")

    missing = [target for target in TARGETS if target not in findings]
    if missing:
        raise ValueError(f"backend response is missing targets: {missing}")

    out: dict[str, TargetExtraction] = {}
    for target in TARGETS:
        cell = findings[target]
        if not isinstance(cell, dict):
            raise ValueError(f"target {target!r} must map to an object")
        state = str(cell.get("state", "")).strip().lower()
        if state not in B23_STATES:
            raise ValueError(f"target {target!r} has unknown state {state!r}")
        raw_confidence = cell.get("confidence", None)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target {target!r} has non-numeric confidence") from exc
        if not np.isfinite(confidence):
            raise ValueError(f"target {target!r} has non-finite confidence")
        if not B23_MIN_REPORTED_CONFIDENCE <= confidence <= B23_MAX_REPORTED_CONFIDENCE:
            raise ValueError(f"target {target!r} confidence {confidence} outside [0,1]")
        evidence = str(cell.get("evidence", "") or "")
        out[target] = TargetExtraction(state=state, confidence=confidence, evidence=evidence)
    return out


def empty_report_extraction() -> dict[str, TargetExtraction]:
    """Every target is unmentioned when there is no report text at all."""
    return {
        target: TargetExtraction(state=STATE_UNMENTIONED, confidence=0.0, evidence="")
        for target in TARGETS
    }


class ExtractionCache:
    """Append-only JSONL cache keyed by normalised report hash.

    A 4,349-report job is long enough that an interruption is likely. Keying on
    the report hash rather than the study UID also collapses the duplicate
    reports that `add_report_groups` already relies on, so identical text is
    only ever sent once.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = str(row.get("report_sha1", ""))
                    if key:
                        self._entries[key] = row

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, row: dict) -> None:
        self._entries[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_anthropic_backend(
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: str | None = None,
) -> ReportLabelBackend:
    """Build an Anthropic-backed extraction callable.

    Imported lazily so the module stays usable -- and testable -- without the
    SDK installed or any network access.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "B23 Anthropic backend requires the 'anthropic' package: pip install anthropic"
        ) from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("B23 Anthropic backend requires ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)

    def _call(system: str, user: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "".join(parts)

    return _call


def extract_report(
    report: str,
    backend: ReportLabelBackend,
    *,
    max_attempts: int = 3,
    sleep_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, TargetExtraction], dict]:
    """Extract one report, retrying only on responses the parser rejects."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >=1")
    normalised = normalize_report(report)
    if not normalised:
        return empty_report_extraction(), {"attempts": 0, "empty_report": True, "raw": ""}

    user = build_user_prompt(report)
    errors: list[str] = []
    for attempt in range(1, int(max_attempts) + 1):
        raw = backend(SYSTEM_PROMPT, user)
        try:
            extraction = parse_extraction_response(raw)
        except ValueError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < max_attempts:
                sleep(float(sleep_seconds) * attempt)
            continue
        return extraction, {"attempts": attempt, "empty_report": False, "raw": raw}
    raise RuntimeError("B23 extraction failed after retries: " + " | ".join(errors))


def build_b23_frame(
    df: pd.DataFrame,
    backend: ReportLabelBackend,
    *,
    cache: ExtractionCache | None = None,
    max_attempts: int = 3,
    progress_every: int = 100,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Label every study in `df`, reusing cached extractions where available."""
    reports = df["Report"].fillna("").astype(str)
    uids = df["StudyInstanceUID"].astype(str)
    hashes = reports.map(report_hash)

    out = pd.DataFrame({"StudyInstanceUID": uids.to_numpy()})
    out["is_gold"] = gold_mask(df).to_numpy(dtype=bool)
    out["has_report"] = reports.map(lambda text: bool(normalize_report(text))).to_numpy(dtype=bool)
    out["report_sha1"] = hashes.to_numpy()

    columns: dict[str, list] = {}
    for target in TARGETS:
        columns[target] = []
        columns[f"{target}__confidence"] = []
        columns[f"{target}__state"] = []
        columns[f"{target}__evidence"] = []

    n_cached = 0
    n_called = 0
    for position, (report, key) in enumerate(zip(reports.tolist(), hashes.tolist()), start=1):
        row = cache.get(key) if cache is not None else None
        if row is not None:
            extraction = {
                target: TargetExtraction(
                    state=str(row["findings"][target]["state"]),
                    confidence=float(row["findings"][target]["confidence"]),
                    evidence=str(row["findings"][target].get("evidence", "")),
                )
                for target in TARGETS
            }
            n_cached += 1
        else:
            extraction, _meta = extract_report(
                report, backend, max_attempts=max_attempts, sleep=sleep
            )
            n_called += 1
            if cache is not None:
                cache.put(
                    key,
                    {
                        "report_sha1": key,
                        "findings": {
                            target: {
                                "state": item.state,
                                "confidence": item.confidence,
                                "evidence": item.evidence,
                            }
                            for target, item in extraction.items()
                        },
                    },
                )
        for target in TARGETS:
            item = extraction[target]
            columns[target].append(item.probability())
            columns[f"{target}__confidence"].append(item.usable_confidence())
            columns[f"{target}__state"].append(item.state)
            columns[f"{target}__evidence"].append(item.evidence)
        if progress_every and position % int(progress_every) == 0:
            print(f"[B23] {position}/{len(reports)} studies | cached={n_cached} called={n_called}")

    for target in TARGETS:
        out[target] = np.asarray(columns[target], dtype=np.float32)
        out[f"{target}__confidence"] = np.asarray(
            columns[f"{target}__confidence"], dtype=np.float32
        )
        out[f"{target}__state"] = columns[f"{target}__state"]
        out[f"{target}__evidence"] = columns[f"{target}__evidence"]
    return out


def _target_audit(frame: pd.DataFrame, target: str, *, min_confidence: float) -> dict:
    subset = frame.loc[~frame["is_gold"].astype(bool)]
    state_col = f"{target}__state"
    conf_col = f"{target}__confidence"
    states = (
        subset[state_col]
        .value_counts()
        .reindex(list(B23_STATES), fill_value=0)
        .astype(int)
        .to_dict()
    )
    usable = subset[state_col].isin([STATE_POSITIVE, STATE_NEGATED]) & subset[conf_col].ge(
        min_confidence
    )
    positive = subset[state_col].eq(STATE_POSITIVE) & subset[conf_col].ge(min_confidence)
    negative = subset[state_col].eq(STATE_NEGATED) & subset[conf_col].ge(min_confidence)
    return {
        "states": states,
        "usable_cells": int(usable.sum()),
        "positive_cells": int(positive.sum()),
        "negative_cells": int(negative.sum()),
        "coverage": float(usable.mean()) if len(subset) else 0.0,
    }


def run_b23_export(
    train_csv: str | Path,
    backend: ReportLabelBackend,
    *,
    out_root: str | Path = "runs/b23_llm_report_labels",
    min_confidence: float = 0.75,
    cache_path: str | Path | None = None,
    max_attempts: int = 3,
    progress_every: int = 100,
) -> dict:
    """Produce a B6-compatible export from LLM extractions."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")

    df = load_train_csv(train_csv)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    cache = ExtractionCache(cache_path or out / "extraction_cache.jsonl")

    structured = build_b23_frame(
        df,
        backend,
        cache=cache,
        max_attempts=max_attempts,
        progress_every=progress_every,
    )
    structured.to_csv(out / "structured_labels.csv", index=False)

    report_only = structured.loc[~structured["is_gold"].astype(bool)].copy()
    training_columns = ["StudyInstanceUID"]
    for target in TARGETS:
        training_columns.extend([target, f"{target}__confidence", f"{target}__state"])
    report_only[training_columns].to_csv(out / "training_targets.csv", index=False)

    per_target = {
        target: _target_audit(structured, target, min_confidence=min_confidence)
        for target in TARGETS
    }
    usable_total = int(sum(item["usable_cells"] for item in per_target.values()))
    possible_total = int(len(report_only) * len(TARGETS))
    audit = {
        "b23_version": B23_VERSION,
        "experiment": B23_EXPERIMENT,
        "n_studies": int(len(structured)),
        "n_gold_audit_only": int(structured["is_gold"].sum()),
        "n_report_only_training": int(len(report_only)),
        "n_reports_present": int(structured["has_report"].sum()),
        "min_confidence_for_usable_cell": float(min_confidence),
        "external_models": True,
        "external_data": False,
        "gold_fitted_calibration": False,
        "gold_rows_in_training_targets": 0,
        "usable_cells_total": usable_total,
        "possible_cells_total": possible_total,
        "cell_coverage": float(usable_total / possible_total) if possible_total else 0.0,
        "unique_reports_labelled": int(len(cache)),
        "targets": per_target,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    policy = {
        "experiment": B23_EXPERIMENT,
        "version": B23_VERSION,
        "purpose": "LLM-extracted multilingual report weak labels replacing the B6 regex parser",
        "states": list(B23_STATES),
        "fixed_soft_labels": {
            STATE_POSITIVE: {"probability": B23_POSITIVE_PROBABILITY, "confidence": "model-reported"},
            STATE_NEGATED: {"probability": B23_NEGATED_PROBABILITY, "confidence": "model-reported"},
            STATE_UNCERTAIN: {
                "probability": B23_UNCERTAIN_PROBABILITY,
                "confidence": B23_IGNORED_STATE_CONFIDENCE,
            },
            STATE_UNMENTIONED: {
                "probability": B23_UNMENTIONED_PROBABILITY,
                "confidence": B23_IGNORED_STATE_CONFIDENCE,
            },
        },
        "gold_usage": "audit only; excluded from training_targets.csv",
        "unmentioned_is_negative": False,
        "supersedes": "none; B6 v1.2.1 remains frozen for historical comparisons",
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return audit


def load_frozen_b23_export(b23_root: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load a completed B23 export with the same guarantees B7 demands of B6."""
    root = Path(b23_root)
    targets_path = root / "training_targets.csv"
    policy_path = root / "policy.json"
    audit_path = root / "audit.json"
    for path in (targets_path, policy_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"B23 export is missing artifact: {path}")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(policy.get("version")) != B23_VERSION or str(audit.get("b23_version")) != B23_VERSION:
        raise ValueError(f"expected B23 v{B23_VERSION} export")
    if int(audit.get("gold_rows_in_training_targets", -1)) != 0:
        raise ValueError("B23 audit does not certify zero gold rows in training_targets.csv")
    if bool(policy.get("unmentioned_is_negative", False)):
        raise ValueError("B23 must not map unmentioned report states to negative")

    frame = pd.read_csv(targets_path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError("B23 training_targets.csv is missing StudyInstanceUID")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B23 training_targets.csv contains duplicate StudyInstanceUID values")
    return frame, policy, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="B23 LLM multilingual report labeller")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-root", default="runs/b23_llm_report_labels")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    backend = make_anthropic_backend(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    audit = run_b23_export(
        args.train_csv,
        backend,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
        cache_path=args.cache,
        max_attempts=args.max_attempts,
        progress_every=args.progress_every,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
