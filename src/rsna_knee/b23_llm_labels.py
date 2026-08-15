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
from B7 onward is trained on those cells, and the frozen B6 state-only baseline
scores 0.7025 on gold while B20 scores 0.6672 -- a fixed map from four parser
states to four constants ranks the expert labels slightly better than the
trained pipeline. That is not a ceiling and not a ratio ("95% of teacher" is
meaningless for AUC); it is an argument about where to look next.

B23 replaces the parser, not the policy. It keeps the same four states, the
same export contract and the same "report silence is not a negative" rule, so
the resulting export is a drop-in alternative supervision source rather than a
new training recipe. B6 v1.2.1 remains frozen for every historical comparison.

Labels are produced by an **openly downloadable checkpoint executed locally**,
not by a hosted API. Competition reproducibility requires the label-generating
function to be identifiable and re-runnable, and the weights served behind a
hosted model name can change without notice. Every export therefore carries a
`ModelProvenance` record pinning repo id, commit revision, dtype, quantisation
and greedy decoding, plus a SHA-256 of the prompt, and `run_b23_export` refuses
to certify an export whose provenance is not reproducible.

The labeller is deliberately backend-injectable: `run_b23_export` accepts any
callable implementing `ReportLabelBackend`, so the extraction contract can be
tested exactly without a GPU or a model download, and a cached run can resume
after an interruption without recomputing completed reports.
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

from .b23_local_llm import (
    BACKEND_LOCAL_TRANSFORMERS,
    BACKEND_OLLAMA,
    EVIDENCE_MAX_CHARS,
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_NUM_CTX,
    OLLAMA_DEFAULT_NUM_PREDICT,
    TruncatedCompletionError,
    BACKEND_LOCAL_VLLM,
    DEFAULT_LOCAL_MODEL,
    ModelProvenance,
    hash_local_weights,
    make_hosted_api_backend,
    make_local_transformers_backend,
    make_local_vllm_backend,
    build_findings_schema,
    make_ollama_backend,
    strip_thinking,
)
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
# B6 v1.2.1 gives every definite call a FIXED confidence of 0.90. B23 matches
# that exactly so the experiment is a parser substitution and nothing else. If
# the model's own confidence were used for thresholding instead, B23 would
# change both the extracted diagnosis AND which cells become supervision --
# two variables at once, with the second governed by an uncalibrated
# self-report. The model's confidence is still recorded, as a diagnostic
# column only. Confidence-aware supervision is a separate experiment.
B23_DEFINITE_STATE_CONFIDENCE = 0.90
B23_MIN_REPORTED_CONFIDENCE = 0.0
B23_MAX_REPORTED_CONFIDENCE = 1.0

DEFAULT_MODEL = DEFAULT_LOCAL_MODEL
DEFAULT_MAX_TOKENS = OLLAMA_DEFAULT_NUM_PREDICT

# These are ABNORMALITY targets, not tear-only targets. The frozen B6 v1.2.1
# regression suite fixes this semantics and B23 must match it exactly, because
# B23 is a parser substitution and not a redefinition of the pathology:
#   "ACL: grade 1 sprain is seen with intact fibers."          -> positive
#   "Mucoid degeneration of the ACL without evidence of tear." -> positive
#   "Myxoid degeneration ... but no definite tear."            -> positive
# See tests/test_b6_report_labels.py::test_b6_negated_tear_does_not_cancel_other_abnormality
TARGET_DEFINITIONS: dict[str, str] = {
    "ACL": "ANY anterior cruciate ligament abnormality: tear, rupture, sprain of any grade, degeneration (mucoid/myxoid), or graft failure.",
    "MCL": "ANY medial collateral ligament abnormality: tear, sprain of any grade, degeneration or injury.",
    "Medial Meniscus": "ANY medial meniscus abnormality: tear, degeneration (myxoid/mucoid/intrasubstance), maceration or extrusion.",
    "Lateral Meniscus": "ANY lateral meniscus abnormality: tear, degeneration (myxoid/mucoid/intrasubstance), maceration or extrusion.",
    "Medial OA": "Osteoarthritis / cartilage loss / degenerative change in the MEDIAL tibiofemoral compartment.",
    "Lateral OA": "Osteoarthritis / cartilage loss / degenerative change in the LATERAL tibiofemoral compartment.",
    "PF OA": "Osteoarthritis / chondromalacia / cartilage loss in the PATELLOFEMORAL compartment.",
    "Effusion": "Joint effusion or increased intra-articular fluid.",
    "Synovitis": "Synovitis, synovial thickening, synovial proliferation or synovial enhancement.",
    "Baker's": "Baker's cyst, popliteal cyst or gastrocnemio-semimembranosus bursal cyst.",
    "Contusion": "Bone contusion, bone bruise or bone marrow oedema of traumatic origin.",
    "Fracture": "Fracture, including occult, insufficiency, avulsion and osteochondral fracture.",
}

# Every rule below answers a measured B6 v1.2.1 failure on the 58-study gold
# surface (see docs/B23_LLM_REPORT_LABELS.md). B6 discards 445 of 696 gold cells
# as uncertain/unmentioned, and those discards hold 121 of the 240 expert
# positives -- half the disease. All 12 sampled review-queue rows failed for one
# reason, `conflicting_definite_evidence`, and reading their evidence spans gives
# the taxonomy the rules below address.
SYSTEM_PROMPT = """You are a careful musculoskeletal radiologist extracting structured findings from knee MRI reports.

Reports may be written in ANY language: English, Spanish, Dutch and Turkish all appear in this corpus. Read the report in its original language. Do not translate before deciding; decide from the original text.

For each of the 12 target findings, assign exactly one state:

- "positive": the report asserts the finding is present.
- "negated": the report states the finding is absent, or the structure is normal/intact.
- "uncertain": the report genuinely hedges AND nothing else in the report resolves it.
- "unmentioned": the report says nothing about this finding, either way.

## Rule 1 - read the findings, never the request

Reports open with sections such as INDICATION, CLINICAL HISTORY, ANTECEDENTES CLINICOS, KLINISCHE INLICHTINGEN, COMPARISON, TECHNIQUE or PROTOCOL. Those record what the clinician SUSPECTED and which sequences were run. They are never evidence of a finding.

"Indication: ACL sprain", "Suspected bone contusion/meniscal tear" and "? anterior cruciate ligament" tell you nothing about what is present. Decide only from the FINDINGS / BEVINDINGEN / HALLAZGOS / BULGULAR section and the IMPRESSION / CONCLUSION / BESLUIT / IMPRESION / IMP section.

## Rule 2 - the impression wins

When the findings section and the impression disagree, follow the impression. Radiologists routinely call a structure grossly intact in the body and then commit to a diagnosis in the conclusion.

"The anterior cruciate ligament as a construct is intact" followed by "1. Low-grade partial tear of the anterior cruciate ligament" is POSITIVE for ACL. That is not a hedge; do not answer "uncertain".

## Rule 3 - a lesion NEAR a structure is not a lesion OF it

- "The ACL is intact. There is a ganglion cyst adjacent to the proximal ACL." -> ACL negated. A neighbouring cyst is not an ACL injury.
- "ACL normal. Avulsive fracture of the tibia at the attachment site of the ACL." -> ACL negated, Fracture positive. The bone is broken; the ligament is not.
- Bone marrow oedema at the medial femoral condyle is a Contusion finding, not an MCL finding.

## Rule 4 - partial, bundle and interstitial tears are still tears

- "Complete rupture of the posterolateral bundle of the ACL. The anteromedial bundle is structurally continuous." -> ACL positive.
- "High-grade partial-thickness tear involving the anterior cruciate ligament" -> ACL positive.
- "Diffuse increased intrasubstantial signal ... intrasubstantial tear. No definite disruption." -> ACL positive.

"tear", "rupture", "rotura", "scheur" or "yirtik" applied to the structure makes it positive regardless of grade or extent.

## Rule 5 - abnormality, not just "tear"

Every target means ANY abnormality of that structure, at any grade. A stated abnormality is POSITIVE even when the report also says the structure is otherwise intact, and even when a tear is explicitly excluded:

- "ACL: grade 1 sprain is seen with intact fibers." -> ACL positive.
- "Mucoid degeneration of the ACL without evidence of tear." -> ACL positive.
- "Mucoide degeneratie van de voorste kruisband ... ACL: intact." -> ACL positive.
- "Myxoid degeneration of the posterior horn of the medial meniscus but no definite tear." -> Medial Meniscus positive.
- "Grade I ligamentous sprain of the medial collateral ligament. The MCL complexes are intact." -> MCL positive.
- "Grade 2 intrasubstance signal not reaching the articular surface." -> meniscus positive (degeneration).

Negating a tear does not negate the finding. Only use "negated" when the report describes NO abnormality of that structure at all.

For the three osteoarthritis targets the same principle applies: chondromalacia, chondrosis, chondropathy, cartilage thinning, chondral ulcers and marginal osteophytes are all POSITIVE for their compartment, at any grade.

## Rule 6 - silence is not absence

If the report is simply silent about a finding, answer "unmentioned". Never infer absence from silence.

A generic normality statement does negate what it conventionally covers: "normal knee MRI", "Diz eklemi ici sivi miktari normal", "Ligamentos cruzados y colaterales dentro de limites normales" negate the structures they name.

## Rule 7 - compartment and laterality are not interchangeable

"Medial OA", "Lateral OA" and "PF OA" are three separate findings, as are the medial and lateral meniscus.

- "tricompartmental osteoarthritis" -> all three OA targets positive.
- "osteoarthritis" with no compartment named -> "uncertain" for all three, not positive.
- Change described in one compartment says nothing about the others; those stay "unmentioned" unless separately addressed.

## Rule 8 - vocabulary is multilingual

Do not miss a finding because it is not in English. Non-exhaustive:

- Effusion: "hydrops", "derrame articular", "eklem ici sivi", "efuzyon", "joint fluid"
- Synovitis: "sinovitis", "hypertrophy of the synovium", "synoviale verdikking"
- Baker\'s cyst: "bakercyste", "quiste popliteo", "popliteal cyst", "bursa semimembranosa gastrocnemia"
- Meniscal tear: "rotura", "scheur", "yirtik", "meniscusscheur"
- Cartilage: "kraakbeen", "cartilago", "kikirdak", "condropatia", "chondrosis"
- Negation: "geen", "no hay", "sin", "yok", "normal", "intact", "bewaard", "conservado"

## Rule 9 - reserve "uncertain" for genuine unresolved hedging

Use "uncertain" only when the report hedges and nothing else settles it: "possible", "suspected", "R/O", "cannot exclude", "vermoeden van", "sospecha", "obs.".

A conflict between two sentences is NOT uncertainty. Resolve it with rules 1 to 5, and answer "uncertain" only if it remains genuinely unresolvable afterwards.

## Output

Report a calibrated confidence in [0,1] for each finding: how certain you are that the STATE you assigned is the correct reading of the report. This is confidence about your extraction, not about the patient\'s disease.

Also give a short verbatim `evidence` span copied from the report (in its original language) that justifies the state. Use an empty string when the state is "unmentioned".

**Keep every `evidence` span under 120 characters.** Quote the shortest phrase that settles the question, not the whole sentence and never a whole paragraph. Twelve long spans overflow the output budget and truncate the answer.

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

        Definite states get B6's fixed 0.90 rather than the model's own number,
        so B23 changes the extracted diagnosis and nothing else. Uncertain and
        unmentioned are pinned to zero so the frozen 0.75 usable-cell threshold
        cannot admit them, matching the B6 v1.2.1 policy.
        """
        if self.state in (STATE_POSITIVE, STATE_NEGATED):
            return B23_DEFINITE_STATE_CONFIDENCE
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
    # Reasoning models (Qwen3 and family) prefix a <think> block. Strip it here
    # as well as in the backend so no backend can leak one into the parse.
    stripped = strip_thinking(text)
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
        if len(evidence) > EVIDENCE_MAX_CHARS:
            # Truncate rather than reject: the span is an audit aid, not
            # supervision, so an over-long quote is not worth failing a report.
            evidence = evidence[:EVIDENCE_MAX_CHARS].rstrip() + "..."
        out[target] = TargetExtraction(state=state, confidence=confidence, evidence=evidence)
    return out


def empty_report_extraction() -> dict[str, TargetExtraction]:
    """Every target is unmentioned when there is no report text at all."""
    return {
        target: TargetExtraction(state=STATE_UNMENTIONED, confidence=0.0, evidence="")
        for target in TARGETS
    }


def extraction_cache_key(report_sha1: str, provenance: ModelProvenance | None) -> str:
    """Cache key covering the report AND the labelling function that read it.

    Keying on the report alone is unsafe: after a prompt edit or a model swap,
    a stale extraction would be replayed while the export records the NEW
    provenance, producing a file that misdescribes how its own labels were
    made. That would quietly defeat the entire provenance system, and it would
    depend on the operator remembering to pass a fresh cache path.

    The key therefore binds the report to the prompt, the model, its pinned
    revision, and the decoding configuration.
    """
    parts = [str(report_sha1)]
    if provenance is not None:
        parts.extend(
            [
                str(provenance.prompt_sha256),
                str(provenance.backend),
                str(provenance.model_id),
                str(provenance.revision),
                str(provenance.decoding),
                str(provenance.max_new_tokens),
                str(provenance.seed),
            ]
        )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class ExtractionCache:
    """Append-only JSONL cache keyed by report AND labelling function.

    A 4,349-report job is long enough that an interruption is likely. Keying on
    the report content rather than the study UID also collapses the duplicate
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
                    key = str(row.get("cache_key", "")) or str(row.get("report_sha1", ""))
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


def make_backend(
    *,
    backend: str = BACKEND_OLLAMA,
    model_id: str = DEFAULT_MODEL,
    revision: str | None = None,
    dtype: str = "bfloat16",
    quantisation: str = "none",
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = 2026,
    weights_path: str | Path | None = None,
    ollama_host: str = OLLAMA_DEFAULT_HOST,
    num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
    think: bool = False,
) -> tuple[ReportLabelBackend, ModelProvenance]:
    """Build a labelling backend together with its provenance record.

    The default is a locally executed, openly downloadable checkpoint, because
    only that can be pinned to an exact artefact. Passing `weights_path`
    additionally digests the downloaded shards, which proves which bytes
    produced the labels even if the hub repository is later re-tagged.
    """
    weights_sha = hash_local_weights(weights_path) if weights_path else None
    if backend == BACKEND_OLLAMA:
        return make_ollama_backend(
            SYSTEM_PROMPT,
            model=model_id,
            host=ollama_host,
            num_ctx=num_ctx,
            max_new_tokens=max_new_tokens,
            seed=seed,
            think=think,
            schema=build_findings_schema(TARGETS),
        )
    if backend == BACKEND_LOCAL_TRANSFORMERS:
        return make_local_transformers_backend(
            SYSTEM_PROMPT,
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            quantisation=quantisation,
            max_new_tokens=max_new_tokens,
            seed=seed,
            weights_sha256=weights_sha,
        )
    if backend == BACKEND_LOCAL_VLLM:
        return make_local_vllm_backend(
            SYSTEM_PROMPT,
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            quantisation=quantisation,
            max_new_tokens=max_new_tokens,
            seed=seed,
            weights_sha256=weights_sha,
        )
    if backend == "hosted_api":
        return make_hosted_api_backend(
            SYSTEM_PROMPT, model_id=model_id, max_new_tokens=max_new_tokens
        )
    raise ValueError(
        "backend must be one of: ollama, local_transformers, local_vllm, hosted_api"
    )


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
        try:
            raw = backend(SYSTEM_PROMPT, user)
        except TruncatedCompletionError:
            # The backend already escalated its output budget as far as the
            # context window allows. Greedy decoding is deterministic, so
            # repeating the identical request would repeat the identical
            # truncation -- fail now with the specific cause intact rather
            # than burying it under identical retries.
            raise
        try:
            extraction = parse_extraction_response(raw)
        except ValueError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < max_attempts:
                sleep(float(sleep_seconds) * attempt)
            continue
        return extraction, {"attempts": attempt, "empty_report": False, "raw": raw}
    raise RuntimeError(
        "B23 extraction failed after retries: "
        + " | ".join(errors)
        + "\n(If every attempt reports the same parse error, the decoding is "
        "deterministic and retrying cannot help -- check num_predict/num_ctx.)"
    )


def build_b23_frame(
    df: pd.DataFrame,
    backend: ReportLabelBackend,
    *,
    provenance: ModelProvenance | None = None,
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
        columns[f"{target}__model_confidence"] = []
        columns[f"{target}__state"] = []
        columns[f"{target}__evidence"] = []

    n_cached = 0
    n_called = 0
    for position, (report, report_sha1) in enumerate(zip(reports.tolist(), hashes.tolist()), start=1):
        key = extraction_cache_key(report_sha1, provenance)
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
                        "cache_key": key,
                        "report_sha1": report_sha1,
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
            columns[f"{target}__model_confidence"].append(float(item.confidence))
            columns[f"{target}__state"].append(item.state)
            columns[f"{target}__evidence"].append(item.evidence)
        if progress_every and position % int(progress_every) == 0:
            print(f"[B23] {position}/{len(reports)} studies | cached={n_cached} called={n_called}")

    for target in TARGETS:
        out[target] = np.asarray(columns[target], dtype=np.float32)
        out[f"{target}__confidence"] = np.asarray(
            columns[f"{target}__confidence"], dtype=np.float32
        )
        out[f"{target}__model_confidence"] = np.asarray(
            columns[f"{target}__model_confidence"], dtype=np.float32
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
    provenance: ModelProvenance | None = None,
    require_reproducible: bool = True,
    limit: int | None = None,
) -> dict:
    """Produce a B6-compatible export from LLM extractions.

    `provenance` pins the exact labelling function. When `require_reproducible`
    is set -- the default, and what competition use needs -- an export whose
    provenance is missing or not reproducible is refused rather than written,
    because unreproducible labels cannot be defended later.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")
    if require_reproducible:
        if provenance is None:
            raise ValueError(
                "a reproducible B23 export requires ModelProvenance; "
                "pass require_reproducible=False only for offline testing"
            )
        if not provenance.reproducible:
            raise ValueError(
                "B23 provenance is not reproducible "
                f"(backend={provenance.backend!r}, model={provenance.model_id!r}, "
                f"revision={provenance.revision!r}). Competition labels must come "
                "from an openly downloadable checkpoint pinned to an exact revision "
                "and decoded greedily."
            )

    df = load_train_csv(train_csv)
    if limit is not None:
        # Smoke-test mode: label a deterministic prefix so the extraction can be
        # inspected by hand before committing to the full corpus. The export is
        # marked partial so it can never be mistaken for a complete one.
        df = df.head(int(limit)).copy()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    cache = ExtractionCache(cache_path or out / "extraction_cache.jsonl")

    structured = build_b23_frame(
        df,
        backend,
        provenance=provenance,
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
        "partial_smoke_test": limit is not None,
        "limit": int(limit) if limit is not None else None,
        "external_model_reproducible": bool(provenance.reproducible) if provenance else False,
        "provenance": provenance.to_dict() if provenance else None,
        "targets": per_target,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    policy = {
        "experiment": B23_EXPERIMENT,
        "version": B23_VERSION,
        "purpose": "LLM-extracted multilingual report weak labels replacing the B6 regex parser",
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
        "gold_usage": "audit only; excluded from training_targets.csv",
        "unmentioned_is_negative": False,
        "confidence_policy": (
            "B6-matched fixed 0.90 for definite states; the model's self-reported "
            "confidence is stored as a diagnostic column only and never thresholds "
            "supervision"
        ),
        "supersedes": "none; B6 v1.2.1 remains frozen for historical comparisons",
        "provenance": provenance.to_dict() if provenance else None,
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return audit


def load_frozen_b23_export(
    b23_root: str | Path, *, require_reproducible: bool = True
) -> tuple[pd.DataFrame, dict, dict]:
    """Load a completed B23 export with the same guarantees B7 demands of B6.

    Also refuses, by default, an export that was not produced by a pinned,
    openly downloadable, locally executed checkpoint -- so an unreproducible
    development export cannot reach training by accident.
    """
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
    if bool(audit.get("partial_smoke_test", False)):
        raise ValueError(
            "this B23 export is a partial smoke test, not a full labelling run; "
            "re-run without --limit before using it for training or a split"
        )
    if require_reproducible and not bool(audit.get("external_model_reproducible", False)):
        raise ValueError(
            "B23 export was not produced by a reproducible openly downloadable "
            "local checkpoint; refusing to use it for competition training"
        )

    frame = pd.read_csv(targets_path)
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError("B23 training_targets.csv is missing StudyInstanceUID")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B23 training_targets.csv contains duplicate StudyInstanceUID values")
    return frame, policy, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B23 multilingual report labeller (local open-weights LLM)"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-root", default="runs/b23_llm_report_labels")
    parser.add_argument(
        "--backend",
        default=BACKEND_OLLAMA,
        choices=[BACKEND_OLLAMA, BACKEND_LOCAL_TRANSFORMERS, BACKEND_LOCAL_VLLM, "hosted_api"],
        help="ollama (default), local_transformers, local_vllm, hosted_api (dev only)",
    )
    parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=OLLAMA_DEFAULT_NUM_CTX,
        help="Ollama context window; too small silently truncates the report",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="enable Qwen3 reasoning mode (slower; off by default for extraction)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="hub repo id of an open checkpoint")
    parser.add_argument(
        "--revision",
        default=None,
        help="exact commit SHA to pin; resolved from the hub when omitted",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantisation", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument(
        "--weights-path",
        default=None,
        help="local checkpoint directory to digest, proving which bytes made the labels",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--cache", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="label only the first N studies, for a smoke test; marks the export partial",
    )
    parser.add_argument(
        "--allow-unreproducible",
        action="store_true",
        help="permit a hosted/unpinned model; the export is then NOT competition-usable",
    )
    args = parser.parse_args()

    backend, provenance = make_backend(
        backend=args.backend,
        model_id=args.model,
        revision=args.revision,
        dtype=args.dtype,
        quantisation=args.quantisation,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        weights_path=args.weights_path,
        ollama_host=args.ollama_host,
        num_ctx=args.num_ctx,
        think=args.think,
    )
    print("[B23] labelling provenance")
    print(provenance.describe())
    if not provenance.reproducible and not args.allow_unreproducible:
        raise SystemExit(
            "[B23] refusing to run: this backend cannot be pinned to an exact artefact.\n"
            "      Use --backend local_transformers with an openly downloadable model,\n"
            "      or pass --allow-unreproducible for a development-only export."
        )

    audit = run_b23_export(
        args.train_csv,
        backend,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
        cache_path=args.cache,
        max_attempts=args.max_attempts,
        progress_every=args.progress_every,
        provenance=provenance,
        require_reproducible=not args.allow_unreproducible,
        limit=args.limit,
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "targets"}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
