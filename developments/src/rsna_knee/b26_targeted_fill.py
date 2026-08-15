"""B26 — targeted supervision fill for balance-flagged targets.

Everything B23 through B25X measured points at one narrow, well-supported
action. Recapping the evidence, because it is what constrains this design:

* **Fill, do not replace.** B24X-Density preserved every B6 cell and added the
  LLM only where B6 was silent. It matched full replacement (`0.7148` vs
  `0.7116`) with the paired interval crossing zero, so replacing B6 decisions
  bought nothing. Replacement is also what failed the B23-v1 specificity gate.
* **The gain is one target.** B25X's 12-target macro moved `+0.0584`; with
  Synovitis excluded the same change was worth `+0.0024` across the other
  eleven. 96.4% of the effect was a single target.
* **That target is identifiable from training labels alone.** The frozen B6
  surface gives Synovitis 399 positive to 17 negative cells. The balance audit
  flags it and nothing else, robustly: any majority-share cut-off between
  80.1% and 95.9%, or any minority-cell floor between 18 and 203, selects the
  same one target.

So B26 labels **only the flagged targets**, and only to fill cells where B6 is
silent. Two consequences follow, and both matter:

The output is roughly one twelfth the size of a full 12-target extraction, so
the run is several times faster per report and the output-budget truncation
that killed the first B23 attempt cannot recur.

And the scope is decided by `targets_needing_fill` from the balance audit, not
by a name in this file. A hard-coded `if target == "Synovitis"` would be
target selection read off a weak-v2 table, which is precisely what the
repository prohibits; deriving it from training-label counts is not.

## What B26 deliberately does not do

It does not seek negatives. Synovitis needs negatives, but instructing the
labeller to find them would be manufacturing the class balance rather than
reading the report, and the resulting labels would be worthless. B26 extracts
states honestly and lets the balance fall where the reports put it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .b23_llm_labels import (
    B23_DEFINITE_STATE_CONFIDENCE,
    B23_IGNORED_STATE_CONFIDENCE,
    B23_NEGATED_PROBABILITY,
    B23_POSITIVE_PROBABILITY,
    B23_STATES,
    B23_UNCERTAIN_PROBABILITY,
    B23_UNMENTIONED_PROBABILITY,
    ExtractionCache,
    TARGET_DEFINITIONS,
    extraction_cache_key,
)
from .b23_local_llm import (
    EVIDENCE_MAX_CHARS,
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_NUM_CTX,
    OLLAMA_DEFAULT_NUM_PREDICT,
    ModelProvenance,
    TruncatedCompletionError,
    make_ollama_backend,
    prompt_sha256,
    strip_thinking,
)
from .constants import TARGETS
from .data import gold_mask, load_train_csv, normalize_report, report_hash
from .supervision_balance import audit_supervision_balance

B26_VERSION = "1.1.0"
B26_EXPERIMENT = "B26_targeted_supervision_fill"
# v1.1.0 adds rule 3b after the v1.0 manual audit found that 50 of 60 sampled
# added negations inferred absence of the target from absence of a *different*
# finding. The v1.0 extraction record (prompt hash, cache, provenance) stands
# unchanged; because the cache key includes the prompt SHA-256, v1.1 cannot
# reuse v1.0 extractions. The current Synovitis path does not need a v1.1
# re-run -- B26.1 adjudicates the existing proposals more cheaply -- but any
# future target flagged by the balance audit gets the corrected prompt.

# Same frozen semantics as B23. B26 changes which targets are read, not what a
# state means, so a fill cell is indistinguishable from a B6 cell downstream.
B26_SYSTEM_PROMPT_HEADER = """You are a careful musculoskeletal radiologist reading a knee MRI report.

You are asked about a SMALL number of specific findings, not all twelve. Answer only about the findings listed below.

Reports may be written in ANY language: English, Spanish, Dutch and Turkish all appear in this corpus. Read the report in its original language.

For each finding assign exactly one state:

- "positive": the report asserts the finding is present.
- "negated": the report states the finding is absent, or the structure is normal/intact.
- "uncertain": the report genuinely hedges AND nothing else in the report resolves it.
- "unmentioned": the report says nothing about this finding, either way.

## Rule 1 - read the findings, never the request

INDICATION, CLINICAL HISTORY, ANTECEDENTES CLINICOS, KLINISCHE INLICHTINGEN, COMPARISON and TECHNIQUE record what was suspected and which sequences were run. They are never evidence of a finding. Decide only from FINDINGS / BEVINDINGEN / HALLAZGOS / BULGULAR and IMPRESSION / CONCLUSION / BESLUIT / IMPRESION.

## Rule 2 - the impression wins

When the findings section and the impression disagree, follow the impression. That is not a hedge; do not answer "uncertain".

## Rule 3 - a lesion NEAR a structure is not a lesion OF it

Attribute a finding only to the structure that is actually abnormal.

## Rule 3b - the absence of a DIFFERENT finding does not negate this one

This is the mirror of rule 3 and it is the most common way to get this task
wrong. Answer "negated" only when the report addresses THIS finding and says
it is absent, or gives a genuinely unqualified global-normal conclusion
covering the whole joint.

None of these negate a finding they do not name:

- "no joint effusion" or "trace effusion"
- "no bone bruise" / "normal bone marrow"
- "normal menisci" / "normal ligaments"
- "no intra-articular body"
- "normal surrounding soft tissues"
- the absence of any single unrelated abnormality

If the report is silent about THIS finding, the answer is "unmentioned", not
"negated" -- even when it negates several neighbouring findings.

## Rule 4 - abnormality, not just severe disease

Each target means ANY abnormality of that structure, at any grade. A stated abnormality is positive even when the report also calls the structure otherwise intact, and even when a severe form is explicitly excluded.

## Rule 5 - silence is not absence

If the report says nothing about a finding, answer "unmentioned". Never infer absence from silence. A generic normality statement does negate what it conventionally covers.

## Rule 6 - reserve "uncertain" for genuine unresolved hedging

"possible", "suspected", "R/O", "cannot exclude", "vermoeden van", "sospecha". A conflict between two sentences is not uncertainty; resolve it with rules 1 to 4 first.

## Output

Give a calibrated confidence in [0,1] that the STATE you assigned is the correct reading of the report -- confidence about your extraction, not about the disease.

Give a short verbatim `evidence` span copied from the report in its original language. Keep it under {evidence_max} characters. Use an empty string when the state is "unmentioned".
"""

B26_SYSTEM_PROMPT_FOOTER = """
Return ONLY a JSON object, no prose and no code fences, with exactly this shape:

{"findings": {"<target name>": {"state": "...", "confidence": 0.0, "evidence": "..."}, ...}}

The `findings` object must contain exactly the target names listed above, spelled exactly as given, and no others."""

# Extra vocabulary for the findings most likely to be flagged. Kept general
# rather than Synovitis-specific so the module does not encode a target name.
TARGET_VOCABULARY: dict[str, str] = {
    "Synovitis": (
        "synovitis, sinovitis, synovial thickening/hypertrophy/proliferation, "
        "'hypertrophy of the synovium', synoviale verdikking, pannus"
    ),
    "Effusion": "hydrops, derrame articular, eklem ici sivi, efuzyon, joint fluid",
    "Baker's": "bakercyste, quiste popliteo, popliteal cyst, bursa semimembranosa gastrocnemia",
    "Contusion": "bone bruise, bone marrow oedema of traumatic origin, kemik iligi odemi",
    "Fracture": "fractuur, fractura, kirik, occult/insufficiency/avulsion fracture",
}


def build_system_prompt(targets: list[str]) -> str:
    """Prompt covering exactly the flagged targets and no others."""
    if not targets:
        raise ValueError("B26 needs at least one target to fill")
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")

    lines = [B26_SYSTEM_PROMPT_HEADER.format(evidence_max=EVIDENCE_MAX_CHARS)]
    lines.append("\n## The findings you are asked about\n")
    for target in targets:
        lines.append(f"- {target}: {TARGET_DEFINITIONS[target]}")
        if target in TARGET_VOCABULARY:
            lines.append(f"    also written as: {TARGET_VOCABULARY[target]}")
    lines.append(B26_SYSTEM_PROMPT_FOOTER)
    return "\n".join(lines)


def build_findings_schema(targets: list[str]) -> dict:
    """JSON schema constraining the decoder to exactly these targets."""
    cell = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": list(B23_STATES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "string"},
        },
        "required": ["state", "confidence", "evidence"],
    }
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "object",
                "properties": {str(t): cell for t in targets},
                "required": [str(t) for t in targets],
            }
        },
        "required": ["findings"],
    }


def parse_targeted_response(text: str, targets: list[str]) -> dict[str, dict]:
    """Validate a response covering exactly `targets`.

    Rejects rather than defaults, for the same reason B23 does: a silently
    defaulted label is far more expensive than a retried request.
    """
    stripped = strip_thinking(text)
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    findings = payload.get("findings")
    if not isinstance(findings, dict):
        raise ValueError("response is missing a 'findings' object")

    missing = [t for t in targets if t not in findings]
    if missing:
        raise ValueError(f"response is missing targets: {missing}")

    out: dict[str, dict] = {}
    for target in targets:
        cell = findings[target]
        if not isinstance(cell, dict):
            raise ValueError(f"target {target!r} must map to an object")
        state = str(cell.get("state", "")).strip().lower()
        if state not in B23_STATES:
            raise ValueError(f"target {target!r} has unknown state {state!r}")
        try:
            confidence = float(cell.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target {target!r} has non-numeric confidence") from exc
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"target {target!r} confidence out of range")
        evidence = str(cell.get("evidence", "") or "")
        if len(evidence) > EVIDENCE_MAX_CHARS:
            evidence = evidence[:EVIDENCE_MAX_CHARS].rstrip() + "..."
        out[target] = {"state": state, "confidence": confidence, "evidence": evidence}
    return out


def state_to_supervision(state: str, confidence: float) -> tuple[float, float]:
    """Map a state to (probability, usable confidence), matching B23 exactly.

    Definite states take B6's fixed 0.90 rather than the model's own number, so
    B26 changes which cells exist and not how supervision is thresholded.
    """
    if state == "positive":
        return B23_POSITIVE_PROBABILITY, B23_DEFINITE_STATE_CONFIDENCE
    if state == "negated":
        return B23_NEGATED_PROBABILITY, B23_DEFINITE_STATE_CONFIDENCE
    if state == "uncertain":
        return B23_UNCERTAIN_PROBABILITY, B23_IGNORED_STATE_CONFIDENCE
    return B23_UNMENTIONED_PROBABILITY, B23_IGNORED_STATE_CONFIDENCE


def _empty(targets: list[str]) -> dict[str, dict]:
    return {t: {"state": "unmentioned", "confidence": 0.0, "evidence": ""} for t in targets}


def run_targeted_fill(
    train_csv: str | Path,
    backend,
    targets: list[str],
    provenance: ModelProvenance,
    *,
    out_root: str | Path,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    progress_every: int = 50,
    max_attempts: int = 3,
) -> dict:
    """Label the flagged targets across the report corpus."""
    if not provenance.reproducible:
        raise ValueError(
            "B26 requires reproducible provenance: an openly downloadable "
            "checkpoint, pinned to an exact revision, decoded greedily"
        )
    df = load_train_csv(train_csv)
    if limit is not None:
        df = df.head(int(limit)).copy()

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    cache = ExtractionCache(cache_path or out / "extraction_cache.jsonl")

    reports = df["Report"].fillna("").astype(str)
    uids = df["StudyInstanceUID"].astype(str).tolist()
    is_gold = gold_mask(df).to_numpy(dtype=bool)

    records: list[dict] = []
    n_called = n_cached = 0
    for position, (uid, report, gold) in enumerate(
        zip(uids, reports.tolist(), is_gold), start=1
    ):
        if not normalize_report(report):
            extraction = _empty(targets)
        else:
            key = extraction_cache_key(report_hash(report), provenance)
            row = cache.get(key)
            if row is not None:
                extraction = {t: row["findings"][t] for t in targets}
                n_cached += 1
            else:
                started = time.monotonic()
                extraction = _extract(report, backend, targets, max_attempts=max_attempts)
                n_called += 1
                cache.put(
                    key,
                    {
                        "cache_key": key,
                        "report_sha1": report_hash(report),
                        "seconds": round(time.monotonic() - started, 3),
                        "findings": extraction,
                    },
                )
        record = {"StudyInstanceUID": uid, "is_gold": bool(gold)}
        for target in targets:
            state = extraction[target]["state"]
            probability, usable = state_to_supervision(
                state, float(extraction[target]["confidence"])
            )
            record[target] = probability
            record[f"{target}__confidence"] = usable
            record[f"{target}__model_confidence"] = float(extraction[target]["confidence"])
            record[f"{target}__state"] = state
            record[f"{target}__evidence"] = extraction[target]["evidence"]
        records.append(record)

        if progress_every and position % int(progress_every) == 0:
            print(f"[B26] {position}/{len(uids)} | cached={n_cached} called={n_called}")

    frame = pd.DataFrame(records)
    frame.to_csv(out / "targeted_labels.csv", index=False)

    non_gold = frame.loc[~frame["is_gold"].astype(bool)]
    per_target = {}
    for target in targets:
        state = non_gold[f"{target}__state"]
        per_target[target] = {
            "positive": int((state == "positive").sum()),
            "negated": int((state == "negated").sum()),
            "uncertain": int((state == "uncertain").sum()),
            "unmentioned": int((state == "unmentioned").sum()),
        }

    audit = {
        "b26_version": B26_VERSION,
        "experiment": B26_EXPERIMENT,
        "targets": list(targets),
        "n_studies": int(len(frame)),
        "n_gold_excluded_from_fill": int(frame["is_gold"].sum()),
        "n_report_only": int(len(non_gold)),
        "external_model_reproducible": True,
        "provenance": provenance.to_dict(),
        "per_target_states": per_target,
        "scope": "partial" if limit is not None else "full",
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _extract(report: str, backend, targets: list[str], *, max_attempts: int) -> dict[str, dict]:
    errors = []
    for attempt in range(1, int(max_attempts) + 1):
        try:
            raw = backend(build_system_prompt(targets), _user_prompt(report))
        except TruncatedCompletionError:
            # Deterministic: repeating the identical request repeats the
            # identical truncation. The backend already escalated its budget.
            raise
        try:
            return parse_targeted_response(raw, targets)
        except ValueError as exc:
            errors.append(f"attempt {attempt}: {exc}")
    raise RuntimeError("B26 extraction failed: " + " | ".join(errors))


def _user_prompt(report: str) -> str:
    return f"Knee MRI report:\n\n<report>\n{report.strip()}\n</report>"


def build_fill_supervision(
    base_targets: np.ndarray,
    base_weights: np.ndarray,
    fill_states: pd.DataFrame,
    fill_targets: list[str],
    *,
    min_confidence: float = 0.75,
) -> dict:
    """Preserve every base cell; add fill only where the base is silent.

    This is the B24X-Density policy. No base decision is dropped and none is
    overridden, so the specificity of the existing supervision is untouched on
    every cell it already covers.
    """
    y = np.array(base_targets, dtype=np.float64, copy=True)
    w = np.array(base_weights, dtype=np.float64, copy=True)
    if y.shape != w.shape or y.shape[1] != len(TARGETS):
        raise ValueError("base supervision must be [n, 12]")
    if len(fill_states) != y.shape[0]:
        raise ValueError("fill states must align row-for-row with base supervision")

    added = {t: 0 for t in fill_targets}
    skipped_occupied = {t: 0 for t in fill_targets}
    for target in fill_targets:
        j = TARGETS.index(target)
        states = fill_states[f"{target}__state"].astype(str).to_numpy()
        confs = fill_states[f"{target}__confidence"].to_numpy(dtype=np.float64)
        silent = w[:, j] <= 0
        usable = np.isin(states, ("positive", "negated")) & (confs >= min_confidence)
        fill_here = silent & usable
        skipped_occupied[target] = int(np.sum(~silent & usable))
        for i in np.flatnonzero(fill_here):
            probability, _ = state_to_supervision(states[i], float(confs[i]))
            y[i, j] = 0.85 if probability > 0.5 else 0.05
            w[i, j] = 0.50 if probability > 0.5 else 1.00
        added[target] = int(fill_here.sum())

    # The fill must be additive: every cell the base supervised keeps exactly
    # its original target and weight.
    base_mask = np.asarray(base_weights) > 0
    if not np.all(w[base_mask] == np.asarray(base_weights)[base_mask]):
        raise RuntimeError("fill overwrote an existing supervision weight")
    if not np.all(y[base_mask] == np.asarray(base_targets)[base_mask]):
        raise RuntimeError("fill overwrote an existing supervision target")

    return {
        "targets": y,
        "weights": w,
        "cells_added": added,
        "cells_skipped_already_supervised": skipped_occupied,
        "base_usable_cells": int(base_mask.sum()),
        "final_usable_cells": int((w > 0).sum()),
        "base_cells_dropped": 0,
        "base_cells_overridden": 0,
    }


def resolve_fill_targets(balance_audit_path: str | Path) -> list[str]:
    """Read the flagged targets from a completed balance audit.

    The scope must come from the audit rather than from this module. A
    hard-coded target name would be selection read off a weak-v2 table;
    training-label counts are not.
    """
    payload = json.loads(Path(balance_audit_path).read_text(encoding="utf-8"))
    flagged = payload.get("targets_needing_fill")
    if flagged is None:
        raise ValueError(f"{balance_audit_path} is not a balance-audit payload")
    if not flagged:
        raise ValueError(
            "the balance audit flagged no targets; there is nothing for B26 to "
            "fill and no experiment to run"
        )
    return list(flagged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B26 targeted supervision fill for balance-flagged targets"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--balance-audit",
        required=True,
        help="balance-audit JSON; its targets_needing_fill decides the scope",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST)
    parser.add_argument("--num-ctx", type=int, default=OLLAMA_DEFAULT_NUM_CTX)
    parser.add_argument("--max-new-tokens", type=int, default=OLLAMA_DEFAULT_NUM_PREDICT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    targets = resolve_fill_targets(args.balance_audit)
    print(f"[B26] balance audit selected {len(targets)} target(s): {', '.join(targets)}")

    system_prompt = build_system_prompt(targets)
    backend, provenance = make_ollama_backend(
        system_prompt,
        model=args.model,
        host=args.ollama_host,
        num_ctx=args.num_ctx,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        schema=build_findings_schema(targets),
    )
    # The prompt covers only these targets, so it hashes differently from B23's
    # and cannot collide with that cache.
    provenance = ModelProvenance(
        **{**provenance.to_dict(), "prompt_sha256": prompt_sha256(system_prompt)}
    )
    print(provenance.describe())

    audit = run_targeted_fill(
        args.train_csv,
        backend,
        targets,
        provenance,
        out_root=args.out_root,
        cache_path=args.cache,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
