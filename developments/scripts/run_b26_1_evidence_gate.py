#!/usr/bin/env python3
"""B26.1 strict evidence adjudication for the raw B26 fill candidates.

This is a post-B26 quality-control gate motivated by the completed 80-case
manual audit.  It does NOT rerun the full report corpus and does NOT create
supervision on its own.  It re-reads only the B6-silent, B26-definite candidate
cells on the exact B20 training surface under a stricter frozen semantic rule.

A candidate is accepted only when the strict second pass returns the SAME
polarity as raw B26.  A polarity flip is recorded but is not used as training
supervision.  This keeps B26.1 a filter rather than a relabeller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.b23_llm_labels import ExtractionCache, extraction_cache_key
from rsna_knee.b23_local_llm import (
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_NUM_CTX,
    ModelProvenance,
    make_ollama_backend,
    prompt_sha256,
    strip_thinking,
)
from rsna_knee.b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
from rsna_knee.b26_targeted_fill import resolve_fill_targets
from rsna_knee.constants import TARGETS
from rsna_knee.data import gold_mask, load_train_csv, report_hash

B26_1_VERSION = "1.0.0"
B26_1_EXPERIMENT = "B26_1_strict_evidence_gate"
STATES = ("positive", "negated", "uncertain", "unmentioned")
DEFINITE = ("positive", "negated")

# Target selection still comes exclusively from the frozen balance audit.  The
# semantic rule is target-specific because it is the quality defect diagnosed
# by the manual B26 review.  If a future balance audit flags a different target,
# a separately reviewed rule must be frozen before using this gate for it.
STRICT_TARGET_RULES = {
    "Synovitis": """
For Synovitis use this strict evidence policy:

POSITIVE only when the report explicitly asserts synovitis or a direct
qualifying abnormality of the synovium, such as synovial thickening,
hypertrophy, proliferation, pannus, or an explicitly inflammatory abnormal
synovium.

NEGATED only when the report explicitly says there is no synovitis, or that the
synovium is normal/unremarkable, OR when the report gives a genuinely
unqualified global normal/no-intra-articular-pathology conclusion that covers
the whole joint rather than one structure.

The following are NOT sufficient to negate Synovitis by themselves:
- no joint effusion or trace effusion;
- no bone bruise / normal bone marrow;
- normal ligaments or menisci;
- no intra-articular body;
- normal surrounding soft tissues;
- absence of one unrelated abnormality.

Effusion alone is not Synovitis.  Synovial-fluid leakage alone is not Synovitis.
Do not convert absence of effusion into absence of synovitis.

If FINDINGS and IMPRESSION/CONCLUSION disagree, the IMPRESSION/CONCLUSION wins.
If the report does not support either a positive or a negated Synovitis state,
return unmentioned.  Use uncertain only for genuine unresolved hedging.
""".strip()
}

SYSTEM_HEADER = """You are a strict evidence adjudicator reading a knee MRI report.

Read the report in its original language.  Decide the requested finding from
FINDINGS and IMPRESSION/CONCLUSION, not from the indication or clinical
history.  Do not infer one pathology from a related but different finding.
Silence is not absence.

This is a quality-control pass.  Precision is more important than coverage.
When the report does not directly support a definite state under the policy
below, return `unmentioned` rather than guessing.
"""

SYSTEM_FOOTER = """
Return ONLY JSON, no prose and no code fence, with exactly this shape:

{"state":"positive|negated|uncertain|unmentioned","evidence":"short verbatim span","reason":"brief reason"}

The evidence must come from the report.  For `unmentioned`, evidence may be an
empty string.
"""


def build_system_prompt(target: str) -> str:
    if target not in STRICT_TARGET_RULES:
        raise ValueError(
            f"B26.1 has no frozen manually reviewed semantic rule for {target!r}; "
            "do not generalize this post-B26 gate to a new target"
        )
    return f"{SYSTEM_HEADER}\n\nTARGET: {target}\n\n{STRICT_TARGET_RULES[target]}\n\n{SYSTEM_FOOTER}"


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": list(STATES)},
            "evidence": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["state", "evidence", "reason"],
    }


def parse_response(text: str) -> dict:
    stripped = strip_thinking(text).strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        stripped = stripped.strip()
    payload = json.loads(stripped)
    state = str(payload.get("state", "")).strip().lower()
    if state not in STATES:
        raise ValueError(f"invalid B26.1 state {state!r}")
    evidence = str(payload.get("evidence", "") or "").strip()
    reason = str(payload.get("reason", "") or "").strip()
    return {"state": state, "evidence": evidence, "reason": reason}


def _user_prompt(report: str) -> str:
    return f"Knee MRI report:\n\n<report>\n{report.strip()}\n</report>"


def main() -> None:
    parser = argparse.ArgumentParser("B26.1 strict evidence adjudication")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--balance-audit", required=True)
    parser.add_argument("--b26-labels", required=True)
    parser.add_argument("--out-root", default="runs/b26_1_gate")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST)
    parser.add_argument("--num-ctx", type=int, default=OLLAMA_DEFAULT_NUM_CTX)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--previous-review",
        default="runs/b26_fill/synovitis_blinded_review_80.csv",
        help="optional previous quality-review CSV; its UIDs are excluded from the fresh review set",
    )
    args = parser.parse_args()

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    fill_targets = resolve_fill_targets(args.balance_audit)
    if len(fill_targets) != 1:
        raise ValueError(
            "B26.1 v1 requires exactly one audit-selected target because its semantic "
            "quality rule was frozen from the completed single-target manual audit"
        )
    target = fill_targets[0]
    if target not in STRICT_TARGET_RULES:
        raise ValueError(f"no frozen B26.1 evidence rule for audit-selected target {target!r}")

    root = Path(args.data_root)
    train = load_train_csv(root / "train.csv")
    train = train.copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

    b6_frame, _b6_policy, _b6_audit = load_frozen_b6_export(args.b6_root)
    uids, base_y, base_w, _summary = prepare_b7_supervision(train, b6_frame)
    uids = [str(uid) for uid in uids]
    j = TARGETS.index(target)

    # B20 training must not contain expert-gold studies.
    gold_by_uid = pd.Series(
        gold_mask(train).to_numpy(dtype=bool),
        index=train["StudyInstanceUID"].astype(str),
    )
    if any(bool(gold_by_uid.loc[uid]) for uid in uids):
        raise RuntimeError("B26.1 exact B20 surface unexpectedly contains expert-gold studies")

    raw = pd.read_csv(args.b26_labels, dtype={"StudyInstanceUID": str})
    if raw["StudyInstanceUID"].duplicated().any():
        raise ValueError("B26 targeted labels contain duplicate StudyInstanceUID rows")
    raw = raw.set_index("StudyInstanceUID", drop=False)
    missing = [uid for uid in uids if uid not in raw.index]
    if missing:
        raise ValueError(f"B26 labels are missing {len(missing)} exact B20 training studies")
    raw = raw.loc[uids].reset_index(drop=True)

    report_by_uid = train.set_index("StudyInstanceUID")["Report"].fillna("").astype(str)
    reports = np.asarray([report_by_uid.loc[uid] for uid in uids], dtype=object)

    raw_state = raw[f"{target}__state"].astype(str).to_numpy()
    raw_conf = raw[f"{target}__confidence"].to_numpy(dtype=float)
    base_used = base_w[:, j] > 0
    candidate = (~base_used) & np.isin(raw_state, DEFINITE) & (raw_conf >= 0.75)
    candidate_idx = np.flatnonzero(candidate)
    if len(candidate_idx) != 631:
        raise RuntimeError(
            f"B26.1 expected the completed B26 fill audit's 631 candidates; got {len(candidate_idx)}"
        )
    if args.limit is not None:
        candidate_idx = candidate_idx[: int(args.limit)]

    system_prompt = build_system_prompt(target)
    backend, provenance = make_ollama_backend(
        system_prompt,
        model=args.model,
        host=args.ollama_host,
        num_ctx=args.num_ctx,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        schema=response_schema(),
    )
    provenance = ModelProvenance(
        **{**provenance.to_dict(), "prompt_sha256": prompt_sha256(system_prompt)}
    )
    if not provenance.reproducible:
        raise ValueError("B26.1 requires reproducible pinned local-model provenance")

    print(f"[B26.1] audit target: {target}")
    print(f"[B26.1] exact B20 candidates: {int(candidate.sum())}")
    print(f"[B26.1] processing now: {len(candidate_idx)}")
    print(provenance.describe())

    cache = ExtractionCache(out / "extraction_cache.jsonl")
    rows = []
    n_cached = n_called = 0
    for k, i in enumerate(candidate_idx, start=1):
        uid = uids[i]
        report = str(reports[i])
        key = extraction_cache_key(report_hash(report), provenance)
        cached = cache.get(key)
        if cached is not None:
            adjudicated = cached["adjudication"]
            n_cached += 1
        else:
            raw_text = backend(system_prompt, _user_prompt(report))
            adjudicated = parse_response(raw_text)
            cache.put(
                key,
                {
                    "cache_key": key,
                    "report_sha1": report_hash(report),
                    "adjudication": adjudicated,
                },
            )
            n_called += 1

        gate_state = str(adjudicated["state"])
        original = str(raw_state[i])
        accepted = bool(gate_state in DEFINITE and gate_state == original)
        polarity_flip = bool(gate_state in DEFINITE and gate_state != original)
        rows.append(
            {
                "StudyInstanceUID": uid,
                "target": target,
                "b26_state": original,
                "b26_evidence": str(raw.loc[i, f"{target}__evidence"]),
                "gate_state": gate_state,
                "gate_evidence": str(adjudicated.get("evidence", "")),
                "gate_reason": str(adjudicated.get("reason", "")),
                "accepted_same_polarity": accepted,
                "polarity_flip_rejected": polarity_flip,
            }
        )
        if args.progress_every and k % int(args.progress_every) == 0:
            print(f"[B26.1] {k}/{len(candidate_idx)} | cached={n_cached} called={n_called}")

    result = pd.DataFrame(rows)
    result.to_csv(out / "adjudicated_candidates.csv", index=False)

    # A partial run is for calibration only and must not produce final surface claims.
    scope = "partial" if args.limit is not None else "full"
    accepted = result["accepted_same_polarity"].astype(bool)
    accepted_pos = int((accepted & (result["gate_state"] == "positive")).sum())
    accepted_neg = int((accepted & (result["gate_state"] == "negated")).sum())
    polarity_flips = int(result["polarity_flip_rejected"].astype(bool).sum())

    base_pos = int(((base_w[:, j] > 0) & (base_y[:, j] > 0.5)).sum())
    base_neg = int(((base_w[:, j] > 0) & (base_y[:, j] < 0.5)).sum())

    audit = {
        "b26_1_version": B26_1_VERSION,
        "experiment": B26_1_EXPERIMENT,
        "scope": scope,
        "target": target,
        "candidate_count_full_surface": int(candidate.sum()),
        "candidate_count_processed": int(len(result)),
        "raw_candidate_positive_full_surface": int((candidate & (raw_state == "positive")).sum()),
        "raw_candidate_negated_full_surface": int((candidate & (raw_state == "negated")).sum()),
        "accepted_same_polarity_positive": accepted_pos,
        "accepted_same_polarity_negated": accepted_neg,
        "accepted_same_polarity_total": int(accepted.sum()),
        "polarity_flips_rejected": polarity_flips,
        "gate_state_counts": {
            str(k): int(v) for k, v in result["gate_state"].value_counts().to_dict().items()
        },
        "base_positive": base_pos,
        "base_negative": base_neg,
        "final_positive_if_quality_gate_passes": (base_pos + accepted_pos) if scope == "full" else None,
        "final_negative_if_quality_gate_passes": (base_neg + accepted_neg) if scope == "full" else None,
        "base_cells_dropped": 0,
        "base_cells_overridden": 0,
        "model_provenance": provenance.to_dict(),
        "policy": (
            "strict second pass on exact B20 B6-silent B26-definite candidates; "
            "accept only same-polarity definite adjudications; polarity flips are rejected"
        ),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Fresh quality-review set: accepted calls only, excluding the 80 cases that
    # motivated this stricter rule so the next audit is not performed on its
    # own prompt-design examples.
    if scope == "full":
        previous_uids: set[str] = set()
        previous = Path(args.previous_review)
        if previous.is_file():
            prev = pd.read_csv(previous, dtype={"StudyInstanceUID": str})
            previous_uids = set(prev["StudyInstanceUID"].astype(str))

        review_pool = result.loc[
            accepted & ~result["StudyInstanceUID"].astype(str).isin(previous_uids)
        ].copy()
        rng = np.random.default_rng(2601)
        neg_pool = review_pool.index[review_pool["gate_state"] == "negated"].to_numpy()
        pos_pool = review_pool.index[review_pool["gate_state"] == "positive"].to_numpy()
        neg_pick = rng.choice(neg_pool, size=min(60, len(neg_pool)), replace=False) if len(neg_pool) else np.array([], dtype=int)
        pos_pick = rng.choice(pos_pool, size=min(20, len(pos_pool)), replace=False) if len(pos_pool) else np.array([], dtype=int)
        picked = np.concatenate([neg_pick, pos_pick])
        review = result.loc[picked].copy() if len(picked) else result.iloc[0:0].copy()
        review["Report"] = review["StudyInstanceUID"].map(report_by_uid)
        review = review.sample(frac=1.0, random_state=2601).reset_index(drop=True) if len(review) else review
        review["review_state_correct"] = ""
        review["review_evidence_supports_state"] = ""
        review["review_comment"] = ""
        review.to_csv(out / "fresh_review_80.csv", index=False)
        audit["fresh_review_rows"] = int(len(review))
        audit["fresh_review_excludes_previous_manual_uids"] = int(len(previous_uids))
        (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(out / "adjudicated_candidates.csv")
    print(out / "audit.json")
    if scope == "full":
        print(out / "fresh_review_80.csv")


if __name__ == "__main__":
    main()
