"""B26.2 — deterministic evidence whitelist over the frozen B26.1 output.

B26.2 is a post-B26.1 label-quality gate. It does not call an LLM, does not
read model predictions, and does not use weak-v2 or gold outcomes.

The fresh B26.1 manual audit showed:
- positive precision 19/20 = 95%;
- negated precision 36/60 = 60%.

The remaining errors were systematic enough to freeze a deterministic filter:
retain a B26.1 positive only when its quoted evidence contains an explicit
Synovitis/synovial-abnormality expression; retain a B26.1 negative only when
its quoted evidence directly negates Synovitis/the synovium OR the full report
contains a vetted unqualified global-normal conclusion.

Everything else is discarded back to "unmentioned". B26.2 never flips
polarity and never creates a new label. It can only remove B26.1 proposals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
from .b26_targeted_fill import resolve_fill_targets
from .constants import TARGETS
from .data import load_train_csv

B26_2_VERSION = "1.0.0"
B26_2_EXPERIMENT = "B26_2_deterministic_evidence_gate"
SUPPORTED_TARGET = "Synovitis"
DEFINITE = ("positive", "negated")

# These expressions were frozen from the completed B26/B26.1 manual audits.
# They intentionally prefer precision to coverage.
POSITIVE_EVIDENCE_PATTERNS = (
    r"\b(?:synovitis|sinovitis|sinovit|snovit)\b",
    r"\b\w*synovialitis\b",
    r"\bsynov(?:ial|iale|ialis|ium|ia)[\w-]*\s+"
    r"(?:thicken\w*|verdick\w*|hypertroph\w*|prolifer\w*|pannus)\b",
    r"\b(?:thicken\w*|verdick\w*|hypertroph\w*|prolifer\w*)\s+"
    r"(?:of\s+the\s+)?synov(?:ium|ial|ia|ialis)\b",
    r"\bhipertrof\w*\s+sinovij\w*\b",
    r"\bsinovyal\s+(?:kal[iı]nla[sş]ma|hipertrofi|proliferasyon)\b",
    r"\bsynoviale?\s+(?:verdikking|hypertrofie|proliferatie)\b",
)

# If an otherwise-positive evidence span contains one of these direct
# negations, it is not accepted as a B26.2 positive.
POSITIVE_NEGATION_PATTERNS = (
    r"\bno\s+(?:evidence\s+of\s+|signs?\s+of\s+)?synovitis\b",
    r"\bwithout\s+synovitis\b",
    r"\babsence\s+of\s+synovitis\b",
    r"\bkein(?:e|en|er|es)?\s+synov(?:itis|ialitis)\b",
    r"\bsin\s+(?:signos\s+de\s+)?sinovitis\b",
    r"\bsinovit\s+yok\b",
    r"\bno\b[^.;:\n]{0,100}\bsynovial\s+"
    r"(?:thickening|hypertrophy|proliferation)\b",
    r"\bkeine?\b[^.;:\n]{0,100}\bsynov(?:ia|ialis|ialitis)\b",
)

DIRECT_NEGATIVE_EVIDENCE_PATTERNS = (
    r"\bno\s+(?:evidence\s+of\s+|signs?\s+of\s+)?synovitis\b",
    r"\bwithout\s+synovitis\b",
    r"\babsence\s+of\s+synovitis\b",
    r"\bkein(?:e|en|er|es)?\s+synov(?:itis|ialitis)\b",
    r"\bsin\s+(?:signos\s+de\s+)?sinovitis\b",
    r"\bsinovit\s+yok\b",
    r"\bsynov(?:ium|ia|ialis)\s+(?:is\s+)?"
    r"(?:normal|unremarkable|unauff[aä]llig)\b",
    r"\bsynovialis\s+nicht\s+verdickt\b",
    r"\bkeine\s+verdickung\s+der\s+synovia\b",
    r"\bno\s+(?:significant\s+)?synovial\s+"
    r"(?:thickening|hypertrophy|proliferation)\b",
    # Allows constructions such as:
    # "No significant knee joint effusion or synovial thickening is identified."
    r"\bno\b[^.;:\n]{0,100}\bsynovial\s+"
    r"(?:thickening|hypertrophy|proliferation)\b",
    r"\bgeen\s+synoviale\s+verdikking\b",
    r"\bsinovyal\s+kal[iı]nla[sş]ma\s+yok\b",
)

# Global normality must be explicit and cover the examination/study as a
# whole. Phrases such as "normal bone marrow", "normal menisci" or "no
# effusion" are deliberately absent.
GLOBAL_NORMAL_REPORT_PATTERNS = (
    r"\bconclusion\s*:\s*normal\b",
    r"\bimpression\s*:\s*normal\b",
    r"\bnormal\s+study\b",
    r"\bnormal\s+exam(?:ination)?\b",
    r"\bnormal\s+mr\s+examination\s+of\s+(?:the\s+)?"
    r"(?:left\s+|right\s+)?knee\b",
    r"\bnormal\s+mri?\s+examination\s+of\s+(?:the\s+)?"
    r"(?:left\s+|right\s+)?knee\b",
    r"\bno\s+significant\s+abnormalit(?:y|ies)\s+identified\b",
    r"\bno\s+significant\s+abnormal\s+findings?\b",
    r"\bno\s+significant\s+findings?\b",
    r"\bno\s+significant\s+abnormality\b",
    r"\bunremarkable\s+(?:knee\s+)?mri\b",
    r"\bnormal\s+mri\s+(?:arthrogram\s+)?of\s+(?:the\s+)?"
    r"(?:left\s+|right\s+)?knee\b",
    r"\botherwise,\s*normal\s+mri\s+of\s+knee\b",
    r"\bgeen\s+afwijkingen\s+aangetoond\b",
)


def _normalise(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = value.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def evidence_is_verbatim(evidence: object, report: object) -> bool:
    """Require quoted evidence to occur in the original report after normalization."""
    e = _normalise(evidence)
    r = _normalise(report)
    return bool(e) and e in r


def accept_positive(evidence: object, report: object) -> tuple[bool, str]:
    e = _normalise(evidence)
    if not evidence_is_verbatim(evidence, report):
        return False, "positive_evidence_not_verbatim"
    if _matches_any(e, POSITIVE_NEGATION_PATTERNS):
        return False, "positive_evidence_is_negated"
    if _matches_any(e, POSITIVE_EVIDENCE_PATTERNS):
        return True, "explicit_positive_synovial_evidence"
    return False, "positive_not_on_whitelist"


def accept_negative(evidence: object, report: object) -> tuple[bool, str]:
    e = _normalise(evidence)
    r = _normalise(report)

    # A direct target-specific negation is sufficient, but it must be a
    # verbatim report span rather than a generated paraphrase.
    if (
        evidence_is_verbatim(evidence, report)
        and _matches_any(e, DIRECT_NEGATIVE_EVIDENCE_PATTERNS)
    ):
        return True, "explicit_negative_synovial_evidence"

    # Otherwise require a vetted global-normal statement in the original
    # report. This is intentionally independent of the LLM's quoted span,
    # because the B26.1 audit found some correct global-normal decisions whose
    # evidence quote was narrower than the actual concluding statement.
    if _matches_any(r, GLOBAL_NORMAL_REPORT_PATTERNS):
        return True, "global_normal_report_conclusion"

    return False, "negative_not_on_whitelist"


def apply_b26_2_filter(candidates: pd.DataFrame, reports_by_uid: pd.Series) -> pd.DataFrame:
    """Filter frozen B26.1 candidates without creating or flipping labels."""
    required = {
        "StudyInstanceUID",
        "target",
        "gate_state",
        "gate_evidence",
        "accepted_same_polarity",
        "polarity_flip_rejected",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"B26.1 candidates missing columns: {missing}")

    out = candidates.copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    if out["StudyInstanceUID"].duplicated().any():
        raise ValueError("B26.1 candidates contain duplicate StudyInstanceUID rows")
    if set(out["target"].astype(str)) != {SUPPORTED_TARGET}:
        raise ValueError("B26.2 v1 supports only the manually audited Synovitis target")

    accept_rows: list[bool] = []
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        if uid not in reports_by_uid.index:
            raise ValueError(f"missing original report for {uid}")

        if not bool(row.accepted_same_polarity):
            accept_rows.append(False)
            reasons.append("b26_1_not_accepted")
            continue
        if bool(row.polarity_flip_rejected):
            accept_rows.append(False)
            reasons.append("b26_1_polarity_flip")
            continue

        state = str(row.gate_state).strip().lower()
        report = reports_by_uid.loc[uid]
        if state == "positive":
            accepted, reason = accept_positive(row.gate_evidence, report)
        elif state == "negated":
            accepted, reason = accept_negative(row.gate_evidence, report)
        else:
            accepted, reason = False, "b26_1_state_not_definite"

        accept_rows.append(bool(accepted))
        reasons.append(reason)

    out["b26_2_accept"] = accept_rows
    out["b26_2_reason"] = reasons
    out["b26_2_state"] = np.where(out["b26_2_accept"], out["gate_state"], "unmentioned")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser("B26.2 deterministic evidence gate")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--balance-audit", required=True)
    parser.add_argument("--b26-1-candidates", required=True)
    parser.add_argument("--out-root", default="runs/b26_2_gate")
    parser.add_argument(
        "--exclude-review",
        action="append",
        default=[],
        help=(
            "review CSV whose StudyInstanceUIDs must be excluded from the fresh "
            "B26.2 review; repeatable"
        ),
    )
    parser.add_argument("--review-seed", type=int, default=2602)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fill_targets = resolve_fill_targets(args.balance_audit)
    if fill_targets != [SUPPORTED_TARGET]:
        raise ValueError(
            f"B26.2 v1 requires audit scope [{SUPPORTED_TARGET!r}], got {fill_targets!r}"
        )

    root = Path(args.data_root)
    train = load_train_csv(root / "train.csv").copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    reports_by_uid = train.set_index("StudyInstanceUID")["Report"].fillna("").astype(str)

    b6_frame, _b6_policy, _b6_audit = load_frozen_b6_export(args.b6_root)
    uids, base_y, base_w, _summary = prepare_b7_supervision(train, b6_frame)
    uids = [str(uid) for uid in uids]
    j = TARGETS.index(SUPPORTED_TARGET)

    candidates_path = Path(args.b26_1_candidates)
    candidates = pd.read_csv(candidates_path, dtype={"StudyInstanceUID": str})
    if len(candidates) != 631:
        raise RuntimeError(
            "B26.2 expects the completed exact B20 B26.1 surface of 631 "
            f"candidates; got {len(candidates)}"
        )
    if int(candidates["accepted_same_polarity"].astype(bool).sum()) != 281:
        raise RuntimeError(
            "B26.2 expects the completed B26.1 gate with 281 same-polarity "
            "accepted candidates"
        )

    filtered = apply_b26_2_filter(candidates, reports_by_uid)
    filtered.to_csv(out_root / "filtered_candidates.csv", index=False)

    accepted = filtered["b26_2_accept"].astype(bool)
    accepted_pos = int((accepted & (filtered["b26_2_state"] == "positive")).sum())
    accepted_neg = int((accepted & (filtered["b26_2_state"] == "negated")).sum())

    base_pos = int(((base_w[:, j] > 0) & (base_y[:, j] > 0.5)).sum())
    base_neg = int(((base_w[:, j] > 0) & (base_y[:, j] < 0.5)).sum())
    final_pos = base_pos + accepted_pos
    final_neg = base_neg + accepted_neg
    final_total = final_pos + final_neg
    majority_share = max(final_pos, final_neg) / final_total if final_total else float("nan")

    audit = {
        "b26_2_version": B26_2_VERSION,
        "experiment": B26_2_EXPERIMENT,
        "target": SUPPORTED_TARGET,
        "input_candidate_count": int(len(filtered)),
        "b26_1_accepted_input": int(candidates["accepted_same_polarity"].astype(bool).sum()),
        "accepted_positive": accepted_pos,
        "accepted_negated": accepted_neg,
        "accepted_total": int(accepted.sum()),
        "rejected_from_b26_1": int(
            candidates["accepted_same_polarity"].astype(bool).sum() - accepted.sum()
        ),
        "reason_counts": {
            str(k): int(v)
            for k, v in filtered["b26_2_reason"].value_counts().to_dict().items()
        },
        "base_positive": base_pos,
        "base_negative": base_neg,
        "final_positive_if_manual_quality_gate_passes": final_pos,
        "final_negative_if_manual_quality_gate_passes": final_neg,
        "final_usable_if_manual_quality_gate_passes": final_total,
        "final_majority_share_if_manual_quality_gate_passes": float(majority_share),
        "effective_positive_loss_mass_weight_0_5": float(final_pos * 0.50),
        "effective_negative_loss_mass_weight_1_0": float(final_neg),
        "base_cells_dropped": 0,
        "base_cells_overridden": 0,
        "training_allowed": False,
        "training_block_reason": (
            "B26.2 requires a fresh manual quality audit before any model training"
        ),
        "policy": (
            "deterministic precision-first whitelist over frozen B26.1 same-polarity "
            "calls; never creates labels, never flips polarity, never replaces B6"
        ),
        "input_sha256": {
            "b26_1_candidates": _sha256(candidates_path),
            "balance_audit": _sha256(Path(args.balance_audit)),
        },
    }

    # Fresh audit excludes every prior reviewed UID supplied by the caller.
    excluded: set[str] = set()
    for review_path in args.exclude_review:
        path = Path(review_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        review = pd.read_csv(path, dtype={"StudyInstanceUID": str})
        excluded.update(review["StudyInstanceUID"].astype(str))

    review_pool = filtered.loc[
        accepted & ~filtered["StudyInstanceUID"].astype(str).isin(excluded)
    ].copy()
    rng = np.random.default_rng(args.review_seed)
    neg_pool = review_pool.index[review_pool["b26_2_state"] == "negated"].to_numpy()
    pos_pool = review_pool.index[review_pool["b26_2_state"] == "positive"].to_numpy()
    neg_pick = (
        rng.choice(neg_pool, size=min(60, len(neg_pool)), replace=False)
        if len(neg_pool)
        else np.array([], dtype=int)
    )
    pos_pick = (
        rng.choice(pos_pool, size=min(20, len(pos_pool)), replace=False)
        if len(pos_pool)
        else np.array([], dtype=int)
    )
    picked = np.concatenate([neg_pick, pos_pick])

    review = filtered.loc[picked].copy() if len(picked) else filtered.iloc[0:0].copy()
    review["Report"] = review["StudyInstanceUID"].map(reports_by_uid)
    if len(review):
        review = review.sample(frac=1.0, random_state=args.review_seed).reset_index(drop=True)
    review["review_state_correct"] = ""
    review["review_evidence_supports_state"] = ""
    review["review_comment"] = ""
    review.to_csv(out_root / "fresh_review_80.csv", index=False)

    audit["fresh_review_rows"] = int(len(review))
    audit["fresh_review_excluded_prior_uids"] = int(len(excluded))
    audit["fresh_review_negated_rows"] = int((review["b26_2_state"] == "negated").sum())
    audit["fresh_review_positive_rows"] = int((review["b26_2_state"] == "positive").sum())
    (out_root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print(out_root / "filtered_candidates.csv")
    print(out_root / "audit.json")
    print(out_root / "fresh_review_80.csv")


if __name__ == "__main__":
    main()
