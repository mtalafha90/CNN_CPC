"""B6 v1.3: give the parser an osteoarthritis vocabulary, and stop it reading
list negations as findings.

Two defects were measured and left unfixed. This fixes both, without touching
`b6_report_labels` — v1.2.1 stays byte-identical because this module never
imports its way into it, only out of it.

## Defect one: there is no working OA vocabulary

Across 4,349 report studies, every one of B6's 2,632 osteoarthritis calls came
through `compartment_aware_oa_context` — the legacy fallback that runs *only
when no alias matched at all*:

```text
                        quoted   unquoted
the three OA targets        24      2,632
the other nine          11,467          0
```

Twenty-four quoted calls across three targets is not a lexicon that works, it
is one that is bypassed. And the fallback is the worst of the three measured
rules against the experts, at 42.9% wrong, while recording a rule name where a
quotation should be.

The fix adds patterns for the anatomy the reports actually name. Because a
pattern only pre-empts the fallback when it *matches*, every cell it places is
a cell that moves from "guessed with no evidence" to "quoted".

Measured on the corpus before being written, with counts as
`studies placed / studies where it widens an existing call`:

```text
\bpatella\b                108 / 21     kept, the largest single gain
\bpatellar\b + guard       118 / 31     kept; unguarded it cost 30 places
\btrochleas\b               16 /  1     kept
```

The guard exists because thirty read windows showed `patellar` attaching to
tendon, bursa and retinaculum far more often than to the joint. Guarding cost
30 of 148 placements and removed 11 of 42 widens — worse on both counts than
predicted, and still worth it, because an unguarded `patellar` labels the
patellar *tendon* as patellofemoral osteoarthritis.

The Spanish and Turkish orthography patterns are kept but **were measured to
place nothing** on this corpus: `troclea`, `mediyal`, `platillo` and their
relatives fired zero times across 4,349 reports. They are retained only
because the hidden test set is a different sample of the same sixteen sites,
and they cost nothing when they do not match. No claim is made for them.

## Defect two: a list negation is read as a finding

83 of B6's 7,039 positive calls quote their own contradiction — a report that
lists `ACL: intact` and is scored positive for ACL. 1.18%, small but wrong in
the direction that matters, since a false positive on a structural target is
exactly what the expert audit punishes.

The guard is deliberately narrow. It fires only when the negation follows the
target phrase **immediately**, which is what a list entry looks like. A clause
such as "meniscal tear, no displacement" contains the same words and must not
flip, because there the negation attaches to something else.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .b6_report_labels import (
    B6Prediction,
    _aliases,
    _clause,
    _classify_mention,
    _component_limited_negative,
    _review_queue,
    _state_to_values,
    _target_audit,
)
from .constants import TARGETS
from .data import gold_mask, load_train_csv, normalize_report
from .report_labels import (
    OA_TARGETS,
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
    predict_target as legacy_predict_target,
)

B6_V13_VERSION = "1.3.0"

# `patellar` and `trochlear` attach to soft tissue far more often than to the
# joint. Without this, the patellar tendon becomes patellofemoral arthritis.
NOT_CARTILAGE = r"(?!\s+(?:tendon|tendin|enthesop|bursit|plica|ligament|retinacul))"

V13_PATTERNS: dict[str, tuple[str, ...]] = {
    "PF OA": (
        r"\bpatella\b",
        r"\bpatellar\b" + NOT_CARTILAGE,
        r"\bpatellae\b",
        r"\btrochleas\b",
        r"\btrochlear\b" + NOT_CARTILAGE,
        # Romance/Turkish orthography: measured at zero on this corpus.
        r"\brotulian\w*\b",
        r"\brotula\b",
        r"\btroclea\w*\b",
    ),
    "Medial OA": (
        r"\bmedial(?: and lateral)? compartments?\b",
        r"\bmedial(?:e|es|en)? femorotibial\w*\b",
        r"\bmediyal\b",
        r"\bcondilo femoral medial\b",
        r"\bplatillo tibial medial\b",
    ),
    "Lateral OA": (
        r"\b(?:medial and )?lateral compartments?\b",
        r"\blateral(?:e|es|en)? femorotibial\w*\b",
        r"\bcondilo femoral lateral\b",
        r"\bplatillo tibial lateral\b",
    ),
}

COMPILED_V13_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    target: tuple(re.compile(pattern, re.I) for pattern in patterns)
    for target, patterns in V13_PATTERNS.items()
}

# The words a report uses to say a listed structure is fine.
ABSENT_WORDS = (
    "none",
    "no",
    "nil",
    "absent",
    "negative",
    "not present",
    "normal",
    "intact",
    "unremarkable",
    "yok",  # Turkish
    "nema",  # South-Slavic
    "sin",  # Spanish
    "senza",  # Italian
    "kein",  # German
    "keine",
    "geen",  # Dutch
)

# Anchored at the start: this only matches text *immediately* after the target
# phrase, which is what "ACL: intact" looks like and "tear, no displacement"
# does not.
LIST_NEGATION_AFTER = re.compile(
    r"^\s*[:\-–—]\s*(?:"
    + "|".join(re.escape(word) for word in ABSENT_WORDS)
    + r")\b",
    re.IGNORECASE,
)


def is_list_negated(text: str, stop: int, *, width: int = 40) -> bool:
    """Whether the target phrase ending at `stop` is immediately called normal."""
    return bool(LIST_NEGATION_AFTER.match(text[stop : stop + int(width)]))


def _collect(
    norm: str,
    target: str,
    spans,
    *,
    guard: bool,
    reason_suffix: str = "",
) -> list[tuple[str, str, str]]:
    """Turn `(start, stop, phrase)` spans into (state, clause, reason) triples."""
    observations: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for start, stop, phrase in spans:
        clause = _clause(norm, start, stop)
        state, reason = _classify_mention(target, clause, phrase)
        if guard and state == STATE_POSITIVE and is_list_negated(norm, stop):
            state, reason = STATE_NEGATED, "list_negation_guard"
        item = (state, clause[:360], reason + reason_suffix)
        if item not in seen:
            observations.append(item)
            seen.add(item)
    return observations


def _alias_spans(norm: str, target: str):
    for phrase in _aliases(target):
        for match in re.finditer(re.escape(phrase), norm, re.I):
            yield match.start(), match.end(), phrase


def _v13_spans(norm: str, target: str):
    for pattern in COMPILED_V13_PATTERNS.get(target, ()):
        for match in pattern.finditer(norm):
            yield match.start(), match.end(), match.group(0)


def alias_observations(norm: str, target: str, *, guard: bool) -> list[tuple[str, str, str]]:
    """What v1.2.1 sees, optionally with the list-negation guard applied."""
    return _collect(norm, target, _alias_spans(norm, target), guard=guard)


def v13_observations(norm: str, target: str) -> list[tuple[str, str, str]]:
    """What the new vocabulary sees. Consulted only where the aliases are silent."""
    return _collect(
        norm, target, _v13_spans(norm, target), guard=True, reason_suffix="_v13"
    )


def _resolve(observations: list[tuple[str, str, str]], target: str) -> B6Prediction:
    """v1.2.1's resolution, unchanged, applied to whatever was observed."""
    positive = [item for item in observations if item[0] == STATE_POSITIVE]
    negative = [item for item in observations if item[0] == STATE_NEGATED]

    if positive and negative:
        global_negative = [
            item for item in negative if not _component_limited_negative(target, item)
        ]
        if not global_negative:
            probability, confidence = _state_to_values(STATE_POSITIVE)
            return B6Prediction(
                probability,
                confidence,
                True,
                STATE_POSITIVE,
                positive[0][1],
                "positive_with_component_limited_normality",
            )
        probability, confidence = _state_to_values(STATE_UNCERTAIN, conflict=True)
        return B6Prediction(
            probability,
            confidence,
            True,
            STATE_UNCERTAIN,
            " || ".join(item[1] for item in observations[:3]),
            "conflicting_definite_evidence",
        )

    for group, state in ((positive, STATE_POSITIVE), (negative, STATE_NEGATED)):
        if group:
            probability, confidence = _state_to_values(state)
            return B6Prediction(
                probability, confidence, True, state, group[0][1], group[0][2]
            )

    probability, confidence = _state_to_values(STATE_UNCERTAIN)
    return B6Prediction(
        probability,
        confidence,
        True,
        STATE_UNCERTAIN,
        observations[0][1],
        observations[0][2],
    )


def predict_target_b6_v13(text: str, target: str, *, use_v13: bool = True) -> B6Prediction:
    """One report, one target. `use_v13=False` reproduces v1.2.1 exactly.

    ## The precedence, and why it is strict

    The new vocabulary answers **only where the aliases are silent**. It is not
    merged into their evidence, and it cannot overturn what they concluded.

    That rule was added after the review found the merge could downgrade a
    confident call. "Patella: normal. Patellofemoral osteoarthritis is
    present." resolves positive under v1.2.1 — one clause, one disease phrase.
    Merging a v1.3 `\\bpatella\\b` match adds a *negated* observation from the
    other clause, the two conflict, and the cell falls to `uncertain` with
    confidence 0.25 instead of 0.90.

    That is backwards. The v1.3 patterns are deliberately broad anatomy words,
    chosen to place cells that had nothing at all; a bare anatomy word calling
    something normal is far weaker evidence than a named disease, and must not
    be allowed to veto it. Letting it would also quietly shrink supervision on
    the three OA targets, which already have the worst coverage of the twelve.

    So the order is: the aliases, then the new vocabulary, then the
    evidence-free fallback — each consulted only if the one before it found
    nothing. The fallback is retained, not deleted; it simply now runs last.
    """
    norm = normalize_report(text)
    if not norm:
        return B6Prediction(0.50, 0.0, False, STATE_UNMENTIONED, "", "empty_report")

    observations = alias_observations(norm, target, guard=use_v13)
    if observations:
        return _resolve(observations, target)

    if use_v13:
        observations = v13_observations(norm, target)

    if not observations and target in OA_TARGETS:
        legacy = legacy_predict_target(norm, target)
        if legacy.state != STATE_UNMENTIONED:
            probability, confidence = _state_to_values(legacy.state)
            return B6Prediction(
                probability,
                confidence,
                True,
                legacy.state,
                "oa_context_parser",
                "compartment_aware_oa_context",
            )

    if not observations:
        return B6Prediction(0.50, 0.0, False, STATE_UNMENTIONED, "", "no_target_evidence")

    return _resolve(observations, target)


def build_b6_v13_frame(df: pd.DataFrame, *, use_v13: bool = True) -> pd.DataFrame:
    """The same table `build_b6_frame` writes, from the v1.3 parser.

    Identical columns and dtypes, so `b23_fill_merge --base` takes it without
    changing a line.
    """
    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"].astype(str)})
    out["is_gold"] = gold_mask(df).to_numpy(dtype=bool)
    reports = df["Report"].fillna("").astype(str)
    out["has_report"] = reports.map(
        lambda text: bool(normalize_report(text))
    ).to_numpy(dtype=bool)

    for target in TARGETS:
        predictions = [
            predict_target_b6_v13(text, target, use_v13=use_v13) for text in reports
        ]
        out[target] = np.asarray(
            [item.probability for item in predictions], dtype=np.float32
        )
        out[f"{target}__confidence"] = np.asarray(
            [item.confidence for item in predictions], dtype=np.float32
        )
        out[f"{target}__state"] = [item.state for item in predictions]
        out[f"{target}__mentioned"] = [item.mentioned for item in predictions]
        out[f"{target}__reason"] = [item.reason for item in predictions]
        out[f"{target}__evidence"] = [item.evidence for item in predictions]
    return out


def change_summary(frozen: pd.DataFrame, updated: pd.DataFrame) -> dict:
    """What v1.3 moved, counted on the report-only studies that train the model."""
    report_only = ~frozen["is_gold"].astype(bool)
    summary: dict = {
        "version": B6_V13_VERSION,
        "studies": int(report_only.sum()),
        "targets": {},
        "totals": {
            "cells_newly_answered": 0,
            "cells_state_changed": 0,
            "fallback_cells_now_quoted": 0,
            "cells_flipped_by_list_negation_guard": 0,
            "cells_silenced": 0,
            # A committed call falling to `uncertain` drops its confidence from
            # 0.90 to 0.25. The strict precedence should make this impossible;
            # it is counted so that "should" is checked rather than trusted.
            "cells_weakened_to_uncertain": 0,
        },
    }
    for target in TARGETS:
        old_state = frozen.loc[report_only, f"{target}__state"]
        new_state = updated.loc[report_only, f"{target}__state"]
        old_reason = frozen.loc[report_only, f"{target}__reason"]
        new_reason = updated.loc[report_only, f"{target}__reason"]

        was_silent = old_state.eq(STATE_UNMENTIONED)
        is_silent = new_state.eq(STATE_UNMENTIONED)
        newly = int((was_silent & ~is_silent).sum())
        changed = int((old_state != new_state).sum())
        requoted = int(
            (
                old_reason.eq("compartment_aware_oa_context")
                & ~new_reason.eq("compartment_aware_oa_context")
                & ~is_silent
            ).sum()
        )
        flipped = int(new_reason.eq("list_negation_guard").sum())
        silenced = int((~was_silent & is_silent).sum())
        weakened = int(
            (
                old_state.isin([STATE_POSITIVE, STATE_NEGATED])
                & new_state.eq(STATE_UNCERTAIN)
            ).sum()
        )

        summary["targets"][target] = {
            "cells_newly_answered": newly,
            "cells_state_changed": changed,
            "fallback_cells_now_quoted": requoted,
            "cells_flipped_by_list_negation_guard": flipped,
            "cells_silenced": silenced,
            "cells_weakened_to_uncertain": weakened,
        }
        for key, value in summary["targets"][target].items():
            summary["totals"][key] += value
    return summary


def run_b6_v13_export(
    train_csv: str | Path,
    *,
    out_root: str | Path = "runs/b6_v13_report_labels",
    min_confidence: float = 0.75,
    max_review: int = 1000,
) -> dict:
    """Write a v1.3 parser export, plus exactly what it changed against v1.2.1."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")
    if max_review < 0:
        raise ValueError("max_review must be >=0")

    df = load_train_csv(train_csv)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    structured = build_b6_v13_frame(df, use_v13=True)
    frozen = build_b6_v13_frame(df, use_v13=False)
    structured.to_csv(out / "structured_labels.csv", index=False)

    report_only = structured.loc[~structured["is_gold"].astype(bool)].copy()
    training_columns = ["StudyInstanceUID"]
    for target in TARGETS:
        training_columns.extend([target, f"{target}__confidence", f"{target}__state"])
    report_only[training_columns].to_csv(out / "training_targets.csv", index=False)

    review = _review_queue(structured, df, max_rows=max_review)
    review.to_csv(out / "review_queue.csv", index=False)

    changes = change_summary(frozen, structured)
    (out / "v13_changes.json").write_text(
        json.dumps(changes, indent=2), encoding="utf-8"
    )

    audit = {
        "b6_version": B6_V13_VERSION,
        "supersedes": "1.2.1",
        "n_studies": int(len(structured)),
        "n_gold_audit_only": int(structured["is_gold"].sum()),
        "n_report_only_training": int((~structured["is_gold"].astype(bool)).sum()),
        "n_reports_present": int(structured["has_report"].sum()),
        "min_confidence_for_usable_cell": float(min_confidence),
        "external_models": False,
        "external_data": False,
        "gold_fitted_calibration": False,
        "gold_rows_in_training_targets": 0,
        "targets": {
            target: _target_audit(structured, target, min_confidence=min_confidence)
            for target in TARGETS
        },
        "v13_changes": changes["totals"],
        "review_queue_rows": int(len(review)),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps({"v13_changes": changes}, indent=2))
    for name in (
        "structured_labels.csv",
        "training_targets.csv",
        "review_queue.csv",
        "audit.json",
        "v13_changes.json",
    ):
        print(out / name)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export B6 v1.3 report labels, and what they change against v1.2.1"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-root", default="runs/b6_v13_report_labels")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-review", type=int, default=1000)
    args = parser.parse_args()

    run_b6_v13_export(
        args.train_csv,
        out_root=args.out_root,
        min_confidence=args.min_confidence,
        max_review=args.max_review,
    )


def compare_versions(reports: pd.Series, target: str) -> pd.DataFrame:
    """What v1.3 changes, report by report, so the diff can be inspected."""
    rows = []
    for index, text in reports.items():
        old = predict_target_b6_v13(str(text), target, use_v13=False)
        new = predict_target_b6_v13(str(text), target, use_v13=True)
        rows.append(
            {
                "index": index,
                "old_state": old.state,
                "new_state": new.state,
                "old_reason": old.reason,
                "new_reason": new.reason,
                "changed": old.state != new.state,
                "newly_placed": old.state == STATE_UNMENTIONED
                and new.state != STATE_UNMENTIONED,
                "now_quoted": old.reason == "compartment_aware_oa_context"
                and new.reason != "compartment_aware_oa_context",
                "evidence": new.evidence,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
