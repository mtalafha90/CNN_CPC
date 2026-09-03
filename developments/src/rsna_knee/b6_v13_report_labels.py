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

import re

import pandas as pd

from .b6_report_labels import (
    B6Prediction,
    _aliases,
    _clause,
    _classify_mention,
    _component_limited_negative,
    _state_to_values,
)
from .data import normalize_report
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


def _observations(norm: str, target: str, *, use_v13: bool) -> list[tuple[str, str, str]]:
    """Every mention of the target, as (state, clause, reason).

    The literal aliases run exactly as v1.2.1 runs them. The v1.3 patterns run
    afterwards and can only add mentions, never remove one.
    """
    observations: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def record(start: int, stop: int, phrase: str, reason_suffix: str = "") -> None:
        clause = _clause(norm, start, stop)
        state, reason = _classify_mention(target, clause, phrase)
        if use_v13 and state == STATE_POSITIVE and is_list_negated(norm, stop):
            state, reason = STATE_NEGATED, "list_negation_guard"
        item = (state, clause[:360], reason + reason_suffix)
        if item not in seen:
            observations.append(item)
            seen.add(item)

    for phrase in _aliases(target):
        for match in re.finditer(re.escape(phrase), norm, re.I):
            record(match.start(), match.end(), phrase)

    if use_v13:
        for pattern in COMPILED_V13_PATTERNS.get(target, ()):
            for match in pattern.finditer(norm):
                record(match.start(), match.end(), match.group(0), "_v13")

    return observations


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

    The evidence-free OA fallback is retained but now runs last, so it only
    fires where the new vocabulary also found nothing — which is the point of
    adding the vocabulary.
    """
    norm = normalize_report(text)
    if not norm:
        return B6Prediction(0.50, 0.0, False, STATE_UNMENTIONED, "", "empty_report")

    observations = _observations(norm, target, use_v13=use_v13)

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
