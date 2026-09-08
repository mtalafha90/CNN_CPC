"""Grade the teacher against the competition's rubric, not against clinical truth.

## The mismatch this exists to close

The competition's labels are **not** report-derived. Two subspecialty MSK
radiologists read each study independently from the images, a third adjudicates,
and the same process produced the test set. The governing rule is a
**specificity bias**: a finding that is borderline or "on the fence" is graded
**NEGATIVE**.

The published thresholds are quantitative:

```text
ACL / MCL     high-grade partial or full-thickness only
              mild signal change, low-grade sprain        -> NEGATIVE
Meniscus      abnormal signal definitely contacting the surface on >= 2
              images, or definite morphological abnormality
              intrasubstance degeneration                 -> NEGATIVE
OA            roughly >= 1 cm of > 50% thickness cartilage loss
              mild chondral change, low-grade chondromalacia -> NEGATIVE
Effusion      moderate or large only; trace and mild      -> NEGATIVE
Baker's       moderate or large only; small and trace     -> NEGATIVE
```

A report teacher tuned for correctness therefore **over-fires**. "Mild effusion",
intrasubstance meniscal degeneration and low-grade chondropathy are all real
findings, correctly extracted, and all competition-negative. That is structured,
target-correlated noise, which is far more damaging than symmetric noise: it
teaches the model to answer a different question from the one it is scored on.

This module is the correction. It reads the same report text the teacher read,
and where the evidence is *explicitly* sub-threshold it turns a positive into a
negative — which under this rubric is not a weakening but the right answer.

## Why NEGATIVE and not UNCERTAIN

The archive's instinct, correctly learned from the v1.3.0 incident, is that
weakening a committed cell is dangerous. It is the opposite here. A trace
effusion is not an unknown; the graders would have written *negative*. Turning
it uncertain would discard a real supervised cell and keep the model ignorant of
the threshold. `DOWNGRADE_STATE` is nonetheless a module constant rather than a
literal, so the choice is one edit and one re-run.

## What it deliberately does not touch

**Contusion and Fracture.** The reported rubric gives them a definition
(marrow oedema without a discrete fracture line; acute cortical break) but no
severity threshold. There is nothing to gate, and inventing one would be the
kind of plausible-sounding change this archive has learned to refuse.

**Synovitis** keeps only the plainest severity words. Its rubric was not
published in the same detail, and it is already the teacher's weakest target.

**Any cell the teacher did not commit.** A gate that promoted or created cells
would be a second mechanism. This one only ever moves `positive -> negated`.

## The multilingual problem, which is not optional

The corpus is roughly Spanish 1,807, English 630, **Turkish 595**, Greek 321,
Cyrillic 220, German 173, French 83. An English-only severity list would gate
about a seventh of the reports and silently leave the rest over-firing — worse
than not gating at all, because the resulting bias would vary by site.

Turkish matters twice over: it is the third-largest language and it places its
qualifiers before the noun (`hafif efüzyon`), while its negation follows
(`efüzyon izlenmedi`).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

SEVERITY_VERSION = "severity_rubric_v1"

STATE_POSITIVE = "positive"
STATE_NEGATED = "negated"

#: What a sub-threshold positive becomes. The rubric grades borderline findings
#: negative, so that is the default rather than "uncertain".
DOWNGRADE_STATE = STATE_NEGATED

#: How much text around the finding is searched for a severity qualifier.
#: Wide enough for "a small amount of joint fluid is seen", narrow enough not to
#: reach the next sentence's finding.
CONTEXT_CHARS = 90

# --- the vocabulary -----------------------------------------------------------
#
# Grouped by meaning rather than by language, so a missing translation is
# visible as a gap in a row rather than hidden in a flat list.

_MILD = (
    # English
    "mild", "minimal", "trace", "small", "slight", "tiny", "minor",
    # Spanish
    "leve", "minimo", "minima", "escaso", "escasa", "pequeno", "pequena",
    "ligero", "ligera", "trazas", "discreto", "discreta",
    # Turkish
    "hafif", "minimal", "az", "kucuk", "silik",
    # Greek (accents stripped by _fold)
    "ipia", "elachisti", "mikri", "mikro",
    # German
    "gering", "geringe", "geringgradig", "leicht", "leichte", "diskret",
    # French
    "leger", "legere", "minime", "discret", "discrete", "petit", "petite",
    # Italian / Portuguese, which appear in small numbers
    "lieve", "modesto", "modesta", "discreta",
)

_LOW_GRADE = (
    "low grade", "low-grade", "grade 1", "grade i", "grade one",
    "grado 1", "grado i", "derece 1", "grad 1", "grado leve",
    "sprain", "strain", "esguince", "distorsion", "zerrung", "entorse",
    "partial low", "signal change without", "signal alteration without",
)

_INTRASUBSTANCE = (
    "intrasubstance", "intra-substance", "intrasustancia", "intrasustancial",
    "intrasubstans", "intrasubstanziell", "intrasubstantiel",
    "degenerative signal", "senal degenerativa", "signal degeneratif",
    "degeneratif sinyal", "mucoid degeneration", "degeneracion mucoide",
    "grade 2 signal", "grado 2", "no alcanza la superficie",
    "does not contact the surface", "does not reach the articular surface",
    "without surface extension", "sin extension a la superficie",
)

_LOW_GRADE_CHONDRAL = (
    "chondromalacia grade 1", "chondromalacia grade i",
    "chondromalacia grade 2", "chondromalacia grade ii",
    "condromalacia grado 1", "condromalacia grado 2",
    "kondromalazi grade 1", "kondromalazi grade 2",
    "chondropathy", "condropatia", "kondropati", "chondropathie",
    "cartilage thinning", "adelgazamiento del cartilago",
    "kikirdak incelmesi", "knorpelverdunnung",
    "focal chondral", "superficial chondral", "partial thickness chondral",
)

#: Which sub-threshold families gate which target. A target absent from this
#: mapping is never downgraded.
SUB_THRESHOLD: dict[str, tuple[str, ...]] = {
    "ACL": _MILD + _LOW_GRADE,
    "MCL": _MILD + _LOW_GRADE,
    "Medial Meniscus": _INTRASUBSTANCE,
    "Lateral Meniscus": _INTRASUBSTANCE,
    "Medial OA": _MILD + _LOW_GRADE_CHONDRAL,
    "Lateral OA": _MILD + _LOW_GRADE_CHONDRAL,
    "PF OA": _MILD + _LOW_GRADE_CHONDRAL,
    "Effusion": _MILD,
    "Baker's": _MILD,
    "Synovitis": _MILD,
    # Contusion and Fracture: the rubric publishes no severity threshold for
    # them, so there is nothing to gate. Deliberately absent.
}

#: Words that mean the finding is emphatically present. When one of these sits
#: closer to the finding than the sub-threshold word, the cell is left alone --
#: "moderate to large effusion with a small Baker's cyst" must not gate the
#: effusion because the sentence also contains "small".
_ABOVE_THRESHOLD = (
    "moderate", "large", "severe", "marked", "gross", "extensive", "massive",
    "full thickness", "full-thickness", "complete", "high grade", "high-grade",
    "grade 3", "grade iii", "grade 4", "grade iv", "bucket handle",
    "moderado", "moderada", "grande", "severo", "severa", "importante",
    "espesor completo", "completa",
    "orta", "buyuk", "belirgin", "siddetli", "tam kat",
    "massig", "gross", "ausgepragt", "hochgradig", "vollstandig",
    "modere", "moderee", "important", "severe", "transfixiant",
)


def _fold(text: str) -> str:
    """Lowercase, strip accents, and normalise whitespace and punctuation.

    Greek and Cyrillic are transliterated only in the sense that accents are
    removed; the vocabulary above spells Greek terms in the Latin form that
    survives folding, which is how they appear in this corpus after the reports
    have been through the same normalisation the teacher uses.
    """
    folded = unicodedata.normalize("NFKD", str(text).lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", folded.replace("-", " ").replace("/", " "))


_META = set(r"\[](){}?*+|^$")


def _anchor_spans(folded: str, anchor: str) -> list[int]:
    """Where the finding is mentioned, for a literal *or* a regex anchor.

    The teacher's own vocabulary (`V13_PATTERNS`) is regular expressions --
    `\\bpatella\\b`, not `patella`. Searching for those literally matches
    nothing, so the gate would have fired on zero cells while reporting
    success, which is the failure this archive keeps paying for.

    A pattern with no metacharacters is folded and matched literally, so a
    plain word list still works and accented spellings still fold.
    """
    if any(ch in _META for ch in anchor):
        try:
            return [m.start() for m in re.finditer(anchor, folded, re.IGNORECASE)]
        except re.error:
            pass  # fall through and treat it as a literal
    needle = _fold(anchor)
    if not needle:
        return []
    starts, at = [], 0
    while True:
        found = folded.find(needle, at)
        if found < 0:
            return starts
        starts.append(found)
        at = found + len(needle)


def _windows(text: str, anchors: tuple[str, ...]) -> list[tuple[str, int]]:
    """The text around each mention of the finding, and where the finding is.

    The offset is returned rather than assumed to be the middle. Near the start
    or end of a report the window is clipped, so the centre of the string is
    not the position of the finding -- and distances measured from the wrong
    origin pick the wrong qualifier, which is the whole decision this module
    makes.
    """
    folded = _fold(text)
    spans: list[tuple[str, int]] = []
    for anchor in anchors:
        for found in _anchor_spans(folded, str(anchor)):
            lo = max(0, found - CONTEXT_CHARS)
            hi = min(len(folded), found + CONTEXT_CHARS)
            spans.append((folded[lo:hi], found - lo))
    return spans


def _nearest(window: str, at: int, terms: tuple[str, ...]) -> int | None:
    """Distance from the finding to the closest of `terms`, or None."""
    best: int | None = None
    for term in terms:
        folded = _fold(term)
        if not folded:
            continue
        start = 0
        while True:
            found = window.find(folded, start)
            if found < 0:
                break
            distance = abs(found - at)
            best = distance if best is None else min(best, distance)
            start = found + 1
    return best


def is_sub_threshold(target: str, text: str, anchors: tuple[str, ...]) -> dict:
    """Does the report describe this finding below the competition's threshold?

    Returns the decision and why, so every downgrade can be read back rather
    than trusted. A sub-threshold word only counts when no above-threshold word
    sits closer to the finding: reports routinely mention several findings of
    different severities in one sentence.
    """
    terms = SUB_THRESHOLD.get(target)
    if not terms:
        return {"downgrade": False, "reason": "no published threshold for this target"}

    windows = _windows(text, anchors)
    if not windows:
        return {"downgrade": False, "reason": "the finding is not mentioned in the text"}

    for window, at in windows:
        below = _nearest(window, at, terms)
        if below is None:
            continue
        above = _nearest(window, at, _ABOVE_THRESHOLD)
        if above is not None and above <= below:
            continue
        matched = next(
            (t for t in terms if _fold(t) in window), ""
        )
        return {
            "downgrade": True,
            "reason": f"sub-threshold qualifier {matched!r} near the finding",
            "matched_term": matched,
            "window": window.strip()[:200],
        }
    return {"downgrade": False, "reason": "no sub-threshold qualifier near the finding"}


def apply_rubric(
    cells: pd.DataFrame,
    reports: dict[str, str],
    anchors: dict[str, tuple[str, ...]],
    *,
    state_column: str = "state",
    downgrade_to: str = DOWNGRADE_STATE,
) -> tuple[pd.DataFrame, dict]:
    """Downgrade every explicitly sub-threshold positive, and audit all of it.

    `cells` needs StudyInstanceUID, target and a state column. `anchors` maps a
    target to the phrases that mean the finding itself, which is the teacher's
    own vocabulary rather than a second one invented here.

    Only `positive -> downgrade_to` is ever written. Negatives, uncertains and
    absent cells are returned untouched.
    """
    required = {"StudyInstanceUID", "target", state_column}
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"cells is missing {', '.join(sorted(missing))}")

    out = cells.copy()
    changes: list[dict] = []
    by_target: Counter = Counter()

    for index, row in out.iterrows():
        if str(row[state_column]) != STATE_POSITIVE:
            continue
        target = str(row["target"])
        text = reports.get(str(row["StudyInstanceUID"]), "")
        if not text:
            continue
        verdict = is_sub_threshold(target, text, anchors.get(target, ()))
        if not verdict["downgrade"]:
            continue
        out.at[index, state_column] = downgrade_to
        by_target[target] += 1
        changes.append(
            {
                "StudyInstanceUID": str(row["StudyInstanceUID"]),
                "target": target,
                "from": STATE_POSITIVE,
                "to": downgrade_to,
                "matched_term": verdict.get("matched_term", ""),
                "window": verdict.get("window", ""),
            }
        )

    positives = int((cells[state_column] == STATE_POSITIVE).sum())
    audit = {
        "version": SEVERITY_VERSION,
        "downgrade_state": downgrade_to,
        "positive_cells_before": positives,
        "cells_downgraded": len(changes),
        "fraction_of_positives_downgraded": (
            len(changes) / positives if positives else 0.0
        ),
        "by_target": dict(sorted(by_target.items())),
        "targets_never_gated": sorted(
            set(cells["target"].unique()) - set(SUB_THRESHOLD)
        ),
        "context_chars": CONTEXT_CHARS,
    }
    return out, audit


def write_audit(audit: dict, changes_path: str | Path, changes: list[dict]) -> None:
    """Every changed cell, with the sentence that changed it."""
    path = Path(changes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(changes).to_csv(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


__all__ = [
    "CONTEXT_CHARS",
    "DOWNGRADE_STATE",
    "SEVERITY_VERSION",
    "SUB_THRESHOLD",
    "apply_rubric",
    "is_sub_threshold",
    "write_audit",
]
