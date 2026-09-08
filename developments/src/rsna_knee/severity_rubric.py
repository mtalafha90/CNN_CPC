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

#: How many words may sit between a severity word and the sub-threshold word it
#: modifies. Zero covers "high-grade sprain"; two covers "high-grade partial
#: thickness tear". Measured in words rather than characters because characters
#: reach across a whole clause: in "large joint effusion with a small Baker's
#: cyst" the word `large` is 27 characters before `small` and modifies nothing
#: of the sort.
COMPOUND_WORDS = 2

#: Retained for the audit's shape only. The decision is scoped by sentence now,
#: not by a character window -- see `_sentences` for why.
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


def _sentences(folded: str) -> list[tuple[str, int]]:
    """Split into sentences, each with its offset in the folded text.

    A severity word qualifies a finding in its own sentence. Measuring by
    character distance across a fixed window instead lets one sentence's
    qualifier reach into the next: "Large joint effusion. Small Baker's cyst."
    put `small` nine characters from `effusion` and `large` twelve, so the
    effusion was downgraded by the cyst's adjective.
    """
    parts, start = [], 0
    for index, ch in enumerate(folded):
        # Sentences, not clauses. A comma does not stop a qualifier describing
        # a finding -- "Intrasubstance degenerative signal, medial meniscus" is
        # one statement -- so splitting here would lose real downgrades. The
        # comma matters only to the compound veto; see `_qualified_above`.
        if ch in ".;\n":
            piece = folded[start:index]
            if piece.strip():
                parts.append((piece, start))
            start = index + 1
    tail = folded[start:]
    if tail.strip():
        parts.append((tail, start))
    return parts or [(folded, 0)]


def _matches(window: str, terms: tuple[str, ...]) -> list[tuple[int, str]]:
    """Every occurrence of any term, as (offset, term)."""
    found: list[tuple[int, str]] = []
    for term in terms:
        needle = _fold(term)
        if not needle:
            continue
        at = 0
        while True:
            hit = window.find(needle, at)
            if hit < 0:
                break
            found.append((hit, term))
            at = hit + 1
    return found


def _qualified_above(window: str, at: int, anchors: tuple[str, ...]) -> str | None:
    """An above-threshold word modifying this one, if any.

    `sprain` means low grade on its own and nothing of the sort in "high-grade
    sprain": there the severity word modifies the sub-threshold word rather
    than competing with it. Distance from the finding cannot see this -- in
    "High-grade MCL sprain" the word `sprain` is nearer the anchor than
    `high grade` is, so nearest-wins downgrades a high-grade tear.

    The two are taken to describe the same finding when few words separate
    them, **not counting this finding's own name**: "high-grade partial
    thickness MCL sprain" is one finding, while "large joint effusion with a
    small Baker's cyst" is two and the gap there is full of the other one.
    """
    # A modifier cannot reach across a comma. "Large joint effusion, small
    # Baker's cyst" leaves only two words between `large` and `small`, close
    # enough for the word gap below to read the effusion's adjective as
    # modifying the cyst's -- and the small cyst then escapes its downgrade.
    # The comma is a clause boundary for *this* question and not for finding
    # the qualifier in the first place, which is why `_sentences` ignores it.
    clause_start = window.rfind(",", 0, at) + 1
    for hit, term in _matches(window[clause_start:at], _ABOVE_THRESHOLD):
        hit += clause_start
        gap = window[hit + len(_fold(term)) : at]
        for start, anchor in sorted(_anchor_matches(gap, anchors), reverse=True):
            gap = gap[:start] + " " + gap[start + len(anchor) :]
        if len(gap.split()) <= COMPOUND_WORDS:
            return term
    return None


def _anchor_matches(window: str, anchors: tuple[str, ...]) -> list[tuple[int, str]]:
    """Where this finding's own name appears, and how much of the text it spans."""
    found: list[tuple[int, str]] = []
    for anchor in anchors:
        for start in _anchor_spans(window, str(anchor)):
            # A regex anchor's matched text is not the pattern, so re-measure.
            text = _fold(anchor)
            if any(ch in _META for ch in str(anchor)):
                try:
                    match = re.search(str(anchor), window[start:], re.IGNORECASE)
                    text = match.group(0) if match else ""
                except re.error:
                    text = ""
            if text:
                found.append((start, text))
    return found


def is_sub_threshold(target: str, text: str, anchors: tuple[str, ...]) -> dict:
    """Does the report describe this finding below the competition's threshold?

    Three rules, in order, each answering a way the previous one fails:

    ```text
    same sentence      a qualifier belongs to the finding it is written with
    not a compound     "high-grade sprain" is not a low-grade sprain
    nearest wins       within one sentence naming several findings, the
                       closest qualifier is the one that applies
    ```

    Returns the decision and its evidence, so every downgrade can be read back
    rather than trusted.
    """
    terms = SUB_THRESHOLD.get(target)
    if not terms:
        return {"downgrade": False, "reason": "no published threshold for this target"}

    folded = _fold(text)
    positions = sorted(
        {pos for anchor in anchors for pos in _anchor_spans(folded, str(anchor))}
    )
    if not positions:
        return {"downgrade": False, "reason": "the finding is not mentioned in the text"}

    sentences = _sentences(folded)
    blocked: str | None = None

    for position in positions:
        sentence, start = next(
            (
                (piece, offset)
                for piece, offset in sentences
                if offset <= position < offset + len(piece)
            ),
            (folded, 0),
        )
        at = position - start

        # The finding's own comma-clause first. A qualifier written with the
        # finding beats one written with a different finding in the same
        # sentence, however few characters away that one happens to sit:
        # "Large joint effusion, small Baker's cyst" puts `small` ten
        # characters from `effusion` and `large` twelve.
        clause_lo = sentence.rfind(",", 0, at) + 1
        clause_hi = sentence.find(",", at)
        clause_hi = len(sentence) if clause_hi < 0 else clause_hi
        scopes = [(sentence[clause_lo:clause_hi], at - clause_lo, "clause")]
        if (clause_lo, clause_hi) != (0, len(sentence)):
            # Only when the clause settles nothing: "Intrasubstance
            # degenerative signal, medial meniscus" is one statement whose
            # qualifier sits in the other clause.
            scopes.append((sentence, at, "sentence"))

        for window, anchor_at, scope in scopes:
            below: list[tuple[int, int, str]] = []
            for hit, term in _matches(window, terms):
                modifier = _qualified_above(window, hit, anchors)
                if modifier is not None:
                    blocked = blocked or f"{modifier!r} qualifies {term!r}"
                    continue
                below.append((abs(hit - anchor_at), hit, term))

            above = _matches(window, _ABOVE_THRESHOLD)
            nearest_above = min((abs(h - anchor_at) for h, _ in above), default=None)

            if not below:
                # An above-threshold word alone settles it; nothing settles
                # nothing, so widen.
                if nearest_above is not None:
                    break
                continue

            below.sort()
            distance, _, matched = below[0]
            if nearest_above is not None and nearest_above <= distance:
                break

            return {
                "downgrade": True,
                "reason": f"sub-threshold qualifier {matched!r} in the same {scope}",
                "matched_term": matched,
                "window": window.strip()[:200],
            }

    if blocked:
        return {
            "downgrade": False,
            "reason": f"above-threshold: {blocked}",
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
        # Every downgraded cell with the term and the sentence that caused it.
        # `rebuild` pops this into rubric_changes.csv rather than letting it
        # into audit.json, where thousands of rows would bury the summary.
        # It is carried here rather than returned separately because a caller
        # that has to ask for the evidence is a caller that will not.
        "changes": changes,
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
