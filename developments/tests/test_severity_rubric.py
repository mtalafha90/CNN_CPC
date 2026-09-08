"""Grading the teacher against the competition's rubric.

This module turns correct positives into negatives, which is the most dangerous
kind of edit this archive makes -- v1.3.0 weakened 360 committed cells by
accident and the tests that were meant to catch it only covered the alias path.

So the tests here are about the two ways it can be wrong in opposite
directions: gating a finding that is emphatically present, and failing to gate
one that is explicitly sub-threshold in a language nobody checked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rsna_knee.severity_rubric import (
    DOWNGRADE_STATE,
    STATE_NEGATED,
    STATE_POSITIVE,
    SUB_THRESHOLD,
    apply_rubric,
    is_sub_threshold,
)

EFFUSION = (
    "effusion", "efusion", "efüzyon", "derrame", "erguss", "epanchement",
    "versamento",
)
MENISCUS = ("meniscus", "menisco", "menisküs", "meniskus", "menisque")
ACL = ("acl", "anterior cruciate", "ligamento cruzado anterior", "on capraz")


def _sub(target, text, anchors):
    return is_sub_threshold(target, text, anchors)["downgrade"]


# --- the rubric's own thresholds ----------------------------------------------


def test_a_trace_effusion_is_negative_under_this_rubric():
    assert _sub("Effusion", "There is a trace joint effusion.", EFFUSION)


def test_a_moderate_effusion_survives():
    assert not _sub("Effusion", "Moderate joint effusion is present.", EFFUSION)


def test_a_large_effusion_survives():
    assert not _sub("Effusion", "Large suprapatellar effusion.", EFFUSION)


def test_intrasubstance_meniscal_signal_is_negative():
    assert _sub(
        "Medial Meniscus",
        "Intrasubstance degenerative signal in the medial meniscus.",
        MENISCUS,
    )


def test_a_surface_contacting_tear_survives():
    assert not _sub(
        "Medial Meniscus",
        "Full thickness tear of the medial meniscus reaching the surface.",
        MENISCUS,
    )


def test_a_low_grade_acl_sprain_is_negative():
    assert _sub("ACL", "Low-grade sprain of the ACL.", ACL)


def test_a_full_thickness_acl_tear_survives():
    assert not _sub("ACL", "Complete full-thickness ACL tear.", ACL)


def test_low_grade_chondromalacia_is_negative_for_oa():
    assert _sub(
        "Medial OA",
        "Chondromalacia grade 2 of the medial compartment.",
        ("medial compartment", "medial femorotibial"),
    )


# --- the failure that matters most: the nearer word wins ----------------------


def test_a_nearby_severe_word_beats_a_distant_mild_one():
    """Reports list several findings of different severities in one sentence."""
    text = "Large joint effusion with a small Baker's cyst."
    assert not _sub("Effusion", text, EFFUSION)


def test_the_same_sentence_still_gates_the_other_finding():
    """The small Baker's must still be caught in that same sentence."""
    text = "Large joint effusion with a small Baker's cyst."
    assert _sub("Baker's", text, ("baker",))


def test_a_severity_word_for_another_finding_does_not_leak_in():
    text = "No effusion. There is a large full-thickness ACL tear."
    assert not _sub("ACL", text, ACL)


# --- multilingual, which is not optional --------------------------------------
#
# Spanish 1,807, English 630, Turkish 595, Greek 321, Cyrillic 220, German 173,
# French 83. An English-only gate would bias the correction by site.


@pytest.mark.parametrize(
    "language,text",
    [
        ("english", "Mild joint effusion."),
        ("spanish", "Leve derrame articular."),
        ("spanish-min", "Minimo derrame articular."),
        ("turkish", "Hafif eklem efüzyonu mevcut."),
        ("german", "Geringer Gelenkerguss."),
        ("french", "Epanchement articulaire de faible abondance, minime."),
        ("italian", "Lieve versamento, minimo."),
    ],
)
def test_a_mild_effusion_is_gated_in_every_major_language(language, text):
    assert _sub("Effusion", text, EFFUSION), language


@pytest.mark.parametrize(
    "language,text",
    [
        ("english", "Moderate joint effusion."),
        ("spanish", "Derrame articular moderado."),
        ("turkish", "Orta derecede eklem efüzyonu."),
        ("german", "Massiger Gelenkerguss."),
    ],
)
def test_a_moderate_effusion_survives_in_every_major_language(language, text):
    assert not _sub("Effusion", text, EFFUSION), language


def test_turkish_qualifiers_precede_the_noun():
    """Turkish puts the qualifier before the noun and its negation after."""
    assert _sub("Effusion", "hafif efüzyon izlendi", EFFUSION)


# --- what it must never touch -------------------------------------------------


def test_contusion_is_never_gated():
    """The rubric publishes no severity threshold for it."""
    assert "Contusion" not in SUB_THRESHOLD
    assert not _sub("Contusion", "Small bone contusion.", ("contusion",))


def test_fracture_is_never_gated():
    assert "Fracture" not in SUB_THRESHOLD
    assert not _sub("Fracture", "Tiny undisplaced fracture.", ("fracture",))


def test_a_finding_never_mentioned_is_not_gated():
    """Absence of text is not evidence of a sub-threshold finding."""
    assert not _sub("Effusion", "The menisci are intact.", EFFUSION)


def test_a_mention_with_no_qualifier_is_not_gated():
    assert not _sub("Effusion", "Joint effusion is present.", EFFUSION)


# --- the table walk -----------------------------------------------------------


def _frame():
    return pd.DataFrame(
        [
            {"StudyInstanceUID": "a", "target": "Effusion", "state": STATE_POSITIVE},
            {"StudyInstanceUID": "b", "target": "Effusion", "state": STATE_POSITIVE},
            {"StudyInstanceUID": "c", "target": "Effusion", "state": STATE_NEGATED},
            {"StudyInstanceUID": "d", "target": "Contusion", "state": STATE_POSITIVE},
            {"StudyInstanceUID": "e", "target": "Effusion", "state": "uncertain"},
        ]
    )


_REPORTS = {
    "a": "Trace joint effusion.",
    "b": "Large joint effusion.",
    "c": "No effusion.",
    "d": "Small bone contusion.",
    "e": "Mild effusion.",
}
_ANCHORS = {"Effusion": EFFUSION, "Contusion": ("contusion",)}


def test_only_the_sub_threshold_positive_moves():
    out, audit = apply_rubric(_frame(), _REPORTS, _ANCHORS)

    assert list(out["state"]) == [
        DOWNGRADE_STATE,  # a: trace -> negated
        STATE_POSITIVE,  # b: large, untouched
        STATE_NEGATED,  # c: already negative
        STATE_POSITIVE,  # d: contusion is never gated
        "uncertain",  # e: not committed, so not eligible
    ]
    assert audit["cells_downgraded"] == 1


def test_it_never_promotes_or_creates_a_cell():
    before = _frame()
    out, _ = apply_rubric(before, _REPORTS, _ANCHORS)

    assert len(out) == len(before)
    moved = (before["state"] != out["state"])
    assert set(before.loc[moved, "state"]) <= {STATE_POSITIVE}


def test_an_uncommitted_cell_is_never_gated():
    """`e` is mild AND uncertain: eligible by text, ineligible by state."""
    out, _ = apply_rubric(_frame(), _REPORTS, _ANCHORS)
    assert out.loc[4, "state"] == "uncertain"


def test_a_study_with_no_report_is_left_alone():
    out, audit = apply_rubric(_frame(), {}, _ANCHORS)
    assert audit["cells_downgraded"] == 0


def test_the_audit_counts_by_target_and_names_what_it_skipped():
    _, audit = apply_rubric(_frame(), _REPORTS, _ANCHORS)

    assert audit["by_target"] == {"Effusion": 1}
    assert "Contusion" in audit["targets_never_gated"]
    assert audit["positive_cells_before"] == 3
    assert audit["fraction_of_positives_downgraded"] == pytest.approx(1 / 3)


def test_a_frame_missing_a_column_is_refused():
    with pytest.raises(ValueError, match="target"):
        apply_rubric(
            pd.DataFrame([{"StudyInstanceUID": "a", "state": STATE_POSITIVE}]),
            _REPORTS,
            _ANCHORS,
        )


def test_the_downgrade_state_is_configurable():
    """One edit and one re-run, if negated turns out to be the wrong call."""
    out, audit = apply_rubric(_frame(), _REPORTS, _ANCHORS, downgrade_to="uncertain")

    assert out.loc[0, "state"] == "uncertain"
    assert audit["downgrade_state"] == "uncertain"


def test_the_default_is_negated_because_the_rubric_says_so():
    assert DOWNGRADE_STATE == STATE_NEGATED


# --- the audit trail ----------------------------------------------------------


def test_every_change_records_the_sentence_that_caused_it():
    verdict = is_sub_threshold("Effusion", "There is a trace effusion.", EFFUSION)

    assert verdict["downgrade"] is True
    assert verdict["matched_term"] == "trace"
    assert "trace effusion" in verdict["window"]


def test_a_refusal_says_why():
    for text, fragment in (
        ("The menisci are intact.", "not mentioned"),
        ("Joint effusion is present.", "no sub-threshold qualifier"),
    ):
        assert fragment in is_sub_threshold("Effusion", text, EFFUSION)["reason"]


def test_an_ungated_target_says_so():
    reason = is_sub_threshold("Fracture", "Tiny fracture.", ("fracture",))["reason"]
    assert "no published threshold" in reason


# --- the teacher's vocabulary is regexes, not words ---------------------------
#
# V13_PATTERNS holds entries like r"\bpatella\b". Searching for those as literal
# substrings matches nothing, so the gate would have reported success while
# firing on zero cells. This is the test that would have caught it.


def test_a_regex_anchor_is_matched_as_a_regex():
    assert _sub("Effusion", "Trace joint effusion.", (r"\beffusion\b",))


def test_a_regex_anchor_still_respects_the_nearer_word():
    assert not _sub("Effusion", "Large joint effusion.", (r"\beffusion\b",))


def test_an_alternation_anchor_works():
    pattern = (r"\b(?:effusion|epanchement)\b",)
    assert _sub("Effusion", "Minimal epanchement articulaire.", pattern)


def test_a_word_boundary_is_honoured():
    """`\\bacl\\b` must not fire inside 'tentacle'."""
    assert not _sub("ACL", "Mild tentacle-like artefact.", (r"\bacl\b",))


def test_a_plain_word_anchor_still_works():
    """A literal list must keep working, accents and all."""
    assert _sub("Effusion", "Leve derrame articular.", ("derrame",))


def test_an_invalid_regex_falls_back_to_a_literal():
    """An unbalanced bracket must not crash a whole teacher rebuild."""
    assert _sub("Effusion", "Trace effusion[ here.", ("effusion[",))


def test_the_real_teacher_vocabulary_matches_real_text():
    """The end-to-end check: the teacher's own anchors, unmodified."""
    from rsna_knee.b55_rubric_teacher import teacher_anchors

    anchors = teacher_anchors()
    assert _sub("Effusion", "There is a trace joint effusion.", anchors["Effusion"])
    assert not _sub("Effusion", "Large joint effusion.", anchors["Effusion"])


def test_every_target_has_anchors_to_look_for():
    """A target with no anchors is a gate that silently never fires."""
    from rsna_knee.b55_rubric_teacher import teacher_anchors
    from rsna_knee.constants import TARGETS

    anchors = teacher_anchors()
    for target in TARGETS:
        assert len(anchors.get(target, ())) >= 2, target


def test_the_gated_targets_all_have_anchors():
    """SUB_THRESHOLD and the anchor vocabulary must agree on what exists."""
    from rsna_knee.b55_rubric_teacher import teacher_anchors

    anchors = teacher_anchors()
    for target in SUB_THRESHOLD:
        assert anchors.get(target), f"{target} is gated but has nothing to find"
