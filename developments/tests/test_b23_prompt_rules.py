"""The B23 prompt is derived from measured B6 failures, so it is tested like code.

Every case below is a real `conflicting_definite_evidence` row from the B6
v1.2.1 review queue, or a real passage from a training report. They are recorded
here so that a later prompt revision cannot silently drop the rule that a
specific measured failure required. These tests assert the *rule is stated*;
only the labeller audit against expert gold can assert that it works.
"""
from __future__ import annotations

import pytest

from rsna_knee.b23_llm_labels import SYSTEM_PROMPT, TARGET_DEFINITIONS, build_user_prompt
from rsna_knee.constants import TARGETS

PROMPT = SYSTEM_PROMPT.lower()


def test_every_target_has_a_definition():
    assert set(TARGET_DEFINITIONS) == set(TARGETS)
    assert all(TARGET_DEFINITIONS[target].strip() for target in TARGETS)


def test_all_nine_rules_are_present_and_numbered():
    for i in range(1, 10):
        assert f"## rule {i} " in PROMPT, f"rule {i} missing"


def test_output_contract_is_stated_last():
    assert SYSTEM_PROMPT.rstrip().endswith("spelled exactly as listed.")
    assert '{"findings":' in SYSTEM_PROMPT
    assert "no prose and no code fences" in PROMPT


# --- Rule 1: the clinical request is not a finding ---------------------------
# B6 review queue, ACL: evidence was "anterior cruciate ligament: intact ||
# acl sprain" -- the second span came from "Indication: ... ACL sprain."
@pytest.mark.parametrize(
    "section", ["indication", "clinical history", "antecedentes clinicos", "klinische inlichtingen"]
)
def test_rule_1_names_the_request_sections_to_ignore(section):
    assert section in PROMPT


def test_rule_1_names_the_findings_sections_to_use():
    for section in ("findings", "bevindingen", "hallazgos", "bulgular", "impression", "besluit"):
        assert section in PROMPT


# --- Rule 2: impression beats findings ---------------------------------------
# B6 review queue, ACL: "the anterior cruciate ligament as a construct is intact
# || low-grade partial tear of the anterior cruciate ligament".
def test_rule_2_resolves_findings_versus_impression_conflicts():
    assert "the impression wins" in PROMPT
    assert "follow the impression" in PROMPT
    assert "that is not a hedge" in PROMPT


# --- Rule 3: adjacent structures ---------------------------------------------
# B6 review queue, ACL: "acl normal || ganglion cyst adjacent to the proximal
# part of acl" and "acl normal || avulsive fracture of tibia ... at the
# attachment site of acl".
def test_rule_3_covers_both_measured_adjacency_failures():
    assert "ganglion cyst adjacent" in PROMPT
    assert "avulsive fracture" in PROMPT
    assert "attachment site" in PROMPT


# --- Rule 4: partial and bundle tears ----------------------------------------
# Training report: "complete rupture of the posterolateral bundle of ACL. The
# anteromedial bundle is structurally continuous."
def test_rule_4_treats_partial_and_bundle_tears_as_positive():
    assert "posterolateral bundle" in PROMPT
    assert "partial-thickness tear" in PROMPT
    for word in ("rotura", "scheur", "yirtik"):
        assert word in PROMPT


# --- Rule 5: the targets mean ABNORMALITY, not "tear" ------------------------
# B23 is a parser substitution, so it must not redefine the pathology. These are
# the exact cases frozen in tests/test_b6_report_labels.py; if the prompt ever
# contradicts them again, B23 would be changing the label semantics rather than
# improving extraction, and every downstream comparison to B6 would be invalid.
FROZEN_B6_POSITIVE_CASES = [
    "acl: grade 1 sprain is seen with intact fibers.",
    "mucoid degeneration of the acl without evidence of tear.",
    "myxoid degeneration of the posterior horn of the medial meniscus but no definite tear.",
]


def test_rule_5_defines_the_targets_as_abnormality_not_tear():
    assert 'abnormality, not just "tear"' in PROMPT
    assert "negating a tear does not negate the finding" in PROMPT
    for target in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"):
        assert TARGET_DEFINITIONS[target].lower().startswith("any ")


@pytest.mark.parametrize(
    "phrase",
    [
        "grade 1 sprain is seen with intact fibers",
        "mucoid degeneration of the acl without evidence of tear",
        "myxoid degeneration of the posterior horn of the medial meniscus but no definite tear",
        "grade i ligamentous sprain of the medial collateral ligament",
    ],
)
def test_rule_5_teaches_the_frozen_b6_positive_cases_as_positive(phrase):
    assert phrase in PROMPT
    # Each is shown in the prompt with an explicit "-> ... positive" verdict.
    tail = PROMPT[PROMPT.index(phrase) : PROMPT.index(phrase) + 400]
    assert "positive" in tail


def test_b23_does_not_contradict_the_frozen_b6_regression_suite():
    """Guard against silently redefining the pathology.

    The B6 suite requires each of these to be POSITIVE. The prompt must never
    tell the labeller to negate them.
    """
    for case in FROZEN_B6_POSITIVE_CASES:
        assert "negated" not in PROMPT[PROMPT.index(case.rstrip(".")) :][:200]


def test_rule_5_keeps_osteoarthritis_degeneration_positive_too():
    assert "the same principle applies" in PROMPT
    for word in ("chondromalacia", "chondrosis", "chondropathy", "osteophytes"):
        assert word in PROMPT


# --- Rule 6: silence is not absence ------------------------------------------
# This is the frozen repository policy and must never be weakened. B6's
# unmentioned bucket is 416 of 696 gold cells at 26.4% gold-positive: about 110
# expert-positive and 306 expert-negative. Mapping the bucket to negative would
# turn those ~110 positives into false negatives.
def test_rule_6_forbids_inferring_absence_from_silence():
    assert "never infer absence from silence" in PROMPT
    assert "silence is not absence" in PROMPT


def test_rule_6_still_allows_generic_normality_statements_to_negate():
    assert "generic normality statement" in PROMPT


# --- Rule 7: compartments ----------------------------------------------------
def test_rule_7_keeps_the_three_osteoarthritis_compartments_separate():
    assert "tricompartmental osteoarthritis" in PROMPT
    assert "with no compartment named" in PROMPT


# --- Rule 8: multilingual vocabulary -----------------------------------------
# The corpus contains at least English, Spanish, Dutch and Turkish. B6's
# Effusion bucket is anti-informative -- P(gold=1 | unmentioned) = 0.714 exceeds
# P(gold=1 | positive) = 0.645 -- which non-English effusion terms explain.
@pytest.mark.parametrize(
    "term",
    ["hydrops", "derrame articular", "efuzyon", "sinovitis", "bakercyste", "quiste popliteo"],
)
def test_rule_8_lists_the_non_english_terms_b6_missed(term):
    assert term in PROMPT


@pytest.mark.parametrize("negation", ["geen", "no hay", "sin", "yok", "bewaard", "conservado"])
def test_rule_8_lists_multilingual_negations(negation):
    assert negation in PROMPT


# --- Rule 9: uncertainty discipline ------------------------------------------
# All 12 sampled B6 review-queue rows were `conflicting_definite_evidence`
# collapsed to uncertain@0.2, i.e. discarded. Rule 9 exists to stop B23
# reproducing that behaviour.
def test_rule_9_forbids_treating_a_two_sentence_conflict_as_uncertainty():
    assert "a conflict between two sentences is not uncertainty" in PROMPT
    assert "genuinely unresolvable" in PROMPT


def test_user_prompt_wraps_the_report_and_lists_every_definition():
    report = "Matige hydrops. Bakercyste met multipele inliggende gewrichtsmuizen."
    prompt = build_user_prompt(report)
    assert "<report>" in prompt and "</report>" in prompt
    assert report in prompt
    for target in TARGETS:
        assert f"- {target}: {TARGET_DEFINITIONS[target]}" in prompt


def test_user_prompt_strips_surrounding_whitespace_only():
    prompt = build_user_prompt("\n\n  Diz eklemi ici sivi miktari normal.  \n\n")
    assert "<report>\nDiz eklemi ici sivi miktari normal.\n</report>" in prompt
