"""Finding the compartment words B6's OA patterns cannot match.

Whatever vocabulary this suggests will end up deciding thousands of labels, so
the counting has to be trustworthy in one specific way: a mention counts as
unplaced only when no compartment pattern reaches it. Counting a placed mention
as unplaced would invent a gap that is not there.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.lexicon_gap_scan import (
    candidate_vocabulary,
    pattern_coverage,
    scan,
    unplaced_mentions,
)


def _reports(mapping):
    return pd.Series(mapping)


# --- placed and unplaced ------------------------------------------------------


def test_a_disease_term_beside_a_known_compartment_is_placed():
    frame, summary = unplaced_mentions(
        _reports({"a": "osteoarthritis of the medial compartment"})
    )
    assert summary["placed_in_a_compartment"] == 1
    assert summary["unplaced"] == 0
    assert frame.empty


def test_a_disease_term_with_no_compartment_is_unplaced():
    frame, summary = unplaced_mentions(_reports({"a": "moderate osteoarthritis"}))
    assert summary["unplaced"] == 1
    assert frame["StudyInstanceUID"].tolist() == ["a"]


def test_the_spanish_trochlea_is_not_matched_by_the_english_pattern():
    """The gap this whole scan exists for: trochlea does not match troclea."""
    spanish = unplaced_mentions(
        _reports({"a": "artrosis focal grado 4 de la troclea femoral"})
    )[1]
    english = unplaced_mentions(
        _reports({"a": "focal grade 4 chondropathy of the femoral trochlea"})
    )[1]

    assert spanish["unplaced"] == 1
    assert english["placed_in_a_compartment"] == 1


def test_a_report_with_no_disease_term_contributes_nothing():
    frame, summary = unplaced_mentions(_reports({"a": "the medial compartment"}))
    assert summary["disease_mentions"] == 0
    assert frame.empty


def test_an_empty_report_is_skipped():
    assert unplaced_mentions(_reports({"a": ""}))[1]["disease_mentions"] == 0


def test_a_compartment_beyond_the_window_does_not_place_the_mention():
    """Otherwise a compartment named three sentences away would count."""
    far = "osteoarthritis" + (" filler" * 60) + " medial compartment"
    assert unplaced_mentions(_reports({"a": far}))[1]["unplaced"] == 1


def test_a_wider_window_can_reach_it():
    far = "osteoarthritis" + (" filler" * 60) + " medial compartment"
    assert unplaced_mentions(_reports({"a": far}), window=600)[1]["placed_in_a_compartment"] == 1


def test_several_mentions_in_one_report_are_counted_separately():
    frame, summary = unplaced_mentions(
        _reports({"a": "osteoarthritis here. " + ("x " * 80) + "chondromalacia there."})
    )
    assert summary["disease_mentions"] == 2
    assert summary["studies_with_an_unplaced_mention"] == 1
    assert len(frame) == 2


# --- the candidate words ------------------------------------------------------


def test_a_word_beside_an_unplaced_mention_becomes_a_candidate():
    frame, _ = unplaced_mentions(
        _reports({"a": "gonartrosis de la troclea femoral", "b": "artrosis troclea"})
    )
    words = {item["word"]: item["studies"] for item in candidate_vocabulary(frame)}

    assert words.get("troclea") == 2


def test_a_word_is_counted_once_per_study_not_once_per_mention():
    """One verbose report must not invent a candidate on its own."""
    frame, _ = unplaced_mentions(
        _reports({"a": "artrosis troclea. " + ("y " * 80) + "condropatia troclea."})
    )
    words = {item["word"]: item["studies"] for item in candidate_vocabulary(frame)}

    assert words.get("troclea") == 1


def test_common_filler_words_are_not_offered_as_anatomy():
    frame, _ = unplaced_mentions(_reports({"a": "artrosis de la troclea"}))
    words = {item["word"] for item in candidate_vocabulary(frame)}

    assert "troclea" in words
    assert "de" not in words and "la" not in words


def test_short_words_are_not_offered():
    frame, _ = unplaced_mentions(_reports({"a": "artrosis abc troclea"}))
    assert "abc" not in {item["word"] for item in candidate_vocabulary(frame)}


def test_the_list_is_capped():
    frame, _ = unplaced_mentions(
        _reports({str(i): f"osteoarthritis wordaaa{chr(97 + i % 26)}zz" for i in range(50)})
    )
    assert len(candidate_vocabulary(frame, top=5)) == 5


def test_no_unplaced_mentions_gives_no_candidates():
    frame, _ = unplaced_mentions(_reports({"a": "medial compartment osteoarthritis"}))
    assert candidate_vocabulary(frame) == []


# --- which patterns are doing any work ----------------------------------------


def test_a_pattern_that_matches_is_counted():
    coverage = pattern_coverage(_reports({"a": "the medial compartment is fine"}))
    assert coverage["Medial OA"][r"\bmedial compartment\b"] == 1


def test_a_pattern_that_never_matches_is_zero():
    coverage = pattern_coverage(_reports({"a": "nothing relevant here"}))
    assert all(count == 0 for count in coverage["PF OA"].values())


def test_every_oa_target_is_covered():
    coverage = pattern_coverage(_reports({"a": "text"}))
    assert set(coverage) == {"Medial OA", "Lateral OA", "PF OA"}


# --- the whole thing ----------------------------------------------------------


def test_the_scan_reads_train_csv_and_writes_its_findings(tmp_path):
    from rsna_knee.constants import TARGETS

    frame = {
        "StudyInstanceUID": ["a", "b"],
        "Report": ["gonartrosis de la troclea femoral", "medial compartment osteoarthritis"],
    }
    for target in TARGETS:
        frame[target] = [None, None]
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame(frame).to_csv(root / "train.csv", index=False)
    out = tmp_path / "scan"

    result = scan(data_root=root, out_root=out)

    assert result["studies"] == 2
    assert result["unplaced"] == 1
    assert result["placed_in_a_compartment"] == 1
    assert json.loads((out / "summary.json").read_text())["unplaced"] == 1
    assert len(pd.read_csv(out / "unplaced_mentions.csv")) == 1


def test_the_scan_writes_nothing_without_an_out_root(tmp_path):
    from rsna_knee.constants import TARGETS

    frame = {"StudyInstanceUID": ["a"], "Report": ["osteoarthritis"]}
    for target in TARGETS:
        frame[target] = [None]
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame(frame).to_csv(root / "train.csv", index=False)

    assert "out_root" not in scan(data_root=root)
    assert not (tmp_path / "scan").exists()


def test_a_missing_train_csv_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan(data_root=tmp_path)


# --- what B6 v1.2.1 currently cannot read -------------------------------------
#
# Spanish, Italian and Portuguese drop the `h` from Greek-derived medical words
# and turn `ph` into `f`: chondral -> condral, trochlea -> troclea, osteophyte ->
# osteofito. B6's patterns were written from English and Germanic spellings, so
# every one of those forms falls through.
#
# These record the state of the frozen parser, not a wish. A B6 v1.3 that closes
# the gap will fail them, and that failure is the point: it makes the change
# visible rather than silent.


@pytest.mark.parametrize(
    "english, romance",
    [
        ("chondromalacia", "condromalacia"),
        ("chondropathy", "condropatia"),
        ("chondral defect", "lesion condral"),
        ("osteophytes", "osteofitos"),
        ("cartilage loss", "perdida de cartilago"),
    ],
)
def test_the_disease_vocabulary_reads_english_but_not_its_romance_form(english, romance):
    from rsna_knee.data import normalize_report
    from rsna_knee.report_labels import OA_DISEASE_RE

    assert OA_DISEASE_RE.search(normalize_report(english))
    assert not OA_DISEASE_RE.search(normalize_report(romance))


@pytest.mark.parametrize(
    "english, romance",
    [
        ("trochlea", "troclea"),
        ("patellar cartilage", "cartilago rotuliano"),
        ("medial femoral condyle", "condilo femoral medial"),
        ("medial tibial plateau", "platillo tibial medial"),
    ],
)
def test_the_compartment_vocabulary_reads_english_but_not_its_romance_form(english, romance):
    from rsna_knee.data import normalize_report
    from rsna_knee.report_labels import OA_CONTEXT_REGEX

    def placed(text):
        norm = normalize_report(text)
        return any(r.search(norm) for rs in OA_CONTEXT_REGEX.values() for r in rs)

    assert placed(english)
    assert not placed(romance)


def test_arthrosis_is_the_one_disease_family_spanish_already_reaches():
    """Which is why B6 answers OA at all on Spanish reports, just not accurately."""
    from rsna_knee.data import normalize_report
    from rsna_knee.report_labels import OA_DISEASE_RE

    for term in ("artrosis", "gonartrosis", "artrose"):
        assert OA_DISEASE_RE.search(normalize_report(term)), term


# --- sizing a proposed v1.3 ---------------------------------------------------
#
# "places" and "widens" are not the same risk. Places reaches a mention nothing
# could reach, which is a repair. Widens adds a compartment to a mention already
# placed, which changes a label the model has trained on. Confusing them would
# make a risky change look like a safe one.


def test_a_pattern_that_reaches_an_unplaced_mention_places_it():
    from rsna_knee.lexicon_gap_scan import simulate

    result = simulate(
        _reports({"a": "chondromalacia patella"}), {"PF OA": (r"\bpatella\b",)}
    )
    row = result["patterns"][0]

    assert row["newly_places_studies"] == 1
    assert row["widens_studies"] == 0
    assert result["studies_newly_placed"] == 1


def test_a_pattern_that_adds_a_target_to_an_already_placed_mention_widens_it():
    from rsna_knee.lexicon_gap_scan import simulate

    # The mention is already placed in Medial OA; the pattern would add PF OA.
    result = simulate(
        _reports({"a": "medial compartment osteoarthritis and patella"}),
        {"PF OA": (r"\bpatella\b",)},
    )
    row = result["patterns"][0]

    assert row["newly_places_studies"] == 0
    assert row["widens_studies"] == 1


def test_a_pattern_matching_a_target_already_placed_does_neither():
    from rsna_knee.lexicon_gap_scan import simulate

    result = simulate(
        _reports({"a": "medial compartment osteoarthritis"}),
        {"Medial OA": (r"\bmedial compartments?\b",)},
    )
    row = result["patterns"][0]

    assert row["newly_places_studies"] == 0
    assert row["widens_studies"] == 0


def test_a_pattern_matching_nothing_scores_zero():
    from rsna_knee.lexicon_gap_scan import simulate

    result = simulate(_reports({"a": "osteoarthritis"}), {"PF OA": (r"\bzebra\b",)})
    assert result["patterns"][0]["newly_places_studies"] == 0


def test_the_conjoined_compartment_pattern_catches_what_the_frozen_one_misses():
    """Today this sentence gives Lateral positive and Medial silence."""
    from rsna_knee.lexicon_gap_scan import simulate

    text = "osteoarthritis of the medial and lateral compartments"
    assert unplaced_mentions(_reports({"a": text}))[1]["unplaced"] == 1

    result = simulate(
        _reports({"a": text}),
        {"Medial OA": (r"\bmedial(?: and lateral)? compartments?\b",)},
    )
    assert result["patterns"][0]["newly_places_studies"] == 1


def test_the_default_candidates_are_used_when_none_are_given():
    from rsna_knee.lexicon_gap_scan import CANDIDATE_PATTERNS, simulate

    result = simulate(_reports({"a": "chondromalacia patella"}))
    assert len(result["patterns"]) == sum(len(v) for v in CANDIDATE_PATTERNS.values())


def test_studies_are_counted_once_however_many_mentions_they_carry():
    from rsna_knee.lexicon_gap_scan import simulate

    result = simulate(
        _reports({"a": "chondromalacia patella. " + ("z " * 80) + "chondrosis patella."}),
        {"PF OA": (r"\bpatella\b",)},
    )
    assert result["patterns"][0]["newly_places_studies"] == 1


# --- reading the windows a pattern would affect -------------------------------


def test_examples_separate_places_from_widens():
    from rsna_knee.lexicon_gap_scan import pattern_examples

    frame = pattern_examples(
        _reports(
            {
                "a": "chondromalacia patella",
                "b": "medial compartment osteoarthritis with patellar chondrosis",
            }
        ),
        "PF OA",
        r"\bpatell\w*\b",
    )
    by_uid = dict(zip(frame["StudyInstanceUID"], frame["effect"]))

    assert by_uid["a"] == "places"
    assert by_uid["b"] == "widens"


def test_a_widen_names_what_was_already_placed():
    from rsna_knee.lexicon_gap_scan import pattern_examples

    frame = pattern_examples(
        _reports({"b": "medial compartment osteoarthritis with patellar chondrosis"}),
        "PF OA",
        r"\bpatell\w*\b",
    )
    assert frame["already"].iloc[0] == "Medial OA"


def test_a_window_where_the_target_is_already_placed_is_not_shown():
    """It changes nothing there, so it is not evidence for or against."""
    from rsna_knee.lexicon_gap_scan import pattern_examples

    frame = pattern_examples(
        _reports({"a": "patellofemoral chondrosis and patella"}), "PF OA", r"\bpatella\b"
    )
    assert frame.empty


def test_the_tendon_case_is_visible_which_is_the_point():
    """A count cannot tell a cartilage patella from a tendon one; the text can."""
    from rsna_knee.lexicon_gap_scan import pattern_examples

    frame = pattern_examples(
        _reports({"a": "medial compartment osteoarthritis, patellar tendon thickening"}),
        "PF OA",
        r"\bpatellar\b",
    )
    assert frame["effect"].iloc[0] == "widens"
    assert "tendon" in frame["window"].iloc[0]


def test_the_limit_is_respected():
    from rsna_knee.lexicon_gap_scan import pattern_examples

    frame = pattern_examples(
        _reports({str(i): "chondromalacia patella" for i in range(30)}),
        "PF OA",
        r"\bpatella\b",
        limit=4,
    )
    assert len(frame) == 4


# --- the guard, derived by reading thirty windows -----------------------------
#
# Every wrong placement in that reading was "patellar" attached to something
# that is not cartilage. These twelve cases are the reading itself, written down
# so the guard cannot drift away from what it was justified by.


@pytest.mark.parametrize(
    "text",
    [
        "grade 2 patellar chondromalacia",
        "patellar apex",
        "patellar kikirdakta kondromalazi",
        "patellar ridge",
        "medial and lateral patellar facets",
        "patellar articular cartilage degeneration",
        "severe chondromalacia patella at the patellar apex",
    ],
)
def test_the_guard_keeps_a_cartilage_patellar(text):
    import re

    from rsna_knee.lexicon_gap_scan import NOT_CARTILAGE

    assert re.search(r"\bpatellar\b" + NOT_CARTILAGE, text, re.I)


@pytest.mark.parametrize(
    "text",
    [
        "patellar bursitis",
        "patellar plicae",
        "quadriceps and patellar tendons",
        "mild patellar tendinosis",
        "patellar enthesopathy",
        "patellar ligament",
        "patellar retinaculum",
    ],
)
def test_the_guard_rejects_a_patellar_that_is_not_cartilage(text):
    import re

    from rsna_knee.lexicon_gap_scan import NOT_CARTILAGE

    assert not re.search(r"\bpatellar\b" + NOT_CARTILAGE, text, re.I)


def test_prepatellar_was_never_a_match_to_begin_with():
    """The word boundary already excludes it; recorded so nobody adds a guard for it."""
    import re

    from rsna_knee.lexicon_gap_scan import NOT_CARTILAGE

    assert not re.search(r"\bpatellar\b" + NOT_CARTILAGE, "prepatellar region", re.I)


def test_the_guard_survives_a_tendon_elsewhere_in_the_sentence():
    """Several correct placements name a patellar tendon in the same sentence."""
    import re

    from rsna_knee.lexicon_gap_scan import NOT_CARTILAGE

    text = "moderate chondromalacia patella. mild patellar tendinosis with edema."
    assert re.search(r"\bpatella\b", text, re.I)
    # The adjective here really is the tendon, and the bone carries the finding.
    assert not re.search(r"\bpatellar\b" + NOT_CARTILAGE, text, re.I)


def test_the_plural_trochleas_is_now_reachable():
    """Reports write "medial and lateral trochleas"; \\btrochlea\\b cannot match it."""
    import re

    from rsna_knee.lexicon_gap_scan import CANDIDATE_PATTERNS
    from rsna_knee.report_labels import OA_CONTEXT_REGEX

    text = "chondrosis along medial and lateral trochleas"
    assert not any(r.search(text) for r in OA_CONTEXT_REGEX["PF OA"])
    assert any(
        re.search(p, text, re.I) for p in CANDIDATE_PATTERNS["PF OA"]
    )
