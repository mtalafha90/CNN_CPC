"""B6 v1.3: an osteoarthritis vocabulary, and a guard against list negations.

The contract that matters most is the first one below: with `use_v13=False`
this module must reproduce the frozen v1.2.1 parser exactly, cell for cell,
including its reason strings and its evidence. Everything else is additive.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rsna_knee.b6_report_labels import predict_target_b6
from rsna_knee.b6_v13_report_labels import (
    B6_V13_VERSION,
    V13_PATTERNS,
    compare_versions,
    is_list_negated,
    predict_target_b6_v13,
)
from rsna_knee.report_labels import (
    OA_TARGETS,
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNMENTIONED,
)

REPORTS = (
    "Tricompartmental osteoarthritis with patella cartilage loss.",
    "ACL: intact. PCL: intact. Medial meniscus: tear.",
    "Medial meniscus tear, no displacement seen.",
    "Complete tear of the anterior cruciate ligament.",
    "Patellar tendinopathy. No joint space narrowing.",
    "Chondromalacia of the trochleas with subchondral oedema.",
    "Moderate medial compartment joint space narrowing and osteophytes.",
    "Artrosis del compartimento medial con osteofitos.",
    "Normal study. No acute abnormality.",
    "",
    "Lateral femorotibial osteoarthritis with cartilage thinning.",
    "Kissing contusions. Effusion present.",
)

TARGETS = ("PF OA", "Medial OA", "Lateral OA", "ACL", "MCL", "Medial Meniscus")


# --- the frozen path must not move --------------------------------------------


@pytest.mark.parametrize("text", REPORTS)
@pytest.mark.parametrize("target", TARGETS)
def test_v121_is_reproduced_exactly(text, target):
    """Same state, same reason, same evidence, same numbers."""
    frozen = predict_target_b6(text, target)
    replay = predict_target_b6_v13(text, target, use_v13=False)

    assert replay.state == frozen.state
    assert replay.reason == frozen.reason
    assert replay.evidence == frozen.evidence
    assert replay.probability == pytest.approx(frozen.probability)
    assert replay.confidence == pytest.approx(frozen.confidence)
    assert replay.mentioned == frozen.mentioned


def test_the_version_is_declared():
    assert B6_V13_VERSION == "1.3.0"


# --- the new vocabulary -------------------------------------------------------


def test_patella_now_places_a_patellofemoral_call():
    text = "Tricompartmental osteoarthritis with patella cartilage loss."
    assert predict_target_b6_v13(text, "PF OA").state == STATE_POSITIVE


def test_a_placed_call_carries_a_quotation_not_a_rule_name():
    """The whole point: it stops being an evidence-free fallback."""
    text = "Full thickness cartilage loss over the patella."
    result = predict_target_b6_v13(text, "PF OA")

    assert result.reason != "compartment_aware_oa_context"
    assert result.evidence != "oa_context_parser"
    assert "patella" in result.evidence


def test_the_patellar_guard_refuses_soft_tissue():
    """Patellar tendon is not patellofemoral osteoarthritis."""
    text = "Patellar tendinopathy at the inferior pole."
    assert predict_target_b6_v13(text, "PF OA").state == STATE_UNMENTIONED


def test_the_guard_covers_every_soft_tissue_word_it_claims_to():
    for word in ("tendon", "tendinosis", "enthesopathy", "bursitis", "ligament"):
        text = f"Patellar {word} noted."
        assert (
            predict_target_b6_v13(text, "PF OA").state == STATE_UNMENTIONED
        ), word


def test_the_guard_does_not_block_a_real_joint_finding():
    text = "Patellar cartilage thinning with subchondral sclerosis."
    assert predict_target_b6_v13(text, "PF OA").state == STATE_POSITIVE


def test_the_trochlear_guard_matches_the_patellar_one():
    assert predict_target_b6_v13("Trochlear tendon strain.", "PF OA").state == (
        STATE_UNMENTIONED
    )


def test_the_compartment_patterns_reach_the_compartment_targets():
    text = "Moderate medial compartment joint space narrowing and osteophytes."
    assert predict_target_b6_v13(text, "Medial OA").state == STATE_POSITIVE


def test_femorotibial_reaches_the_lateral_target():
    text = "Lateral femorotibial osteoarthritis with cartilage thinning."
    assert predict_target_b6_v13(text, "Lateral OA").state == STATE_POSITIVE


def test_the_new_patterns_only_touch_the_three_oa_targets():
    assert set(V13_PATTERNS) == set(OA_TARGETS)


def test_a_non_oa_target_is_untouched_by_v13():
    text = "Complete tear of the anterior cruciate ligament."
    assert (
        predict_target_b6_v13(text, "ACL").state
        == predict_target_b6_v13(text, "ACL", use_v13=False).state
    )


# --- the list-negation guard --------------------------------------------------


def test_a_list_entry_calling_the_target_intact_is_a_negation():
    assert is_list_negated("acl: intact. pcl: intact.", stop=3)


def test_a_dash_list_is_recognised_too():
    assert is_list_negated("acl - normal", stop=3)


def test_a_negation_that_follows_other_words_does_not_count():
    """"tear, no displacement" negates the displacement, not the tear."""
    assert not is_list_negated("medial meniscus tear, no displacement", stop=15)


def test_the_guard_flips_a_list_negated_positive():
    text = "Findings: ACL: intact. Medial meniscus: tear."
    guarded = predict_target_b6_v13(text, "ACL")
    assert guarded.state == STATE_NEGATED


def test_the_guard_leaves_a_genuine_positive_alone():
    text = "Medial meniscus tear, no displacement seen."
    assert predict_target_b6_v13(text, "Medial Meniscus").state == STATE_POSITIVE


def test_the_guard_is_off_when_v13_is_off():
    """It is part of v1.3, not a silent change to the frozen parser."""
    text = "Findings: ACL: intact. Medial meniscus: tear."
    assert (
        predict_target_b6_v13(text, "ACL", use_v13=False).state
        == predict_target_b6(text, "ACL").state
    )


# --- edges --------------------------------------------------------------------


def test_an_empty_report_is_unmentioned():
    result = predict_target_b6_v13("", "PF OA")
    assert result.state == STATE_UNMENTIONED
    assert result.reason == "empty_report"


def test_the_fallback_still_runs_where_the_vocabulary_finds_nothing():
    """v1.3 pre-empts the fallback; it does not delete it.

    Spanish word order puts the compartment after the noun, so none of the
    v1.3 patterns match here and the legacy context rule still has to answer.
    """
    text = "Artrosis del compartimento medial con osteofitos."
    result = predict_target_b6_v13(text, "Medial OA")

    assert result.state != STATE_UNMENTIONED
    assert result.reason == "compartment_aware_oa_context"


def test_a_disease_with_no_compartment_is_declined_by_both_versions():
    """"gonartrosis" names no compartment, so neither version may guess one."""
    text = "Gonartrosis avanzada."
    assert predict_target_b6_v13(text, "Medial OA").state == STATE_UNMENTIONED
    assert predict_target_b6(text, "Medial OA").state == STATE_UNMENTIONED


def test_v13_never_removes_a_call_v121_made():
    """It may change a state, but it may not silence an answered cell."""
    for text in REPORTS:
        for target in TARGETS:
            old = predict_target_b6_v13(text, target, use_v13=False)
            new = predict_target_b6_v13(text, target, use_v13=True)
            if old.mentioned:
                assert new.mentioned, (text, target)


# --- the diff -----------------------------------------------------------------


def test_the_comparison_reports_what_changed():
    frame = compare_versions(pd.Series(list(REPORTS)), "PF OA")

    assert len(frame) == len(REPORTS)
    assert set(["old_state", "new_state", "changed", "newly_placed", "now_quoted"]).issubset(
        frame.columns
    )
    assert bool(frame["newly_placed"].any())


def test_the_comparison_marks_a_fallback_becoming_a_quotation():
    frame = compare_versions(
        pd.Series(["Tricompartmental osteoarthritis with patella cartilage loss."]),
        "PF OA",
    )
    assert bool(frame.loc[0, "now_quoted"]) or not bool(frame.loc[0, "changed"])


# --- the export ---------------------------------------------------------------


def _train_csv(tmp_path):
    from rsna_knee.constants import TARGETS as ALL_TARGETS

    rows = []
    for index, text in enumerate(REPORTS):
        row = {"StudyInstanceUID": f"study{index:03d}", "Report": text}
        for target in ALL_TARGETS:
            # One gold study, so the export has both populations to split.
            row[target] = 1 if index == 0 else None
        rows.append(row)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_the_export_frame_matches_the_frozen_one_in_shape(tmp_path):
    from rsna_knee.b6_report_labels import build_b6_frame
    from rsna_knee.b6_v13_report_labels import build_b6_v13_frame
    from rsna_knee.data import load_train_csv

    df = load_train_csv(_train_csv(tmp_path))
    frozen = build_b6_frame(df)
    updated = build_b6_v13_frame(df)

    assert list(updated.columns) == list(frozen.columns)
    assert len(updated) == len(frozen)


def test_the_export_frame_reproduces_v121_when_v13_is_off(tmp_path):
    from rsna_knee.b6_report_labels import build_b6_frame
    from rsna_knee.b6_v13_report_labels import build_b6_v13_frame
    from rsna_knee.data import load_train_csv

    df = load_train_csv(_train_csv(tmp_path))
    pd.testing.assert_frame_equal(build_b6_v13_frame(df, use_v13=False), build_b6_frame(df))


def test_the_change_summary_counts_what_moved(tmp_path):
    from rsna_knee.b6_v13_report_labels import build_b6_v13_frame, change_summary
    from rsna_knee.data import load_train_csv

    df = load_train_csv(_train_csv(tmp_path))
    summary = change_summary(
        build_b6_v13_frame(df, use_v13=False), build_b6_v13_frame(df, use_v13=True)
    )

    from rsna_knee.constants import TARGETS as ALL_TARGETS

    assert summary["totals"]["cells_newly_answered"] > 0
    assert summary["totals"]["cells_silenced"] == 0
    assert set(summary["targets"]) == set(ALL_TARGETS)


def test_the_export_writes_the_files_the_merge_needs(tmp_path):
    import json as _json

    from rsna_knee.b6_v13_report_labels import run_b6_v13_export

    out = tmp_path / "export"
    audit = run_b6_v13_export(_train_csv(tmp_path), out_root=out)

    for name in (
        "structured_labels.csv",
        "training_targets.csv",
        "review_queue.csv",
        "audit.json",
        "v13_changes.json",
    ):
        assert (out / name).is_file(), name

    assert audit["b6_version"] == "1.3.0"
    assert audit["supersedes"] == "1.2.1"
    assert _json.loads((out / "v13_changes.json").read_text())["version"] == "1.3.0"


def test_the_export_keeps_gold_out_of_the_training_targets(tmp_path):
    from rsna_knee.b6_v13_report_labels import run_b6_v13_export

    out = tmp_path / "export"
    run_b6_v13_export(_train_csv(tmp_path), out_root=out)
    training = pd.read_csv(out / "training_targets.csv")

    assert "study000" not in set(training["StudyInstanceUID"].astype(str))
    assert len(training) == len(REPORTS) - 1


def test_a_bad_confidence_is_refused(tmp_path):
    from rsna_knee.b6_v13_report_labels import run_b6_v13_export

    with pytest.raises(ValueError, match="min_confidence"):
        run_b6_v13_export(_train_csv(tmp_path), out_root=tmp_path / "x", min_confidence=2.0)
