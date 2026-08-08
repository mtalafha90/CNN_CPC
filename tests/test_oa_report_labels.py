from __future__ import annotations

from rsna_knee.report_labels import (
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNMENTIONED,
    predict_target,
)


def test_medial_oa_detects_compartment_cartilage_loss_without_cross_talk():
    text = (
        "MEDIAL COMPARTMENT: Medial compartment cartilage: Full thickness cartilage loss "
        "along the medial femoral condyle. LATERAL COMPARTMENT: Lateral compartment "
        "cartilage appears intact."
    )
    assert predict_target(text, "Medial OA").state == STATE_POSITIVE
    assert predict_target(text, "Lateral OA").state == STATE_NEGATED


def test_lateral_oa_recognizes_explicit_normal_chondral_statement():
    text = "In the lateral compartment, there is no focal chondrosis or chondral injury."
    assert predict_target(text, "Lateral OA").state == STATE_NEGATED


def test_pf_oa_detects_patellofemoral_cartilage_disease():
    text = (
        "PATELLOFEMORAL COMPARTMENT: High-grade cartilage loss along the medial patellar "
        "facet and high-grade cartilage loss at the lateral trochlea."
    )
    assert predict_target(text, "PF OA").state == STATE_POSITIVE


def test_explicit_medial_compartment_oa_is_positive():
    assert predict_target("Conclusion: OA of medial compartment.", "Medial OA").state == STATE_POSITIVE


def test_tricompartmental_oa_maps_to_all_three_compartments():
    text = "Tricompartmental osteoarthritis with marginal osteophytes."
    for target in ("Medial OA", "Lateral OA", "PF OA"):
        assert predict_target(text, target).state == STATE_POSITIVE


def test_generic_meniscal_degeneration_does_not_create_oa_label():
    text = "Medial meniscus demonstrates degenerative signal without a surfacing tear."
    for target in ("Medial OA", "Lateral OA", "PF OA"):
        assert predict_target(text, target).state == STATE_UNMENTIONED


def test_compartment_mention_alone_remains_unmentioned():
    text = "The medial compartment is reviewed. The medial meniscus is intact."
    assert predict_target(text, "Medial OA").state == STATE_UNMENTIONED
