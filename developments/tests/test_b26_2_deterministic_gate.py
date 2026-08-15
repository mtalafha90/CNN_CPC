from __future__ import annotations

import pandas as pd
import pytest

from rsna_knee.b26_2_deterministic_gate import (
    B26_2_VERSION,
    SUPPORTED_TARGET,
    accept_negative,
    accept_positive,
    apply_b26_2_filter,
    evidence_is_verbatim,
)


def test_version_is_frozen():
    assert B26_2_VERSION == "1.0.0"


@pytest.mark.parametrize(
    ("evidence", "report"),
    [
        ("Diffuse synovitis.", "Findings: Diffuse synovitis."),
        ("Reizsynovialitis", "Beurteilung: Reizsynovialitis."),
        ("hipertrofiju sinovije", "Nalaz pokazuje hipertrofiju sinovije."),
        ("hafif snovit", "Bulgular: hafif snovit izlendi."),
        ("synovial thickening", "There is synovial thickening."),
        ("synoviale Verdikking", "Er is synoviale Verdikking."),
    ],
)
def test_positive_whitelist_accepts_direct_synovial_abnormality(evidence, report):
    accepted, reason = accept_positive(evidence, report)
    assert accepted
    assert reason == "explicit_positive_synovial_evidence"


@pytest.mark.parametrize(
    "evidence",
    [
        "Small joint effusion.",
        "Kleine synoviale Aussackung posterior des Innenmeniskus.",
        "Synovial fluid leakage.",
        "No joint effusion.",
    ],
)
def test_positive_whitelist_rejects_related_but_nonqualifying_findings(evidence):
    report = f"Findings: {evidence}"
    accepted, _reason = accept_positive(evidence, report)
    assert not accepted


def test_positive_whitelist_rejects_negated_synovitis():
    evidence = "No evidence of synovitis."
    accepted, reason = accept_positive(evidence, f"Findings: {evidence}")
    assert not accepted
    assert reason == "positive_evidence_is_negated"


@pytest.mark.parametrize(
    ("evidence", "report"),
    [
        ("Synovialis nicht verdickt", "Synovialis nicht verdickt."),
        ("keine Verdickung der Synovia", "Es zeigt sich keine Verdickung der Synovia."),
        ("No synovitis.", "Findings: No synovitis."),
        (
            "No significant knee joint effusion or synovial thickening is identified.",
            "No significant knee joint effusion or synovial thickening is identified.",
        ),
        ("Synovium is unremarkable.", "The synovium is unremarkable."),
        ("sinovit yok", "Bulgular: sinovit yok."),
    ],
)
def test_negative_whitelist_accepts_direct_target_specific_negation(evidence, report):
    accepted, reason = accept_negative(evidence, report)
    assert accepted
    assert reason == "explicit_negative_synovial_evidence"


@pytest.mark.parametrize(
    "report",
    [
        "Conclusion: Normal.",
        "Impression: Normal.",
        "Normal study.",
        "No significant abnormality identified.",
        "No significant abnormal finding in this study.",
        "Normal MR examination of the right knee.",
        "Normal MRI arthrogram of the left knee.",
        "Otherwise, normal MRI of knee.",
        "Geen afwijkingen aangetoond.",
    ],
)
def test_negative_whitelist_accepts_vetted_global_normal_conclusion(report):
    accepted, reason = accept_negative("No joint effusion.", report + " No joint effusion.")
    assert accepted
    assert reason == "global_normal_report_conclusion"


@pytest.mark.parametrize(
    "report",
    [
        "No joint effusion.",
        "Trace effusion only.",
        "Normal bone marrow.",
        "Normal menisci and ligaments.",
        "No intra-articular body.",
        "Surrounding soft tissues are normal.",
        "No intra-articular injury.",
        "The joint capsule is not thickened.",
        "No significant perimeniscal inflammation.",
    ],
)
def test_negative_whitelist_rejects_measured_spurious_negations(report):
    accepted, reason = accept_negative(report, report)
    assert not accepted
    assert reason == "negative_not_on_whitelist"


def test_evidence_must_be_verbatim_for_direct_evidence_paths():
    assert evidence_is_verbatim("Synovialis nicht verdickt", "Synovialis nicht verdickt.")
    assert not evidence_is_verbatim("Synovialis nicht verdickt", "No joint effusion.")
    accepted, reason = accept_negative("Synovialis nicht verdickt", "No joint effusion.")
    assert not accepted
    assert reason == "negative_not_on_whitelist"


def _row(**overrides):
    row = {
        "StudyInstanceUID": "1",
        "target": SUPPORTED_TARGET,
        "gate_state": "positive",
        "gate_evidence": "Diffuse synovitis.",
        "accepted_same_polarity": True,
        "polarity_flip_rejected": False,
    }
    row.update(overrides)
    return row


def test_filter_can_only_remove_b26_1_calls_not_create_or_flip():
    frame = pd.DataFrame(
        [
            _row(StudyInstanceUID="1"),
            _row(
                StudyInstanceUID="2",
                gate_state="negated",
                gate_evidence="No joint effusion.",
            ),
            _row(
                StudyInstanceUID="3",
                accepted_same_polarity=False,
            ),
            _row(
                StudyInstanceUID="4",
                gate_state="negated",
                gate_evidence="Synovium is unremarkable.",
            ),
        ]
    )
    reports = pd.Series(
        {
            "1": "Findings: Diffuse synovitis.",
            "2": "Findings: No joint effusion.",
            "3": "Findings: Diffuse synovitis.",
            "4": "Findings: Synovium is unremarkable.",
        }
    )
    out = apply_b26_2_filter(frame, reports)

    assert out["b26_2_accept"].tolist() == [True, False, False, True]
    assert out["b26_2_state"].tolist() == [
        "positive",
        "unmentioned",
        "unmentioned",
        "negated",
    ]
    assert out.loc[2, "b26_2_reason"] == "b26_1_not_accepted"


def test_filter_refuses_a_target_not_manually_reviewed():
    frame = pd.DataFrame([_row(target="ACL")])
    reports = pd.Series({"1": "ACL is torn."})
    with pytest.raises(ValueError, match="supports only"):
        apply_b26_2_filter(frame, reports)
