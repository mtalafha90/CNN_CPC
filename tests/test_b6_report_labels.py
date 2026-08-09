from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsna_knee.b6_report_labels import predict_target_b6, run_b6_export
from rsna_knee.constants import TARGETS
from rsna_knee.report_labels import (
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
)


def test_b6_acl_states_handle_positive_negative_uncertain_and_unmentioned():
    positive = predict_target_b6("There is a complete tear of the ACL.", "ACL")
    negative = predict_target_b6("The ACL is intact and continuous.", "ACL")
    uncertain = predict_target_b6("Possible partial tear of the ACL.", "ACL")
    absent = predict_target_b6("Small joint effusion. No other finding.", "ACL")

    assert positive.state == STATE_POSITIVE
    assert positive.probability > 0.9
    assert positive.confidence >= 0.8

    assert negative.state == STATE_NEGATED
    assert negative.probability < 0.1
    assert negative.confidence >= 0.8

    assert uncertain.state == STATE_UNCERTAIN
    assert uncertain.confidence < positive.confidence

    assert absent.state == STATE_UNMENTIONED
    assert absent.confidence == 0.0


def test_b6_multilingual_aliases_are_accent_insensitive():
    french = predict_target_b6("Rupture du ligament croisé antérieur.", "ACL")
    italian = predict_target_b6("Versamento articolare moderato.", "Effusion")
    german = predict_target_b6("Vorderes Kreuzband intakt.", "ACL")
    turkish = predict_target_b6("Ön çapraz bağ yırtığı izlenmektedir.", "ACL")

    assert french.state == STATE_POSITIVE
    assert italian.state == STATE_POSITIVE
    assert german.state == STATE_NEGATED
    assert turkish.state == STATE_POSITIVE


def test_b6_target_local_scope_ignores_neighboring_intact_structures():
    pred = predict_target_b6(
        "Quadriceps tendon intact. Patellar tendon intact. Intercondylar compartment: ACL: high grade tear.",
        "ACL",
    )
    assert pred.state == STATE_POSITIVE
    assert pred.reason == "explicit_structural_abnormality"


def test_b6_pathology_can_coexist_with_intact_fibers():
    pred = predict_target_b6("ACL: grade 1 sprain is seen with intact fibers.", "ACL")
    assert pred.state == STATE_POSITIVE


def test_b6_uncertain_duplicate_does_not_cancel_definite_positive():
    pred = predict_target_b6(
        "There is a complete tear of the ACL. Impression: possible ACL tear.",
        "ACL",
    )
    assert pred.state == STATE_POSITIVE


def test_b6_uncertain_indication_does_not_cancel_definite_negative():
    pred = predict_target_b6(
        "Clinical indication: assess for ACL tear. Findings: ACL is intact and continuous.",
        "ACL",
    )
    assert pred.state == STATE_NEGATED


def test_b6_detects_genuinely_opposing_definite_evidence():
    pred = predict_target_b6("The ACL is intact. There is also a complete ACL tear.", "ACL")
    assert pred.state == STATE_UNCERTAIN
    assert pred.reason == "conflicting_definite_evidence"
    assert pred.confidence <= 0.25


def test_b6_export_excludes_gold_from_training_targets(tmp_path):
    rows = []
    reports = [
        "Complete ACL tear with joint effusion.",
        "ACL intact. No joint effusion.",
        "Possible medial meniscus tear.",
        "Fracture of the tibial plateau.",
    ]
    for i, report in enumerate(reports):
        row = {"StudyInstanceUID": f"study-{i}", "Report": report}
        for target in TARGETS:
            row[target] = np.nan
        if i == 0:
            for target in TARGETS:
                row[target] = float(target == "ACL")
        rows.append(row)

    train_csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    out = tmp_path / "b6"
    payload = run_b6_export(train_csv, out_root=out, max_review=100)

    assert payload["n_studies"] == 4
    assert payload["n_gold_audit_only"] == 1
    assert payload["n_report_only_training"] == 3
    assert payload["gold_rows_in_training_targets"] == 0
    assert payload["external_models"] is False
    assert payload["gold_fitted_calibration"] is False

    structured = pd.read_csv(out / "structured_labels.csv")
    training = pd.read_csv(out / "training_targets.csv")
    assert len(structured) == 4
    assert len(training) == 3
    assert "study-0" not in set(training["StudyInstanceUID"])

    policy = json.loads((out / "policy.json").read_text())
    assert policy["version"] == "1.1"
    assert policy["unmentioned_is_negative"] is False
    assert policy["gold_usage"].startswith("audit only")
