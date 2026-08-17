from __future__ import annotations

import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee.dataset_contract_audit import (
    audit_series_table,
    audit_train_table,
    report_script_profile,
)


def _train_frame() -> pd.DataFrame:
    rows = []
    for uid, report in [
        ("s0", "Normal knee examination."),
        ("s1", "تمزق في الرباط"),
        ("s2", "Сустав без выпота"),
        ("s3", "ACL tear"),
    ]:
        row = {"StudyInstanceUID": uid, "Report": report}
        row.update({target: float("nan") for target in TARGETS})
        rows.append(row)
    rows[0]["ACL"] = 1.0
    for j, target in enumerate(TARGETS):
        rows[1][target] = float(j % 2)
    rows[2]["Effusion"] = 0.0
    rows[2]["Fracture"] = 1.0
    return pd.DataFrame(rows)


def test_train_audit_distinguishes_any_label_from_fully_labeled():
    train = _train_frame()
    summary, per_target, histogram, scripts, labeled = audit_train_table(train)
    assert summary["training_studies"] == 4
    assert summary["repository_gold_any_label_studies"] == 3
    assert summary["fully_labeled_12_studies"] == 1
    assert summary["partially_labeled_studies"] == 2
    assert summary["zero_official_label_studies"] == 1
    assert set(labeled["StudyInstanceUID"]) == {"s0", "s1", "s2"}
    assert int(histogram.loc[histogram["official_label_count"].eq(12), "studies"].iloc[0]) == 1
    acl = per_target.loc[per_target["target"].eq("ACL")].iloc[0]
    assert int(acl["labeled_cells"]) == 2
    assert int(acl["positive_cells"]) == 1
    assert int(acl["negative_cells"]) == 1
    assert {"Latin", "Arabic", "Cyrillic"}.issubset(set(scripts["script_bucket"]))


def test_report_script_profile_is_script_not_language():
    assert report_script_profile("ACL tear")["bucket"] == "Latin"
    assert report_script_profile("تمزق")["bucket"] == "Arabic"
    assert report_script_profile("перелом")["bucket"] == "Cyrillic"
    assert report_script_profile("1234 !!!")["bucket"] == "Empty/no-letters"


def test_series_audit_counts_studies_and_metadata():
    train = _train_frame()
    series = pd.DataFrame([
        {"StudyInstanceUID": "s0", "SeriesInstanceUID": "a", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"},
        {"StudyInstanceUID": "s0", "SeriesInstanceUID": "b", "Fluid_Sensitive": 0, "Fat_Suppression": 0, "Anatomical_Plane": "Coronal"},
        {"StudyInstanceUID": "s1", "SeriesInstanceUID": "c", "Fluid_Sensitive": 1, "Fat_Suppression": 0, "Anatomical_Plane": "Axial"},
        {"StudyInstanceUID": "s2", "SeriesInstanceUID": "d", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"},
    ])
    summary, per_study, metadata = audit_series_table(series, train)
    assert summary["series_rows"] == 4
    assert summary["studies_with_series"] == 3
    assert summary["studies_without_series"] == 1
    assert int(per_study.loc[per_study["StudyInstanceUID"].eq("s0"), "series_count"].iloc[0]) == 2
    plane = metadata.loc[metadata["field"].eq("Anatomical_Plane")]
    assert int(plane.loc[plane["value"].eq("Sagittal"), "series"].iloc[0]) == 2
