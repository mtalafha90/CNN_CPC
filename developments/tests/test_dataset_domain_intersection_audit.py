from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee.dataset_domain_intersection_audit import (
    build_study_domain_table,
    manufacturer_family,
    script_b6_crosstab,
    summarize_cohorts,
)


def _train_fixture() -> pd.DataFrame:
    rows = []
    reports = [
        "Normal knee report",
        "Ελληνική αναφορά γόνατος",
        "Кириллический отчёт колена",
    ]
    for i, report in enumerate(reports):
        row = {"StudyInstanceUID": f"S{i}", "Report": report}
        for target in TARGETS:
            row[target] = np.nan
        rows.append(row)
    for target in TARGETS:
        rows[0][target] = 1 if target == "ACL" else 0
    return pd.DataFrame(rows)


def _header_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "StudyInstanceUID": "S0", "SeriesInstanceUID": "A", "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": True, "Fat_Suppression": True, "dicom_files": 120,
                "manufacturer": "Siemens Healthineers", "manufacturer_model": "ModelA",
                "magnetic_field_strength_t": 3.0, "mr_acquisition_type": "3D",
            },
            {
                "StudyInstanceUID": "S0", "SeriesInstanceUID": "B", "Anatomical_Plane": "Coronal",
                "Fluid_Sensitive": False, "Fat_Suppression": False, "dicom_files": 30,
                "manufacturer": "Siemens Healthineers", "manufacturer_model": "ModelA",
                "magnetic_field_strength_t": 3.0, "mr_acquisition_type": "2D",
            },
            {
                "StudyInstanceUID": "S1", "SeriesInstanceUID": "C", "Anatomical_Plane": "Axial",
                "Fluid_Sensitive": True, "Fat_Suppression": True, "dicom_files": 28,
                "manufacturer": "GE MEDICAL SYSTEMS", "manufacturer_model": "ModelB",
                "magnetic_field_strength_t": 1.5, "mr_acquisition_type": "2D",
            },
            {
                "StudyInstanceUID": "S2", "SeriesInstanceUID": "D", "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": False, "Fat_Suppression": False, "dicom_files": 220,
                "manufacturer": "Philips Healthcare", "manufacturer_model": "ModelC",
                "magnetic_field_strength_t": 1.5, "mr_acquisition_type": "3D",
            },
        ]
    )


def test_manufacturer_family_aliases():
    assert manufacturer_family("SIEMENS") == "Siemens"
    assert manufacturer_family("Siemens Healthineers") == "Siemens"
    assert manufacturer_family("GE MEDICAL SYSTEMS") == "GE"
    assert manufacturer_family("GEHC") == "GE"
    assert manufacturer_family("Philips Healthcare") == "Philips"
    assert manufacturer_family("TOSHIBA") == "Canon/Toshiba"
    assert manufacturer_family("CANON_MEC") == "Canon/Toshiba"
    assert manufacturer_family("Hitachi Medical Corporation") == "Fujifilm/Hitachi"


def test_study_domain_and_cohort_contract():
    study, header = build_study_domain_table(
        _train_fixture(), _header_fixture(), b6_active_uids=["S1"]
    )
    assert len(study) == 3
    s0 = study.set_index("StudyInstanceUID").loc["S0"]
    s1 = study.set_index("StudyInstanceUID").loc["S1"]
    s2 = study.set_index("StudyInstanceUID").loc["S2"]

    assert bool(s0["repository_gold"])
    assert s0["b6_status"] == "gold_not_in_b6"
    assert bool(s0["any_3d"]) and bool(s0["any_gt100"])
    assert s1["b6_status"] == "active"
    assert s1["report_script_bucket"] == "Greek"
    assert not bool(s1["any_3d"])
    assert s2["b6_status"] == "inactive"
    assert s2["report_script_bucket"] == "Cyrillic"
    assert bool(s2["any_gt200"])

    cohorts = summarize_cohorts(study).set_index("cohort")
    assert int(cohorts.loc["gold", "studies"]) == 1
    assert int(cohorts.loc["report_only_b6_active", "studies"]) == 1
    assert int(cohorts.loc["report_only_b6_inactive", "studies"]) == 1
    assert float(cohorts.loc["report_only_b6_active", "fraction_studies_any_3d"]) == 0.0
    assert float(cohorts.loc["report_only_b6_inactive", "fraction_studies_any_gt200"]) == 1.0

    cross = script_b6_crosstab(study)
    assert set(cross["report_script_bucket"]) == {"Greek", "Cyrillic"}
    assert set(header["manufacturer_family"]) == {"Siemens", "GE", "Philips"}
