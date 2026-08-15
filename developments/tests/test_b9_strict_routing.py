import pandas as pd

from rsna_knee.strict_routing import (
    STRICT_ROUTING_POLICY,
    build_strict_series_index,
    routing_audit,
)


def _frame():
    return pd.DataFrame(
        [
            # Two sagittal structural series: historical dual routing invents
            # a sagittal_fluid assignment; B9 must leave that slot missing.
            ("study", "sag_struct_1", False, False, "Sagittal"),
            ("study", "sag_struct_2", False, False, "Sagittal"),
            # Two coronal fluid series: historical dual routing invents a
            # coronal_structural assignment; B9 must leave it missing.
            ("study", "cor_fluid_1", True, True, "Coronal"),
            ("study", "cor_fluid_2", True, True, "Coronal"),
            # Unknown contrast is not allowed to masquerade as structural.
            ("study", "ax_unknown", pd.NA, pd.NA, "Axial"),
        ],
        columns=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ],
    ).astype({"Fluid_Sensitive": "boolean", "Fat_Suppression": "boolean"})


def test_strict_routing_never_crosses_semantic_slots():
    index = build_strict_series_index(_frame(), ["study"])["study"]

    assert index["sagittal_fluid"] is None
    assert index["sagittal_structural"] in {"sag_struct_1", "sag_struct_2"}
    assert index["coronal_fluid"] in {"cor_fluid_1", "cor_fluid_2"}
    assert index["coronal_structural"] is None
    assert index["axial_fluid"] is None
    assert index["axial_structural"] is None


def test_routing_audit_exposes_historical_cross_contrast_substitution():
    audit = routing_audit(_frame(), ["study"])

    assert audit["routing_policy"] == STRICT_ROUTING_POLICY
    assert audit["strict_semantic_mismatches"] == 0
    assert audit["legacy_semantic_mismatches"] == 3
    assert audit["removed_cross_contrast_substitutions"] == 3
    assert audit["per_stream"]["sagittal_fluid"]["legacy_semantic_mismatch"] == 1
    assert audit["per_stream"]["coronal_structural"]["legacy_semantic_mismatch"] == 1
    assert audit["per_stream"]["axial_structural"]["legacy_semantic_mismatch"] == 1


def test_strict_selection_is_deterministic_with_multiple_valid_candidates():
    frame = pd.DataFrame(
        [
            ("s", "f1", True, True, "Sagittal"),
            ("s", "f2", True, False, "Sagittal"),
            ("s", "q1", False, False, "Sagittal"),
            ("s", "q2", False, True, "Sagittal"),
        ],
        columns=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ],
    )
    first = build_strict_series_index(frame, ["s"])
    second = build_strict_series_index(frame, ["s"])
    assert first == second
    assert first["s"]["sagittal_fluid"] == "f1"
    assert first["s"]["sagittal_structural"] == "q1"
