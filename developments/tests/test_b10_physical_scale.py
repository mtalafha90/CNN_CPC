import numpy as np
import pandas as pd

from rsna_knee.physical_scale import (
    B10_PHYSICAL_POLICY,
    derive_policy_from_geometry,
    physical_policy_digest,
    resample_volume_inplane,
    selected_series_signature,
    validate_physical_scale_policy,
)


def _geometry_frame():
    rows = []
    for plane, stream in [
        ("Sagittal", "sagittal_fluid"),
        ("Coronal", "coronal_fluid"),
        ("Axial", "axial_fluid"),
    ]:
        rows.extend(
            [
                {
                    "StudyInstanceUID": "s1",
                    "stream": stream,
                    "plane": plane,
                    "SeriesInstanceUID": f"{plane}-1",
                    "row_spacing_mm": 0.20,
                    "col_spacing_mm": 0.30,
                    "row_fov_mm": 160.0,
                    "col_fov_mm": 180.0,
                },
                {
                    "StudyInstanceUID": "s2",
                    "stream": stream,
                    "plane": plane,
                    "SeriesInstanceUID": f"{plane}-2",
                    "row_spacing_mm": 0.40,
                    "col_spacing_mm": 0.50,
                    "row_fov_mm": 200.0,
                    "col_fov_mm": 220.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_b10_policy_uses_plane_medians_and_is_label_free():
    frame = _geometry_frame()
    policy = derive_policy_from_geometry(
        frame,
        source_study_count=2,
        selected_series_signature_value="abc",
        min_geometry_coverage=0.95,
    )
    assert policy["policy_name"] == B10_PHYSICAL_POLICY
    assert policy["uses_gold_labels"] is False
    assert policy["routing_mode"] == "dual"
    for plane in ("Sagittal", "Coronal", "Axial"):
        assert np.allclose(policy["planes"][plane]["target_spacing_mm"], [0.30, 0.40])
        assert np.allclose(policy["planes"][plane]["target_fov_mm"], [180.0, 200.0])
    assert policy["policy_sha256"] == physical_policy_digest(policy)
    validate_physical_scale_policy(policy)


def test_resample_volume_inplane_produces_canonical_physical_grid():
    frame = _geometry_frame()
    policy = derive_policy_from_geometry(
        frame,
        source_study_count=2,
        selected_series_signature_value="abc",
        min_geometry_coverage=0.95,
    )
    volume = np.ones((3, 100, 120), dtype=np.float32)
    out, applied = resample_volume_inplane(
        volume,
        source_spacing_mm=(0.6, 0.5),
        plane="Sagittal",
        policy=policy,
    )
    assert applied is True
    expected_h = round(180.0 / 0.30)
    expected_w = round(200.0 / 0.40)
    assert out.shape == (3, expected_h, expected_w)
    assert np.isfinite(out).all()
    assert out.max() <= 1.0
    assert out.max() > 0.99


def test_missing_pixel_spacing_keeps_legacy_geometry():
    frame = _geometry_frame()
    policy = derive_policy_from_geometry(
        frame,
        source_study_count=2,
        selected_series_signature_value="abc",
        min_geometry_coverage=0.95,
    )
    volume = np.arange(2 * 8 * 10, dtype=np.float32).reshape(2, 8, 10)
    out, applied = resample_volume_inplane(
        volume,
        source_spacing_mm=None,
        plane="Axial",
        policy=policy,
    )
    assert applied is False
    assert np.array_equal(out, volume)


def test_selected_series_signature_is_deterministic_and_sensitive_to_mapping():
    index_a = {
        "b": {"sagittal_fluid": "2"},
        "a": {"sagittal_fluid": "1"},
    }
    index_b = {
        "a": {"sagittal_fluid": "1"},
        "b": {"sagittal_fluid": "2"},
    }
    sig_a = selected_series_signature(index_a, ["a", "b"])
    sig_b = selected_series_signature(index_b, ["b", "a"])
    assert sig_a == sig_b

    index_b["b"]["sagittal_fluid"] = "3"
    assert selected_series_signature(index_b, ["a", "b"]) != sig_a
