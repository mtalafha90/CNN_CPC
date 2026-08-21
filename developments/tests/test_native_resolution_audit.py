from __future__ import annotations

import pandas as pd

from rsna_knee.native_resolution_audit import _subset_summary


def test_padding_feasibility_is_computed_from_native_90pct_crop():
    frame = pd.DataFrame(
        {
            "rows": [512, 384, 320],
            "columns": [512, 384, 320],
            "matrix": ["512x512", "384x384", "320x320"],
            "pixel_spacing_row_mm": [0.33, 0.5, 0.6],
            "pixel_spacing_col_mm": [0.33, 0.5, 0.6],
            "fov_row_mm": [168.96, 192.0, 192.0],
            "fov_col_mm": [168.96, 192.0, 192.0],
            "slice_thickness_mm": [3.4, 3.0, 3.0],
            "spacing_between_slices_mm": [4.6, 3.5, 3.5],
            "manufacturer": ["Philips", "GE", "Siemens"],
            "manufacturer_model": ["A", "B", "C"],
            "magnetic_field_strength_t": [1.5, 3.0, 1.5],
        }
    )
    summary = _subset_summary(frame, crop_fraction=0.90, canvases=(288, 384, 464, 512))
    feasibility = {row["canvas"]: row["coverage"] for row in summary["padding_canvas_feasibility"]}

    # round(0.9*512)=461, round(0.9*384)=346, round(0.9*320)=288
    assert feasibility[288] == 1 / 3
    assert feasibility[384] == 2 / 3
    assert feasibility[464] == 1.0
    assert summary["smallest_tested_canvas_covering_99pct"] == 464


def test_spacing_heterogeneity_is_not_confused_with_matrix_coverage():
    frame = pd.DataFrame(
        {
            "rows": [512] * 20,
            "columns": [512] * 20,
            "matrix": ["512x512"] * 20,
            "pixel_spacing_row_mm": [0.30] * 10 + [0.60] * 10,
            "pixel_spacing_col_mm": [0.30] * 10 + [0.60] * 10,
            "fov_row_mm": [153.6] * 10 + [307.2] * 10,
            "fov_col_mm": [153.6] * 10 + [307.2] * 10,
            "slice_thickness_mm": [3.0] * 20,
            "spacing_between_slices_mm": [4.0] * 20,
            "manufacturer": ["X"] * 20,
            "manufacturer_model": ["Y"] * 20,
            "magnetic_field_strength_t": [1.5] * 20,
        }
    )
    summary = _subset_summary(frame, crop_fraction=0.90, canvases=(464, 512))
    assert summary["smallest_tested_canvas_covering_99pct"] == 464
    assert summary["pixel_spacing_p95_p05_ratio_row"] > 1.25
    assert "PixelSpacing is materially heterogeneous" in summary["decision_note"]
