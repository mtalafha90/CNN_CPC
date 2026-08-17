from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.dataset_header_audit import _slice_bin, orientation_from_iop, summarize_headers


def test_orientation_from_iop_maps_cardinal_planes():
    sagittal = orientation_from_iop([0, 1, 0, 0, 0, 1])
    coronal = orientation_from_iop([1, 0, 0, 0, 0, 1])
    axial = orientation_from_iop([1, 0, 0, 0, 1, 0])
    assert sagittal["normal_plane"] == "Sagittal"
    assert coronal["normal_plane"] == "Coronal"
    assert axial["normal_plane"] == "Axial"
    assert sagittal["obliquity_deg"] == 0.0
    assert coronal["obliquity_deg"] == 0.0
    assert axial["obliquity_deg"] == 0.0


def test_slice_bins_freeze_phase2_tail_boundaries():
    assert _slice_bin(48) == "<=48"
    assert _slice_bin(49) == "49-78"
    assert _slice_bin(78) == "49-78"
    assert _slice_bin(79) == "79-100"
    assert _slice_bin(100) == "79-100"
    assert _slice_bin(101) == "101-200"
    assert _slice_bin(200) == "101-200"
    assert _slice_bin(201) == ">200"


def test_summarize_headers_tracks_orientation_mismatch_and_tail():
    rows = [
        {
            "StudyInstanceUID": "s1", "SeriesInstanceUID": "a", "Anatomical_Plane": "Axial",
            "Fluid_Sensitive": True, "Fat_Suppression": True, "dicom_files": 30,
            "header_read_ok": True, "number_of_frames": 1, "normal_plane": "Axial",
            "orientation_matches_supplied_plane": True, "obliquity_deg": 0.0,
            "manufacturer": "A", "manufacturer_model": "M1", "magnetic_field_strength_t": 3.0,
            "mr_acquisition_type": "2D", "transfer_syntax_uid": "1", "photometric_interpretation": "MONOCHROME2",
            "bits_allocated": 16, "bits_stored": 12, "pixel_representation": 0,
            "pixel_spacing_row_mm": 0.5, "pixel_spacing_col_mm": 0.5,
            "fov_row_mm": 128.0, "fov_col_mm": 128.0, "slice_thickness_mm": 3.0,
            "spacing_between_slices_mm": 3.3, "rows": 256, "columns": 256,
        },
        {
            "StudyInstanceUID": "s2", "SeriesInstanceUID": "b", "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": False, "Fat_Suppression": False, "dicom_files": 320,
            "header_read_ok": True, "number_of_frames": 1, "normal_plane": "Coronal",
            "orientation_matches_supplied_plane": False, "obliquity_deg": 2.0,
            "manufacturer": "B", "manufacturer_model": "M2", "magnetic_field_strength_t": 1.5,
            "mr_acquisition_type": "3D", "transfer_syntax_uid": "2", "photometric_interpretation": "MONOCHROME2",
            "bits_allocated": 16, "bits_stored": 16, "pixel_representation": 1,
            "pixel_spacing_row_mm": 0.4, "pixel_spacing_col_mm": 0.4,
            "fov_row_mm": 102.4, "fov_col_mm": 102.4, "slice_thickness_mm": 0.5,
            "spacing_between_slices_mm": 0.5, "rows": 256, "columns": 256,
        },
    ]
    summary, categorical, tail, orientation = summarize_headers(pd.DataFrame(rows))
    assert summary["series"] == 2
    assert summary["orientation_mismatches_supplied_plane"] == 1
    assert int(tail.loc[tail["slice_bin"].eq(">200"), "series"].sum()) == 1
    assert not categorical.empty
    assert not orientation.empty
