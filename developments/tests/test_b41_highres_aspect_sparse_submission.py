from __future__ import annotations

import pytest

import rsna_knee.b41_highres_aspect_sparse_submission as submission
from rsna_knee.b35_training import B35_EXPECTED_CELLS, B35_EXPECTED_SERIES
from rsna_knee.b41_highres_aspect_sparse_mil import (
    B41_EXPERIMENT,
    B41_IMAGE_SIZE,
    B41_RESIZE_POLICY,
    B41_VERSION,
)
from rsna_knee.b41_highres_aspect_sparse_training import B41_EPOCHS
from rsna_knee.phase9_supervision import REPORT_ONLY_STUDIES


def _config() -> dict:
    return {
        "competition_mode": False,
        "requested_gpus": 1,
        "device": "cpu",
        "precision": "fp32",
        "runtime_budget_hours": 1.0,
        "runtime_reserve_minutes": 30.0,
        "seed": 2026,
        "num_workers": 0,
        "pin_memory": False,
        "series_cache_mb_per_worker": 0,
        "strict_dicom_inference": True,
        "b7_n_slices": 16,
        "b7_image_size": 448,
        "b7_triplet_gap": 1,
        "b7_eval_batch_size": 1,
        "b7_eval_tta_offsets": [-1, 0, 1],
        "b37_grid_size": 6,
        "b37_top_k": 8,
        "b37_temperature": 1.0,
        "b37_local_aux_weight": 1.0,
        "b37_encoder_trainable_stages": 1,
        "b37_encoder_lr_scale": 0.05,
        "b37_encoder_chunk_size": 4,
        "b20_crop_focus_enabled": True,
        "b20_crop_focus_version": "joint_focus_center_crop_only_v1",
        "b20_crop_focus_crop_fraction": 0.90,
        "b41_resize_policy": "aspect_preserving_pad",
        "b41_pad_value": 0.0,
    }


def _payload() -> dict:
    return {
        "experiment": B41_EXPERIMENT,
        "version": B41_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B41_EPOCHS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "sparse_mil": {"grid_size": 6, "top_k": 8, "dense_slices": 32},
        "preprocessing": {
            "resize_policy": B41_RESIZE_POLICY,
            "preserves_in_plane_aspect_ratio": True,
            "image_size": B41_IMAGE_SIZE,
            "padding": {"value": 0.0},
        },
    }


def test_b41_submission_contract_requires_aspect_preserving_geometry() -> None:
    submission.require_b41_submission_contract(_config())
    changed = _config()
    changed["b41_resize_policy"] = "square_stretch"
    with pytest.raises(ValueError, match="b41_resize_policy"):
        submission.require_b41_submission_contract(changed)


def test_b41_submission_contract_rejects_changed_tta() -> None:
    changed = _config()
    changed["b7_eval_tta_offsets"] = [0]
    with pytest.raises(ValueError, match="tta_offsets"):
        submission.require_b41_submission_contract(changed)


def test_b41_checkpoint_contract_accepts_completed_fixed_e2() -> None:
    submission._require_b41_checkpoint_contract(_payload())


def test_b41_checkpoint_contract_rejects_square_resize_metadata() -> None:
    payload = _payload()
    payload["preprocessing"] = dict(payload["preprocessing"])
    payload["preprocessing"]["resize_policy"] = "square_stretch"
    with pytest.raises(ValueError, match="aspect-preserving"):
        submission._require_b41_checkpoint_contract(payload)


def test_b41_checkpoint_contract_rejects_wrong_epoch() -> None:
    payload = _payload()
    payload["completed_epochs"] = 3
    with pytest.raises(ValueError, match="fixed-E2"):
        submission._require_b41_checkpoint_contract(payload)
