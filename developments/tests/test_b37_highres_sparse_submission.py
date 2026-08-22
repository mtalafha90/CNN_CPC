from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import rsna_knee.b37_highres_sparse_submission as submission
from rsna_knee.b35_training import B35_EXPECTED_CELLS, B35_EXPECTED_SERIES, sha256_file
from rsna_knee.b37_highres_sparse_mil import B37_IMAGE_SIZE, B37_VERSION
from rsna_knee.b37_highres_sparse_training import B37_EPOCHS, B37_EXPERIMENT
from rsna_knee.constants import TARGETS
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
    }


def _payload(base_checkpoint_sha256: str) -> dict:
    return {
        "experiment": B37_EXPERIMENT,
        "version": B37_VERSION,
        "fixed_endpoint": True,
        "completed_epochs": B37_EPOCHS,
        "gold_studies_used_in_gradient": 0,
        "gold_labels_used": False,
        "training_studies": REPORT_ONLY_STUDIES,
        "training_series": B35_EXPECTED_SERIES,
        "training_supervision_cells": B35_EXPECTED_CELLS,
        "encoder_sha256_initial": "initial",
        "encoder_sha256_final": "final",
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "sparse_mil": {"grid_size": 6, "top_k": 8, "dense_slices": 32},
        "preprocessing": {"image_size": B37_IMAGE_SIZE},
    }


def test_b37_submission_dataset_is_strict_and_deterministic(tmp_path) -> None:
    dataset_config = submission._b37_test_dataset_config(_config(), tmp_path)
    assert dataset_config.data_root == str(tmp_path)
    assert dataset_config.split == "test"
    assert dataset_config.image_size == 448
    assert dataset_config.n_slices == 16
    assert dataset_config.strict_dicom is True
    assert dataset_config.tta_center_offsets == ()
    assert dataset_config.noise_std == 0.0
    assert dataset_config.slice_dropout == 0.0
    assert dataset_config.center_jitter == 0
    assert dataset_config.series_cache_mb == 0


def test_b37_submission_contract_rejects_changed_memory_or_tta() -> None:
    config = _config()
    submission.require_b37_submission_contract(config)

    changed_workers = dict(config)
    changed_workers["num_workers"] = 1
    with pytest.raises(ValueError, match="num_workers=0"):
        submission.require_b37_submission_contract(changed_workers)

    changed_offsets = dict(config)
    changed_offsets["b7_eval_tta_offsets"] = [0]
    with pytest.raises(ValueError, match="tta_offsets"):
        submission.require_b37_submission_contract(changed_offsets)


def test_b37_checkpoint_contract_rejects_unmoved_encoder() -> None:
    payload = _payload("base")
    payload["encoder_sha256_final"] = payload["encoder_sha256_initial"]
    with pytest.raises(ValueError, match="encoder-tail"):
        submission._require_b37_checkpoint_contract(payload)


def test_generate_b37_submission_averages_three_probability_views_and_writes_manifest(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "b37_model.pt"
    base_checkpoint = tmp_path / "base_model.pt"
    checkpoint.write_bytes(b"b37")
    base_checkpoint.write_bytes(b"base")
    payload = _payload(sha256_file(base_checkpoint))
    captured: dict[str, object] = {}

    class FakeRuntime:
        device = torch.device("cpu")
        num_workers = 0
        pin_memory = False
        amp_dtype = None

        def describe(self) -> str:
            return "fake cpu runtime"

        def loader_kwargs(self, *, seed: int) -> dict:
            captured["loader_seed"] = seed
            return {}

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, volumes, present, series_meta, position):
            values = volumes[:, 0, 0, 0, 0].reshape(-1, 1)
            return SimpleNamespace(logits=values.repeat(1, len(TARGETS)))

    class FakeDataset:
        def __init__(self, uids, index, dataset_config, **kwargs) -> None:
            captured["dataset_uids"] = uids
            captured["dataset_index"] = index
            captured["dataset_config"] = dataset_config
            captured["dataset_kwargs"] = kwargs

    def fake_batch(uid: str) -> dict:
        views = torch.tensor([0.0, 1.0, 2.0]).reshape(1, 3, 1, 1, 1, 1, 1)
        return {
            "study_uid": [uid],
            "volumes": views,
            "slice_position": torch.zeros((1, 3, 1, 1)),
            "present": torch.ones((1, 1)),
            "series_meta": torch.zeros((1, 1, 3), dtype=torch.long),
        }

    batches = [fake_batch("study-a"), fake_batch("study-b")]
    monkeypatch.setattr(submission, "resolve_runtime", lambda config: FakeRuntime())
    monkeypatch.setattr(
        submission,
        "load_b37_checkpoint",
        lambda checkpoint, *, base_checkpoint, device: (FakeModel(), payload),
    )
    monkeypatch.setattr(
        submission,
        "load_test_csv",
        lambda path: pd.DataFrame({"StudyInstanceUID": ["study-a", "study-b"]}),
    )
    monkeypatch.setattr(submission, "load_series_csv", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        submission,
        "backfill_series_metadata",
        lambda series, root, split: (series, {"repaired": 0}),
    )
    monkeypatch.setattr(
        submission,
        "build_variable_series_index",
        lambda series, uids: {
            "study-a": [{"series_uid": "1"}],
            "study-b": [{"series_uid": "2"}],
        },
    )
    monkeypatch.setattr(submission, "B37HighResSparseDataset", FakeDataset)
    monkeypatch.setattr(submission, "DataLoader", lambda *args, **kwargs: batches)
    monkeypatch.setattr(submission, "_release_memory", lambda: None)

    output = tmp_path / "submission.csv"
    result = submission.generate_b37_submission(
        _config(),
        data_root=tmp_path,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=output,
    )

    assert result == output
    assert captured["dataset_uids"] == ["study-a", "study-b"]
    assert captured["dataset_kwargs"] == {
        "crop_focus_policy": {
            "version": "joint_focus_center_crop_only_v1",
            "crop_fraction": 0.9,
        },
        "center_offsets": (-1, 0, 1),
    }
    dataset_config = captured["dataset_config"]
    assert dataset_config.split == "test"
    assert dataset_config.image_size == 448

    frame = pd.read_csv(output)
    assert frame["StudyInstanceUID"].tolist() == ["study-a", "study-b"]
    expected_probability = float(
        np.mean(torch.sigmoid(torch.tensor([0.0, 1.0, 2.0])).numpy())
    )
    assert np.allclose(frame[TARGETS].to_numpy(float), expected_probability)

    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["test_rows"] == 2
    assert manifest["tta_center_offsets"] == [-1, 0, 1]
    assert (
        manifest["prediction"]
        == "B37 combined sparse-MIL logits; raw sigmoid probability"
    )
    assert manifest["checkpoint_base_sha256_verified"] is True
