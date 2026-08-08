from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rsna_knee.constants import TARGETS
from rsna_knee.training import train_fold


class _TinyKneeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, len(TARGETS))

    def forward(self, volumes, present):
        # Keep the same production input contract while making CI inexpensive.
        scalar = volumes.mean(dim=(1, 2, 3, 4, 5), keepdim=False).unsqueeze(1)
        return self.head(scalar)


def _write_multiframe(path: Path, value: int) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows = 12
    ds.Columns = 12
    ds.NumberOfFrames = 5
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.InstanceNumber = 1
    pixels = np.stack(
        [np.full((12, 12), value + frame, dtype=np.uint16) for frame in range(5)]
    )
    ds.PixelData = pixels.tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def _make_dataset(root: Path) -> None:
    rows = []
    series_rows = []
    n_gold = 12
    n_weak = 6
    for i in range(n_gold + n_weak):
        uid = f"study{i:02d}"
        series_uid = f"series{i:02d}"
        row = {"StudyInstanceUID": uid, "Report": f"unique synthetic report {i}"}
        for j, target in enumerate(TARGETS):
            row[target] = float((i + j) % 2) if i < n_gold else np.nan
        rows.append(row)
        series_rows.append(
            {
                "StudyInstanceUID": uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": True,
                "Fat_Suppression": True,
                "Anatomical_Plane": "Sagittal",
            }
        )
        _write_multiframe(root / "train_images" / uid / series_uid / "image.dcm", i + 1)

    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)


def test_synthetic_fold_runs_preflight_nested_training_and_artifact_export(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    _make_dataset(root)

    import rsna_knee.training as training

    monkeypatch.setattr(training, "_build_model", lambda spec, config, device: _TinyKneeModel().to(device))
    monkeypatch.setattr(
        training,
        "macro_auc_from_arrays",
        lambda y_true, y_score: (0.5, np.full(y_score.shape[1], 0.5, dtype=float)),
    )

    config = {
        "data_root": str(root),
        "train_csv": "train.csv",
        "train_series_csv": "train_series.csv",
        "output_dir": str(tmp_path / "runs"),
        "competition_mode": True,
        "requested_gpus": 1,
        "runtime_budget_hours": 0.25,
        "runtime_reserve_minutes": 0,
        "pretrained": False,
        "allow_external_pretrained": False,
        "n_folds": 3,
        "seed": 2026,
        "device": "cpu",
        "precision": "fp32",
        "num_workers": 0,
        "persistent_workers": False,
        "n_slices": 1,
        "image_size": 16,
        "batch_size": 2,
        "inference_batch_size": 2,
        "oof_batch_size": 2,
        "weak_oof_batch_size": 2,
        "encoder_batch_size": 2,
        "gradient_checkpointing": False,
        "normalize_input": False,
        "transformer_layers": 1,
        "transformer_heads": 8,
        "transformer_ff_mult": 2.0,
        "pathology_layers": 1,
        "epochs": 1,
        "max_train_batches_per_epoch": 1,
        "patience": 1,
        "preflight_before_train": True,
        "preflight_sample_size": 2,
        "preflight_max_decode_failure_rate": 0.05,
        "preflight_max_file_decode_failure_rate": 0.05,
        "series_cache_mb_per_worker": 0,
        "tta_center_offsets": [0],
        "validation_tta_offsets": [0],
        "weak_oof_tta_offsets": [0],
        "finish_inference_safety_factor": 1.0,
        "finish_seconds_per_study_floor": 0.001,
        "finish_seconds_per_study_fallback": 0.001,
        "finish_bootstrap_reserve_seconds": 0,
        "finish_serialization_reserve_seconds": 0,
        "finish_loader_startup_reserve_seconds": 0,
        "prediction_initial_batch_guard_seconds": 0.01,
        "n_bootstrap": 2,
        "rank_loss_weight": 0.0,
        "cotrain_stage1_root": None,
        "cotrain_stage1_candidates": None,
    }

    checkpoint = train_fold(config, 0)
    fold_dir = checkpoint.parent
    assert checkpoint.is_file()
    for name in [
        "oof.csv",
        "oof_center.csv",
        "weak_oof.csv",
        "history.csv",
        "selection.json",
        "training_diagnostics.json",
        "supervision_plan.json",
        "runtime.json",
        "bootstrap.json",
    ]:
        assert (fold_dir / name).is_file(), name

    selection = pd.read_json(fold_dir / "selection.json", typ="series")
    assert selection["stage"] == "stage1"
    assert selection["validation_tta_offsets"] == [0]
