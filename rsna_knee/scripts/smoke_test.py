"""End-to-end rehearsal on synthetic data.

Run this before you start a real training run. It fabricates a small DICOM
dataset with realistic geometry and metadata, then exercises the entire chain —
caching, schema discovery, folds, training, out-of-fold scoring and submission
writing — in a couple of minutes on a CPU.

If this passes, any failure on the real data is about the data, not the code.

    python scripts/smoke_test.py --work-dir /tmp/rsna_smoke
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS = [
    "acl_tear",
    "pcl_tear",
    "medial_meniscus_tear",
    "lateral_meniscus_tear",
    "cartilage_defect",
    "bone_marrow_lesion",
    "joint_effusion",
]

# Plane geometry and typical timings, so the detection logic gets a real test.
PROTOCOL = [
    ("sagittal", [0, 1, 0, 0, 0, -1], 30.0, 3000.0, "FS"),
    ("coronal", [1, 0, 0, 0, 0, -1], 80.0, 4000.0, "FS"),
    ("axial", [1, 0, 0, 0, 1, 0], 10.0, 600.0, ""),
]


def write_synthetic_dicoms(root: Path, exam_ids: list[str], slices: int = 8, size: int = 64) -> None:
    """Write a small DICOM tree in the usual study/series/instance layout."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    rng = np.random.default_rng(0)
    for exam in exam_ids:
        study_uid = generate_uid()
        for series_index, (_, orientation, echo, repetition, options) in enumerate(PROTOCOL):
            series_dir = root / exam / f"series{series_index}"
            series_dir.mkdir(parents=True, exist_ok=True)
            series_uid = generate_uid()
            for slice_index in range(slices):
                dataset = Dataset()
                dataset.file_meta = FileMetaDataset()
                dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dataset.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
                dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()

                # A real UID, not the folder name: identity comes from the
                # directory layout, and a non-conforming UID upsets pydicom.
                dataset.StudyInstanceUID = study_uid
                dataset.SeriesInstanceUID = series_uid
                dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
                dataset.Modality = "MR"
                dataset.ImageOrientationPatient = orientation
                # Positions advance along the slice normal, so ordering is testable.
                dataset.ImagePositionPatient = [
                    float(slice_index * 4) if axis == series_index else 0.0 for axis in range(3)
                ]
                dataset.InstanceNumber = slice_index + 1
                dataset.EchoTime = echo
                dataset.RepetitionTime = repetition
                dataset.ScanOptions = options
                dataset.PixelSpacing = [0.3, 0.3]
                dataset.SliceThickness = 3.0
                dataset.Rows = size
                dataset.Columns = size
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                pixels = rng.integers(0, 3000, size=(size, size), dtype=np.uint16)
                dataset.PixelData = pixels.tobytes()

                pydicom.dcmwrite(
                    series_dir / f"{slice_index:03d}.dcm", dataset, enforce_file_format=True
                )


def write_synthetic_csvs(data_dir: Path, exam_ids: list[str]) -> None:
    """Write train.csv, the reports and a wide sample submission."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"StudyInstanceUID": exam_ids})
    frame["PatientID"] = [f"patient{index // 2}" for index in range(len(exam_ids))]
    for index, label in enumerate(LABELS):
        # Vary the prevalence so class balancing and rare-label folds get tested.
        prevalence = 0.5 / (index + 1)
        frame[label] = (rng.random(len(exam_ids)) < prevalence).astype(int)
    frame.to_csv(data_dir / "train.csv", index=False)

    languages = ["Normal knee MRI.", "Rupture du LCA.", "Riss des Innenmeniskus.", "前十字韧带撕裂"]
    pd.DataFrame(
        {
            "StudyInstanceUID": exam_ids,
            "report": [languages[i % len(languages)] for i in range(len(exam_ids))],
        }
    ).to_csv(data_dir / "train_reports.csv", index=False)

    sample = pd.DataFrame({"StudyInstanceUID": exam_ids[:4]})
    for label in LABELS:
        sample[label] = 0.5
    sample.to_csv(data_dir / "sample_submission.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehearse the pipeline on synthetic data")
    parser.add_argument("--work-dir", default="/tmp/rsna_smoke")
    parser.add_argument("--exams", type=int, default=12)
    parser.add_argument("--keep", action="store_true", help="Do not delete the work directory")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    data_dir = work_dir / "data"
    dicom_dir = work_dir / "dicom"
    data_dir.mkdir(parents=True)

    exam_ids = [f"study{index:03d}" for index in range(args.exams)]
    print(f"Writing {args.exams} synthetic exams to {work_dir} ...")
    write_synthetic_dicoms(dicom_dir, exam_ids)
    write_synthetic_csvs(data_dir, exam_ids)

    from rsnaknee.config import load_config
    from rsnaknee.infer import run_inference
    from rsnaknee.preprocess import build_cache
    from rsnaknee.train import combine_oof, prepare_frames, train_fold

    print("\n--- Caching ---")
    manifest = build_cache(dicom_dir, work_dir / "cache", size=64, max_slices=8, workers=1)
    planes = sorted(manifest["plane"].unique())
    assert planes == ["axial", "coronal", "sagittal"], f"plane detection failed: {planes}"
    weightings = sorted(manifest["weighting"].unique())
    assert weightings == ["pd", "t1", "t2"], f"weighting detection failed: {weightings}"
    print(f"Planes detected: {planes}; weightings: {weightings}")

    config = load_config(
        None,
        [
            f"paths.data_dir={data_dir}",
            f"paths.cache_dir={work_dir / 'cache'}",
            f"paths.output_dir={work_dir / 'run'}",
            "data.image_size=32",
            "data.depth=4",
            "data.max_series=3",
            "data.n_folds=3",
            "data.num_workers=0",
            "model.backbone=resnet18",
            "model.pretrained=false",
            "model.embed_dim=64",
            "model.slice_layers=1",
            "model.slice_heads=4",
            "train.epochs=1",
            "train.batch_size=2",
            "train.accumulate=1",
            "train.amp_dtype=fp32",
            "train.channels_last=false",
            "train.ema_decay=0.9",
            "inference.batch_size=2",
            "inference.half=false",
        ],
    )

    print("\n--- Schema and folds ---")
    frame, manifest, schema = prepare_frames(config)
    assert schema.labels == LABELS, f"schema discovery failed: {schema.labels}"
    assert schema.group_column == "PatientID", "patient grouping was not detected"
    print(f"Discovered {schema.num_labels} labels, grouping by {schema.group_column}")

    print("\n--- Training one fold ---")
    report = train_fold(0, frame, manifest, schema, config, teacher_columns=[])
    print(f"macro AUC on a random-noise dataset: {report['macro_auc']:.3f} (~0.5 is expected)")

    combine_oof(config, schema, frame)

    print("\n--- Inference ---")
    config.paths.sample_submission_csv = str(data_dir / "sample_submission.csv")
    submission = run_inference(config, [0], str(work_dir / "submission.csv"))
    assert list(submission.columns) == ["StudyInstanceUID"] + LABELS, submission.columns.tolist()
    assert submission[LABELS].to_numpy().min() >= 0.0
    assert submission[LABELS].to_numpy().max() <= 1.0
    print(submission.head())

    if not args.keep:
        shutil.rmtree(work_dir)
    print("\nSmoke test passed: the full pipeline runs end to end.")


if __name__ == "__main__":
    main()
