from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from rsna_knee.constants import TARGETS

SOURCES = [
    {
        "study_uid": "EXTVAL_ACL_001",
        "series_uid": "EXTVAL_ACL_001_SAG_PDW",
        "filename": "acl_tear_pdw.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/5/51/MRT_VKB-Riss_PDW.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:MRT_VKB-Riss_PDW.jpg",
        "author": "Hellerhoff",
        "license": "CC BY-SA 3.0",
        "known_finding": "Anterior cruciate ligament rupture",
        "plane": "Sagittal",
        "fluid_sensitive": True,
        "fat_suppression": False,
        "known_targets": {"ACL": 1.0},
    },
    {
        "study_uid": "EXTVAL_MEDMEN_001",
        "series_uid": "EXTVAL_MEDMEN_001_COR_PDW",
        "filename": "medial_meniscus_tear_pdw.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/2/25/Proton_density_MRI_of_a_grade_2_medial_meniscal_tear.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Proton_density_MRI_of_a_grade_2_medial_meniscal_tear.jpg",
        "author": "Nicolas Lefevre, Jean Francois Naouri, Serge Herman, Antoine Gerometta, Shahnaz Klouche, Yoann Bohu",
        "license": "CC BY 4.0",
        "known_finding": "Grade 2 medial meniscal tear",
        "plane": "Coronal",
        "fluid_sensitive": True,
        "fat_suppression": False,
        "known_targets": {"Medial Meniscus": 1.0},
    },
    {
        "study_uid": "EXTVAL_BAKER_001",
        "series_uid": "EXTVAL_BAKER_001_AX_MR",
        "filename": "baker_cyst.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/32/MRT_Bakerzyste.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:MRT_Bakerzyste.jpg",
        "author": "Hellerhoff",
        "license": "CC BY-SA 3.0",
        "known_finding": "Baker cyst in a patient with ACL rupture",
        "plane": "Axial",
        "fluid_sensitive": True,
        "fat_suppression": True,
        "known_targets": {"ACL": 1.0, "Baker's": 1.0},
    },
    {
        "study_uid": "EXTVAL_REFERENCE_001",
        "series_uid": "EXTVAL_REFERENCE_001_SAG_PDFS",
        "filename": "reference_sagittal_pdfs.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/d/dc/Knee_MRI_PD_TSE_FS_Sagittal.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Knee_MRI_PD_TSE_FS_Sagittal.jpg",
        "author": "Ptrump16",
        "license": "CC BY-SA 4.0",
        "known_finding": "Reference sagittal PD TSE FS knee MRI; source provides no pathology label",
        "plane": "Sagittal",
        "fluid_sensitive": True,
        "fat_suppression": True,
        "known_targets": {},
    },
]


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CNN-CPC-validation-fixture/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _orientation(plane: str) -> list[float]:
    if plane == "Sagittal":
        return [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if plane == "Coronal":
        return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    if plane == "Axial":
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    raise ValueError(f"unsupported plane: {plane}")


def _uid(namespace: str) -> str:
    return f"2.25.{uuid.uuid5(uuid.NAMESPACE_URL, namespace).int}"


def _write_dicom(jpeg_bytes: bytes, destination: Path, source: dict) -> None:
    from PIL import Image, ImageOps
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
    image = ImageOps.fit(image, (384, 384))
    base = np.asarray(image, dtype=np.uint16) * np.uint16(257)
    # The public files are single published slices, not original DICOM series.
    # Repeat the same slice to exercise the production multi-frame DICOM path.
    frames = np.stack([base] * 7, axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4.1"
    ds.file_meta.MediaStorageSOPInstanceUID = _uid(source["study_uid"] + "/sop")
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = _uid(source["study_uid"])
    ds.SeriesInstanceUID = _uid(source["series_uid"])
    ds.Modality = "MR"
    ds.PatientName = "EXTERNAL^VALIDATION"
    ds.PatientID = source["study_uid"]
    ds.SeriesDescription = "Open-license knee MRI validation fixture"
    ds.Rows = int(frames.shape[1])
    ds.Columns = int(frames.shape[2])
    ds.NumberOfFrames = int(frames.shape[0])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.ImageOrientationPatient = _orientation(source["plane"])
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.InstanceNumber = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 3.0
    ds.SpacingBetweenSlices = 3.0
    ds.PixelData = frames.tobytes()
    pydicom.dcmwrite(destination, ds, enforce_file_format=True)


def _write_csvs(root: Path, records: list[dict]) -> None:
    validation_rows = []
    series_rows = []
    source_rows = []
    for source in records:
        row = {
            "StudyInstanceUID": source["study_uid"],
            "Report": source["known_finding"],
            "KnownFinding": source["known_finding"],
        }
        for target in TARGETS:
            row[target] = source["known_targets"].get(target, np.nan)
        validation_rows.append(row)
        series_rows.append(
            {
                "StudyInstanceUID": source["study_uid"],
                "SeriesInstanceUID": source["series_uid"],
                "Fluid_Sensitive": source["fluid_sensitive"],
                "Fat_Suppression": source["fat_suppression"],
                "Anatomical_Plane": source["plane"],
            }
        )
        source_rows.append(
            {
                "StudyInstanceUID": source["study_uid"],
                "source_filename": source["filename"],
                "source_page": source["source_page"],
                "image_url": source["image_url"],
                "author": source["author"],
                "license": source["license"],
                "known_finding": source["known_finding"],
                "sha256": source["sha256"],
            }
        )

    import pandas as pd

    pd.DataFrame(validation_rows).to_csv(root / "validation.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "validation_series.csv", index=False)
    pd.DataFrame(source_rows).to_csv(root / "sources.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fixtures/external_validation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.output)
    source_dir = root / "source_jpgs"
    source_dir.mkdir(parents=True, exist_ok=True)

    materialized = []
    for source in SOURCES:
        source = dict(source)
        jpg_path = source_dir / source["filename"]
        dcm_path = root / "validation_images" / source["study_uid"] / source["series_uid"] / "image.dcm"
        if args.overwrite or not jpg_path.is_file():
            jpeg_bytes = _download(source["image_url"])
            jpg_path.write_bytes(jpeg_bytes)
        else:
            jpeg_bytes = jpg_path.read_bytes()
        source["sha256"] = hashlib.sha256(jpeg_bytes).hexdigest()
        if args.overwrite or not dcm_path.is_file():
            _write_dicom(jpeg_bytes, dcm_path, source)
        materialized.append(source)

    _write_csvs(root, materialized)
    (root / "materialization.json").write_text(
        json.dumps(
            {
                "studies": len(materialized),
                "purpose": "external technical validation only",
                "warning": "single published slices are repeated into synthetic multi-frame DICOMs; do not use for leaderboard/scientific AUC",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(root)


if __name__ == "__main__":
    main()
