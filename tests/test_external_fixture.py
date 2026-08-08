from pathlib import Path

import pandas as pd
import pydicom


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "external_validation"


def test_external_four_study_test_fixture_is_complete_and_decodable():
    test = pd.read_csv(ROOT / "test.csv")
    series = pd.read_csv(ROOT / "test_series.csv")

    assert len(test) == 4
    assert test["StudyInstanceUID"].is_unique
    assert len(series) == 4
    assert series[["StudyInstanceUID", "SeriesInstanceUID"]].duplicated().sum() == 0
    assert set(test["StudyInstanceUID"].astype(str)) == set(series["StudyInstanceUID"].astype(str))

    for row in series.itertuples(index=False):
        path = (
            ROOT
            / "test_images"
            / str(row.StudyInstanceUID)
            / str(row.SeriesInstanceUID)
            / "image.dcm"
        )
        assert path.is_file(), path
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array
        assert pixels.shape == (7, 384, 384)
        assert str(ds.Modality) == "MR"
        assert str(ds.PatientID) == str(row.StudyInstanceUID)


def test_external_sparse_validation_labels_do_not_invent_negatives():
    validation = pd.read_csv(ROOT / "validation.csv")
    assert len(validation) == 4
    reference = validation.loc[
        validation["StudyInstanceUID"].eq("EXTVAL_REFERENCE_001")
    ].iloc[0]
    target_columns = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
        "Baker's", "Contusion", "Fracture",
    ]
    assert reference[target_columns].isna().all()

    acl = validation.loc[validation["StudyInstanceUID"].eq("EXTVAL_ACL_001")].iloc[0]
    assert acl["ACL"] == 1.0
    assert acl[[column for column in target_columns if column != "ACL"]].isna().all()
