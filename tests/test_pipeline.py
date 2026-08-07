import numpy as np
import pandas as pd

from rsna_knee.constants import SUBMISSION_COLUMNS, TARGETS
from rsna_knee.data import make_balanced_gold_folds
from rsna_knee.evaluation import macro_auc_from_arrays
from rsna_knee.inference import validate_submission
from rsna_knee.report_labels import predict_target


def test_macro_auc_perfect():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
    p = y * 0.9 + (1 - y) * 0.1
    assert macro_auc_from_arrays(y, p)[0] == 1.0


def test_macro_auc_ignores_unannotated_cells():
    y = np.array([[0, np.nan], [1, np.nan], [0, 1], [1, 0]], dtype=float)
    p = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.9], [0.8, 0.1]], dtype=float)
    score, per_target = macro_auc_from_arrays(y, p)
    assert score == 1.0
    assert np.allclose(per_target, [1.0, 1.0], equal_nan=True)


def test_report_negation():
    assert predict_target("Complete tear of the anterior cruciate ligament.", "ACL").probability > 0.8
    assert predict_target("No tear of the anterior cruciate ligament.", "ACL").probability < 0.2


def test_gold_folds_and_submission():
    rows = []
    for i in range(18):
        row = {"StudyInstanceUID": str(i), "Report": f"report {i}"}
        for j, target in enumerate(TARGETS):
            row[target] = int((i + j) % 3 == 0)
        rows.append(row)
    rows.append({"StudyInstanceUID": "u", "Report": "unlabeled", **{t: np.nan for t in TARGETS}})
    folds = make_balanced_gold_folds(pd.DataFrame(rows), n_splits=3)
    assert folds.iloc[-1] == -1
    submission = pd.DataFrame([["x", *([0.5] * 12)]], columns=SUBMISSION_COLUMNS)
    validate_submission(submission)
