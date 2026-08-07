import numpy as np
import pandas as pd

from rsna_knee.constants import SUBMISSION_COLUMNS, TARGETS
from rsna_knee.data import make_balanced_gold_folds
from rsna_knee.inference import validate_submission
from rsna_knee.metrics import macro_auc
from rsna_knee.report_labels import predict_target


def test_macro_auc_perfect():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    p = y * 0.9 + (1 - y) * 0.1
    assert macro_auc(y, p) == 1.0


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
    sub = pd.DataFrame([["x", *([0.5] * 12)]], columns=SUBMISSION_COLUMNS)
    validate_submission(sub)
