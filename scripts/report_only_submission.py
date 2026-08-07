#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from rsna_knee.constants import SUBMISSION_COLUMNS, TARGETS
from rsna_knee.data import load_test_csv
from rsna_knee.inference import validate_submission
from rsna_knee.report_labels import label_dataframe

parser = argparse.ArgumentParser()
parser.add_argument("--test-csv", required=True)
parser.add_argument("--out", default="submission_report_only.csv")
args = parser.parse_args()

df = load_test_csv(args.test_csv)
probs, _ = label_dataframe(df)
out = pd.DataFrame(probs, columns=TARGETS)
out.insert(0, "StudyInstanceUID", df["StudyInstanceUID"].astype(str))
out = out[SUBMISSION_COLUMNS]
validate_submission(out)
out.to_csv(args.out, index=False)
print(Path(args.out).resolve())
