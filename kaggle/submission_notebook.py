"""Kaggle code-competition submission notebook.

Paste each numbered block into its own notebook cell, in order.

This competition is scored by running your notebook against hidden test data,
so nothing is uploaded as a finished CSV. The notebook attaches your code and
your trained model as Kaggle datasets, reads the competition's own test files,
and writes `/kaggle/working/submission.csv`.

The notebook runs with the internet turned off, so nothing may be downloaded
while it runs. That rules out the DINOv3 encoder, whose weights are fetched on
demand -- submit the report-aligned model.

Before running: attach three things to the notebook, using the Add Input panel.

    1. the competition data
    2. a dataset holding this repository
    3. a dataset holding your trained model.pt

Cell 1 prints where everything landed, so the later cells can be pointed at it.
"""

# ---------------------------------------------------------------- cell 1 ----
# Look at what is attached. Run this first and read the output.

import os
from pathlib import Path

for root in sorted(Path("/kaggle/input").glob("*")):
    print("===", root.name)
    shown = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print("   ", path.relative_to(root))
            shown += 1
            if shown >= 12:
                print("    ... more files not shown")
                break


# ---------------------------------------------------------------- cell 2 ----
# Point the notebook at your code and your model.
# Change these two lines to match the folder names printed by cell 1.

import sys
from pathlib import Path

CODE_ROOT = Path("/kaggle/input/cnn-cpc-code")      # <-- your code dataset
MODEL_PATH = Path("/kaggle/input/cnn-cpc-model/model.pt")  # <-- your model.pt

# The repository may sit one level down inside the dataset.
if not (CODE_ROOT / "model").is_dir():
    candidates = [p for p in CODE_ROOT.glob("*") if (p / "model").is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"could not find the repository under {CODE_ROOT}; "
            "check the folder name printed by cell 1"
        )
    CODE_ROOT = candidates[0]

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"no model file at {MODEL_PATH}")

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "developments" / "src"))
print("code :", CODE_ROOT)
print("model:", MODEL_PATH)


# ---------------------------------------------------------------- cell 3 ----
# Find the competition data. It is whichever attached folder holds test.csv.

DATA_ROOT = None
for root in sorted(Path("/kaggle/input").glob("*")):
    if (root / "test.csv").is_file():
        DATA_ROOT = root
        break
    nested = [p for p in root.glob("*") if (p / "test.csv").is_file()]
    if nested:
        DATA_ROOT = nested[0]
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "no attached folder contains test.csv -- attach the competition data"
    )

print("data :", DATA_ROOT)
for name in ("test.csv", "test_series.csv", "sample_submission.csv"):
    print(f"   {name}: {'found' if (DATA_ROOT / name).is_file() else 'MISSING'}")


# ---------------------------------------------------------------- cell 4 ----
# Check the model loads before spending time on the images.

from model._implementation import read_config
from model.architecture import load

config = read_config(str(CODE_ROOT / "config" / "current_model.yaml"))
config["data_root"] = str(DATA_ROOT)

model, payload = load(str(MODEL_PATH), device="cpu")
print("encoder     :", payload.get("encoder_source", "report-aligned"))
print("epochs done :", payload.get("completed_epochs"))
print("frozen      :", payload.get("encoder_frozen"))

if str(payload.get("encoder_source", "report-aligned")) == "dinov3":
    raise RuntimeError(
        "this checkpoint uses DINOv3, whose weights download on demand; "
        "the notebook runs offline, so submit the report-aligned model"
    )

del model  # the next cell rebuilds it on the GPU


# ---------------------------------------------------------------- cell 5 ----
# Predict the hidden test set and write the submission.

from testing.test import predict_test_set

submission = predict_test_set(
    config,
    checkpoint=str(MODEL_PATH),
    out_path="/kaggle/working/submission.csv",
)
print("wrote", submission)


# ---------------------------------------------------------------- cell 6 ----
# Look at the result before submitting.

import pandas as pd

frame = pd.read_csv("/kaggle/working/submission.csv")
print("rows   :", len(frame))
print("columns:", len(frame.columns))
print(frame.head())

# Every column should vary across studies. A column that does not is a warning
# sign: the model would be giving every knee the same answer for that finding.
scores = frame.drop(columns=["StudyInstanceUID"])
spread = (scores.max() - scores.min()).sort_values()
print("\nsmallest spreads:")
print(spread.head(3))
if spread.max() < 0.01:
    print("\nWARNING: every column is nearly constant -- check the model")
