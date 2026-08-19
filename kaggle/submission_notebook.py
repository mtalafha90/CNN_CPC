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
    3. a dataset holding your trained checkpoints

Cell 1 prints where everything landed, so the later cells can be pointed at it.

Cell 2 has the one setting worth editing: `SUBMIT`, the list of checkpoint
filenames this run should use. Everything else searches rather than assumes.
"""

# ---------------------------------------------------------------- cell 1 ----
# Look at what is attached. Run this first and read the output.
#
# Only the top level of each folder is listed, on purpose. The competition data
# holds hundreds of thousands of scan files, and walking all of them takes many
# minutes for no benefit -- the folder names are all the later cells need.

import os
from pathlib import Path

for root in sorted(Path("/kaggle/input").glob("*")):
    print("===", root.name)
    shown = 0
    with os.scandir(root) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            print(f"    {'dir ' if entry.is_dir() else 'file'} {entry.name}")
            shown += 1
            if shown >= 15:
                print("    ... more not shown")
                break


# ---------------------------------------------------------------- cell 2 ----
# Find your code, your model and the competition data.
#
# Nothing is hard-coded except the competition name, because Kaggle nests
# inputs differently depending on how they were attached.  Your own datasets
# are small, so searching them is cheap; the competition folder is not, so it
# is only checked at the top level.

import sys
from pathlib import Path

BASE = Path("/kaggle/input")
MINE = BASE / "datasets" if (BASE / "datasets").is_dir() else BASE
COMP = BASE / "competitions" / "rsna-knee-abnormality-detection"

CODE_ROOT = None
for marker in MINE.rglob("architecture.py"):
    if marker.parent.name == "model":
        CODE_ROOT = marker.parent.parent
        break
if CODE_ROOT is None:
    print("Could not find model/architecture.py. Your datasets contain:")
    for path in sorted(MINE.rglob("*"))[:40]:
        print("   ", path.relative_to(MINE))
    raise FileNotFoundError("repository not found -- see the listing above")

# Name the models you are submitting, and nothing else is used. One model
# scores on its own; several are averaged, which is the cheapest reliable way
# to gain a little accuracy -- when the members genuinely differ.
#
# This is a list rather than "whatever .pt is attached" on purpose. A model
# dataset accumulates checkpoints over a competition, so sweeping up all of
# them lets the dataset's contents decide the experiment: adding a file for
# later turns today's single-model run into an ensemble without saying so.
# Naming them means the notebook submits what you meant to submit.
SUBMIT = ["model_finetuned.pt"]

available = {path.name: path for path in sorted(MINE.rglob("*.pt"))}
if not available:
    raise FileNotFoundError("no .pt file found -- is the model dataset attached?")

missing = [name for name in SUBMIT if name not in available]
if missing:
    raise FileNotFoundError(
        f"SUBMIT names {missing}, which is not attached. Available: "
        f"{sorted(available)}"
    )

MODEL_PATHS = [available[name] for name in SUBMIT]
MODEL_PATH = MODEL_PATHS[0]

unused = sorted(set(available) - set(SUBMIT))
if unused:
    print("not submitting:", ", ".join(unused))

DATA_ROOT = COMP if (COMP / "test.csv").is_file() else None
if DATA_ROOT is None:
    for child in sorted(COMP.glob("*")):
        if child.is_dir() and (child / "test.csv").is_file():
            DATA_ROOT = child
            break
if DATA_ROOT is None:
    print("test.csv not found. The competition folder holds:")
    for item in sorted(COMP.glob("*"))[:20]:
        print("   ", "dir " if item.is_dir() else "file", item.name)
    raise FileNotFoundError("could not find test.csv")

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "developments" / "src"))
print("code :", CODE_ROOT)
for path in MODEL_PATHS:
    print("model:", path)
print("data :", DATA_ROOT)


# ---------------------------------------------------------------- cell 3 ----
# Check the model loads before spending time on the images.

from model._implementation import read_config
from model.architecture import load

config = read_config(str(CODE_ROOT / "config" / "current_model.yaml"))
config["data_root"] = str(DATA_ROOT)

# A checkpoint's filename is a label someone typed; the payload is the record
# the training run wrote. When the two disagree, the payload is right -- so
# print what each file actually is before spending an hour proving it later.
for path in MODEL_PATHS:
    model, payload = load(str(path), device="cpu")
    before = payload.get("encoder_sha256_initial")
    after = payload.get("encoder_sha256_final")
    print(f"--- {path.name}")
    print("  encoder    :", payload.get("encoder_source", "report-aligned"))
    print("  epochs done:", payload.get("completed_epochs"))
    # The fingerprints are taken before and after training, so they cannot
    # agree unless the encoder really did stay still.
    print("  fine-tuned :", bool(before and after and before != after))
    print("  stages free:", payload.get("encoder_trainable_stages", 0))

    if str(payload.get("encoder_source", "report-aligned")) == "dinov3":
        raise RuntimeError(
            f"{path.name} uses DINOv3, whose weights download on demand; "
            "the notebook runs offline, so submit the report-aligned model"
        )
    del model  # cell 4 rebuilds it on the GPU

print(f"\nsubmitting {len(MODEL_PATHS)} model(s)")


# ---------------------------------------------------------------- cell 4 ----
# Predict the hidden test set and write the submission.

from testing.test import predict_test_set

submission = predict_test_set(
    config,
    checkpoint=[str(p) for p in MODEL_PATHS],
    out_path="/kaggle/working/submission.csv",
)
print("wrote", submission)


# ---------------------------------------------------------------- cell 5 ----
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
