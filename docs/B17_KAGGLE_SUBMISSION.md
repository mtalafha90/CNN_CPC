# B17 Kaggle submission workflow

> **Snapshot — 2026-08-13.** Package `0.26.1`. B17 is the current reused-gold development champion at macro AUC `0.6425890153`. This document covers hidden-test inference only; it does not change B17 training or model selection.

## Why a dedicated code-competition path is required

The downloaded release contains only:

```text
local test studies     3
local test series     15
```

Those rows are suitable for an inference smoke test but are not the hidden leaderboard surface. The real test must be evaluated when Kaggle runs the committed notebook against the hidden competition data.

The B17 submission command is:

```text
rsna-knee-b17-submit
```

It loads only a fully valid completed B17 checkpoint, reconstructs every recognized test MRI series from `test_series.csv`, applies the frozen B17 TTA `[-1,0,1]`, writes the official 13-column `submission.csv`, and writes a provenance manifest next to it.

## Local smoke test

After pulling `main`:

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected version:

```text
0.26.1
```

Run:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b17-submit \
  --config configs/b17_frozen_encoder.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b17_frozen_encoder/b17_model.pt \
  --out runs/b17_frozen_encoder/submission_smoke.csv
```

Expected outputs:

```text
runs/b17_frozen_encoder/submission_smoke.csv
runs/b17_frozen_encoder/submission_smoke.csv.manifest.json
```

The local file should contain exactly three rows because the public placeholder `test.csv` has three studies. This file should not be treated as a leaderboard submission.

## Kaggle artifact bundle

The hidden-test notebook must have access to both the source package and the B17 checkpoint without internet access. A robust approach is to create a private Kaggle Dataset containing:

```text
CNN_CPC/
  configs/b17_frozen_encoder.yaml
  src/rsna_knee/...
  pyproject.toml
  runs/b17_frozen_encoder/b17_model.pt
```

Do not include competition DICOM data in this private dataset; the notebook should attach the official competition data separately.

## Kaggle notebook inference cell

Assuming the private artifact dataset is mounted at:

```text
/kaggle/input/cnn-cpc-b17-artifacts/CNN_CPC
```

and the competition data are mounted at:

```text
/kaggle/input/rsna-knee-abnormality-detection
```

run:

```bash
%cd /kaggle/input/cnn-cpc-b17-artifacts/CNN_CPC
!python -m pip install -e . --no-deps -q
```

Then generate the hidden-test submission:

```bash
!rsna-knee-b17-submit \
  --config /kaggle/input/cnn-cpc-b17-artifacts/CNN_CPC/configs/b17_frozen_encoder.yaml \
  --data-root /kaggle/input/rsna-knee-abnormality-detection \
  --checkpoint /kaggle/input/cnn-cpc-b17-artifacts/CNN_CPC/runs/b17_frozen_encoder/b17_model.pt \
  --out /kaggle/working/submission.csv
```

Inspect before committing:

```python
import pandas as pd
p = "/kaggle/working/submission.csv"
df = pd.read_csv(p)
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.isna().sum().sum())
```

Required columns:

```text
StudyInstanceUID
ACL
MCL
Medial Meniscus
Lateral Meniscus
Medial OA
Lateral OA
PF OA
Effusion
Synovitis
Baker's
Contusion
Fracture
```

The notebook output file for submission is:

```text
/kaggle/working/submission.csv
```

## B17 inference contract

The submission path rejects a checkpoint unless it certifies:

```text
completed B17 epochs                 5
encoder frozen                       true
encoder SHA unchanged                true
gold gradients                        0
gold early stopping                  false
gold checkpoint selection            false
additional label smoothing           0
robust loss                           none
```

Inference is frozen at:

```text
all recognized real MRI series
16 2.5D positions / series
224 x 224
ImageNet normalization
TTA center offsets [-1,0,1]
MRI only at test time
```

The code does not silently fall back to a different TTA policy. If the configured runtime budget cannot finish safely, it aborts rather than changing the frozen inference recipe.

## Submission governance

The hidden Kaggle score is independent evidence relative to the repeatedly reused 58-study gold development set. Record the public/private score without altering B17 after seeing it. Any subsequent training change must be a separately versioned experiment.
