# Local Real-Data Training Runbook

> **Current stage — 2026-08-09:** B0-B4.3 and fixed ensemble evaluations are complete. **B5 image-report representation training has completed all four predefined epochs; the frozen B5 gold probe is now the current task.** See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for measured results.

This runbook is for the verified local workstation.

## Environment

```text
Conda environment: rsna-knee
GPU: NVIDIA RTX A4500 Laptop GPU
precision: bf16
one visible GPU
```

Verified paths:

```text
repo:      /media/talafha/Disk_1/CNN_CPC
data root: /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection
```

Start a terminal:

```bash
export REPO="/media/talafha/Disk_1/CNN_CPC"
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
conda activate rsna-knee
```

## Pull only between stages

```bash
git checkout main
git pull --ff-only origin main
python -m pip install -e .
```

Do not pull code into an already running training process.

## Verified data gates

These checks are already complete for the current local data unless data/DICOM/routing code changes:

```text
studies                 4,407
gold                       58
report-only              4,349
series                  24,371
train preflight         121/121 selected streams decoded
local test preflight     14/14 selected streams decoded
full audit           21,886/21,886 selected series decoded
DICOM files         732,554/732,556 decoded
selected series lost         0
```

## Completed experiment record

```text
B0 random                         0.4762536432
B1 strong SSL                    0.5030284974
B2 lower encoder LR              0.4993244663
B3 pathology-aware MIL           0.4944652486
B4 frozen SSL + classical        0.5137567459
B4.1 shared policy               0.4847792672
B4.2 grouped policies            0.4901328905
B4.3 two-way CV selector         0.4966083942
B1+B4 fixed raw 50:50            0.5050
B1+B4 fixed rank 50:50           0.5167
B5                               training complete; probe pending
```

B4 remains the best clean standalone point estimate. The B1+B4 rank ensemble is numerically higher but statistically tied with B4.

## Strong SSL checkpoint

The completed competition-only strong SSL checkpoint is:

```text
runs/ssl_strong/ssl_encoder.pt
```

Coverage:

```text
8 epochs
8,000 batches
24,000 study draws
~5.52 corpus passes
238,274 active 2.5D examples
```

## B5 representation training — complete

B5 used all 4,349 report-only competition studies and excluded all 58 gold studies.

Completed checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Training summary:

```text
epochs                  4
batches               4000
study draws          16000
active 2.5D examples 158886
loss          5.5204 -> 4.7049
image loss    3.0068 -> 2.8937
report NCE    4.6031 -> 3.2901
report cosine 0.8015 -> 0.5924
final encoder LR       1e-6
final head LR          1e-6
budget limited          false
```

All logged objectives improved monotonically. Do not assign a B5 AUC from the pretraining loss; the frozen gold probe below is the performance test.

## Current task: inspect B5 artifacts

```bash
cat runs/b5_report_ssl/policy.json
cat runs/b5_report_ssl/report_semantics.json
cat runs/b5_report_ssl/coverage.json
cat runs/b5_report_ssl/history.json
```

Check that:

- source is competition-only;
- no gold studies were used for B5 representation training;
- `completed_epochs` is 4;
- the encoder checkpoint exists;
- losses are finite;
- `budget_limited` is false for all completed epochs.

## Extract B5 frozen gold features

```bash
mkdir -p runs/b5_frozen_probe

rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --split train \
  --scope gold \
  --out runs/b5_frozen_probe/gold_features.npz
```

Sanity check:

```bash
python - <<'PY'
import numpy as np
p=np.load('runs/b5_frozen_probe/gold_features.npz', allow_pickle=False)
print('uids:', p['study_uids'].shape)
print('features:', p['features'].shape)
print('present:', p['present'].shape)
print('finite:', np.isfinite(p['features']).all())
PY
```

Expected:

```text
uids      (58,)
features  (58, 6, 2304)
present   (58, 6)
finite    True
```

## Run the unchanged original B4 probe on B5 features

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000

cat runs/b5_frozen_probe/evaluation.json
```

Do not switch to B4.1/B4.2/B4.3. The first B5 test must use the original B4 protocol to isolate the representation change.

## Compare B5 with B4

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b4_frozen_ssl/oof.csv \
  --compare-oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b4_vs_b5.json

cat runs/b4_vs_b5.json
```

Orientation:

```text
A = B4 image-only strong SSL representation
B = B5 image-report representation
```

Positive `median_difference` and `probability_b_better > 0.5` favor B5.

## Current files worth preserving

```text
runs/stage1_random/
runs/ssl_strong/
runs/stage1_ssl_strong/
runs/stage1_ssl_b2/
runs/stage1_ssl_b3/
runs/b4_frozen_ssl/
runs/b4_1_shared_ssl/
runs/b4_2_grouped_ssl/
runs/b4_3_crossval_ssl/
runs/fixed_ensembles/
runs/b5_report_ssl/
```

Do not delete completed OOF files; they are the audit trail for the experiment campaign.

## Warnings

The scikit-learn `penalty='l2'` FutureWarning seen in B4-family runs is non-fatal for current results. PyTorch Transformer/nested-tensor warnings are also optimization warnings unless accompanied by an actual failure.

## Interpretation rule

The same 58 gold studies have now informed multiple method decisions. Treat the current campaign as **model-selection CV**. Do not optimize B5 or ensemble weights post hoc from these same outer labels without declaring a new controlled experiment.
