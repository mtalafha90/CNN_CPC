# Local Real-Data Training Runbook

> **Current stage — 2026-08-10:** package `0.13.0`. **B7.1 full-corpus weak supervision is the current best standalone development model at macro AUC `0.5644802945`. The fixed B5+B7.1 rank ensemble is rejected. B8 spatial-anatomy learning is currently training.** See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for measured results.

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

These checks are complete for the current local data unless data/DICOM/routing code changes:

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
B1+B4 fixed rank                 0.5167
B5 image-report SSL              0.5243650851
B7-v1 weak supervision           0.5397724412
B7.1 full coverage               0.5644802945   CURRENT LEADER
B5+B7.1 fixed rank ensemble      0.5540141184   REJECTED
B8 spatial anatomy               pending        TRAINING
```

## Strong SSL checkpoint

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

## B5 representation checkpoint

```text
runs/b5_report_ssl/b5_encoder.pt
```

B5 development result:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
```

B5 is retained as the report-aligned representation baseline and initialization source for B7.

## B6 frozen weak-label artifacts

```text
runs/b6_report_labels_v121/
├── training_targets.csv
├── policy.json
└── audit.json
```

Frozen training scope:

```text
report-only rows             4349
active studies               3120
usable cells                14123
positive cells               6871
negative cells               7252
```

Do not patch the B6 parser from B7/B8 gold outcomes.

## B7-v1 reference

Checkpoint:

```text
runs/b7_weak_supervision/b7_model.pt
```

Result:

```text
macro AUC = 0.5397724412
```

Its 500-batch epoch cap yielded only about 1.28 nominal corpus passes.

## B7.1 full coverage — retained leader

Checkpoint:

```text
runs/b7_1_full_coverage/b7_model.pt
```

Training completed four full passes:

```text
epochs                 4
batches/epoch       1560
study draws/epoch   3120
active cells/epoch 14123
positive            6871
negative            7252
loss 0.752419 -> 0.612758
budget limited      false
```

Gold development result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

Paired B7-v1 -> B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

## Fixed B5+B7.1 rank ensemble — rejected

```text
ensemble AUC        0.5540141184
B7.1 AUC            0.5644802945
P(ensemble > B7.1)  0.3054
```

Do not search other blend weights, raw averages or target-specific mixtures.

## Current task: B8 spatial-anatomy training

B8 initializes from B7.1 and preserves 2x2 within-slice ConvNeXt spatial tokens:

```text
B7.1 MRI memory = 96 tokens/study
B8 MRI memory   = 384 tokens/study
```

B8 keeps the B6 weak-label policy, target balancing, full 3,120-study epoch coverage, four epochs and learning rates unchanged.

Training command:

```bash
rsna-knee-b8 \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --b71-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b8_spatial_anatomy
```

Expected outputs:

```text
runs/b8_spatial_anatomy/
├── b8_model.pt
├── history.json
├── policy.json
└── supervision_plan.json
```

The checkpoint is saved after every completed epoch.

## When B8 training finishes

Inspect first:

```bash
cat runs/b8_spatial_anatomy/history.json
cat runs/b8_spatial_anatomy/supervision_plan.json
```

Expected for every complete full epoch:

```text
batches                         1560
study draws                     3120
active supervision cells       14123
positive cells                  6871
negative cells                  7252
```

Confirm that the loss is finite and that no epoch was unexpectedly budget-limited.

Do **not** run the B8 gold evaluation until these artifacts have been inspected.

## B8 gold evaluation — only after artifact inspection

```bash
rsna-knee-b8-eval \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b8_spatial_anatomy/b8_model.pt \
  --out-root runs/b8_spatial_anatomy/gold_eval
```

If DataLoader shutdown is noisy, a runtime-only config with `num_workers: 0` and `persistent_workers: false` may be used for evaluation without changing the scientific experiment.

Primary benchmark:

```text
B7.1 = 0.5644802945
```

Primary statistical comparison: paired B7.1 -> B8 bootstrap with 5,000 study-level replicates.

## Current files worth preserving

```text
runs/stage1_random/
runs/ssl_strong/
runs/stage1_ssl_strong/
runs/stage1_ssl_b2/
runs/stage1_ssl_b3/
runs/b4_frozen_ssl/
runs/b5_report_ssl/
runs/b5_frozen_probe/
runs/b6_report_labels_v121/
runs/b7_weak_supervision/
runs/b7_1_full_coverage/
runs/b8_spatial_anatomy/
```

Do not delete completed prediction/OOF/evaluation files; they are the experiment audit trail.

## Warnings

PyTorch Transformer nested-tensor warnings are optimization warnings unless accompanied by an actual failure. A DataLoader worker teardown error after the final checkpoint has already been written should be distinguished from an optimization-time failure by checking `history.json` and the saved checkpoint.

## Interpretation rule

The same 58 gold studies have informed multiple method decisions. Treat the campaign as **model-selection CV**. Do not tune B8 spatial grid size, anatomy-prior strength, target-specific priors, epochs, weak-label weights or ensemble weights post hoc from the first B8 gold result without declaring a new development experiment.
