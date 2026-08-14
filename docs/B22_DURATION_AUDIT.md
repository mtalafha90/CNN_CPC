# B22 — pre-resize crop training-duration audit

> **Status — 2026-08-14:** IMPLEMENTED / NOT YET RUN. B20 remains the active working model. B22 is exploratory and cannot promote itself from reused gold.

## Question

B21 tested the corrected pre-resize 90% crop at a fixed epoch-2 endpoint and failed the predeclared gold acceptance gate. B22 asks one narrower post-hoc question:

```text
Does the B21 pre-resize crop pipeline require more downstream training than B20?
```

B22 therefore changes **training duration only** relative to the full-data B21 recipe.

## Frozen training recipe

```text
initializer                    historical B16 report-aligned encoder
encoder                        frozen
B6-active training studies     3120
usable B6 cells               14123
positive / negative            6871 / 7252
eligible MRI series           17475
crop fraction                  0.90
crop stage                     native array before resize
normalization support          cropped native field
output resolution              224 x 224
training epochs                5
cosine scheduler horizon       5
expert evaluation in training  none
checkpoint selection training  none
```

Every epoch is saved:

```text
runs/b22_duration_audit/candidates/epoch_1.pt
runs/b22_duration_audit/candidates/epoch_2.pt
runs/b22_duration_audit/candidates/epoch_3.pt
runs/b22_duration_audit/candidates/epoch_4.pt
runs/b22_duration_audit/candidates/epoch_5.pt
```

B22 requires the completed B21 acceptance JSON and refuses to start unless it certifies that B21 consumed its one-look gold comparison and failed promotion.

## Why retrain from epoch 1

The B21 full-data checkpoint does not contain the optimizer/scaler/scheduler states needed to continue E3-E5 exactly. B22 therefore retrains the entire five-epoch trajectory from the same frozen initialization and seeds. This gives a coherent E1-E5 trajectory.

## E2 reproducibility safeguard

Before E3-E5 are interpreted, B22 E2 must reproduce the prior B21 E2 expert macro AUC within:

```text
absolute tolerance = 0.005
```

The prior B21 E2 macro AUC is:

```text
0.6573196516459231
```

If the newly retrained E2 differs by more than the tolerance, the duration audit aborts because the later epochs are no longer a clean extension of the B21 trajectory.

Historical B20 is also replayed and must satisfy its existing canonical replay tolerance.

## Gold trajectory role

After all five checkpoints are trained, one exploratory command evaluates E1-E5 together on the 58 reused expert studies.

This is **not** a new promotion gate. Gold has already been reused throughout historical development and was consumed once for the predeclared B21 acceptance test. Therefore:

```text
best B22 epoch on reused gold     exploratory only
per-target AUCs                   descriptive only
B20 working-model replacement     forbidden from B22 alone
additional B22 retuning           forbidden from this trajectory
```

The audit can answer whether later epochs recover relative to E2, but it cannot establish independent superiority.

## Run commands

Update/install first:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
```

Run focused tests:

```bash
pytest -q tests/test_b22_duration_protocol.py
```

Train the five-epoch trajectory:

```bash
rsna-knee-b22-duration \
  --config configs/b22_duration_audit.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --b21-acceptance runs/b21_full_acceptance/gold_acceptance/acceptance.json \
  --out-root runs/b22_duration_audit
```

Then run the single exploratory trajectory audit:

```bash
rsna-knee-b22-gold-audit \
  --config configs/b22_duration_audit.yaml \
  --data-root "$DATA_ROOT" \
  --b20-checkpoint runs/b20_crop_focus/b20_model.pt \
  --candidate-root runs/b22_duration_audit/candidates \
  --b21-acceptance runs/b21_full_acceptance/gold_acceptance/acceptance.json \
  --out-root runs/b22_duration_audit/gold_trajectory
```

Outputs:

```text
runs/b22_duration_audit/history.json
runs/b22_duration_audit/gold_trajectory/trajectory.json
runs/b22_duration_audit/gold_trajectory/trajectory_predictions.csv
```

## Interpretation plan

The primary diagnostic is the global macro-AUC trajectory:

```text
E1 -> E2 -> E3 -> E4 -> E5
```

If E3-E5 remain below E2, the pre-resize crop does not appear to benefit from longer downstream training under this recipe.

If one or more later epochs recover materially above E2, that supports a crop-by-duration interaction hypothesis, but because the finding is post-hoc on reused gold it must be validated independently before changing the working model.
