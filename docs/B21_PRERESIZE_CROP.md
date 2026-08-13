# B21 — pre-resize crop optimization

> **Status — 2026-08-13:** IMPLEMENTED / NOT YET RUN. B20 remains the active working model.

## Why this experiment exists

The historical B20 crop is applied in `CropFocusedVariableSeriesKneeDataset.__getitem__` after the parent dataset has already created 224x224 triplets. Its implemented geometry is therefore:

```text
native MRI -> resize 224 -> center crop about 202x202 -> resize 224
```

B21 fixes that ordering without changing the 90% field-of-view fraction:

```text
native MRI -> center crop 90% at native resolution -> single resize 224
```

This is a defect-fix experiment, not an assumption that cropping must improve AUC.

## Development-surface leakage audit

The initial plan was to reuse the historical B16 report-aligned encoder while moving model ranking to the frozen weak-v2 holdout. That would be invalid: historical B16 explicitly trained on all 4,349 non-gold MRI/report pairs, which includes the 623 weak-v2 holdout studies.

The frozen weak-v2 contract requires every model scored there to exclude every holdout StudyInstanceUID from training. Therefore historical B16 is not an eligible representation for this comparison.

B15 MRI SSL is eligible: its SSL pool already excluded both the 58 gold studies and all 623 weak-v2 holdout studies.

The corrected representation path is:

```text
B15 MRI SSL encoder
  -> NEW weak-v2-safe B16 report alignment
       excludes 58 gold studies
       excludes 623 weak-v2 holdout studies
       uses 3,726 remaining MRI/report studies
  -> frozen safe encoder
  -> matched B20-v2 control and B21 candidate
```

The safe report-alignment implementation is `rsna_knee.b16_v2_report_ssl` and produces:

```text
runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt
```

## Frozen weak-v2 split

```text
weak-v2 train studies     2497
weak-v2 holdout studies    623
report-group overlap         0
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The 623 holdout studies are excluded from safe report alignment and from both matched downstream training arms. The 58 expert studies are also excluded from development gradients and are not used for epoch/model selection.

The historical B20 checkpoint cannot be scored on weak-v2 because it trained on all 3,120 active B6 studies, including the weak-v2 holdout. A newly trained matched B20-v2 control is therefore required.

## Matched arms

### B20-v2 control

```text
encoder             frozen weak-v2-safe B16 report encoder
training studies    2497 weak-v2 train only
crop                90% centered
crop stage          after resize 224, matching historical B20 implementation
training length     exactly 2 epochs
scheduler horizon   5 epochs, matching historical B20 E1/E2 LR trajectory
checkpoint choice   none; E2 predeclared
weak-v2 holdout     not used in gradients
expert/gold use     none during development
```

### B21 candidate

Identical to the matched control except:

```text
crop stage          native MRI array before resize 224
```

The two arms use the same safe encoder, seeded hierarchy initialization, DataLoader seed, weak-train UIDs, B6 supervision, target-balancing derivation, all-series mapping, optimizer, augmentations, 90% crop fraction, 224 output resolution and fixed E2 endpoint.

The run stops after epoch 2 but keeps the historical B20 cosine-scheduler horizon `T_max=5`. This preserves B20's first-two-epoch learning-rate trajectory rather than introducing a new two-epoch schedule.

## Declared normalization-support difference

B21 has one additional, small preprocessing consequence that must be recorded before the first run.

`_normalise_volume` derives its intensity window from the 1st and 99th percentiles of the array it receives. Therefore the two matched arms do **not** derive that window from exactly the same pixel population:

```text
B20-v2 control:
full native volume -> percentile normalization -> resize 224 -> crop 90% -> resize 224

B21:
full native volume -> crop 90% -> percentile normalization on cropped volume -> resize 224
```

Accordingly, the first B21 experiment should be interpreted as a test of the corrected **pre-resize 90% crop pipeline**, whose intervention includes both:

1. moving the 90% field-of-view restriction upstream of the 224 resize; and
2. deriving the percentile normalization window from that cropped field of view.

At the frozen `0.90` crop fraction this normalization-support difference is expected to be small because only a thin border is removed and percentile normalization is robust. A separate knee-like phantom simulation supplied during review estimated roughly a `+1.58` low-percentile shift and about a `+1.25%` normalization-range change at `0.90`. These are planning diagnostics from synthetic phantoms, **not measurements on the competition dataset and not performance evidence**.

This does **not** block the first B21 run. It does mean that a future crop-fraction sweep must not reuse this implementation as if crop fraction were the only changing variable. Before testing `0.85`, `0.80`, `0.75`, `0.70`, or another crop fraction, preprocessing must be refactored so both arms derive the normalization window from the same full native volume, then apply the crop after normalization but before the final resize.

For the current B21-v1 experiment:

```text
crop fraction sweep under current normalization order   FORBIDDEN
current frozen 0.90 B21 run                            ALLOWED
normalization-support difference                        DECLARED INTERVENTION
```

## What is deliberately not changed

- no robust-loss experiment;
- no extra label smoothing;
- no transformer/aggregation change;
- no resolution change;
- no crop-fraction sweep;
- no ensembling;
- no expert-guided epoch selection;
- no occlusion-guided retraining.

These hypotheses are deferred until the crop-order experiment is resolved.

## Development comparison

After both fixed-E2 runs complete, evaluate them together on the same untouched 623-study weak-v2 holdout. The evaluator reports each strict 12-target weak macro AUC and a paired study-bootstrap comparison:

```text
candidate minus control
95% paired interval
bootstrap probability candidate > control
```

Weak-v2 measures agreement with frozen B6 report supervision, not expert truth. A favorable weak-v2 result does not automatically replace B20.

## Promotion rule

```text
B20 remains the working model while B21 is under development.
```

If B21 is favorable under the predeclared weak-v2 comparison, freeze it first. Only then may one predeclared expert B20-vs-B21 acceptance comparison be performed. No gold result may be used to tune B21 before that freeze.

## Pre-run gates

The following local artifacts must exist before the campaign starts:

```text
runs/b15_mri_ssl/b15_ssl_encoder.pt
runs/weak_holdout_v2/
runs/b6_report_labels_v121/
runs/b12_variable_series/audit/series_policy.json
```

The weak-v2-safe B16 report-alignment stage must complete **all four exact full-coverage passes**. `load_b16_v2_report_encoder` is intentionally strict: a truncated, budget-limited, or fewer-than-four-epoch checkpoint is invalid and the matched B20-v2/B21 runs must not start from it.

## Run order

### 1. Build the weak-v2-safe B16 report encoder

```bash
rsna-knee-b16-v2-report-ssl \
  --config configs/b16_full_report_alignment.yaml \
  --data-root "$DATA_ROOT" \
  --b15-ssl-checkpoint runs/b15_mri_ssl/b15_ssl_encoder.pt \
  --weak-holdout-root runs/weak_holdout_v2 \
  --out-root runs/b16_v2_safe_report/report_ssl
```

### 2. Train the matched historical-B20 preprocessing control

```bash
rsna-knee-b20-v2-control \
  --config configs/b21_preresize_crop.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --weak-holdout-root runs/weak_holdout_v2 \
  --report-ssl-checkpoint runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt \
  --out-root runs/b20_v2_control
```

### 3. Train B21

```bash
rsna-knee-b21 \
  --config configs/b21_preresize_crop.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --weak-holdout-root runs/weak_holdout_v2 \
  --report-ssl-checkpoint runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt \
  --out-root runs/b21_preresize_crop
```

### 4. Run the paired weak-v2 comparison

```bash
rsna-knee-b21-weak-eval \
  --config configs/b21_preresize_crop.yaml \
  --data-root "$DATA_ROOT" \
  --control-checkpoint runs/b20_v2_control/b20_v2_control.pt \
  --candidate-checkpoint runs/b21_preresize_crop/b21_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --weak-holdout-root runs/weak_holdout_v2 \
  --out-root runs/b21_preresize_crop/weak_v2_comparison
```

## Current model roles

```text
B17  fixed-epoch historical reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  controlled pre-resize crop candidate; not promoted
```
