# B21 — pre-resize crop optimization

> **Status — 2026-08-13:** IMPLEMENTED / NOT YET RUN. B20 remains the active working model.

## Why this experiment exists

The current B20 crop is applied in `CropFocusedVariableSeriesKneeDataset.__getitem__` after the parent dataset has already created 224x224 triplets. Therefore the historical B20 geometry is effectively:

```text
native MRI -> resize to 224 -> center crop to about 202 -> resize to 224
```

B21 fixes that ordering without changing the field-of-view fraction:

```text
native MRI -> center crop to 90% -> single resize to 224
```

The aim is not to assume that cropping must improve AUC. It is to remove a real preprocessing defect and test crop ordering as one controlled variable.

## Validation redesign

The 58 expert studies are no longer used to rank this development candidate. The existing frozen weak-v2 split is used instead:

```text
weak-v2 train studies     2497
weak-v2 holdout studies    623
report-group overlap         0
```

Every checkpoint scored on weak-v2 must exclude all 623 holdout UIDs from training. Because the historical B20 checkpoint was trained on all 3,120 active B6 studies, it is not a valid weak-v2 comparator.

Therefore the experiment contains two newly trained matched arms.

## Matched arms

### B20-v2 control

```text
encoder             frozen B16 report-aligned encoder
training studies    2497 weak-v2 train only
crop                90% centered
crop stage          after resize to 224, matching historical B20
training length     exactly 2 epochs
checkpoint choice   none; epoch 2 is predeclared
expert/gold use     none during development
```

### B21 candidate

Identical to the control except:

```text
crop stage          native MRI array before resize to 224
```

The control and candidate use the same initialization seed, training UIDs, B6 supervision, target-balancing derivation, all-series mapping, optimizer, augmentation, frozen encoder and fixed two-epoch schedule.

## Why two epochs are fixed in advance

B18 and B20 both selected epoch 2 in every outer fold of their cross-fitted epoch-selection audits. The new experiment therefore does not search epochs. Epoch 2 is the endpoint by protocol, which removes the expert-guided checkpoint-selection loop from this comparison.

## What is deliberately not changed

- no robust-loss experiment;
- no label smoothing;
- no transformer/aggregation change;
- no resolution change;
- no crop-fraction sweep;
- no ensembling;
- no expert-guided epoch selection;
- no occlusion-guided retraining.

These remain later hypotheses. B21 tests crop ordering first.

## Development comparison

After both fixed-E2 runs complete, evaluate them together on the same 623-study weak-v2 holdout. The evaluator reports each strict 12-target weak macro AUC and a paired study-bootstrap difference:

```text
candidate minus control
95% paired interval
bootstrap probability candidate > control
```

This surface measures agreement with the frozen B6 report teacher, not expert truth. A favorable weak-v2 result does not automatically replace B20.

## Promotion rule

```text
B20 remains the working model while B21 is under development.
```

If B21 is favorable on the predeclared weak-v2 comparison, freeze it first. Only then may one predeclared B20-vs-B21 expert acceptance check be performed. Until that happens, no gold result may be used to tune B21.

## Commands

Use the frozen artifacts already produced by the project:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
```

Train matched post-resize control:

```bash
python -m rsna_knee.b21_training \
  --help
```

The module defaults to the B21 candidate. The registered control entry point, once package wiring is installed, is `rsna-knee-b20-v2-control`; the candidate entry point is `rsna-knee-b21`.

Paired weak-v2 evaluation is implemented in:

```bash
python -m rsna_knee.b21_weak_eval --help
```

## Current model roles

```text
B17  fixed-epoch historical reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  controlled pre-resize crop candidate; not promoted
```
