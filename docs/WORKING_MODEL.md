# Active working model

> **Decision — 2026-08-13:** B20 remains the active working model. B21 is a controlled optimization candidate and does not replace B20 unless it passes the predeclared development and acceptance protocol.

## Active model

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
implemented geometry   native MRI -> resize 224 -> center crop 90% -> resize 224
cosine/vignette mask   no
encoder                frozen historical B16 report-aligned encoder
```

The crop-order audit found that B20's historical crop is applied after the normal 224x224 triplet resize. B20 is preserved unchanged so that its existing checkpoint remains a reproducible working baseline.

## Controlled optimization candidate

B21 fixes the crop ordering without changing the 90% field-of-view fraction:

```text
B20 historical: native MRI -> resize 224 -> crop 90% -> resize 224
B21 candidate:  native MRI -> crop 90% -> single resize 224
```

B21 development is deliberately moved away from repeated use of the 58 expert studies.

```text
weak-v2 train studies      2497
weak-v2 holdout studies     623
fixed training epochs         2
scheduler horizon              5  # preserves historical B20 E1/E2 LR trajectory
expert epoch selection        no
gold development looks        0
```

### Representation-leakage correction

The historical B16 report encoder cannot be used for weak-v2 model ranking because B16 trained on all 4,349 non-gold MRI/report pairs, including the 623 weak-v2 holdout studies. The optimization protocol therefore first builds a **weak-v2-safe B16 report encoder** from the already-safe B15 MRI-SSL checkpoint while excluding all 623 weak-v2 holdout UIDs and all 58 gold UIDs.

```text
B15 MRI SSL encoder
  -> weak-v2-safe B16 report alignment on 3,726 studies
  -> frozen safe encoder
  -> matched B20-v2 control and B21 candidate
```

The historical B20 checkpoint also cannot be scored fairly on weak-v2 because it was trained on all 3,120 active B6 studies, including the weak-v2 holdout. Therefore B21 is compared against a newly trained matched B20-v2 control using the same safe encoder, the same 2,497 weak-train studies and the same fixed two-epoch schedule. Crop ordering is the intended difference between those two development arms.

Canonical protocol: [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md).

## Validation state of the current working model

B20 nested epoch-selection audit:

```text
all-58 selected macro AUC           0.6671593555313430
cross-fitted selected epochs        [2,2,2]
cross-fitted OOF macro AUC          0.6671593555313430
measured epoch-selection optimism   0.0
strict selected epochs              [2,5,2]
strict OOF macro AUC                0.6351640998170208
fixed epoch-5 macro AUC             0.6577823350159498
```

B18 full-FOV comparator nested audit:

```text
historical selected statistic       0.6654496134246369
post-hoc replay epoch-2 macro AUC   0.6655517376076434
cross-fitted selected epochs        [2,2,2]
cross-fitted OOF macro AUC          0.6655517376076434
measured epoch-selection optimism   0.0
strict selected epochs              [2,5,2]
strict OOF macro AUC                0.6475369755138950
fixed epoch-5 / B17 endpoint        0.6425890152580378
```

The B20-vs-B18 cross-fitted difference is approximately `+0.0016076`, which is too small to claim predictive superiority on the repeatedly reused 58-study development surface.

## Model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  pre-resize crop optimization candidate; not promoted
```

## Governance

- Do not modify the historical B20 checkpoint or reinterpret its preprocessing after the fact.
- Do not use the historical B16 encoder for weak-v2 B21 ranking; it saw the weak-v2 holdout during report alignment.
- Build the weak-v2-safe B16 report encoder first and use that exact encoder for both matched development arms.
- B20-v2 control and B21 must use the same weak-train UIDs, initialization, optimizer, augmentation, frozen encoder, crop fraction, five-epoch LR horizon and fixed E2 endpoint.
- Robust losses, aggregation changes, ensembling, resolution increases and occlusion-guided retraining are deferred until the crop-order experiment is resolved.
- If B21 is favorable on weak-v2, freeze it before one predeclared expert acceptance comparison.
- Until that acceptance step succeeds, B20 remains the working model.
