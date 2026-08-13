# B21 — pre-resize crop optimization

> **Status — 2026-08-14:** COMPLETED WEAK-V2 DEVELOPMENT EXPERIMENT / PASSED FROZEN PAIRED GATE. B20 remains the active working model. B21-v1 is frozen for the full-data acceptance step.

## Frozen intervention

```text
historical B20: native MRI -> resize 224 -> crop 90% -> resize 224
B21-v1:         native MRI -> crop 90% -> percentile normalization -> resize 224
```

The `0.90` crop is frozen. Because percentile normalization is computed after the raw crop in B21, normalization support is also part of the declared B21-v1 intervention. Crop-fraction sweeps under this implementation are forbidden.

## Leakage-safe development design

Historical B16 saw the weak-v2 holdout during report alignment and therefore could not be used for weak-v2 ranking. The development experiment used:

```text
B15 MRI SSL
 -> weak-v2-safe B16 report alignment on 3,726 studies
    excludes all 623 weak-v2 holdout studies
    excludes all 58 gold studies
 -> same frozen safe encoder for both downstream arms
```

The matched B20-v2 and B21-v1 arms used the same 2,497 weak-v2 training studies, 11,248 usable cells, 13,974 series, seeded hierarchy, optimizer, augmentation, fixed epoch-2 endpoint and five-epoch scheduler horizon. Gold development use was zero.

## Completed result

```text
B20-v2 control macro AUC        0.7298727911214620
B21-v1 macro AUC                0.7410090411495206
raw B21 - control              +0.0111362500280586
paired median                  +0.0109814528626144
paired 95% CI        [+0.0001624069723888,+0.0226346589736566]
P(B21 > control)                0.9758888434818145
valid bootstrap reps           4894 / 5000
```

This is positive paired evidence on the frozen B6 teacher-agreement surface, not expert-truth validation. Target-wise results are descriptive only and cannot be used to retune B21-v1.

## Frozen decision

```text
weak-v2 gate                 PASSED
B21-v1 preprocessing         FROZEN
second weak-v2 tuning round  FORBIDDEN
working model                B20
```

The only permitted next step is the full-data refit plus one-look expert acceptance in [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md).

Canonical completed development artifacts:

```text
runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt
runs/b20_v2_control/b20_v2_control.pt
runs/b21_preresize_crop/b21_model.pt
runs/b21_preresize_crop/weak_v2_comparison/comparison.json
```
