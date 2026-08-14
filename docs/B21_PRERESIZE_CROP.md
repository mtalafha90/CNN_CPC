# B21 — pre-resize crop optimization

> **Status — 2026-08-14:** COMPLETE. B21-v1 passed the leakage-safe weak-v2 development gate, was refit on the full B6 surface, and then **failed the single predeclared gold acceptance comparison. B21-v1 is not promoted; B20 remains the active working model.**

## Frozen intervention

```text
historical B20: native MRI -> resize 224 -> crop 90% -> resize 224
B21-v1:         native MRI -> crop 90% -> percentile normalization -> resize 224
```

The `0.90` crop was frozen. Because percentile normalization is computed after the raw crop in B21, normalization support was also part of the declared B21-v1 intervention. Crop-fraction sweeps under this implementation were forbidden.

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

## Weak-v2 development result

```text
B20-v2 control macro AUC        0.7298727911214620
B21-v1 macro AUC                0.7410090411495206
raw B21 - control              +0.0111362500280586
paired median                  +0.0109814528626144
paired 95% CI        [+0.0001624069723888,+0.0226346589736566]
P(B21 > control)                0.9758888434818145
valid bootstrap reps           4894 / 5000
```

This was positive paired evidence on the frozen B6 teacher-agreement surface, not expert-truth validation.

## Full-data acceptance result

B21 was then refit on the full 3,120-study B6 surface using the same historical B16 frozen encoder as B20, exact fixed-E2 training, 17,475 series, and no gold-guided checkpoint selection.

The one-look expert comparison produced:

```text
B20 canonical macro AUC         0.6671593555
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired median                  -0.0095857726
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

Predeclared decisions:

```text
promotion_rule_passed              false
scientific_superiority_supported   false
```

Therefore B21-v1 was rejected for promotion.

Canonical acceptance record: [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md).

## Follow-up duration audit

B22 subsequently retrained the same pre-resize formulation for five epochs to test whether B21 had simply been stopped too early. The E2 checkpoint reproduced B21 E2 within `+0.0001073` macro AUC and later epochs did not improve expert ranking:

```text
E1  0.6135270850
E2  0.6574269018  <- best
E3  0.6387456622
E4  0.6136783995
E5  0.6282683534
```

Thus longer downstream training did not rescue the pre-resize crop.

Canonical duration record: [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md).

## Final B21-v1 decision

```text
weak-v2 development gate       PASSED
full-data gold acceptance      FAILED
longer-duration rescue         NOT SUPPORTED
B21-v1 promotion               REJECTED
working model                  B20
```

The B21/B22 sequence is retained as a controlled negative result showing that improved agreement with the B6 teacher did not transfer to the reused expert-gold ranking for this preprocessing change.

## Canonical artifacts

```text
runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt
runs/b20_v2_control/b20_v2_control.pt
runs/b21_preresize_crop/b21_model.pt
runs/b21_preresize_crop/weak_v2_comparison/comparison.json
runs/b21_full_acceptance/b21_full_model.pt
runs/b21_full_acceptance/gold_acceptance/acceptance.json
runs/b22_duration_audit/gold_trajectory/trajectory.json
```
