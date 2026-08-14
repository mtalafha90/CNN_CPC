# Active working model

> **Decision — 2026-08-14:** **B20 remains the active working model.** B21-v1 passed the frozen weak-v2 development gate but failed the predeclared full-data gold acceptance comparison and is not promoted.

## Active model

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
implemented geometry   native MRI -> resize 224 -> center crop 90% -> resize 224
cosine/vignette mask   no
encoder                frozen historical B16 report-aligned encoder
canonical gold score   0.667159355531343
```

Historical B20 is preserved unchanged.

## B21-v1 completed result

B21 changed the spatial ordering to:

```text
native MRI -> center crop 90% -> percentile normalization -> single resize 224
```

The leakage-safe weak-v2 development comparison favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21 pre-resize macro AUC        0.7410090411
raw B21 - control              +0.0111362500
paired 95% CI        [+0.0001624070,+0.0226346590]
```

B21 was then refit on the full 3,120-study B6 surface using the same historical B16 frozen encoder as B20, exact fixed E2 endpoint, 17,475 series, and no gold-guided checkpoint selection.

The single predeclared gold acceptance result was:

```text
B20 canonical macro AUC         0.6671593555
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired median                  -0.0095857726
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

The B20 replay sanity check passed (`+0.0002473` from canonical, tolerance `0.005`).

Predeclared decisions:

```text
promotion_rule_passed              false
scientific_superiority_supported   false
```

Therefore **B21-v1 is not promoted**.

Canonical acceptance record: [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md).

## Interpretation

The B21 experiment demonstrates a meaningful mismatch between the frozen weak-v2 teacher-agreement surface and expert-gold ranking for a near-neighbor preprocessing change. A positive weak-v2 development result did not transfer to the expert-gold global metric.

Accordingly, future optimization should not simply continue to maximize weak-v2 AUC and assume expert performance will follow. The next modelling campaign should first improve or replace the development-selection strategy.

Target-level B20/B21 results from the consumed gold look are descriptive only and must not be used for target mixing, crop-fraction retuning, or a second B21-v1 variant.

## Historical B20/B18 audit context

```text
B20 cross-fitted epoch selections       [2,2,2]
B20 cross-fitted OOF macro AUC          0.6671593555313430
B20 measured epoch-selection optimism   0.0

B18 cross-fitted epoch selections       [2,2,2]
B18 replay OOF macro AUC                0.6655517376076434
B18 measured epoch-selection optimism   0.0
```

The B20-vs-B18 difference remains too small to establish predictive superiority on the repeatedly reused 58-study gold surface.

## Model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  weak-v2-passed but gold-acceptance-failed candidate; NOT PROMOTED
```

## Governance

- Keep historical B20 unchanged.
- Do not run another B21-v1 gold acceptance look.
- Do not build a target-wise B20/B21 mixture from the 58 expert studies.
- Do not reopen B21-v1 crop fraction, normalization order, loss, architecture, aggregation, or resolution based on the consumed gold result.
- Do not treat weak-v2 teacher agreement as a sufficient proxy for expert truth for future near-neighbor model selection.
- Preserve the B21 artifacts as a negative controlled experiment.
- Hidden competition evaluation remains the independent predictive-performance signal.
