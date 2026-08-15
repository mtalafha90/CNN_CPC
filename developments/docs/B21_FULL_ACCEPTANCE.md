# B21 full-data acceptance protocol

> **Status — 2026-08-14:** COMPLETE. Weak-v2 gate passed, full-data refit completed, and the single predeclared gold acceptance look was consumed. **B21 was not promoted. B20 remains the active working model.**

## Frozen weak-v2 result

The leakage-safe matched comparison on the frozen 623-study weak-v2 holdout favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21 pre-resize macro AUC        0.7410090411
raw B21 - control              +0.0111362500
paired median                  +0.0109814529
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
valid bootstrap reps           4894 / 5000
```

This was teacher-agreement evidence, not expert truth.

## Completed full-data refit

The frozen full-data B21 candidate used:

```text
initializer                    historical B16 report-aligned encoder
encoder                        frozen
B6-active training studies     3120
usable B6 cells               14123
positive / negative            6871 / 7252
eligible MRI series           17475
crop fraction                  0.90
crop stage                     native array before resize
output resolution              224 x 224
training endpoint              fixed epoch 2
cosine scheduler horizon       5 epochs
expert checkpoint selection    disabled
gold labels in gradients       0
```

Both epochs had exact full coverage. The encoder SHA remained
`b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96`.

Canonical checkpoint:

```text
runs/b21_full_acceptance/b21_full_model.pt
```

## Completed one-look gold acceptance

The historical B20 replay passed the predefined sanity guard:

```text
canonical B20 macro AUC         0.6671593555
replayed B20 macro AUC          0.6674066371
replay - canonical             +0.0002472815
allowed replay tolerance        0.005
```

The frozen B21 candidate scored:

```text
B21 global macro AUC            0.6573196516
B21 - canonical B20            -0.0098397039
B21 - replayed B20             -0.0100869854
```

Paired bootstrap comparison:

```text
median B21 - B20               -0.0095857726
95% CI               [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
valid bootstrap reps            5000 / 5000
```

Predeclared rules therefore give:

```text
promotion_rule_passed              false
scientific_superiority_supported   false
```

## Decision

**B21-v1 is rejected for promotion. B20 remains the active working model.**

The result also demonstrates an important development-surface mismatch: B21 improved agreement with the frozen B6 teacher on weak-v2 but did not improve the expert-gold global ranking. Weak-v2 should therefore not be treated as a reliable surrogate for expert-truth model selection for near-neighbor architecture/preprocessing changes.

Target-level AUCs from this one-look comparison are descriptive only and must not be used to build a target-wise B20/B21 mixture, retune crop fraction, or define a second B21-v1 gold-guided variant.

## Governance after completion

- Do not run another B21-v1 gold acceptance look.
- Do not promote B21-v1.
- Do not target-mix B20 and B21 from the 58-study result.
- Preserve `runs/b21_full_acceptance/gold_acceptance/acceptance.json` as the canonical acceptance artifact.
- Keep B20 unchanged as the working checkpoint.
- Future optimization should first address the development/validation-surface problem rather than continuing to optimize directly against weak-v2 teacher agreement.

The 58 expert studies were already reused during historical B20 development, so this remains a governance/development comparison rather than pristine independent validation. Hidden competition evaluation remains the independent predictive-performance signal.
