# Fixed B5 + B7.1 rank ensemble

> **Status — 2026-08-10:** **COMPLETE / REJECTED.**

## Purpose

B5 and B7.1 are materially different retained models:

- B5: report-aligned MRI representation followed by the unchanged nested frozen-feature PCA/logistic-regression probe;
- B7.1: end-to-end pathology-query MRI model trained directly from frozen B6 weak labels with full active-corpus coverage.

Their output probability scales are therefore not directly calibrated to one another. A fixed rank-space ensemble was used to avoid tuning calibration or blend weights on the 58-study development set.

## Frozen ensemble rule

For each of the 12 targets independently:

1. rank B5 predictions across the evaluation studies using average ranks for ties;
2. convert ranks to percentile ranks in `(0,1]`;
3. rank B7.1 predictions identically;
4. output exactly `0.5 * B5_rank + 0.5 * B7.1_rank`.

The same rule applies to all 12 targets.

No target-specific model selection was allowed. No raw-vs-rank search, no blend-weight search, no calibration fit, and no use of gold labels in constructing the ensemble predictions.

## Completed result

The fixed ensemble produced:

```text
macro AUC = 0.5540141184
95% CI   = [0.4959089903, 0.6097647325]
n         = 58
bootstrap = 5000/5000 usable
```

Per-target AUC:

```text
ACL               0.6390931373
MCL               0.3945578231
Medial Meniscus   0.6658653846
Lateral Meniscus  0.6453416149
Medial OA         0.5751937984
Lateral OA        0.4980657640
PF OA             0.6241956242
Effusion          0.5819875776
Synovitis         0.6302270012
Baker's           0.4556159420
Contusion         0.4581646424
Fracture          0.4798611111
```

The retained B7.1 standalone model remains higher:

```text
B7.1 macro AUC        0.5644802945
rank ensemble AUC     0.5540141184
point difference      -0.0104661761
```

Paired comparison using A=B7.1 and B=the fixed rank ensemble:

```text
median(ensemble - B7.1) = -0.0105429030
95% paired CI           = [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)      = 0.3054
valid replicates        = 5000/5000
```

## Decision

Reject the fixed B5+B7.1 rank ensemble as the campaign leader. Retain B7.1 alone as the main standalone development model.

Do not use this result to search 60:40, 70:30, target-specific mixtures, raw probability blends, calibration transforms, or other ensemble weights on the same 58 studies. The predeclared ensemble question is closed.

## Development caveat

The decision to test this ensemble followed observed B5/B7.1 development results, so its performance is an additional development estimate on the same 58 studies rather than independent validation.
