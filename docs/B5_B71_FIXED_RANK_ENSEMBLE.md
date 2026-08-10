# Fixed B5 + B7.1 rank ensemble

> **Status — 2026-08-10:** **PREDECLARED / EVALUATION PENDING.**

## Purpose

B5 and B7.1 are materially different retained models:

- B5: report-aligned MRI representation followed by the unchanged nested frozen-feature PCA/logistic-regression probe;
- B7.1: end-to-end pathology-query MRI model trained directly from frozen B6 weak labels with full active-corpus coverage.

Their output probability scales are therefore not directly calibrated to one another. A fixed rank-space ensemble is used to avoid tuning calibration or blend weights on the 58-study development set.

## Frozen ensemble rule

For each of the 12 targets independently:

1. rank B5 predictions across the evaluation studies using average ranks for ties;
2. convert ranks to percentile ranks in `(0,1]`;
3. rank B7.1 predictions identically;
4. output exactly `0.5 * B5_rank + 0.5 * B7.1_rank`.

The same rule applies to all 12 targets.

No target-specific model selection is allowed. No raw-vs-rank search, no blend-weight search, no calibration fit, and no use of gold labels in constructing the ensemble predictions.

## Development caveat

The decision to test this ensemble follows observed B5/B7.1 development results, so its performance is an additional development estimate on the same 58 studies rather than independent validation. The ensemble must be evaluated once under this fixed rule and then recorded without tuning the weight from the result.
