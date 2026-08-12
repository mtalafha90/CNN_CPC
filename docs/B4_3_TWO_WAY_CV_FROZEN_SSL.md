# B4.3 — Two-way cross-validated target-wise frozen-SSL classifiers

> **Status — 2026-08-12:** **COMPLETED / REJECTED.** B4.3 remains a historical downstream-selector ablation. B13 is now the reused-gold development champion; completed B15 did not replace it.

B4.3 kept one downstream policy per target but replaced B4's single tiny inner-fold selector with symmetric two-way cross-validation across the two non-outer folds.

## Selection protocol

For each untouched outer fold, every candidate was evaluated by swapping train/validation roles across the two non-outer folds, concatenating the held-out predictions, selecting one policy per target, refitting on all non-outer gold, and predicting the outer fold once.

Grid remained:

```text
feature mode   all / prior
PCA components 4 / 8 / 12 / 16
logistic C     0.1 / 1.0
```

## Result

```text
B4.3 macro AUC       0.4966083942
95% CI              [0.4419461475,0.5494817895]
median(B4.3-B4)     -0.0169979675
95% paired CI       [-0.0583054282,+0.0255299046]
P(B4.3 > B4)         0.2182
```

Decision: **rejected**. Increasing the policy-selection validation sample did not improve the frozen-feature candidate. Together B4.1-B4.3 closed further downstream selector redesign on the same 58 gold studies.

## Current successor context

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15's very large frozen weak-v2 gain did not improve global expert-gold AUC. Current development therefore focuses on auditing weak-supervision states rather than returning to gold-driven classifier selection.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).