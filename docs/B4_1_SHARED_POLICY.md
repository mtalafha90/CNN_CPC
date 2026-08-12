# B4.1 — Shared-policy frozen-SSL classical model

> **Status — 2026-08-12:** **COMPLETED / REJECTED.** B4.1 remains a historical selector-stability ablation. B13 is now the reused-gold development champion; completed B15 did not replace it.

B4.1 tested whether B4's target-wise policy selection was too noisy by forcing one common `(feature_mode, PCA components, C)` tuple per outer fold.

## Leakage contract

For each outer fold, one shared candidate was selected by inner macro AUC using non-outer gold, then separate target classifiers were refit on all non-outer gold and evaluated once on the untouched outer fold.

Grid:

```text
feature mode   all / prior
PCA components 4 / 8 / 12 / 16
logistic C     0.1 / 1.0
```

## Result

```text
B4.1 macro AUC       0.4847792672
95% CI              [0.4324147314,0.5371776207]
median(B4.1-B4)     -0.0292928028
95% paired CI       [-0.0739762338,+0.0160555905]
P(B4.1 > B4)         0.1084
```

Decision: **rejected**. One global downstream policy was too rigid for the heterogeneous targets.

## Current successor context

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 strongly improved frozen weak-v2 teacher agreement but not expert-gold macro AUC. This further argues against reopening gold-driven downstream selector searches.

Do not retune B4.1 policy grids or use target-wise winners from reused gold. Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).