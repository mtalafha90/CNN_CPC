# B4.1 — Shared-policy frozen-SSL classical model

B4.1 tested whether B4's target-wise policy selection was too noisy by forcing one common `(feature_mode, PCA components, C)` tuple per outer fold.

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Leakage contract

For each outer fold:

1. keep outer gold untouched;
2. split the two remaining gold folds into selection-train and inner-selection roles;
3. fit every shared candidate on selection-train;
4. choose one candidate by inner **macro AUC across all 12 targets**;
5. refit 12 separate target classifiers with that shared policy on all non-outer gold;
6. predict the untouched outer fold once.

Outer labels never influence policy selection.

## Representation and search

B4.1 reuses `runs/b4_frozen_ssl/gold_features.npz` from the frozen competition-only strong SSL encoder.

Grid:

```text
feature mode:   all, prior
PCA components: 4, 8, 12, 16
logistic C:     0.1, 1.0
```

## Reproduction

```bash
rsna-knee-b4-shared \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_1_shared_ssl \
  --n-bootstrap 5000
```

## Final result

```text
pooled macro AUC = 0.4847792672
95% CI           = [0.4324147314, 0.5371776207]
```

B4.1 also lost to B1 (`P(B4.1 > B1)=0.2948`). Directly against B4, using A=B4 and B=B4.1:

```text
paired median difference = -0.0292928028
95% CI                   = [-0.0739762338, +0.0160555905]
P(B4.1 > B4)             = 0.1084
```

## Decision

**Rejected.** One global downstream policy is too rigid for the 12 heterogeneous knee pathologies. B4's pathology-specific flexibility matters, even though its inner selectors are noisy.
