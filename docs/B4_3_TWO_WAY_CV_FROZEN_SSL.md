# B4.3 — Two-way cross-validated target-wise frozen-SSL classifiers

B4.3 kept one downstream policy per target but replaced B4's single tiny inner-fold selector with symmetric two-way cross-validation across the two non-outer folds.

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Selection protocol

For an untouched outer fold:

1. train a candidate on non-outer fold A and predict fold B;
2. train the same candidate on B and predict A;
3. concatenate both held-out prediction blocks;
4. compute target AUC over all non-outer studies (`~38-40` gold studies);
5. choose one policy per target by that cross-validated AUC;
6. refit the selected target classifier on all non-outer gold;
7. predict the outer fold once.

Outer labels are never used in policy selection.

## Representation and grid

B4.3 reuses `runs/b4_frozen_ssl/gold_features.npz`. The strong competition-only SSL ConvNeXt remains frozen.

```text
feature mode:   all, prior
PCA components: 4, 8, 12, 16
logistic C:     0.1, 1.0
```

No new hyperparameters were introduced.

## Reproduction

```bash
rsna-knee-b4-crossval \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_3_crossval_ssl \
  --n-bootstrap 5000
```

## Final result

```text
pooled macro AUC = 0.4966083942
95% CI           = [0.4419461475, 0.5494817895]
```

Against B4, using A=B4 and B=B4.3:

```text
paired median difference = -0.0169979675
95% CI                   = [-0.0583054282, +0.0255299046]
P(B4.3 > B4)             = 0.2182
```

B4.3 was also statistically tied/slightly below B1 (`P(B4.3 > B1)=0.425`).

## Decision

**Rejected.** Increasing the policy-selection validation sample did not improve the frozen-feature candidate. Together, B4.1-B4.3 show that further downstream selector redesign on these same 58 gold studies is unlikely to be productive and risks meta-overfitting the validation campaign.

The next experiment, B5, changes the representation instead while keeping the original B4 probe fixed.
