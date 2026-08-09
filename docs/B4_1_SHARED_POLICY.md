# B4.1 — Shared-policy frozen-SSL classical model

B4.1 is a lower-variance follow-up to B4.

## Motivation

B4 freezes the competition-only SSL ConvNeXt encoder and trains low-capacity PCA + logistic-regression classifiers on cached gold-study features. Its first nested run improved the pooled point estimate relative to B1, but the target-wise hyperparameter choices were unstable across the three tiny inner folds.

B4.1 keeps the same frozen feature cache and changes only model selection:

- B4: 12 independent target-wise searches per outer fold.
- B4.1: one shared `(feature_mode, PCA components, C)` tuple selected per outer fold by inner **macro AUC**.

The 12 targets still have separate logistic-regression coefficients after the shared policy is selected.

## Leakage contract

For each outer fold:

1. The outer gold fold is untouched.
2. The remaining two gold folds are split into one selection-training fold and one inner-selection fold.
3. Every shared candidate is fitted on the selection-training fold and scored on the inner fold across all 12 targets.
4. The single candidate with the best inner macro AUC is chosen.
5. Twelve target-specific classifiers are refitted with that same policy on all non-outer gold studies.
6. Predictions are produced for the untouched outer fold.

Outer labels never influence policy selection.

## Representation

B4.1 reuses the B4 feature cache. The encoder remains:

- strong in-domain SSL ConvNeXt Tiny;
- trained only on competition training images;
- completely frozen for B4/B4.1;
- no external pretrained weights.

Each of six MRI streams contributes mean, standard-deviation, and max pooled 768-dimensional slice embeddings, for 2304 pooled values per stream plus explicit stream-presence indicators in the classifier design matrix.

## Default shared search grid

- feature mode: `all`, `prior`
- PCA components: `4, 8, 12, 16`
- logistic `C`: `0.1, 1.0`

This is only 16 shared candidates per outer fold, rather than 16 candidates independently selected for each of 12 targets.

## Run

```bash
rsna-knee-b4-shared \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_1_shared_ssl \
  --n-bootstrap 5000
```

Then compare against B1 with the standard OOF evaluator.
