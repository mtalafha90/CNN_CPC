# Roadmap after B12.1

> **Status — 2026-08-11:** B14 is completed and rejected globally. B13 remains the retained development champion. The next major representation hypothesis is **B15: ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchical aggregation**.

## Current reference state

```text
B7.1 macro AUC        0.5644802945
B12 macro AUC         0.5660915179
B13 macro AUC         0.6293565948   retained development champion
B14 macro AUC         0.6197914249   completed / rejected globally

B13 vs B12
median difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
P(B13 > B12)            0.9920

B13 vs B7.1
median difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
P(B13 > B7.1)           0.9808

B14 vs B13
raw macro difference    -0.0095651699
median difference       -0.0093726931
95% paired CI           [-0.0469823411,+0.0250137870]
P(B14 > B13)             0.2924
```

## Governing rules

1. The 58 fully labelled studies are a development/model-selection surface, not pristine independent validation.
2. Primary model selection remains global macro ROC AUC across 12 targets.
3. Paired 5,000-replicate aligned bootstrap remains required for controlled local comparisons.
4. Do not construct target-specific winners from per-target AUCs.
5. Do not tune slice counts, series caps, thresholds, pooling heads, ImageNet variants, normalization, LR, epoch count or ensemble weights from the reused gold surface.
6. No gold labels enter gradients, early stopping or checkpoint selection.
7. Any B15 SSL stage must exclude all 58 gold studies from SSL optimization.
8. The independent Kaggle hidden-test/leaderboard remains more valuable than repeated local tuning.

## Completed B13

B13 combines:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
standard ImageNet mean/std normalization
16 slice tokens / series -> one learned series token
K series tokens -> study Transformer -> pathology queries
```

Frozen gold macro AUC: `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.

B13 remains the retained global model.

## Completed B14 — full slice-token memory

### Hypothesis tested

B13's one-token-per-series compression might discard focal slice-level information before pathology-specific attention can use it.

### Single change versus B13

```text
B13
16 slices / series -> 1 generic learned series token
K tokens -> study Transformer -> pathology queries

B14
NO per-series compression
K x 16 slice tokens -> study Transformer -> pathology queries
```

### Training result

B14 completed all four frozen epochs with exact study/series coverage.

```text
epoch 1 loss  0.7346330162
epoch 2 loss  0.6606430862
epoch 3 loss  0.6074723502
epoch 4 loss  0.5822778610
```

B14 final loss was lower than B13 (`0.5822778610` vs `0.6132239342`), but this stronger fit to B6 supervision did not improve the primary metric.

### Gold result

```text
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
```

Paired against B13:

```text
raw B14-B13        -0.0095651699
median difference  -0.0093726931
95% paired CI      [-0.0469823411,+0.0250137870]
P(B14 > B13)        0.2924
```

### B14 decision

The paired CI crosses zero, so the two models are statistically unresolved on the 58-study development surface. However, B14 has a lower global point estimate, only `0.2924` probability of superiority, higher token-memory cost and slower training.

```text
B14 -> REJECT GLOBALLY
B13 -> RETAIN
```

Do not run B14 epoch 5 and do not construct target-wise B13/B14 mixtures.

## Next major hypothesis — B15

### Motivation

B14 shows that increasing downstream token memory/capacity is not sufficient. The strongest remaining representation hypothesis is to adapt the successful ImageNet encoder to knee MRI before weakly supervised downstream training.

### Intended structure

```text
ImageNet ConvNeXt-Tiny
        |
        v
competition knee-MRI self-supervised adaptation
        |
        v
B13 one-token-per-series hierarchical architecture
        |
        v
frozen B6 weak-supervision training recipe
```

### Required safeguards

Before B15 implementation/training, freeze the SSL objective and data policy. At minimum:

```text
58 gold studies excluded from SSL optimization
no gold labels in SSL
no B6 labels in SSL
no report labels in SSL unless explicitly declared as a different experiment
no gold-based SSL checkpoint selection
no gold-based SSL hyperparameter sweep
same B13 downstream architecture unless separately predeclared
same B6 downstream supervision surface
same 17475-series downstream mapping
same 4 downstream epochs
same TTA [-1,0,1]
```

Candidate competition-only MRI SSL families to consider before freezing B15:

```text
same-study / cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
teacher-student self-distillation
```

Only one B15 protocol should be selected and frozen before gold evaluation.

## Later candidates

A larger foundation-encoder change can be reserved for **B16** if B15 is unsuccessful and development budget still justifies it.

Scanner/protocol robustness becomes **B17** and remains optional/diagnostic.

## Independent competition signal

The preferred competition path remains:

```text
retain B13 now
      |
      +--> independent test inference / Kaggle submission
      |
      +--> B15 only as one controlled major representation experiment
```

A leaderboard result is the next genuinely independent signal and should not be turned into a high-frequency tuning loop.

## Experiments explicitly not allowed from B14 gold

```text
B14 epoch extension
slice-count sweep
B13/B14 per-target winner mixture
B13/B14 ensemble-weight search
learning-rate sweep
ImageNet normalization/version sweep
series-count cap selected on gold
threshold tuning
```

The goal remains a higher macro AUC through interpretable global representation improvements, not uncontrolled optimization to 58 repeatedly reused cases.
