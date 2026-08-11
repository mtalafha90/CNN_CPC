# Roadmap after B12.1

> **Status — 2026-08-11:** DEVELOPMENT REOPENED FOR ONE CONTROLLED HIGH-UPSIDE EXPERIMENT. B13 remains the retained development champion. **B14 ImageNet full slice-token aggregation is now active.**

## Current reference state

```text
B7.1 macro AUC        0.5644802945
B12 macro AUC         0.5660915179
B13 macro AUC         0.6293565948   retained development champion

B13 vs B12
median difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
P(B13 > B12)            0.9920

B13 vs B7.1
median difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
P(B13 > B7.1)           0.9808

B14                    implemented / active / pending
```

## Governing rules

1. The 58 fully labelled studies are a development/model-selection surface, not pristine independent validation.
2. Primary model selection remains global macro ROC AUC across 12 targets.
3. Paired 5,000-replicate aligned bootstrap remains required for B14 versus B13.
4. Do not construct target-specific winners from per-target AUCs.
5. Do not tune slice counts, series caps, thresholds, pooling heads, ImageNet variants, normalization, LR, epoch count or ensemble weights from the B14 gold result.
6. No gold labels enter gradients, early stopping or checkpoint selection.
7. B14 must complete the exact four-epoch study/series coverage contract before gold evaluation.
8. After B14, prefer an independent Kaggle signal unless there is a strong new global hypothesis.

## Completed B13

B13 combines:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
standard ImageNet mean/std normalization
16 slice tokens / series -> one learned series token
K series tokens -> study Transformer -> pathology queries
```

Frozen gold macro AUC: `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.

## Active B14 — full slice-token memory

### Hypothesis

B13's one-token-per-series compression may discard focal slice-level information before pathology-specific attention can use it.

### Single change versus B13

```text
B13
16 slices / series -> 1 generic learned series token
K tokens -> study Transformer -> pathology queries

B14
NO per-series compression
K x 16 slice tokens -> study Transformer -> pathology queries
```

B14 reuses the already tested B12 full-token architecture but uses B13's ImageNet encoder protocol.

### Frozen B14 controls

```text
same ImageNet ConvNeXt-Tiny weights
same ImageNet normalization
same 3120 studies
same 14123 supervised cells
same 6871 positive / 7252 negative cells
same 17475 real MRI series
same series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
same 16 positions / series
same 224x224 resize
same metadata embeddings
same seed and loader seed offsets
same batch size 2
same optimizer / LR / weight decay
same augmentation
same four epochs
same TTA [-1,0,1]
same 5000 bootstrap replicates
```

### Primary decision

The primary result is the paired global comparison `B14-B13`.

```text
clearly better -> B14 becomes retained champion
tied           -> retain both globally; do not target-wise mix
clearly worse  -> reject B14 and retain B13
```

## After B14

Do not automatically run another local experiment. First decide whether the B14 result materially changes the model direction.

If another major representation experiment is justified, the next candidate is **B15: ImageNet -> competition-MRI self-supervised adaptation -> frozen downstream recipe**. That experiment must be specified before touching gold and must exclude the 58 gold labels from SSL optimization.

A larger foundation-encoder change can be reserved for a later **B16** only if the development budget still justifies it. Scanner/protocol robustness becomes **B17** and remains optional/diagnostic.

## Independent competition signal

After B14 decision:

```text
retain one global model
      |
      v
freeze architecture / preprocessing / series policy / TTA
      |
      v
competition test inference
      |
      v
submission.csv
      |
      v
Kaggle leaderboard
```

The hidden test/leaderboard remains the next genuinely independent performance signal.

## Experiments explicitly not allowed from B14 gold

```text
8-epoch extension chosen because of B14 score
slice-count sweep
B13/B14 per-target winner mixture
B13/B14 ensemble-weight search
learning-rate sweep
ImageNet normalization/version sweep
series-count cap selected on gold
threshold tuning
```

The goal is a higher macro AUC through interpretable global representation changes, not uncontrolled optimization to 58 repeatedly reused cases.
