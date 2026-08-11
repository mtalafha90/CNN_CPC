# Roadmap after B12.1

> **Status — 2026-08-11:** UPDATED DEVELOPMENT ROADMAP. B13 is now the standalone ImageNet encoder-protocol experiment. The previously planned stronger competition-only SSL experiment moves to B14; optional scanner/protocol robustness moves to B15.

## Current reference state

```text
B7.1 macro AUC        0.5644802945   retained benchmark
B12 macro AUC         0.5660915179   highest point estimate
B12 vs B7.1 paired    median +0.0023747526
95% paired CI         [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
B12.1                 implemented / pending
B13                   implemented / training ready
```

## Governing rules

1. The 58 fully labelled studies are a development/model-selection surface, not pristine independent validation.
2. Primary model selection remains global macro ROC AUC across 12 targets.
3. Paired 5,000-replicate bootstrap comparisons remain required.
4. Do not construct target-specific winners from per-target AUCs.
5. Do not tune thresholds, series caps, routing rules, pooling heads, pseudo-label rules, normalization variants or ensemble weights on the 58-study surface.
6. Do not choose checkpoints or epoch counts from gold performance.
7. Experiment-specific frozen controls must remain auditable.
8. Keep the remaining number of major development experiments small.

## Stage 1 — B12.1 hierarchical aggregation

B12.1 tests:

```text
16 slice tokens / real MRI series
        -> learned 8-head series attention pool
        -> 1 token / real series
K real-series tokens
        -> study Transformer
        -> pathology queries
```

It uses the exact frozen B12 series surface and B5 competition-only encoder initialization.

## Stage 2 — B13 ImageNet encoder protocol

B13 reuses the exact B12.1 hierarchy but replaces the encoder protocol:

```text
B12.1
B5 competition-only SSL encoder

B13
torchvision ConvNeXt-Tiny IMAGENET1K_V1
+ standard ImageNet mean/std normalization
```

B13 has separate trainer/evaluator/checkpoint identities. It keeps the same 3,120 studies, 14,123 supervised cells, 17,475-series mapping, optimizer, LR, augmentation, four epochs and TTA.

The primary comparison is B13 versus B12.1 once both prediction files exist. B13 may be trained before B12.1 finishes; interpretation waits for the paired comparison.

## Stage 3 — conditional B12.2

B12.2 remains optional. Run it only if B12.1/B13 results still support further all-series architectural refinement.

Scientific question:

```text
all real MRI series
      -> series representations
      -> pathology-conditioned series attention
      -> pathology queries
```

This must remain one global learned architecture, not hand-coded target-specific routing chosen from gold results.

## Stage 4 — B14 stronger competition-only MRI SSL

The previously planned stronger in-domain representation experiment is now **B14** so B13 remains unambiguous.

Candidate objective family, to be frozen before evaluation:

```text
same-study cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
```

B14 remains competition-only and must use zero gold labels in SSL optimization.

## Stage 5 — B15 scanner/protocol robustness, optional

Only run B15 if diagnostics indicate a remaining acquisition/domain robustness problem. Candidate perturbations:

```text
intensity and contrast variation
resolution/downsampling perturbation
acquisition-quality degradation
metadata dropout
```

Do not reintroduce target-wise B10 physical-normalization hybrids selected on gold.

## Stage 6 — final model freeze

Once the major experiment ladder is complete:

1. select one global retained model using the predeclared development metric and paired comparisons;
2. freeze architecture;
3. freeze preprocessing and encoder protocol;
4. freeze series policy;
5. freeze TTA/inference policy;
6. freeze checkpoint-selection rule;
7. generate competition test predictions;
8. create the Kaggle submission.

## Compact decision tree

```text
B12.1 hierarchical aggregation
       |
       +-----------------------------+
       |                             |
       v                             v
B13 ImageNet protocol         B12.2 [conditional]
       |                             |
       +-------------+---------------+
                     v
          B14 stronger in-domain SSL
                     |
                     v
          B15 robustness [optional]
                     |
                     v
              FINAL MODEL FREEZE
                     |
                     v
              KAGGLE SUBMISSION
```

## Experiments explicitly not planned

```text
blind extra epochs
target-wise B7.1/B12/B12.1/B13 winner mixtures
ImageNet-weight or normalization sweeps selected on gold
B10 target-wise physical-normalization hybrids
B11/B11.1 pseudo-label threshold variants
series-count caps selected on gold
pooling-head sweeps selected on gold
ensemble-weight searches selected on gold
```

The purpose of this roadmap is to finish with a small number of interpretable, globally controlled experiments and then obtain an independent competition signal.
