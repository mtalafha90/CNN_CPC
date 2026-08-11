# Roadmap after B12.1

> **Status — 2026-08-11:** PREDECLARED DEVELOPMENT ROADMAP. This document limits the number and scope of major experiments after B12.1 so that repeated use of the 58-study gold development surface does not become uncontrolled local tuning.

## Current reference state

```text
B7.1 macro AUC        0.5644802945   retained benchmark
B12 macro AUC         0.5660915179   highest point estimate
B12 vs B7.1 paired    median +0.0023747526
95% paired CI         [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
B12.1                 active / pending
```

B12 is statistically tied with B7.1, not confirmed superior. B12.1 tests whether explicit hierarchical compression of each real MRI series improves the all-series representation.

## Governing rules

1. The 58 fully labelled studies are a **development/model-selection surface**, not pristine independent validation.
2. Primary model selection remains **global macro ROC AUC across 12 targets**.
3. Paired 5,000-replicate bootstrap comparisons remain required for model-to-model interpretation.
4. Do not construct target-specific winners from per-target AUCs.
5. Do not tune thresholds, series caps, routing rules, pooling heads, pseudo-label rules, or ensemble weights on the 58-study surface.
6. Do not choose checkpoints or epoch counts from gold performance.
7. B6 supervision, preprocessing and experiment-specific frozen controls must remain auditable.
8. The remaining number of major development experiments should stay small.

## Stage 1 — finish B12.1

B12.1 is the current experiment:

```text
16 slice tokens / real MRI series
        -> learned 8-head series attention pool
        -> 1 token / real series
K real-series tokens
        -> study Transformer
        -> pathology queries
```

It uses the exact frozen B12 surface:

```text
training studies        3120
supervised cells       14123
positive cells          6871
negative cells          7252
real MRI series        17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

B12.1 is compared against both B12 and B7.1 after four complete epochs.

## Stage 2 — conditional B12.2

B12.2 is **not automatic**.

### Run B12.2 only if

B12.1 remains competitive with or improves on B12 and therefore provides continued support for the all-series architecture branch.

### Skip B12.2 if

B12.1 is clearly worse than B12. In that case, close local aggregation refinement and move directly to B13.

### Scientific question

B12 and B12.1 both build one study representation that is ultimately shared across pathologies. B12.2 would test whether each pathology benefits from learning which acquired MRI series are relevant.

Conceptual structure:

```text
all real MRI series
      -> series representations
      -> pathology-conditioned series attention
      -> ACL query
      -> MCL query
      -> medial/lateral meniscus queries
      -> OA queries
      -> effusion/synovitis/Baker's/contusion/fracture queries
```

This is a **single global learned architecture**, not hand-coded pathology routing. The 58 gold cases must not be used to decide that one target should use B12 while another uses B12.1 or B12.2.

## Stage 3 — B13 stronger competition-only MRI SSL

B13 is the main remaining representation experiment and should be pursued after the B12 architecture branch is resolved.

### Motivation

Across the experiment history, better learned representations have been among the most productive changes. B13 therefore targets the encoder initialization rather than adding more weak-label heuristics or hard routing rules.

### Candidate objectives

The exact B13 recipe must be frozen before gold evaluation, but the intended family is:

```text
same-study cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
```

Potentially useful principles:

- positive pairs should come from different acquisitions of the same knee when scientifically appropriate;
- negatives must avoid accidental leakage or obvious same-study false negatives;
- all pretraining remains competition-only / permitted data only;
- zero gold labels enter SSL optimization;
- the retained post-B12 architecture should be initialized from B13 instead of B5 for the controlled downstream test.

B13 should be treated as a major experiment, not tuned target-by-target.

## Stage 4 — B14 scanner/protocol robustness, optional

B14 should only be run if diagnostics after B13 suggest that acquisition heterogeneity remains a meaningful limitation.

Candidate perturbations:

```text
intensity and contrast variation
resolution/downsampling perturbation
acquisition-quality degradation
metadata dropout
```

The goal is robustness to realistic scanner/protocol variation, not fixed physical rescaling. B10-style physical normalization should not be reintroduced as a target-specific hybrid based on the 58 gold studies.

If no clear robustness problem remains after B13, **skip B14**.

## Stage 5 — final model freeze

Once the major experiment ladder is complete:

1. select one **global** retained model based on the predeclared development metric and paired comparisons;
2. freeze architecture;
3. freeze preprocessing;
4. freeze series policy;
5. freeze TTA/inference policy;
6. freeze checkpoint-selection rule;
7. generate competition test predictions;
8. create the Kaggle submission.

Do not reopen local architecture or target-specific tuning after the final freeze unless an external independent result justifies a new experimental cycle.

## Stage 6 — independent competition signal

The hidden competition test/leaderboard result is the next genuinely independent signal available to this project. Development AUC improvements on the repeatedly reused 58-study set must not be treated as equivalent to leaderboard improvement.

## Compact decision tree

```text
B12.1 hierarchical series aggregation
   |
   |-- clearly worse than B12 -----------------------+
   |                                                  |
   |                                                  v
   |                                               B13 SSL
   |
   |-- competitive / better than B12
              |
              v
   B12.2 pathology-conditioned series attention
              |
              v
           B13 SSL
              |
              v
   B14 robustness [optional, diagnostics only]
              |
              v
        FINAL MODEL FREEZE
              |
              v
        KAGGLE SUBMISSION
```

## Experiments explicitly not planned

Unless new independent evidence appears, do not spend the remaining development budget on:

```text
blind extra epochs
B7.1/B12 target-wise winner models
B10 target-wise physical-normalization hybrids
B11/B11.1 pseudo-label threshold variants
manual pathology routing from gold AUCs
series-count caps selected on gold
pooling-head sweeps selected on gold
ensemble-weight searches selected on gold
```

The purpose of this roadmap is to finish the project with a small number of interpretable, globally controlled experiments and then obtain an independent competition signal.
