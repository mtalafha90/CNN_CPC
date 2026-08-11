# Roadmap after B12.1

> **Status — 2026-08-11:** UPDATED AFTER B13 COMPLETION. B13 is the retained development champion. B12.1 is skipped for the competition path, and the immediate next stage is B13-v1 freeze plus Kaggle submission. Additional local experiments are deferred until an independent competition signal or a clear technical diagnostic justifies reopening development.

## Current reference state

```text
B7.1 macro AUC        0.5644802945
B12 macro AUC         0.5660915179
B13 macro AUC         0.6293565948   new development champion

B13 vs B12
median difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
P(B13 > B12)            0.9920

B13 vs B7.1
median difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
P(B13 > B7.1)           0.9808

B12.1                 implemented / skipped for competition path
```

Both B13 paired confidence intervals are above zero on the repeatedly reused 58-study development surface.

## Governing rules

1. The 58 fully labelled studies are a **development/model-selection surface**, not pristine independent validation.
2. Primary model selection remains global macro ROC AUC across 12 targets.
3. Paired 5,000-replicate bootstrap comparisons remain required for interpretation.
4. Do not construct target-specific winners from per-target AUCs.
5. Do not tune thresholds, series caps, routing rules, pooling heads, pseudo-label rules, ImageNet variants, normalization variants, epoch counts or ensemble weights on the 58-study surface.
6. Do not choose checkpoints from gold performance.
7. Experiment-specific frozen controls must remain auditable.
8. Prefer an independent competition signal over further sequential local tuning.

## Completed B13

B13 uses the hierarchical learned series-token architecture with:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
+ standard ImageNet mean/std normalization
```

Frozen training surface:

```text
training studies        3120
supervised cells       14123
positive cells          6871
negative cells          7252
real MRI series        17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

All four epochs completed with exact full study and series coverage. Frozen gold macro AUC is `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.

## B12.1 decision — skip for competition workflow

B12.1 remains implemented and reproducible, but it will not be trained solely to complete the clean causal ablation.

This means the project cannot isolate:

```text
hierarchical architecture + B5 initialization
versus
hierarchical architecture + ImageNet protocol
```

Therefore do not claim that ImageNet alone caused the entire B13 gain.

Skipping B12.1 is an explicit tradeoff: preserving development budget and avoiding another sequential decision on the same 58 gold cases is considered more valuable for the competition than completing the ablation.

## Immediate stage — freeze B13-v1

Freeze:

```text
architecture
ImageNet encoder protocol
input normalization
B12 all-series mapping
B6 supervision-derived trained checkpoint
224x224 preprocessing
16 2.5D positions per series
metadata embeddings
TTA [-1,0,1]
checkpoint runs/b13_imagenet/b13_model.pt
```

Do not retrain B13-v1 for extra epochs and do not select an alternative checkpoint from the gold set.

## Next stage — Kaggle submission

The next high-value task is:

```text
freeze B13-v1
      |
      v
run competition test inference
      |
      v
validate submission schema and probabilities
      |
      v
create submission.csv
      |
      v
submit to Kaggle
      |
      v
use leaderboard as the next independent signal
```

The leaderboard result should be treated as more informative than another local variant on the repeatedly reused development set.

## Deferred experiments

### B12.2 — pathology-conditioned series attention

**Deferred.** Do not run before the first B13 competition submission.

If reopened later, it must remain one global learned architecture rather than target-wise routing chosen from gold results.

### B14 — stronger competition-only MRI SSL

**Deferred.** The previously planned stronger in-domain representation experiment remains available only if an independent competition result suggests representation quality is still limiting.

Candidate family, to be frozen before any future evaluation:

```text
same-study cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
```

### B15 — scanner/protocol robustness

**Deferred / optional.** Only reopen if diagnostics or independent competition evidence indicate acquisition/domain robustness is a real limitation.

Candidate perturbations:

```text
intensity and contrast variation
resolution/downsampling perturbation
acquisition-quality degradation
metadata dropout
```

## Decision after leaderboard result

After the first valid B13 submission:

- If leaderboard performance supports B13, keep the model frozen and prioritize reproducibility/final inference packaging.
- If leaderboard performance is unexpectedly weak, inspect technical/data-domain diagnostics first before opening a new model experiment.
- Reopen B14/B15 only with a clear hypothesis grounded in an independent signal.
- Do not use the leaderboard to create uncontrolled per-target or high-frequency tuning loops.

## Experiments explicitly not planned now

```text
B12.1 purely for completeness
blind extra B13 epochs
B13 learning-rate sweep
ImageNet weight/version sweep
normalization sweep
target-wise B7.1/B12/B13 mixtures
B10 target-wise physical-normalization hybrids
B11/B11.1 pseudo-label threshold variants
series-count caps selected on gold
pooling-head sweeps selected on gold
ensemble-weight searches selected on gold
```

## Compact current roadmap

```text
B13 RETAINED
     |
     v
FREEZE B13-v1
     |
     v
KAGGLE TEST INFERENCE
     |
     v
SUBMISSION / LEADERBOARD
     |
     +---- supported -> keep frozen / finalize
     |
     +---- unexpected weakness -> diagnose first
                                  |
                                  +-> B14/B15 only if justified
```

The purpose of this roadmap is to stop local optimization at a statistically resolved development improvement and obtain the next independent competition signal.
