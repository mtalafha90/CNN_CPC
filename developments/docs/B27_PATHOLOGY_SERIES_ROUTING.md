# B27 — pathology-specific acquisition routing

> **Status — 2026-08-16:** FIXED-E2 TRAINING COMPLETE / STRUCTURALLY SUPERSEDED BEFORE EXPERT EVALUATION. **B20 remains active.** B27 has not been evaluated on the reused 58-study expert surface.

## Motivation

B27 returned to the imaging side after the B26 supervision-repair branch closed. B20 already has 12 pathology query tokens that cross-attend to contextualised real-series memory, so B27 made a deliberately small change: an additive pathology-specific attention-logit bias from acquisition metadata.

```text
B20
series content + metadata embeddings
 -> shared study Transformer
 -> pathology-query cross attention

B27
same path
 + pathology-specific route bias from
   plane / fluid sensitivity / fat suppression
```

## Original B27 routing

For target `t` and series `k`:

```text
routing_bias(t,k)
  = plane_bias(t, plane_k)
  + fluid_bias(t, fluid_k)
  + fat_bias(t, fat_k)
```

The tables were zero-initialised and added only 84 trainable parameters:

```text
12 x (3 plane + 2 fluid + 2 fat) = 84
```

Unknown metadata received zero routing bias. With all routing tables at zero, B27 was functionally equivalent to B20.

## Frozen training contract

```text
training studies            3120
usable B6 cells            14123
positive / negative        6871 / 7252
eligible MRI series        17475
B16 encoder                 frozen
encoder SHA                 b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
B20 crop                    90% post-resize crop only
optimizer / LR              unchanged
augmentation                unchanged
loader seed                 unchanged
scheduler horizon           5 epochs
training endpoint           fixed E2
expert labels in gradients  0
expert checkpoint selection none
```

## Completed B27 training

The run completed the exact frozen surface in both epochs:

```text
E1 loss              0.7422715253
E1 runtime            1751.3 s
E1 routing max |b|    0.0136499135

E2 loss              0.6487928373
E2 runtime            1777.0 s
E2 routing max |b|    0.0165207200

full wall time        3534.2 s (~58.9 min)
```

Every epoch saw exactly 3,120 studies, 17,475 series and 14,123 supervision cells. The encoder SHA remained unchanged and no gold labels were used.

## Pre-outcome structural audit

Before running Ollama or any reused-expert diagnostic, the learned routing table showed that the fluid and fat tables were exactly identical target by target. A direct audit of the complete training metadata then confirmed perfect collinearity:

```text
series                         17475
fluid_id == fat_id             17475
fluid_id != fat_id                 0
fraction identical               1.0

pair (1,1)                     7459
pair (2,2)                    10016
```

Therefore the B27 training surface contains only one empirical sequence-metadata degree of freedom across those two fields. Because the two routing tables started at zero and saw identical categories on every training series, they received identical gradients and remained identical.

B27 consequently implemented, on this surface, approximately:

```text
plane + 2 x paired-sequence signal
```

rather than three independent metadata axes.

This is a structural identifiability issue, not a failed optimization run. The training itself is valid, but the 84-parameter routing representation is redundant for this dataset.

## Decision before outcome inspection

```text
B27 training                    valid
B27 reused-expert evaluation   NOT RUN
B27 Ollama route review        NOT RUN
B27 model promotion            not considered
B27 representation             structurally superseded
successor                      B27.1
```

The correction was made before observing B27 expert performance, so B27.1 is not an outcome-driven retune.

## B27.1 correction

B27.1 collapses the perfectly collinear fluid/fat terms into one paired-sequence term:

```text
B27.1 route = plane + paired_sequence
parameters  = 12 x (3 + 2) = 60
```

The paired categories are:

```text
1 = structural + not-fat-suppressed
2 = fluid-sensitive + fat-suppressed
```

Unknown or discordant future/test combinations receive zero paired-sequence routing bias because no such combination existed on the frozen training surface.

Canonical successor record:

```text
developments/docs/B27_1_COLLINEARITY_SAFE_ROUTING.md
```

## Runtime policy

Both B27 and B27.1 use a hard budget of at most 8.25 h and at least a 30-minute internal reserve, below the 9-hour competition ceiling. B27's observed E2 runtime of about 59 minutes confirms a large safety margin.

## Ollama governance

The local `qwen3:14b` model is audit-only. It is not used in MRI training or competition inference and cannot modify routing values, labels, thresholds, epochs or model selection.

## Evaluation governance

The historical 623-study weak-v2 partition is not a holdout for B27/B27.1 because both use all 3,120 B20 training studies. The 58 expert studies are heavily reused development data and selected historical B20 checkpoints. They remain post-hoc development evidence only.

Hidden competition evaluation remains the independent predictive-performance signal.
