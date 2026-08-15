# Roadmap after B12.1

> **Status — 2026-08-12:** B13 remains the retained reused-gold development champion. B14 was rejected globally. B15 completed the planned ImageNet -> knee-MRI SSL -> B13 hierarchy experiment, passed the frozen weak-v2 gate decisively, but did not improve global reused-gold macro AUC. The next stage is a B6 report-state audit before any new supervision experiment.

## Current reference state

```text
B7.1 gold macro AUC       0.5644802945
B12 gold macro AUC        0.5660915179
B13 gold macro AUC        0.6293565948   RETAINED / CHAMPION
B14 gold macro AUC        0.6197914249   REJECTED GLOBALLY
B15 gold macro AUC        0.6209002783   NO GLOBAL IMPROVEMENT

B15 weak-v2               0.7319060415
B13-v2 control weak-v2    0.5652498118
paired weak median        +0.1675245839
95% paired weak CI        [+0.1124433208,+0.2165156305]
P(B15 > control)           1.0000
```

## Governing rules

1. The 58 fully labelled studies are a repeatedly reused development/model-selection surface, not independent validation.
2. Primary selection remains global macro ROC AUC across 12 targets.
3. Do not construct target-specific winners from per-target AUCs.
4. Do not tune slice counts, thresholds, normalization, LR, epoch count or ensemble weights from reused gold.
5. No gold labels enter gradients, early stopping or checkpoint selection.
6. Weak-v2 is frozen and measures B6 teacher agreement, not expert truth.
7. Any model scored on weak-v2 must exclude every weak-v2 holdout UID from training.
8. Weak-v2 bootstrap is strict: accepted replicates define all 12 target AUCs.
9. Do not retune B15 from its one-look gold confirmation.
10. The hidden Kaggle evaluation remains the next genuinely independent performance signal.

## Completed B13 / B14

```text
B13
ImageNet ConvNeXt-Tiny
one learned token per series
gold macro AUC 0.6293565948
-> RETAIN

B14
same ImageNet protocol
full K x 16 slice-token memory
gold macro AUC 0.6197914249
-> REJECT GLOBALLY
```

B14 reached lower B6 training loss (`0.5822778610`) than B13 (`0.6132239342`) without improving macro AUC. This closed the immediate “more downstream token memory” branch.

## Completed diagnostic — exact B13 slice exposure

The corrected audit reproduced the actual B13 2.5D sampler on all 17,475 eligible non-gold series:

```text
series audited/readable  17475 / 17475
slices/series median     30 (p95 50, max 320)
eval unique fraction     median 100.0%
complete eval exposure   95.9%
eval max skipped run     median 0.0 slices (p95 0.0)
```

Decision:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

In-plane resolution remains a separate possible later question.

## Frozen weak holdout v2

Historical v1 is superseded; no B15/control model was trained on it.

Actual frozen v2:

```text
surface                   weak_b6_holdout_v2
active studies            3120
train studies             2497
holdout studies            623
holdout usable cells      2875
positive / negative    1407 / 1468
report-group overlap         0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

It was frozen before B15/control training and is not eligible for regeneration after seeing results.

## Completed B15 — MRI-domain SSL

B15 tested:

```text
ImageNet ConvNeXt-Tiny
        |
        v
competition knee-MRI same-study contrastive adaptation
        |
        v
B13 one-token-per-series hierarchy
        |
        v
frozen B6 downstream recipe on v2 weak-train studies
```

SSL data contract:

```text
competition studies     4407
minus gold                58
minus v2 holdout         623
SSL studies             3726
SSL real MRI series    20534
```

All four SSL epochs completed exact full passes. Loss decreased `2.70946 -> 2.47569`. Checkpoint selection remained the frozen final epoch; loss was not used for model selection.

## Matched downstream arms

Both newly trained downstream models used:

```text
2497 studies
13974 real MRI series
11248 B6 usable cells
5464 positive / 5784 negative
1249 batches/epoch
4 exact full epochs
```

Control:

```text
ImageNet -> B13 hierarchy
```

Candidate:

```text
ImageNet -> MRI SSL -> B13 hierarchy
```

B13-v2 control final loss: `0.6622741637`.  
B15 final loss: `0.6065262400`.  
Training loss was not a selection metric.

## B15 weak-v2 gate — completed

```text
B13-v2 control macro AUC   0.5652498118
B15 macro AUC              0.7319060415
raw B15-control           +0.1666562297
paired median             +0.1675245839
95% paired CI             [+0.1124433208,+0.2165156305]
P(B15 > control)           1.0000
valid paired replicates    4921 / 5000
passes predeclared gate    true
```

The predeclared gate required positive raw delta, positive paired median and `P>=0.95`; all conditions passed.

## B15 one-look reused-gold confirmation — completed

```text
B15 gold macro AUC      0.6209002783
95% CI                 [0.5706720829,0.6675892903]
B13 gold macro AUC      0.6293565948
raw B15-B13            -0.0084563164
```

The very large weak-v2 improvement did not transfer to a global expert-gold improvement. B13 therefore remains the development champion. B15 is closed as a tuning target.

## Main lesson after B15

The campaign has now tested several ways of improving the image side:

- stronger competition-only MRI SSL;
- report-aligned representation learning;
- more weak-training coverage;
- spatial tokens;
- exact routing;
- physical scaling;
- all-series modeling;
- hierarchical series aggregation;
- ImageNet initialization;
- full slice-token memory;
- MRI-domain contrastive adaptation.

B15 shows that the image representation can become much better at reproducing the frozen report-derived weak targets without improving expert-gold macro AUC. This elevates **supervision semantics/noise/sparsity** as the next bottleneck to investigate directly.

## Immediate next step — B6 report-state audit

Before another GPU training run, audit these parser states against expert truth on the already-reused gold cases:

```text
positive
negated
uncertain
unmentioned
```

For each target/state compute counts, expert-positive and expert-negative fractions, coverage, and appropriate predictive values.

The audit should answer whether ignored states contain reliable information and whether that relationship is global or strongly target-dependent.

Do **not** blindly interpret report silence as a negative. A report can omit a finding without explicitly ruling it out.

## If the state audit supports a new supervision experiment

Define a new version/name before training. Possible controlled options include:

- soft/low-weight treatment of selected uncertain/unmentioned states;
- confidence-aware weak loss;
- robust loss under noisy weak supervision;
- a better independently audited report labeler.

B6 v1.2.1 itself remains frozen for historical reproducibility.

## Later hypotheses

After the supervision audit, later global experiments may include:

- richer image-report representation learning with the stronger encoder;
- higher in-plane resolution, distinct from slice count;
- robust-loss methods;
- carefully justified architecture changes;
- multi-seed/global ensembles with fixed, non-target-specific rules.

Larger foundation encoders remain a separate hypothesis, but B15 argues that simply improving weak-teacher agreement is not enough.

## Explicitly not allowed

```text
B14 epoch extension
B15 SSL/epoch/LR retuning from gold
target-wise B13/B14/B15 mixtures
gold-selected slice count
gold-selected weak-label state weights
gold-selected thresholds
gold-selected ensemble weights
retrospective weak validation of checkpoints trained on holdout studies
regenerating weak-v2 based on model results
calling weak teacher agreement expert truth
calling the reused 58 studies independent validation
claiming a numerical B6 AUC ceiling
blindly mapping unmentioned reports to negative
```

## Current decision chain

```text
B13 retained champion
        |
        v
B6 report-state audit
        |
        v
separately frozen supervision successor only if justified
        |
        v
controlled comparison
        |
        v
Kaggle hidden signal
```

The goal remains a higher global macro AUC through controlled, reproducible improvements rather than increasingly fine tuning to 58 repeatedly reused expert-labelled studies.