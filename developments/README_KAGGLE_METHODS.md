# RSNA Knee Abnormality Detection — Public Code Methodology Review

**Repository:** `mtalafha90/CNN_CPC`  
**Snapshot:** 2026-08-12  
**Purpose:** methodology context and repository-measured development evidence, not a leaderboard claim.

> Canonical measured results are in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md). **B13 is the reused-gold development champion at macro AUC `0.6293565948`. B15 passed the frozen weak-v2 teacher-agreement gate (`0.7319060415` vs matched control `0.5652498118`) but its one-look reused-gold result was `0.6209002783`; therefore B13 remains retained.**

## Problem structure

```text
4,407 training studies
58 fully gold-labelled studies
4,349 report-only studies
24,371 series rows
12 study-level targets
primary metric: macro ROC AUC
```

This is a weak/semi-supervised multi-series MRI problem with an extremely small trusted expert-labelled development set.

## Main lessons from the repository experiments

### Reports are useful as training supervision, not inference inputs

Final inference remains MRI-only. The first fold-safe report teacher reached only `0.49245` macro OOF and was rejected as a general 12-target teacher.

Reports subsequently became useful through two different paths:

- **B5:** image-report semantic representation alignment;
- **B6 onward:** structured positive / negated / uncertain / unmentioned weak target states.

### Unmentioned is not negative

B6 v1.2.1 training export:

```text
active weakly labelled studies  3120
usable cells                   14123
positive cells                  6871
negative cells                  7252
```

B7-B15 weak supervision uses:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

This distinction remains important after B15. The next diagnostic is explicitly testing how each report state relates to expert truth; the project does not assume that report silence is an explicit negative.

### In-domain MRI representation learning helped, but representation is not the whole bottleneck

The early sequence showed progressive representation gains:

```text
B0 random init             0.4762536432
B1 strong MRI SSL          0.5030284974
B4 frozen SSL probe        0.5137567459
B5 image-report SSL        0.5243650851
```

Direct weak supervision and full corpus coverage then raised the development point estimate:

```text
B7-v1                      0.5397724412
B7.1                       0.5644802945
```

B13 later combined all-series hierarchical aggregation with ImageNet ConvNeXt initialization and reached `0.6293565948`, the current champion.

### More downstream token capacity did not help

B14 kept every `K x 16` slice token instead of B13's one-token-per-series compression:

```text
B13 gold macro AUC         0.6293565948
B14 gold macro AUC         0.6197914249
paired median B14-B13     -0.0093726931
95% paired CI             [-0.0469823411,+0.0250137870]
P(B14 > B13)               0.2924
```

B14 also reached a lower B6 training loss than B13, so fitting the existing weak targets harder did not produce a global expert-gold gain.

### Slice-count undersampling is not the main B13 bottleneck

The exact 17,475-series audit reproduced B13's real 2.5D sampler:

```text
eval unique fraction     median 100.0%
complete eval exposure   95.9%
eval max skipped run     median 0.0 slices (p95 0.0)
```

This closes the immediate 24/32/48-slice-count sweep. In-plane resolution is a separate question.

## Frozen weak-v2 validation surface

To reduce repeated direct model selection on the 58 gold studies, a report-group-safe weak holdout was frozen before B15/control training:

```text
surface                   weak_b6_holdout_v2
weak-train studies        2497
holdout studies            623
holdout usable cells      2875
positive / negative    1407 / 1468
report-group overlap         0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The weak surface measures **B6 teacher agreement, not expert truth**. Bootstrap replicates are accepted only when all 12 target AUCs are defined.

## B15: clean MRI-domain SSL test

B15 tested whether knee-MRI adaptation of the successful ImageNet encoder improved downstream ranking.

SSL pool:

```text
4407 competition studies
-58 gold studies
-623 weak-v2 holdout studies
=3726 SSL studies
20534 eligible real MRI series
```

All four SSL epochs completed exact full passes.

Matched downstream arms then used exactly:

```text
2497 studies
13974 real MRI series
11248 B6 cells
5464 positive / 5784 negative
4 full epochs
```

Control:

```text
ImageNet -> B13 hierarchy
```

Candidate:

```text
ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy
```

## B15 weak-v2 result

```text
B13-v2 control             0.5652498118
B15                       0.7319060415
raw delta                 +0.1666562297
paired median             +0.1675245839
95% paired CI             [+0.1124433208,+0.2165156305]
P(B15 > control)           1.0000
```

The predeclared gate passed decisively. All usable paired bootstrap replicates favored B15.

## B15 expert-gold confirmation

Passing weak-v2 earned B15 one reused-gold development evaluation:

```text
B15 gold macro AUC         0.6209002783
95% CI                    [0.5706720829,0.6675892903]
B13 gold macro AUC         0.6293565948
raw B15-B13              -0.0084563164
```

Thus the large improvement in weak-teacher agreement did **not** transfer to a global expert-label improvement.

The correct interpretation is not that MRI SSL failed. Instead, B15 shows that an MRI representation can become much better at matching the weak target surface without improving the expert-gold macro objective. That makes weak-supervision semantics/noise/sparsity a high-priority bottleneck to audit.

## Current measured ladder

| Candidate | Gold macro AUC | Status |
|---|---:|---|
| B0 | `0.4762536432` | baseline |
| B1 | `0.5030284974` | retained reference |
| B2 | `0.4993244663` | rejected |
| B3 | `0.4944652486` | rejected |
| B4 | `0.5137567459` | retained ablation |
| B5 | `0.5243650851` | representation baseline |
| B7-v1 | `0.5397724412` | coverage ablation |
| B7.1 | `0.5644802945` | historical benchmark |
| B5+B7.1 rank | `0.5540141184` | rejected |
| B8 | `0.5300962807` | rejected |
| B9 | `0.5334962669` | rejected |
| B10 | `0.5523982721` | rejected |
| B11.1 | `0.5506902702` | rejected |
| B12 | `0.5660915179` | retained historical reference |
| **B13** | **`0.6293565948`** | **development champion** |
| B14 | `0.6197914249` | rejected globally |
| **B15** | **`0.6209002783`** | **weak-v2 gate passed; no global gold improvement** |

B11-v1 failed viability; B12.1 was implemented but skipped.

## Current methodological priority

The next experiment should **not** be another B15 hyperparameter sweep. First audit B6 report states against expert truth:

```text
positive
negated
uncertain
unmentioned
```

For each target/state, quantify counts, expert-positive fraction, expert-negative fraction and coverage. Only if that audit supports additional information should a separately versioned weak-supervision successor be defined.

Other later controlled directions include robust weak-label losses, richer image-report representation learning, improved report labelling, higher in-plane resolution and global ensembles after structure is settled.

## Validation discipline

Do not:

- select target-specific winners;
- optimize blend weights from the reused 58 labels;
- retune B6 parser rules/weak-label weights from downstream outcomes;
- retune B15 SSL epochs/LR/TTA from its gold confirmation;
- regenerate weak-v2 based on model performance;
- interpret weak-v2 AUC as expert truth;
- map all unmentioned findings to negative by assumption;
- describe local development performance as a hidden-test or leaderboard guarantee.

The hidden competition evaluation remains the next genuinely independent performance signal.