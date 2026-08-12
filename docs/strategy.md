# Modeling strategy

> **Snapshot — 2026-08-12.** **B13 remains the reused-gold development champion at macro AUC `0.6293565948`. B15 passed the frozen weak-v2 teacher-agreement gate decisively but reached `0.6209002783` on its single reused-gold confirmation, so it did not replace B13. The immediate strategy is now supervision-state diagnosis before another training experiment.** Canonical results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Core principle

`CNN_CPC` treats the challenge as a weakly supervised multi-series knee MRI problem with only 58 fully labelled expert development studies. The strategy prioritizes leakage control, supervision quality, representation quality, complete data usage and controlled hypothesis testing before increasing model complexity.

The project now has two local ranking surfaces with different meanings:

```text
58-study gold surface -> expert-labelled development/model selection
623-study weak-v2     -> B6 report-teacher agreement only
```

Neither is the hidden Kaggle test set.

## Experiment evidence so far

| Candidate | Gold macro AUC | Interpretation |
|---|---:|---|
| B0 random | `0.4762536432` | weak baseline |
| B1 strong MRI SSL | `0.5030284974` | useful in-domain representation |
| B4 frozen SSL + classical | `0.5137567459` | representation separability improved |
| B5 image-report SSL | `0.5243650851` | report-aligned representation baseline |
| B7-v1 direct B6 supervision | `0.5397724412` | direct weak supervision helped |
| B7.1 full coverage | `0.5644802945` | full weak-corpus coverage helped |
| B8 spatial anatomy | `0.5300962807` | rejected |
| B9 strict routing | `0.5334962669` | rejected |
| B10 physical scale | `0.5523982721` | rejected |
| B11.1 quantile teacher | `0.5506902702` | rejected |
| B12 all real series | `0.5660915179` | retained historical reference |
| **B13 ImageNet + hierarchy** | **`0.6293565948`** | **development champion** |
| B14 full slice-token memory | `0.6197914249` | rejected globally |
| **B15 MRI-domain SSL + hierarchy** | **`0.6209002783`** | **weak-v2 gate passed; no global gold improvement** |

B11-v1 failed its pseudo-label viability gate; B12.1 was implemented but skipped.

## 1. Reports are training supervision only

Final inference remains MRI-only. Reports have served two distinct roles:

- B5: semantic representation alignment without 12-target report labels;
- B6 onward: structured positive / negated / uncertain / unmentioned weak target states.

Frozen B6 v1.2.1 scope:

```text
report-only rows                  4349
active weakly labelled studies    3120
usable cells                     14123
positive cells                    6871
negative cells                    7252
```

Global downstream policy through B15:

| state | soft target | base weight |
|---|---:|---:|
| positive | 0.85 | 0.50 |
| negated | 0.05 | 1.00 |
| uncertain | ignored | 0.00 |
| unmentioned | ignored | 0.00 |

B6 v1.2.1 remains frozen for historical reproducibility.

## 2. B7.1 established the importance of complete weak-corpus coverage

B7.1 changed only the training coverage from B7-v1 and reached `0.5644802945`. It demonstrated that direct weak supervision benefited from four full passes over all 3,120 active B6 studies.

That result remains an important training-design lesson even though later B13 superseded its architecture/performance.

## 3. B8-B12 explored data semantics, capacity and supervision completion

The intermediate experiments established several negative or neutral results:

- B8: adding coarse 2x2 within-slice spatial tokens did not improve global AUC;
- B9: exact fluid/structural routing fixed a real metadata inconsistency but did not improve global AUC;
- B10: in-plane physical-scale normalization did not improve globally;
- B11-v1: absolute pseudo-label thresholds failed viability;
- B11.1: target-wise quantile pseudo-label tails passed viability but did not improve globally;
- B12: retaining every real MRI acquisition was viable and tied the prior benchmark.

These experiments remain valuable because they narrowed the search space without target-wise post-hoc mixing.

## 4. B13 changed the campaign level

B13 combines:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
+ ImageNet normalization
+ every eligible real MRI series
+ 16 sampled 2.5D positions/series
+ learned attention pool -> one token/series
+ study Transformer
+ pathology-query prediction heads
+ frozen B6 supervision
```

Gold result:

```text
macro AUC = 0.6293565948
95% CI   = [0.5789896351,0.6775867717]
```

It remains the retained global development model.

## 5. B14 and the exact slice audit narrowed the image-side hypotheses

B14 retained the full `K x 16` slice-token memory and fit B6 more strongly than B13, yet scored `0.6197914249`. The exact slice audit also found essentially complete evaluation exposure for ordinary MRI series.

Together these results argue against two immediate explanations for B13's remaining error:

```text
not enough downstream slice-token capacity
not enough sampled through-plane slice coverage
```

They do not rule out in-plane resolution, representation quality or supervision limitations.

## 6. Frozen weak-v2 introduced a pre-gold ranking gate

The v2 surface was frozen before B15/control training:

```text
weak-train studies       2497
weak holdout studies      623
holdout cells            2875
report-group overlap        0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

Its strict bootstrap only accepts a replicate if all 12 target AUCs are defined. It is explicitly a **teacher-agreement** surface.

## 7. B15 tested MRI-domain adaptation cleanly

B15 SSL used 3,726 competition studies after excluding all 58 gold cases and all 623 weak-v2 holdout cases. It trained four exact full passes over 20,534 real MRI series.

Matched downstream arms both used exactly:

```text
2497 studies
13974 series
11248 usable B6 cells
5464 positive / 5784 negative
4 complete epochs
```

The intended difference was encoder initialization:

```text
control: ImageNet
B15:    ImageNet -> knee-MRI same-study contrastive SSL
```

## 8. B15 passed weak-v2 but not gold

Weak-v2:

```text
B13-v2 control          0.5652498118
B15                    0.7319060415
raw delta              +0.1666562297
paired median          +0.1675245839
95% paired CI          [+0.1124433208,+0.2165156305]
P(B15 > control)        1.0000
```

The predeclared gate passed.

One-look reused gold:

```text
B15                    0.6209002783
B13                    0.6293565948
raw B15-B13           -0.0084563164
```

Thus B15 became much better at reproducing the weak teacher without improving the global expert-gold ranking. This is now one of the most important observations in the campaign.

## 9. Strategic interpretation

The current bottleneck should not be described simply as “the encoder is too weak.” B15 shows the encoder/representation can be adapted enough to produce a large gain on the weak target surface.

A more plausible next question is whether the available weak labels encode the expert target ordering well enough, especially given sparse state coverage and instance-dependent report mention/negation behavior.

This is a hypothesis to audit, not a reason to declare the labels unusable or to claim a numerical ceiling.

## 10. Immediate next step — report-state audit

Before another GPU training run, measure how the frozen B6 states relate to expert truth:

```text
positive
negated
uncertain
unmentioned
```

For every target and state, quantify counts, expert-positive fraction, expert-negative fraction, coverage and appropriate predictive values.

Particularly important questions:

```text
How often is an unmentioned finding actually expert-positive?
How reliable are explicit negatives by target?
Do uncertain states contain useful low-confidence ranking information?
Are the relationships target-dependent enough to make a global policy inappropriate?
```

Do not convert unmentioned to negative by assumption.

## 11. Only then define the next model hypothesis

If the audit supports additional supervision information, define a separately named/frozen successor policy. Possible families include soft/low-weight ignored states, confidence-aware/robust weak losses, or a better independently evaluated report labeler.

If the audit does not support such a policy, return to other global hypotheses such as in-plane resolution, richer image-report representation learning, or carefully controlled model diversity.

## 12. Validation discipline

Do not:

- tune target-specific weak-label rules from gold outcomes;
- select target-wise B13/B15 winners;
- retune B15 SSL epochs/LR/TTA from its gold result;
- regenerate weak-v2 after seeing B15;
- optimize ensemble weights on the reused gold set;
- call weak-v2 AUC expert performance;
- call the 58-study development result a hidden-test guarantee.

The next genuinely independent performance signal is the hidden Kaggle evaluation.