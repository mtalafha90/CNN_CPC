# Weak B6 validation holdout v2

> **Status — 2026-08-12:** **FROZEN / USED FOR THE COMPLETED B15 MATCHED GATE.** v1 remains historical and superseded. Package `0.24.1` preserves strict all-12-target bootstrap semantics.

## Purpose

The weak holdout is a pre-gold model-ranking surface constructed from frozen B6 report-derived labels. It is designed to reduce repeated direct selection on the 58 expert-labelled development studies.

It measures **agreement with the B6 report teacher, not expert truth**. Absolute weak-v2 AUC must never be reported as expert validation or leaderboard performance.

## Why v1 was superseded

The first report-group-safe 20% split was frozen before any B15/control training and produced:

```text
surface                    weak_b6_holdout_v1
active studies             3120
train studies              2496
holdout studies             624
train report groups        2430
holdout report groups       609
report-group overlap          0
holdout usable cells       2697
holdout positive cells     1257
holdout negative cells     1440
Synovitis                  70 positive / 1 negative
manifest SHA-256
fdbc02f88e5a4eff31783b4242890e943609d5c783bd54aca38af8a89e7e0968
```

A single Synovitis negative made ordinary 12-target study bootstrap unnecessarily unstable. No B15 or matched-control model was trained on v1, so it was superseded **before model fitting**, without using model predictions or gold performance.

## v2 split policy

`weak_b6_holdout_v2` was selected using only frozen B6 labels and normalized report groups:

1. exact 3,120 active non-gold B6 studies and 14,123 usable cells;
2. duplicate normalized reports grouped so no report group can straddle train/holdout;
3. each group represented by 24 counts: positive and negative for 12 targets;
4. deterministic candidate search with seed `2026`;
5. rare-class floor of at least four examples on each side when globally feasible;
6. objective balances requested holdout size and all target/class counts;
7. no gold labels, MRI predictions or model performance used during selection.

Frozen defaults:

```text
holdout fraction        0.20
seed                    2026
minimum class count     4 per side where globally feasible
candidate splits        4096
report-group overlap    0 required
uses gold labels        false
uses model predictions  false
```

## Actual frozen v2 realization

```text
surface                   weak_b6_holdout_v2
status                    FROZEN before B15/control training
active studies            3120
train studies             2497
holdout studies            623
actual holdout fraction   0.1996794872
train report groups       2426
holdout report groups      613
report-group overlap         0
all usable cells         14123
holdout usable cells      2875
holdout positive cells    1407
holdout negative cells    1468
gold studies in surface      0
uses gold labels          false
uses model predictions    false
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

Per-target holdout counts:

| Target | Positive | Negative | Total |
|---|---:|---:|---:|
| ACL | 110 | 218 | 328 |
| MCL | 53 | 226 | 279 |
| Medial Meniscus | 236 | 105 | 341 |
| Lateral Meniscus | 97 | 234 | 331 |
| Medial OA | 95 | 72 | 167 |
| Lateral OA | 78 | 74 | 152 |
| PF OA | 142 | 77 | 219 |
| Effusion | 283 | 156 | 439 |
| Synovitis | 77 | 4 | 81 |
| Baker's | 111 | 101 | 212 |
| Contusion | 83 | 95 | 178 |
| Fracture | 42 | 106 | 148 |

The candidate search produced `3730/4096` feasible splits; the selected split had class-balance score `0.0034161873` and 623 holdout studies versus the nominal target of 624.

## Strict macro-AUC bootstrap

The weak surface uses a fixed estimand:

```text
bootstrap studies with replacement
-> compute all 12 target AUCs
-> reject replicate if any target AUC is undefined
-> accepted macro = mean of exactly 12 target AUCs
```

Always report:

```text
n_bootstrap
n_valid_replicates
valid_replicate_fraction
strict_all_12_targets = true
```

Because Synovitis has only four negatives, a small fraction of bootstrap samples omit that class. Rejecting those replicates preserves the fixed 12-target macro definition.

## Training contract

Any checkpoint evaluated on weak-v2 must exclude all 623 holdout UIDs from downstream optimization. Existing historical B13/B14 checkpoints were trained on all 3,120 active B6 studies and cannot be retrospectively called weak-v2 validation models.

For B15, the MRI SSL stage was even stricter: all 58 gold studies and all 623 weak-v2 holdout studies were excluded from SSL images.

## Completed matched B15 experiment

### Downstream training surface

Both newly trained arms used exactly:

```text
weak-train studies       2497
eligible real MRI series 13974
usable B6 cells          11248
positive cells            5464
negative cells            5784
batches/epoch             1249
epochs                       4
```

Control:

```text
ImageNet -> B13 hierarchy
```

Candidate:

```text
ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy
```

Every epoch in both arms had exact full study/series coverage and no budget limitation.

## B13-v2 control result

```text
macro AUC              0.5652498118
95% CI                [0.5361620323,0.5924683768]
strict valid bootstrap 4913 / 5000
valid fraction         0.9826
```

## B15 result

```text
macro AUC              0.7319060415
95% CI                [0.6903737595,0.7675416396]
strict valid bootstrap 4913 / 5000
valid fraction         0.9826
```

## Predeclared paired gate

B15 passed only if all three conditions were true:

```text
raw macro delta > 0
paired median delta > 0
P(B15 > B13-v2-control) >= 0.95
```

Observed:

```text
raw B15-control         +0.1666562297
paired median           +0.1675245839
95% paired CI           [+0.1124433208,+0.2165156305]
P(B15 > control)         1.0000
valid paired replicates  4921 / 5000
valid fraction           0.9842
passes gate              true
```

All usable paired bootstrap replicates favored B15. `P=1.0` is an empirical bootstrap probability, not mathematical certainty.

Gate artifact:

```text
runs/b15_mri_ssl/weak_eval/b13_v2_vs_b15.json
```

## What happened after the gate

The predeclared rule allowed the winner exactly one look at the repeatedly reused 58-study expert-gold development surface.

B15 gold confirmation:

```text
macro AUC      0.6209002783
95% CI        [0.5706720829,0.6675892903]
```

Historical B13 gold reference:

```text
B13            0.6293565948
raw B15-B13   -0.0084563164
```

Thus the large weak-teacher improvement did **not** transfer to a global expert-gold improvement. B13 remains the development champion.

## Interpretation

Weak-v2 succeeded at its intended purpose: it provided a model-ranking gate independent of direct gold-label feedback for B15. It also exposed an important limitation: a model can improve B6 teacher agreement dramatically without improving expert-gold macro AUC.

That discrepancy is evidence to investigate the supervision interface. It is **not** permission to retune the weak holdout or to treat weak-v2 as expert truth.

## Current rules

Do not:

- regenerate v2 based on B15/control outcomes;
- tune weak-v2 class floors, candidate count or seed after seeing performance;
- evaluate checkpoints that trained on v2 holdout UIDs and call the result validation;
- use weak-v2 target winners to build target-specific mixtures;
- interpret `0.7319` as expert-label performance.

The next evidence-driven step is a B6 report-state audit on the already-reused gold surface before any new supervision policy is defined.