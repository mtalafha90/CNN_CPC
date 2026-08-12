# Raising macro AUC beyond B13

> **Status — 2026-08-12.** B13 remains the reused-gold development champion at macro AUC `0.6293565948`. B14 was rejected globally at `0.6197914249`. B15 passed the frozen weak-v2 gate by a very large margin but its one-look reused-gold confirmation was `0.6209002783`, so the next priority shifts from more representation tuning to a direct audit of the weak-supervision states.

## Current evidence

```text
B13 gold macro AUC       0.6293565948   retained champion
B14 gold macro AUC       0.6197914249   rejected globally
B15 gold macro AUC       0.6209002783   no global improvement

B15 weak-v2              0.7319060415
B13-v2 control weak-v2   0.5652498118
paired weak median       +0.1675245839
95% paired weak CI       [+0.1124433208,+0.2165156305]
P(B15 > control)          1.0000
```

The key new result is the **weak/gold divergence**. B15 learned a representation that ranks B6 report-derived labels much better, yet this did not improve the primary expert-gold macro AUC.

## What B14 already ruled against

B14 retained full `K x 16` downstream slice-token memory and fit B6 more strongly than B13:

```text
B14 final B6 loss  0.5822778610
B13 final B6 loss  0.6132239342
```

Yet B14 gold macro AUC remained lower. This argues against simply adding downstream token capacity or fitting the current weak labels harder.

## What the exact B13 slice audit ruled against

The corrected 17,475-series audit found:

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

Do not launch a 24/32/48-slice sweep from the reused gold surface. In-plane resolution remains a different question.

## What B15 established

### Stronger MRI-domain representation learning is possible

B15 used ImageNet initialization followed by same-study, multi-instance knee-MRI contrastive adaptation on 3,726 competition studies while excluding all 58 gold studies and all 623 weak-v2 holdout studies.

All four SSL passes completed exactly over 20,534 eligible series. The downstream control and B15 candidate then trained on the same 2,497-study weak-train partition with identical B13 hierarchy/optimization.

### The frozen weak-v2 gate was decisive

```text
control weak-v2      0.5652498118
B15 weak-v2          0.7319060415
raw delta           +0.1666562297
paired median       +0.1675245839
95% paired CI       [+0.1124433208,+0.2165156305]
P(B15 > control)     1.0000
```

### The expert-gold transfer was absent globally

```text
B13 gold             0.6293565948
B15 gold             0.6209002783
raw B15-B13         -0.0084563164
```

This does not prove a representation ceiling. It shows that **optimizing compatibility with the current B6 weak supervision is not sufficient** for improving expert-label ranking.

## What the B6 audit implies — and does not imply

Frozen B6 gold audit:

```text
sensitivity          0.9748
specificity          0.6061
positive precision   0.6905
NPV                  0.9639
balanced accuracy    0.7904
coverage             0.3606
```

These values establish noisy, sparse and asymmetric supervision. They do **not** establish a `0.75-0.80` downstream AUC ceiling.

Under idealized class-conditional label noise, ranking can even be preserved by an affine transformation of the posterior. The real concern here is that report noise is likely instance-dependent and target-dependent: what is mentioned, negated or omitted depends on the study, radiologist and finding.

## Priority 1 — B6 report-state audit

Before training another model, inspect all four parser states against expert truth on the already-reused gold studies:

```text
positive
negated
uncertain
unmentioned
```

For every target/state quantify:

```text
number of cells
expert-positive count/fraction
expert-negative count/fraction
coverage
precision / NPV where meaningful
state-specific uncertainty
```

Key quantities include:

```text
P(expert positive | B6 positive)
P(expert positive | B6 negated)
P(expert positive | B6 uncertain)
P(expert positive | B6 unmentioned)
```

This audit is diagnostic. It must not be converted directly into a post-hoc target-specific winner scheme.

### Why unmentioned deserves audit rather than hard-negative conversion

The current B6 policy correctly distinguishes `negated` from `unmentioned`. Report silence is not equivalent to an explicit negative. Do **not** blindly map all unmentioned findings to negative.

Only if the audit supports a systematic relationship should a separately named/frozen successor supervision policy use unmentioned or uncertain states, preferably with conservative soft targets/low weights rather than unsupported hard labels.

## Priority 2 — separately versioned supervision successor, only if audit supports it

Potential controlled directions include:

- soft/low-weight use of selected uncertain or unmentioned states;
- target-global state weights declared before evaluation;
- robust loss formulations such as symmetric/robust cross-entropy or early-learning regularization;
- explicit modeling of label confidence rather than treating every usable report cell equally apart from the existing global weights.

Any such experiment must have a new name/version. **B6 v1.2.1 remains frozen** for historical B7-B15 comparisons.

## Priority 3 — richer report information without forcing sparse target labels

B5 already demonstrated that full report semantics can shape MRI representations. A future experiment can revisit image-report learning with the stronger ImageNet/B15-era encoder while keeping the report branch training-only.

The scientific question should be representation alignment to the report, not another gold-tuned 12-target report classifier.

## Priority 4 — better report labelling

A stronger report labeler may be valuable, but improvement should come from a small, deliberately annotated report set or a separately validated labeler rather than another round of parser tuning against the same 58 image-level gold labels.

If expert report annotation is collected, preserve a held-out subset for measuring the labeler itself.

## Priority 5 — robust weak-label optimization

After label-state quality is understood, test one robust-loss hypothesis at a time. Avoid combining robust loss, new labels, new representation and larger resolution in one experiment, because attribution would be impossible.

## Priority 6 — in-plane resolution

The exact slice audit closes **slice-count** undersampling, not `224x224` in-plane detail. Higher in-plane resolution remains a plausible later experiment, particularly for focal ligament/meniscal findings.

It should be a global predeclared comparison, not a target-specific resolution selection based on reused gold.

## Priority 7 — architecture/capacity only after supervision is addressed

B8 and B14 both show that more downstream token structure/capacity can fit training well without improving macro AUC. Larger Transformers, more pathology layers or more slice tokens are therefore lower priority than supervision quality.

## Priority 8 — global ensembles after structure is settled

Multi-seed or architecture-diverse ensembles may reduce variance. They should use globally fixed weights/rules and must not use gold-selected target-specific mixtures.

## Frozen validation discipline

Weak-v2 remains frozen:

```text
train studies             2497
holdout studies            623
holdout cells             2875
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

Do not regenerate it after B15.

The reused 58 studies remain development confirmation only. The next genuinely independent signal is the hidden Kaggle evaluation.

## Explicitly prohibited

```text
target-wise B13/B14/B15 winners
gold-selected weak-label state weights
gold-selected slice counts
gold-selected thresholds
gold-selected ensemble weights
B15 SSL epoch/LR retuning from its gold result
retrospective weak validation of checkpoints trained on holdout studies
regenerating weak-v2 based on performance
calling weak-v2 teacher agreement expert truth
calling reused gold independent validation
claiming a numerical B6 AUC ceiling from balanced accuracy
blindly mapping unmentioned report states to negative
```

## Current recommended sequence

```text
B13 remains champion
        |
        v
B6 report-state audit
        |
        v
Does audit justify additional weak information?
   | no                         | yes
   v                            v
prioritize other          define frozen/versioned
representation or         supervision successor
resolution hypotheses           |
   \____________________________/
                |
                v
      frozen model comparison
                |
                v
        hidden Kaggle signal
```

The goal remains higher **global macro ROC AUC** through controlled, reproducible improvements rather than increasingly fine tuning to 58 repeatedly reused expert-labelled cases.