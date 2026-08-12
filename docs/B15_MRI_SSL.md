# B15 — ImageNet to knee-MRI SSL to B13 hierarchy

> **Status — 2026-08-12:** **COMPLETED. Weak-v2 gate PASSED; reused-gold global improvement NOT ESTABLISHED.** Package `0.24.1`. B13 remains the reused-gold development champion.

## Scientific question

B14 showed that retaining all `K x 16` slice tokens did not improve global macro AUC, and the exact 17,475-series audit rejected slice-count undersampling as the primary B13 bottleneck. B15 therefore tested one representation question:

```text
Does adapting the successful ImageNet ConvNeXt-Tiny encoder to competition knee MRI
before the unchanged B13 weakly-supervised hierarchy improve global 12-target ranking?
```

## Frozen weak-v2 surface

The v2 split was frozen before B15 or matched-control training.

```text
surface                   weak_b6_holdout_v2
status                    FROZEN before B15/control training
active B6 studies         3120
weak-train studies        2497
holdout studies            623
holdout usable cells      2875
positive / negative    1407 / 1468
train report groups       2426
holdout report groups      613
report-group overlap         0
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The weak surface measures **agreement with the B6 report teacher, not expert truth**. Its manifest must not be regenerated based on performance.

## Matched downstream arms

### B13-v2 control

```text
ImageNet ConvNeXt-Tiny
        -> B13 hierarchical one-token-per-series model
        -> B6 v1.2.1 training on the 2,497 v2 weak-train studies
```

### B15 candidate

```text
ImageNet ConvNeXt-Tiny
        -> knee-MRI same-study multi-instance contrastive SSL
        -> same B13 hierarchical model
        -> same B6 v1.2.1 training on the same 2,497 studies
```

Both arms construct the same seeded downstream hierarchy before loading encoder state. Downstream architecture, MRI sampling, B6 soft-target policy, train-only target-balancing derivation, optimizer, augmentation, four epochs and TTA are identical. The intended model difference is encoder initialization.

## B15 SSL leakage contract

B15 used the stricter image-held-out SSL pool:

```text
competition studies            4407
fully labelled gold             -58
frozen v2 weak holdout          -623
------------------------------------
B15 SSL studies                3726
eligible real MRI series      20534
```

Forbidden during SSL:

```text
gold studies/images
v2 holdout studies/images
B6 labels
report labels
gold or weak-v2 feedback for checkpoint selection
```

Every eligible repaired real MRI series was retained. Five distributed 2.5D positions were sampled per acquisition and two distributed positions per real series entered the contrastive examples. Same-knee examples were positives; examples from other studies in the batch were negatives. This is described as **MICLe-style same-study contrastive adaptation**, not an exact reproduction of a published implementation.

Frozen SSL optimization:

```text
ImageNet initialization       IMAGENET1K_V1
input normalization          ImageNet mean/std
SSL epochs                   4 full passes
study batch                  2
sampled positions/series     5
used positions/series        2
projection dim               256
encoder LR                   5e-5
projector LR                 5e-4
minimum LR                   1e-6
weight decay                 1e-4
temperature                  0.15
grad clip                    1.0
train gap choices            [1,2]
center jitter                +/-2
```

Final-epoch checkpoint selection was frozen in advance; SSL loss was not used to choose among epochs.

## Completed SSL training

All four SSL epochs matched the exact expected full-coverage contract:

| Epoch | Loss | Batches | Studies | Series | 2.5D examples | Full coverage | Budget limited |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `2.7094607696` | 1863 | 3726 | 20534 | 41068 | true | false |
| 2 | `2.5811344701` | 1863 | 3726 | 20534 | 41068 | true | false |
| 3 | `2.5187829415` | 1863 | 3726 | 20534 | 41068 | true | false |
| 4 | `2.4756854072` | 1863 | 3726 | 20534 | 41068 | true | false |

Checkpoint:

```text
runs/b15_mri_ssl/b15_ssl_encoder.pt
```

The monotonic SSL-loss decrease certifies stable optimization but is **not** itself evidence of downstream pathology performance.

## Frozen downstream recipe

```text
architecture                 B13 hierarchical one-token-per-series
weak-train studies           2497
eligible train series       13974
usable B6 cells             11248
positive / negative      5464 / 5784
slices/series                16
image size                   224
batch size                   2
full batches/epoch           1249
transformer layers           2
transformer heads            8
pathology layers             1
dropout                      0.25
encoder LR                   1e-5
head LR                      1e-4
minimum LR                   1e-6
weight decay                 1e-4
epochs                       4
TTA                          [-1,0,1]
```

B6 policy remains:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

Target-balance multipliers were recomputed from the 2,497 weak-train studies only, so holdout labels did not influence training class weighting.

## Matched B13-v2 control training

All four epochs had exact `2497` studies, `13974` series, `11248` supervised cells, `5464` positive cells, `5784` negative cells and `1249` batches, with full study/series coverage and no budget limiting.

```text
epoch 1 loss  0.7764664802
epoch 2 loss  0.6929244099
epoch 3 loss  0.6857204294
epoch 4 loss  0.6622741637
```

Checkpoint:

```text
runs/b13_v2_control/b13_v2_control.pt
```

## B15 downstream training

B15 used the exact same downstream surface and schedule:

```text
epoch 1 loss  0.7645658738
epoch 2 loss  0.6782294669
epoch 3 loss  0.6336512997
epoch 4 loss  0.6065262400
```

Every epoch had the exact same full-coverage counts as the control. Training loss is not used for model selection.

Checkpoint:

```text
runs/b15_mri_ssl/downstream/b15_model.pt
```

## Frozen weak-v2 evaluations

### B13-v2 matched control

```text
macro AUC              0.5652498118
95% CI                [0.5361620323,0.5924683768]
strict valid bootstrap 4913 / 5000
```

Per-target AUC:

```text
ACL                0.5578815680
MCL                0.4723660043
Medial Meniscus    0.5874697337
Lateral Meniscus   0.4762534144
Medial OA          0.7118421053
Lateral OA         0.7252252252
PF OA              0.5124382660
Effusion           0.6248527680
Synovitis          0.4740259740
Baker's            0.6706805816
Contusion          0.4656943564
Fracture           0.5042677448
```

### B15

```text
macro AUC              0.7319060415
95% CI                [0.6903737595,0.7675416396]
strict valid bootstrap 4913 / 5000
```

Per-target AUC:

```text
ACL                0.6638448707
MCL                0.6974453164
Medial Meniscus    0.7093623890
Lateral Meniscus   0.6411578113
Medial OA          0.8077485380
Lateral OA         0.7813582814
PF OA              0.7528809219
Effusion           0.7573616019
Synovitis          0.6461038961
Baker's            0.8051021318
Contusion          0.6806594800
Fracture           0.8398472597
```

These are teacher-agreement AUCs, not expert-label AUCs.

## Predeclared paired weak-v2 gate

The decision rule was frozen before seeing the weak-v2 result. B15 passed only if all three held:

```text
raw macro delta B15-control > 0
paired median delta > 0
P(B15 > B13-v2-control) >= 0.95
```

Observed:

```text
raw difference B15-control  +0.1666562297
paired median difference    +0.1675245839
95% paired CI               [+0.1124433208,+0.2165156305]
P(B15 > control)             1.0000
valid paired replicates      4921 / 5000
valid fraction               0.9842
passes gate                  true
```

All valid paired bootstrap replicates favored B15. `P=1.0` is an empirical bootstrap result, not a claim of absolute certainty.

Gate artifact:

```text
runs/b15_mri_ssl/weak_eval/b13_v2_vs_b15.json
```

Passing this gate earned B15 exactly one evaluation on the repeatedly reused 58-study expert-gold development surface.

## One-look reused-gold confirmation

```text
B15 macro AUC       0.6209002783
95% CI             [0.5706720829,0.6675892903]
n studies           58
bootstrap           5000 / 5000 usable
```

Per-target B15 AUC:

```text
ACL                0.5661764706
MCL                0.6462585034
Medial Meniscus    0.5973557692
Lateral Meniscus   0.6658385093
Medial OA          0.5085271318
Lateral OA         0.5551257253
PF OA              0.5997425997
Effusion           0.8012422360
Synovitis          0.6845878136
Baker's            0.6739130435
Contusion          0.5492577598
Fracture           0.6027777778
```

Historical B13 reference:

```text
B13 macro AUC       0.6293565948
B15 macro AUC       0.6209002783
raw B15-B13        -0.0084563164
```

B15 therefore did **not** replace B13 as the global reused-gold development champion. Marginal confidence intervals alone do not establish that B15 is statistically inferior; the campaign decision is simply that no global improvement was demonstrated.

Target-wise B13/B15 differences are descriptive only. They must not be used to construct a hybrid model after observing the gold labels.

## Scientific interpretation

B15 produced a large, precisely positive improvement in ranking the frozen report-derived weak labels, yet essentially no global improvement on expert gold. This is a crucial negative/diagnostic result:

1. MRI-domain SSL can substantially improve compatibility with the weak-supervision target surface.
2. Stronger weak-teacher agreement is not sufficient for stronger expert-label ranking.
3. The next bottleneck investigation should focus on the supervision interface rather than simply adding more downstream capacity, more SSL epochs, or more slice tokens.
4. This does **not** prove a numerical weak-label ceiling and does not prove B15 is intrinsically a worse MRI representation.

## Decision

```text
B13  gold 0.6293565948  RETAIN / DEVELOPMENT CHAMPION
B14  gold 0.6197914249  REJECT GLOBALLY
B15  weak 0.7319060415  WEAK-V2 GATE PASS
B15  gold 0.6209002783  NO GLOBAL GOLD IMPROVEMENT
```

B15 is closed as a tuning target. Do not change its SSL epochs, learning rates, downstream epoch count, TTA, hierarchy, or target-specific mixtures from the gold outcome and still call the result B15.

## Next evidence-driven step

Before another model is trained, audit the frozen B6 report states against expert truth on the already-reused gold surface:

```text
positive
negated
uncertain
unmentioned
```

For each target/state, quantify counts, expert-positive fraction, expert-negative fraction, coverage and appropriate predictive values. Only if that audit supports a new treatment should a separately named/frozen supervision successor be defined. In particular, **do not blindly map unmentioned findings to negative**.

The hidden Kaggle evaluation remains the next genuinely independent performance signal.