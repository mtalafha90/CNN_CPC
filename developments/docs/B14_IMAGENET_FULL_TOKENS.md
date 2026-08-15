# B14 — ImageNet full slice-token aggregation

> **Status — 2026-08-12:** **COMPLETED / REJECTED GLOBALLY.** B13 remains the reused-gold development champion after B15 also failed to improve global gold macro AUC.

## Scientific question

B13 compresses each real MRI series from 16 encoded slice tokens to one learned series token before the study Transformer. B14 tested whether retaining the full slice-token memory would improve global macro ROC AUC.

```text
B13: 16 slice tokens -> learned series pool -> 1 token/series -> study Transformer
B14: 16 slice tokens -> no series compression -> K x 16 tokens -> study Transformer
```

Everything else used the same B13 ImageNet ConvNeXt-Tiny protocol, ImageNet normalization, frozen B6 supervision, all-series mapping, 224x224 preprocessing, four full epochs, TTA `[-1,0,1]`, and zero gold gradients/early stopping.

Frozen full-series SHA:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Completed training

```text
epoch 1 loss  0.7346330162
epoch 2 loss  0.6606430862
epoch 3 loss  0.6074723502
epoch 4 loss  0.5822778610
```

Every epoch covered exactly 3,120 studies, 14,123 supervised cells and 17,475 real MRI series with no budget limitation.

B14 fit B6 more strongly than B13 (`0.5822778610` versus `0.6132239342` final loss), but training loss is not a model-selection metric.

## Reused-gold development result

```text
B14 macro AUC       0.6197914249
95% CI             [0.5706800512,0.6693542716]
B13 macro AUC       0.6293565948
raw B14-B13        -0.0095651699
paired median      -0.0093726931
95% paired CI      [-0.0469823411,+0.0250137870]
P(B14 > B13)        0.2924
```

Per-target B14 AUCs, descriptive only:

```text
ACL 0.5122549020
MCL 0.4693877551
Medial Meniscus 0.6454326923
Lateral Meniscus 0.6881987578
Medial OA 0.5116279070
Lateral OA 0.5783365571
PF OA 0.5997425997
Effusion 0.8347826087
Synovitis 0.7419354839
Baker's 0.6884057971
Contusion 0.5465587045
Fracture 0.6208333333
```

The paired interval crosses zero, so the reused gold surface does not prove B14 is statistically inferior. Operationally, B14 is rejected because it has a lower point estimate, low probability of superiority, greater memory cost and no global benefit.

## Successor result — B15

B15 replaced the capacity hypothesis with ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy. It passed frozen weak-v2 decisively:

```text
B13-v2 control weak-v2  0.5652498118
B15 weak-v2             0.7319060415
paired median           +0.1675245839
95% CI                  [+0.1124433208,+0.2165156305]
P(B15 > control)         1.0000
```

But its single reused-gold confirmation was:

```text
B15 gold                0.6209002783
B13 gold                0.6293565948
raw B15-B13            -0.0084563164
```

Thus B13 remains champion. B14 and B15 together reinforce that better fitting/compatibility with the current weak target surface is not sufficient for higher expert-gold macro AUC.

## Decision discipline

Do not run B14 epoch 5, tune slice count, change learning rates, construct B13/B14/B15 target-wise hybrids, or search ensemble weights using reused gold.

Current next step: audit B6 report states (`positive`, `negated`, `uncertain`, `unmentioned`) against expert truth before defining another supervision experiment. See [`B15_MRI_SSL.md`](B15_MRI_SSL.md), [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md), and [`VALIDATION.md`](VALIDATION.md).