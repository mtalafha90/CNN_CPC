# B7.1 — full-corpus weak-supervision coverage

> **Status — 2026-08-10:** **COMPLETE / CURRENT BEST STANDALONE DEVELOPMENT MODEL.** Macro ROC AUC `0.5644802945`, 95% bootstrap CI `[0.5052432984, 0.6229422178]`. The subsequent fixed B5+B7.1 rank ensemble was rejected. B8 spatial-anatomy learning is currently training.

## Motivation

B7-v1 produced macro AUC `0.5397724412` on the 58-study gold development set. Before the B7-v1 gold result was used to define the next experiment, the training audit had already exposed a coverage limitation: B7-v1 trained on 3,120 active weakly labelled studies but capped each epoch at 500 batches with batch size 2, giving only 1,000 study draws per epoch and about 1.28 nominal corpus passes over four epochs.

B7.1 tests that pre-identified limitation directly.

## Single scientific change

```text
b7_max_batches_per_epoch: 500 -> 1560
```

With 3,120 active studies and batch size 2, 1,560 batches are one complete shuffled pass through the active weak-training pool. Four epochs therefore provide four nominal full corpus passes.

Everything else remains fixed from B7-v1: B5 encoder initialization, frozen B6 v1.2.1 labels, global asymmetric soft-target policy, target balancing, six-stream 2.5D ConvNeXt + cross-sequence Transformer + pathology-query architecture, four epochs, learning rates, cosine schedule, augmentation, three-view gold TTA and 5,000 bootstrap replicates.

## Training result

Training completed all four predefined epochs with no budget limiting.

| Epoch | Loss | Batches | Study draws | Active cells | Positive | Negative | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.7524191749` | 1560 | 3120 | 14123 | 6871 | 7252 | 1792.09 |
| 2 | `0.6651707418` | 1560 | 3120 | 14123 | 6871 | 7252 | 1789.91 |
| 3 | `0.6391165589` | 1560 | 3120 | 14123 | 6871 | 7252 | 1890.68 |
| 4 | `0.6127582232` | 1560 | 3120 | 14123 | 6871 | 7252 | 1911.79 |

Totals:

```text
active studies          3120
usable cells           14123
study draws            12480
nominal corpus passes    4.0
budget limited         false
```

The loss decreased monotonically and the complete weak-supervision cell set was seen once per epoch.

Checkpoint:

```text
runs/b7_1_full_coverage/b7_model.pt
```

## Gold development evaluation

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
n         = 58
bootstrap = 5000/5000 usable
```

Per-target AUC:

| Target | B7.1 AUC |
|---|---:|
| ACL | `0.5159313725` |
| MCL | `0.4693877551` |
| Medial Meniscus | `0.5841346154` |
| Lateral Meniscus | `0.5950310559` |
| Medial OA | `0.4604651163` |
| Lateral OA | `0.5764023211` |
| PF OA | `0.5817245817` |
| Effusion | `0.6484472050` |
| Synovitis | `0.6654719235` |
| Baker's | `0.5452898551` |
| Contusion | `0.5398110661` |
| Fracture | `0.5916666667` |

The exact point improvement over B7-v1 is `+0.0247078534`; over B5 it is `+0.0401152095`.

## Paired comparison: B7-v1 -> B7.1

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
valid replicates   = 5000/5000
```

Interpretation: full-corpus coverage is strongly favored by the paired bootstrap, but the 95% paired interval still crosses zero.

## Paired comparison: B5 -> B7.1

```text
median difference = +0.0399233552
95% paired CI     = [-0.0301354430, +0.1092349994]
P(B7.1 > B5)      = 0.8716
valid replicates   = 5000/5000
```

Interpretation: B7.1 is the highest standalone development point estimate, while statistical superiority remains inconclusive on only 58 studies.

## Fixed B5+B7.1 rank ensemble — completed and rejected

A single global 50:50 percentile-rank ensemble was documented before evaluation.

Result:

```text
B7.1 macro AUC          0.5644802945
fixed rank ensemble     0.5540141184
point difference       -0.0104661761
```

Paired B7.1 -> ensemble:

```text
median(ensemble-B7.1) = -0.0105429030
95% paired CI         = [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)     = 0.3054
```

Decision: reject the ensemble as the campaign leader. Do not search 60:40, 70:30, raw-probability, calibrated or target-specific alternatives on the same 58 studies.

## Current follow-up: B8 spatial anatomy

B8 is a substantive architecture experiment rather than a blend/tuning experiment. It initializes from this exact B7.1 checkpoint and changes MRI memory from globally pooled slice tokens to a 2x2 spatial grid per sampled slice:

```text
B7.1: 6 x 16 x 1    = 96 tokens/study
B8:   6 x 16 x 2x2  = 384 tokens/study
```

B8 keeps the B6 supervision policy, target balancing, 3,120-study full coverage, four epochs and learning rates unchanged. It adds learned region-position embeddings and fixed gentle pathology stream/slice attention priors.

**Current status: B8 training is in progress. No B8 gold score is recorded yet.**

## Decision

Retain B7.1 as the current main standalone development model until a separately named experiment beats it under a fixed comparison.

Do not tune target-specific weak-label weights, target-specific model winners or ensemble weights from the 58 development labels.

## Validation caveat

The B6 gold audit informed the global B7 supervision policy, and the same 58 gold studies have supported repeated development decisions. Therefore B7.1 is a **development estimate**, not pristine independent validation. Gold labels did not enter B7.1 gradients or early stopping.
