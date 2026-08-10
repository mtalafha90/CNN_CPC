# B7.1 — full-corpus weak-supervision coverage

> **Status — 2026-08-10:** **COMPLETE / BEST STANDALONE DEVELOPMENT POINT ESTIMATE.**

## Motivation

B7-v1 produced macro AUC `0.5397724412` on the 58-study gold development set, versus B5 `0.5243650851`. Before the B7-v1 gold result was inspected, its supervision audit had already exposed a coverage limitation: B7-v1 trained on 3,120 active weakly labelled studies but capped each epoch at 500 batches with batch size 2, giving only 1,000 study draws per epoch and about 1.28 nominal corpus passes over four epochs.

B7.1 tests that pre-identified limitation directly.

## Single scientific change

B7.1 changes only:

```text
b7_max_batches_per_epoch: 500 -> 1560
```

With 3,120 active studies and batch size 2, 1,560 batches are one complete shuffled pass through the active weak-training pool. Four epochs therefore provide four nominal full corpus passes.

Everything else remains fixed from B7-v1: B5 encoder initialization, frozen B6 v1.2.1 labels, the global asymmetric soft-target policy, target balancing, six-stream 2.5D ConvNeXt + cross-sequence Transformer + pathology-query architecture, four epochs, learning rates, cosine schedule, augmentation, three-view gold TTA, and 5,000 bootstrap replicates.

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
active studies       3120
usable cells        14123
study draws         12480
nominal corpus passes 4.0
budget limited       false
```

The loss decreased monotonically and the complete weak-supervision cell set was seen once per epoch.

Checkpoint:

```text
runs/b7_1_full_coverage/b7_model.pt
```

## Gold development evaluation

B7.1 achieved:

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

Using A=B7-v1 and B=B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
valid replicates   = 5000/5000
```

Interpretation: full-corpus coverage is favored strongly in the paired bootstrap, but the 95% paired interval still crosses zero.

## Paired comparison: B5 -> B7.1

Using A=B5 and B=B7.1:

```text
median difference = +0.0399233552
95% paired CI     = [-0.0301354430, +0.1092349994]
P(B7.1 > B5)      = 0.8716
valid replicates   = 5000/5000
```

Interpretation: B7.1 is now the highest standalone development point estimate. The paired evidence favors B7.1 over B5, but remains statistically inconclusive on only 58 studies.

## Decision

Retain B7.1 as the current main standalone model. Do not tune target-specific weak-label weights, target-specific model winners, or ensemble weights from these 58 labels.

A subsequent fixed ensemble, if tested, must use one global predeclared rule across all 12 targets. Because B5 uses nested logistic-probe probabilities and B7.1 uses neural weak-supervision probabilities, a fixed 50:50 per-target rank average is the preferred calibration-robust ensemble test. No weight search is allowed.

## Validation caveat

The B6 gold audit informed the global B7 supervision policy, and the same 58 gold studies have supported repeated development decisions. Therefore B7.1 is a **development estimate**, not pristine independent validation. Gold labels did not enter B7.1 gradients or early stopping.
