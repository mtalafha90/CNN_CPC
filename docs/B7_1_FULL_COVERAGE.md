# B7.1 — full-corpus weak-supervision coverage

> **Status — 2026-08-12:** **COMPLETED / RETAINED HISTORICAL BENCHMARK.** B7.1 gold macro AUC is `0.5644802945`. It was later surpassed by B13 (`0.6293565948`). B15 has since shown that much stronger B6-teacher agreement does not necessarily transfer to expert-gold macro AUC.

## Motivation

B7-v1 produced macro AUC `0.5397724412` but capped each epoch at 500 batches with batch size 2, giving only about 1.28 nominal corpus passes over four epochs. B7.1 tested the pre-identified coverage limitation directly.

## Single scientific change

```text
b7_max_batches_per_epoch: 500 -> 1560
```

With 3,120 active studies and batch size 2, 1,560 batches are one complete shuffled pass through the active weak-training pool. Four epochs therefore provide four nominal full-corpus passes.

Everything else remained fixed from B7-v1: B5 encoder initialization, frozen B6 v1.2.1 labels, asymmetric soft-target policy, target balancing, six-stream 2.5D ConvNeXt + Transformer + pathology-query model, optimizer, augmentation, TTA and bootstrap.

## Completed training

| Epoch | Loss | Batches | Study draws | Active cells | Positive | Negative |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.7524191749` | 1560 | 3120 | 14123 | 6871 | 7252 |
| 2 | `0.6651707418` | 1560 | 3120 | 14123 | 6871 | 7252 |
| 3 | `0.6391165589` | 1560 | 3120 | 14123 | 6871 | 7252 |
| 4 | `0.6127582232` | 1560 | 3120 | 14123 | 6871 | 7252 |

All epochs completed without budget limiting.

Checkpoint:

```text
runs/b7_1_full_coverage/b7_model.pt
```

## Reused-gold result

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984,0.6229422178]
```

Paired versus B7-v1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876,+0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

Paired versus B5:

```text
median difference = +0.0399233552
95% paired CI     = [-0.0301354430,+0.1092349994]
P(B7.1 > B5)      = 0.8716
```

Full corpus coverage therefore remained a strongly favored training-design change, although the 58-study paired intervals were wide.

## Fixed B5+B7.1 rank ensemble

The predeclared global 50:50 rank ensemble scored `0.5540141184`, below B7.1, with `P(ensemble>B7.1)=0.3054`. That branch remains closed; no blend-weight search followed.

## Successor context through B15

```text
B12 gold  0.5660915179
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 created a matched B13-v2 control on a frozen 2,497-study weak-train partition and applied ImageNet -> knee-MRI same-study contrastive SSL before the same B13 downstream hierarchy.

Weak-v2 teacher agreement:

```text
control  0.5652498118
B15     0.7319060415
paired median +0.1675245839
95% CI [+0.1124433208,+0.2165156305]
```

Yet B15 did not improve the expert-gold global metric. This changes the interpretation of the B7-era success: complete weak-corpus coverage matters, but stronger optimization of the current weak target surface is not by itself sufficient for expert-label improvement.

## Current decision

B7.1 remains an important historical benchmark and a clean demonstration of full weak-training coverage. It is not the current champion.

Do not retune B7.1 target weights, epochs, or target-specific model mixtures from the reused gold set.

The current next evidence step is a direct audit of B6 report states (`positive`, `negated`, `uncertain`, `unmentioned`) against expert truth before a separately versioned supervision successor is defined.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B15 record: [`B15_MRI_SSL.md`](B15_MRI_SSL.md).