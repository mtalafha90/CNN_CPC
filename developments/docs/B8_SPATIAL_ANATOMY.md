# B8 — pathology-aware spatial anatomy learning

> **Status — 2026-08-12:** **COMPLETED / REJECTED GLOBALLY.** B8 gold macro AUC was `0.5300962807`, below B7.1. B13 is now the reused-gold development champion, and B15 has since shifted the immediate diagnostic priority toward weak-supervision quality.

## Scientific question

B7.1 globally pools every sampled 2.5D slice to one vector. B8 tested whether retaining coarse within-slice spatial structure and adding gentle pathology-specific priors would improve pathology ranking while leaving the successful B7.1 weak-supervision recipe fixed.

```text
B7.1 MRI memory = 6 x 16 x 1   = 96 tokens/study
B8 MRI memory   = 6 x 16 x 2x2 = 384 tokens/study
```

B8 reused the B7.1 ConvNeXt weights, added learned region-position embeddings, and kept broad soft stream/slice priors rather than hard masks.

## Frozen supervision/training controls

```text
B6 v1.2.1
3120 active weak studies
14123 usable cells
6871 positive / 7252 negative
4 full epochs
same optimizer/LR/augmentation
TTA [-1,0,1]
zero gold gradients
zero gold early stopping
```

## Completed training

```text
epoch 1 loss  0.6707552306
epoch 2 loss  0.6445401128
epoch 3 loss  0.6186956850
epoch 4 loss  0.5997290100
```

Training was stable and monotonic.

## Reused-gold result

```text
B8 macro AUC          0.5300962807
95% CI               [0.4723014866,0.5867732651]
B7.1 macro AUC        0.5644802945
median(B8-B7.1)      -0.0335501423
95% paired CI        [-0.0900453633,+0.0223997827]
P(B8 > B7.1)          0.1156
```

B8 improved only 3 of 12 target point estimates. Those target-level differences are descriptive and were not used for target-wise winner selection.

Decision: reject B8 as a global replacement. The paired interval crosses zero, so no definitive inferiority claim is made, but there is no evidence to retain B8 over B7.1.

## Successor context through B15

```text
B9 gold   0.5334962669
B10 gold  0.5523982721
B12 gold  0.5660915179
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B14 later showed that even much more slice-token memory did not improve global AUC. B15 then improved frozen weak-v2 teacher agreement dramatically (`0.7319060415` vs matched control `0.5652498118`) without improving expert-gold macro AUC.

Together, these results make another spatial-token/prior sweep a low-priority direction compared with direct supervision-state diagnosis.

## Decision discipline

Do not tune B8 grid size, anatomy priors, prior strength, epochs, per-target B7.1/B8 winners or blend weights from reused gold.

Current next step: audit B6 `positive`, `negated`, `uncertain`, and `unmentioned` report states against expert truth before defining a new supervision policy.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).