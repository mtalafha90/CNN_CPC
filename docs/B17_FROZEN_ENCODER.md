# B17 — frozen B16 report-aligned encoder

> **Status — 2026-08-13:** COMPLETED / RETAINED BY PREDECLARED GLOBAL POINT-ESTIMATE RULE. Package `0.26.0`. B17 reused-gold macro AUC is `0.6425890153`; the gain over B16 is positive but statistically unresolved.

## Motivation

B16 was the reused-gold development champion at `0.6349770242`, but the paired comparison with B13 was unresolved. The post-B15 diagnostic also showed that a coarse B6 report-state ranking baseline reaches `0.7024597743` on the repeatedly reused gold surface. This is an information reference rather than an MRI ceiling, but it motivated testing whether downstream noisy B6 gradients were degrading useful report-aligned MRI features.

## Scientific question

```text
Does freezing the completed B16 report-aligned ConvNeXt encoder during short,
fixed B6 downstream training preserve expert-relevant MRI representation better
than B16 end-to-end fine-tuning?
```

## Representation path

```text
ImageNet ConvNeXt-Tiny
        -> B15 same-study knee-MRI SSL
        -> B16 full-report semantic alignment
        -> FROZEN MRI encoder
        -> B13/B16 hierarchical one-token-per-series model
        -> B6 positive/negated supervision
```

The report branch remains training-only. Test-time inference is MRI-only.

## Frozen B17-v1 contract

### Encoder

```text
source checkpoint
runs/b16_full_report/report_ssl/b16_report_encoder.pt

requires_grad                     false for every encoder parameter
optimizer membership              false
encoder training mode             false
runtime encoder checkpointing     false
encoder LR                        0
```

A deterministic SHA-256 fingerprint was checked after every epoch. It remained unchanged throughout training:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

### Downstream surface

```text
B6-active studies          3120
usable B6 cells           14123
positive cells             6871
negative cells             7252
eligible real series      17475
batches / epoch            1560
max series / study           14
```

Frozen series mapping:

```text
runs/b12_variable_series/audit/series_policy.json
SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

### B6 supervision

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

No gold-derived state probabilities were used.

### Architecture / augmentation

Unchanged B13/B16 hierarchy:

```text
16 sampled 2.5D positions / series
224 x 224
hierarchical learned one-token-per-series pooling
8-head series pooling
2-layer study Transformer
1 pathology-query layer
dropout 0.25
batch size 2
TTA [-1,0,1]
```

### Optimization

```text
encoder LR              0
head LR                  1e-4
minimum LR               1e-6
weight decay             1e-4
grad clip                1.0
epochs                    5 exact full passes
additional smoothing     0
robust loss              none
gold early stopping      none
gold checkpoint choice   none
weak-v2 gate             none
```

B17 reused the B16 seed offsets for hierarchy/head construction and DataLoader setup. Relative to B16 it changed both encoder optimization (`fine-tuned -> frozen`) and fixed training length (`4 -> 5` epochs), so this is a frozen-short-training protocol comparison rather than a mathematically pure one-variable freezing ablation.

## Completed training

All five epochs completed the exact frozen full-surface contract:

| Epoch | B6 loss | Head LR | Batches | Studies | Cells | Series | Encoder SHA stable | Full coverage | Budget limited |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | `0.7371836930` | `9.0546341e-5` | 1560 | 3120 | 14123 | 17475 | true | true | false |
| 2 | `0.6336947483` | `6.5796341e-5` | 1560 | 3120 | 14123 | 17475 | true | true | false |
| 3 | `0.6087776578` | `3.5203659e-5` | 1560 | 3120 | 14123 | 17475 | true | true | false |
| 4 | `0.5862506992` | `1.0453659e-5` | 1560 | 3120 | 14123 | 17475 | true | true | false |
| 5 | `0.5667051629` | `1.0e-6` | 1560 | 3120 | 14123 | 17475 | true | true | false |

Final checkpoint:

```text
runs/b17_frozen_encoder/b17_model.pt
```

The final B17 B6 loss (`0.5667051629`) is close to B16's final end-to-end loss (`0.5675074643`), showing that the trainable hierarchy/head can fit the frozen B6 surface nearly as strongly without changing the encoder. Training loss is not a model-selection metric.

## Single reused-gold development result

B17 completed its one frozen gold look with all 12 target AUCs defined and all 5000 bootstrap replicates usable:

```text
B17 macro AUC        0.6425890153
95% CI              [0.5935606351, 0.6887356582]
B16 macro AUC        0.6349770242
raw B17-B16          +0.0076119910
paired median        +0.0074330332
95% paired CI        [-0.0188853047, +0.0332991195]
P(B17 > B16)          0.7110
n                     58 studies
bootstrap             5000 / 5000 usable
```

Per-target B17 AUCs, descriptive only:

```text
ACL                0.4938725490
MCL                0.4376417234
Medial Meniscus    0.6947115385
Lateral Meniscus   0.6472049689
Medial OA          0.6992248062
Lateral OA         0.5976789168
PF OA              0.6435006435
Effusion           0.8385093168
Synovitis          0.8100358423
Baker's            0.7463768116
Contusion          0.5398110661
Fracture           0.5625000000
```

Descriptive B17-B16 target deltas:

```text
ACL               -0.00735
MCL               -0.02041
Medial Meniscus   +0.02885
Lateral Meniscus  -0.02112
Medial OA         +0.00465
Lateral OA        +0.01354
PF OA             +0.00644
Effusion          +0.00248
Synovitis         +0.05854
Baker's           +0.07065
Contusion         -0.03104
Fracture          -0.01389
```

These target-level differences are descriptive only and must not be used for B16/B17 target mixing.

Outputs:

```text
runs/b17_frozen_encoder/gold_confirmation/gold_predictions.csv
runs/b17_frozen_encoder/gold_confirmation/eval.json
```

The Transformer nested-tensor messages during evaluation are benign PyTorch warnings caused by `norm_first=True`; evaluation completed normally.

## Decision

The predeclared rule stated that B17 replaces B16 if its **global macro-AUC point estimate is higher**. Therefore:

```text
B17  0.6425890153
B16  0.6349770242
      -----------
delta +0.0076119910
```

Accordingly:

```text
B17 RETAIN by predeclared point-estimate rule
B17 = current reused-gold development champion
B16 = retained historical reference / statistically unresolved with B17
```

The paired interval crosses zero and `P(B17>B16)=0.711`. The evidence therefore does **not** establish true superiority. The scientifically correct interpretation is that B17 and B16 remain statistically unresolved on the repeatedly reused 58-study development set, while B17 has the higher frozen point estimate.

## Scientific interpretation

B17 gives modest support to the hypothesis that preserving the report-aligned MRI encoder can be preferable to continuing to adapt it directly to sparse/noisy B6 supervision. The effect is small and cannot be attributed solely to freezing because B17 also used five fixed epochs rather than B16's four.

The result is nevertheless directionally consistent with the idea that useful representation learning and weak-label fitting should be separated more carefully. A future robust-loss or label-smoothing experiment must be separately versioned and motivated before another gold look rather than tuned as a continuation of B17.

## Governance after the gold look

B17 is now closed. Do not perform any of the following from this gold result:

```text
no epoch-6 extension
no label-smoothing tuning
no ELR/SCE selection from gold
no head-LR tuning from gold
no target-specific B16/B17 winner mixing
no gold checkpoint selection
no regeneration of weak-v2
```

The 58-study gold surface is a repeatedly reused development/model-selection set, not independent validation. The hidden competition evaluation remains the next genuinely independent performance signal.
