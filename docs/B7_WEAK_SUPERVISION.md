# B7 — B5-initialized MRI model trained on frozen B6 weak labels

> **Status — 2026-08-10:** **COMPLETE / RETAINED COVERAGE ABLATION.** B7-v1 completed the frozen four-epoch recipe and reached macro ROC AUC `0.5397724412` on the 58-study development set. It is no longer the campaign leader: B7.1 full-corpus coverage reached `0.5644802945`. B8 spatial-anatomy learning is currently training.

## Measured B7-v1 result

Training completed all four predefined epochs with 500 batches per epoch and no runtime-budget limiting:

```text
epoch 1 loss = 0.8393994069
epoch 2 loss = 0.7084098365
epoch 3 loss = 0.6809013321
epoch 4 loss = 0.6433874173
```

The run made 4,000 study draws total. With 3,120 active weakly labelled studies and batch size 2, this corresponds to about `1.28` nominal corpus passes.

Supervision scope:

```text
report-only studies             4349
active weakly labelled studies  3120
inactive zero-usable studies    1229
usable target cells            14123
positive cells                  6871
negative cells                  7252
MRI-filter losses                   0
```

The fixed target-balancing policy produced approximately equal total balanced supervision mass (`890.625`) for each of the 12 targets.

Gold development evaluation with TTA `[-1,0,1]`:

```text
macro AUC = 0.5397724412
95% CI   = [0.4733481702, 0.6035621405]
bootstrap = 5000/5000 valid
n = 58
```

Per-target AUC:

| Target | B7-v1 AUC |
|---|---:|
| ACL | `0.4828431373` |
| MCL | `0.3945578231` |
| Medial Meniscus | `0.5576923077` |
| Lateral Meniscus | `0.5341614907` |
| Medial OA | `0.4480620155` |
| Lateral OA | `0.5899419729` |
| PF OA | `0.6396396396` |
| Effusion | `0.6211180124` |
| Synovitis | `0.6535244922` |
| Baker's | `0.5181159420` |
| Contusion | `0.4723346829` |
| Fracture | `0.5652777778` |

## Comparison with B5

B5 macro AUC:

```text
0.5243650851
```

B7-v1 point delta:

```text
+0.0154073561
```

Paired B5 -> B7-v1:

```text
median difference = +0.0155102430
95% paired CI     = [-0.0607472600, +0.0889531461]
P(B7-v1 > B5)     = 0.6678
```

Interpretation: B7-v1 improved the point estimate but the paired evidence was statistically inconclusive.

## Goal

B7 is the first experiment that uses the 4,349 report-only studies as direct target-level MRI supervision rather than only for representation alignment.

Architecture:

```text
6 MRI streams
-> 2.5D ConvNeXt slice encoder initialized from B5
-> slice-position + stream embeddings
-> cross-sequence Transformer
-> 12 interacting pathology queries
-> cross-attention to MRI memory
-> 12 logits
```

B5 initializer:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Frozen B6 source:

```text
runs/b6_report_labels_v121/training_targets.csv
```

Gold studies are excluded from the B7 training loss and are not used for early stopping.

## Why asymmetric supervision

The frozen B6 gold audit produced high sensitivity/NPV but lower positive precision. B7-v1 therefore uses one global asymmetric policy across all 12 targets:

| B6 state | soft target | base weight |
|---|---:|---:|
| positive | `0.85` | `0.50` |
| negated | `0.05` | `1.00` |
| uncertain | ignored | `0.00` |
| unmentioned | ignored | `0.00` |

Only B6 cells with confidence `>=0.75` are used. These values are frozen.

## Target-balanced loss

B7 applies one fixed multiplier per target:

```text
target_multiplier_j = mean(total_base_mass) / total_base_mass_j
```

The optimization objective is weighted soft-label BCE. There is no gold-based ranking loss or gold calibration.

## Fixed B7-v1 recipe

```text
n_slices                 16
image_size               224
batch_size               2
epochs                    4
max_batches_per_epoch     500
encoder_lr                1e-5
head_lr                   1e-4
weight_decay              1e-4
grad_clip                 1.0
TTA                       [-1,0,1]
bootstrap replicates      5000
```

## Leakage/development contract

B7 enforces:

- competition-only B5 initializer;
- B5 gold studies used = 0;
- no external image pretraining;
- B6 exactly v1.2.1;
- B6 training export contains zero gold rows;
- B6 confidence threshold remains `0.75`;
- B6 UID set matches the official non-gold UID set;
- gold labels never enter B7 gradient;
- no gold-based early stopping.

Because the B6 gold audit informed the global B7 supervision policy, B7 performance on the same 58 studies is a **development estimate**, not independent validation.

## Why B7.1 followed

Before interpreting the B7-v1 gold result as final, the supervision/training audit exposed the 500-batch coverage limitation. B7.1 was therefore declared as a separate experiment changing only:

```text
500 -> 1560 batches/epoch
```

B7.1 result:

```text
macro AUC = 0.5644802945
```

Paired B7-v1 -> B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

B7-v1 is therefore retained as the direct weak-supervision/coverage ablation, while B7.1 is the current main standalone model.

## Current downstream status

The predeclared B5+B7.1 rank ensemble was tested once and rejected (`0.5540141184` versus B7.1 `0.5644802945`). No blend search followed.

The current substantive follow-up is B8 spatial-anatomy learning, which initializes from B7.1 and retains 2x2 within-slice spatial tokens before pathology attention. B8 training is in progress; no B8 gold score is recorded yet.

## Reproduction

```bash
rsna-knee-b7 \
  --config configs/b7.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b7_weak_supervision
```

Gold development evaluation:

```bash
rsna-knee-b7-eval \
  --config configs/b7.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b7_weak_supervision/b7_model.pt \
  --out-root runs/b7_weak_supervision/gold_eval
```

## Decision rule

B7-v1 is frozen. Do not alter supervision weights, soft labels, training duration or architecture and still call the result B7-v1. B7.1 and B8 are separately named development experiments.
