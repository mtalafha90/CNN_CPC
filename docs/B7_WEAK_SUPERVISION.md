# B7 — B5-initialized MRI model trained on frozen B6 weak labels

> **Status — 2026-08-10:** **COMPLETED / GOLD DEVELOPMENT EVALUATED.** B7-v1 completed the frozen four-epoch training recipe and reached macro ROC AUC `0.5397724412` on the 58-study gold development set, 95% bootstrap CI `[0.4733481702, 0.6035621405]`. This is the highest standalone point estimate so far. The gold set remains a development set because the B6 gold audit informed the fixed global B7-v1 supervision policy.

## Measured B7-v1 result

Training completed all four predefined epochs with 500 batches per epoch and was not runtime-budget limited:

```text
epoch 1 loss = 0.8393994069
epoch 2 loss = 0.7084098365
epoch 3 loss = 0.6809013321
epoch 4 loss = 0.6433874173
```

The run made 4,000 study draws total. With 3,120 active weakly labelled studies and batch size 2, this corresponds to about `1.28` nominal corpus passes. This coverage limitation was identified before the gold evaluation and can motivate a separately named coverage experiment, but B7-v1 itself is frozen.

Supervision scope:

```text
report-only studies             4,349
active weakly labelled studies  3,120
inactive zero-usable studies    1,229
usable target cells            14,123
positive cells                  6,871
negative cells                  7,252
MRI-filter losses                   0
```

The fixed target-balancing policy produced approximately equal total balanced supervision mass (`890.625`) for each of the 12 targets.

Gold development evaluation with TTA `[-1, 0, 1]`:

```text
macro AUC = 0.5397724411790162
95% CI    = [0.47334817023085735, 0.6035621405473302]
bootstrap = 5000/5000 valid
n          = 58 studies
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

Relative to B5 (`0.5243650851`), B7-v1 increases the macro point estimate by approximately `+0.01541`. This difference must be evaluated with a paired study-level bootstrap before claiming an improvement because both methods were evaluated on the same 58 studies.

## Goal

B7 is the first experiment that uses the 4,349 report-only studies as direct target-level MRI supervision rather than only for representation alignment.

The model is the repository's existing `KneeMILNet`:

```text
6 MRI streams
  -> 2.5D ConvNeXt slice encoder initialized from B5
  -> slice-position + stream embeddings
  -> cross-sequence Transformer
  -> 12 interacting pathology queries
  -> 12 logits
```

The B5 encoder is initialized from:

```text
runs/b5_report_ssl/b5_encoder.pt
```

The weak labels come only from frozen B6 v1.2.1:

```text
runs/b6_report_labels_v121/training_targets.csv
```

Gold studies are excluded from the B7 training loss and are not used for early stopping.

## Why asymmetric supervision

The frozen B6 gold audit covered 251 usable cells and produced:

```text
TP = 116
TN = 80
FP = 52
FN = 3

pooled positive precision = 0.6905
pooled sensitivity        = 0.9748
pooled specificity        = 0.6061
pooled NPV                = 0.9639
pooled balanced accuracy  = 0.7904
```

B6 is therefore much stronger at explicit negatives than at explicit positives. B7-v1 uses a single global asymmetric policy across all 12 targets:

| B6 state | B7 soft target | B7 base weight |
|---|---:|---:|
| positive | `0.85` | `0.50` |
| negated | `0.05` | `1.00` |
| uncertain | ignored | `0.00` |
| unmentioned | ignored | `0.00` |

Only B6 cells with confidence `>=0.75` are used.

These values are frozen in code and `configs/b7.yaml`. Changing them requires a new named experiment rather than silently retuning B7-v1.

## Target-balanced loss

B7 first applies the asymmetric cell weights above, then computes one fixed multiplier per target:

```text
target_multiplier_j = mean(total_base_mass) / total_base_mass_j
```

Thus, over the full B7 training pool, each pathology has equal expected total supervision mass despite large differences in report-label frequency.

The optimization loss is weighted soft-label BCE. There is no gold-based ranking loss or gold calibration in B7-v1.

## Fixed training recipe

`configs/b7.yaml` declares:

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
TTA for gold evaluation   [-1, 0, 1]
bootstrap replicates      5000
```

The ConvNeXt encoder uses the B5 checkpoint's input-normalization contract. No external image weights are introduced.

## Leakage / development contract

B7 training enforces all of the following:

- B5 checkpoint must be the competition-only `b5_image_report_tfidf_svd` variant;
- B5 must certify `gold_studies_used=0`;
- B5 must certify no external image pretraining;
- B6 must be exactly v1.2.1;
- B6 audit must certify zero gold rows in `training_targets.csv`;
- B6 confidence threshold must remain `0.75`;
- the B6 training UID set must match the official non-gold UID set exactly;
- studies with no usable B6 target cells are excluded from B7 optimization;
- gold labels never enter the B7 gradient;
- no gold-based early stopping is used.

The B6 gold audit did inform the global B7-v1 positive/negative reliability policy. Therefore B7 performance on the same 58 gold studies is explicitly a **development estimate**, not untouched independent validation.

## Install and test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull
python -m pip install -e .

pytest -q \
  tests/test_b6_report_labels.py \
  tests/test_b6_gold_audit.py \
  tests/test_b7_weak_supervision.py
```

## Train B7-v1

```bash
rsna-knee-b7 \
  --config configs/b7.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b7_weak_supervision
```

The run writes:

```text
runs/b7_weak_supervision/
├── b7_model.pt
├── history.json
├── policy.json
└── supervision_plan.json
```

`b7_model.pt` is refreshed after every completed epoch.

## Gold development evaluation

```bash
rsna-knee-b7-eval \
  --config configs/b7.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b7_weak_supervision/b7_model.pt \
  --out-root runs/b7_weak_supervision/gold_eval
```

Outputs:

```text
runs/b7_weak_supervision/gold_eval/
├── gold_predictions.csv
└── eval.json
```

## Decision rule

B7-v1 is now frozen. Do not alter its supervision weights, soft labels, training duration, or architecture and still call the result B7-v1. Any coverage extension or materially different policy must be recorded as a new experiment (for example B7.1) and interpreted as additional development on the same 58-study set.
