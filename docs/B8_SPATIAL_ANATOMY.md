# B8 — pathology-aware spatial anatomy learning

> **Status — 2026-08-10:** **REAL-DATA TRAINING COMPLETE / GOLD DEVELOPMENT EVALUATION PENDING.** The B8-v1 recipe was frozen before training and before its first 58-study gold development evaluation. All four predefined full-corpus epochs completed successfully with no budget limiting. No B8 gold score has been recorded yet.

## Motivation

B7.1 is the current best standalone development model:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

B7.1 improved the point estimate over B7-v1 after correcting a pre-identified training-coverage limitation. Its six-stream ConvNeXt/Transformer/pathology-query architecture still globally pools every sampled 2.5D slice to one vector before MRI-token attention. That means within-slice spatial information is discarded before pathology-specific evidence selection.

B8 tests whether retaining coarse within-slice spatial structure improves pathology learning while keeping the successful B7.1 weak-supervision recipe fixed.

## Single architecture direction

B7.1 MRI memory:

```text
6 streams x 16 slices x 1 pooled token = 96 MRI tokens
```

B8 MRI memory:

```text
6 streams x 16 slices x 2x2 regions = 384 MRI tokens
```

The B8 ConvNeXt encoder reuses the B7.1 weights. Instead of applying only global average pooling, B8 takes the final ConvNeXt feature map, adaptive-pools it to a `2x2` grid, applies the same learned ConvNeXt classifier normalization, and emits four spatial tokens per sampled 2.5D slice.

Each token receives the inherited slice-position embedding, inherited stream embedding, and a new learned region-position embedding. The inherited MRI Transformer contextualizes the 384-token memory and the inherited 12 pathology queries cross-attend to it.

## Soft anatomy priors

B8 applies a fixed additive attention-logit prior for each pathology query. The prior is deliberately soft:

- preferred MRI streams have prior weight `1.0`;
- non-preferred streams retain prior weight `0.75`;
- focal internal structures receive only a broad center-slice preference with floor `0.80`;
- diffuse/fluid findings are slice-neutral;
- no MRI stream or slice is hard-masked.

Predeclared stream preferences:

| Target | Preferred streams |
|---|---|
| ACL | sagittal fluid, sagittal structural, coronal fluid |
| MCL | coronal fluid, coronal structural |
| Medial Meniscus | sagittal fluid/structural, coronal fluid/structural |
| Lateral Meniscus | sagittal fluid/structural, coronal fluid/structural |
| Medial OA | coronal structural/fluid, sagittal structural |
| Lateral OA | coronal structural/fluid, sagittal structural |
| PF OA | axial fluid/structural, sagittal structural |
| Effusion | fluid-sensitive sagittal/coronal/axial |
| Synovitis | fluid-sensitive sagittal/coronal/axial |
| Baker's | sagittal fluid, axial fluid, coronal fluid |
| Contusion | fluid-sensitive sagittal/coronal/axial |
| Fracture | structural sagittal/coronal/axial |

These priors were defined from general knee MRI anatomy and sequence sensitivity, not from target-specific B5/B7/B7.1 development AUCs.

### Why no fixed in-plane quadrant prior

The current preprocessing does not certify a canonical left/right or anterior/posterior pixel orientation across every selected series. Therefore B8 does not hard-code a quadrant as medial, lateral, anterior or posterior. The fixed prior is uniform across the four in-plane regions, while the region embeddings and pathology queries learn spatial preferences from weak supervision.

## Initialization

B8 initializes from the completed B7.1 checkpoint:

```text
runs/b7_1_full_coverage/b7_model.pt
```

The loader enforces:

```text
implementation variant          b7_b5_init_b6_asymmetric_weak_v1
experiment name                 B7.1_full_coverage
completed epochs                4
batches per epoch               1560
training studies                3120
training usable cells           14123
gold studies in gradient        0
gold early stopping             0
```

All compatible B7.1 parameters are copied into B8. Only the new region embedding and fixed anatomy-bias buffer are absent from B7.1 and initialized by B8.

## Frozen weak supervision

B8 keeps B6 v1.2.1 unchanged:

```text
active weakly labelled studies = 3120
usable cells                   = 14123
positive cells                  = 6871
negative cells                  = 7252
```

The asymmetric global policy remains:

| B6 state | Soft target | Base weight |
|---|---:|---:|
| positive | 0.85 | 0.50 |
| negated | 0.05 | 1.00 |
| uncertain | ignored | 0.00 |
| unmentioned | ignored | 0.00 |

Target-balance multipliers match the frozen B7.1 supervision mass contract.

## Frozen B8-v1 training recipe

Configuration:

```text
configs/b8_spatial_anatomy.yaml
```

Key values:

```text
spatial grid                 2x2
MRI memory tokens/study      384
batch size                   2
epochs                       4
batches/epoch                1560
study draws/epoch            3120
encoder LR                   1e-5
head LR                      1e-4
minimum LR                   1e-6
weight decay                 1e-4
grad clip                    1.0
anatomy prior strength       1.0
nonpreferred stream prior    0.75
slice prior floor            0.80
```

No gold labels are used for training loss or early stopping.

## Completed real-data training

Command:

```bash
rsna-knee-b8 \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --b71-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b8_spatial_anatomy
```

Checkpoint:

```text
runs/b8_spatial_anatomy/b8_model.pt
```

Training history:

| Epoch | Loss | Batches | Study draws | Active cells | Positive | Negative | Seconds | Budget limited |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.6707552306` | 1560 | 3120 | 14123 | 6871 | 7252 | 2239.67 | false |
| 2 | `0.6445401128` | 1560 | 3120 | 14123 | 6871 | 7252 | 3071.56 | false |
| 3 | `0.6186956850` | 1560 | 3120 | 14123 | 6871 | 7252 | 3256.41 | false |
| 4 | `0.5997290100` | 1560 | 3120 | 14123 | 6871 | 7252 | 2724.49 | false |

Totals:

```text
completed epochs          4
batches                 6240
study draws            12480
nominal corpus passes      4.0
usable cells/epoch       14123
positive cells/epoch      6871
negative cells/epoch      7252
training seconds        11292.13
budget limited            false
```

The weak-training loss decreased monotonically from `0.6707552306` to `0.5997290100`. This confirms stable optimization but is not evidence by itself that gold macro AUC improved.

The Transformer nested-tensor warning and the PyTorch warning about differing `key_padding_mask`/`attn_mask` dtypes were non-fatal; the frozen B8-v1 run completed all predefined epochs. The mask-dtype warning should be cleaned up in a later implementation revision, not by changing this completed B8-v1 checkpoint before its evaluation.

## Gold development evaluation — next step

The training artifacts have now been inspected and satisfy the frozen contract. The first B8-v1 gold evaluation may proceed without changing model parameters:

```bash
rsna-knee-b8-eval \
  --config configs/b8_spatial_anatomy.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b8_spatial_anatomy/b8_model.pt \
  --out-root runs/b8_spatial_anatomy/gold_eval
```

A runtime-only `num_workers: 0` / `persistent_workers: false` copy of the config may be used if worker teardown is noisy; this does not change the scientific model.

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Primary statistical test: paired B7.1 -> B8 study-level bootstrap with 5,000 replicates.

## Decision rule

B8-v1 is evaluated once under this frozen rule. Do not search spatial grid sizes, anatomy-prior strengths, target-specific priors, epochs or blend weights on the 58-study development labels and still call the result B8-v1.

Because B8 was designed after prior development results on the same 58 studies, its gold score will be a further development estimate, not independent validation.
