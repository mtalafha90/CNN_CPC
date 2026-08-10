# Modeling strategy

> **Snapshot: 2026-08-10.** **B7.1 remains the best standalone development model at macro AUC `0.5644802945`. B8 spatial-anatomy learning is rejected at `0.5300962807`. B9 strict semantic sequence routing is the active predeclared experiment.** Canonical results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Core principle

`CNN_CPC` treats the challenge as a weakly supervised multi-sequence MRI problem with only 58 fully labelled development studies. The strategy prioritizes supervision quality, leakage control, representation quality, exact data semantics, full-corpus weak supervision, and runtime discipline before increasing model complexity.

## Experiment evidence so far

| Candidate | Macro AUC | Interpretation |
|---|---:|---|
| B0 random | `0.4763` | weak baseline |
| B1 strong MRI SSL | `0.5030` | useful in-domain representation |
| B4 frozen SSL + classical | `0.5138` | representation separability improved |
| B5 image-report SSL | `0.524365` | report-aligned representation helped modestly |
| B7-v1 direct B6 supervision | `0.539772` | direct weak supervision helped |
| **B7.1 full coverage** | **`0.564480`** | **current best standalone development model** |
| B5+B7.1 fixed rank | `0.554014` | rejected versus B7.1 |
| B8 spatial anatomy | `0.530096` | rejected; spatial-prior branch closed |
| **B9 strict routing** | pending | active label-free data-contract experiment |

## 1. Reports are training supervision only

Final inference remains MRI-only. B5 uses report semantics for representation learning; B6 converts reports to positive / negated / uncertain / unmentioned states. B7/B7.1/B9 train directly from the frozen B6 target cells.

Frozen B6 v1.2.1 scope:

```text
report-only rows                  4349
active weakly labelled studies    3120
usable cells                     14123
positive cells                    6871
negative cells                    7252
```

Global asymmetric policy:

| state | soft target | base weight |
|---|---:|---:|
| positive | 0.85 | 0.50 |
| negated | 0.05 | 1.00 |
| uncertain | ignored | 0.00 |
| unmentioned | ignored | 0.00 |

The parser and supervision policy are frozen.

## 2. B7.1 established the strongest current architecture

B7.1 uses:

```text
6 MRI streams
-> 16 sampled 2.5D slices/stream
-> ConvNeXt slice encoder initialized from B5
-> slice-position + stream embeddings
-> cross-sequence Transformer
-> 12 pathology queries
-> cross-attention to MRI memory
-> 12 logits
```

The only change from B7-v1 was full active-pool coverage:

```text
500 -> 1560 batches/epoch
1000 -> 3120 study draws/epoch
4 complete corpus passes
```

Result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

This is the architecture B9 returns to.

## 3. B8 result closes the spatial-prior branch

B8 preserved a 2x2 ConvNeXt grid per sampled slice and increased MRI memory from 96 to 384 tokens/study. Optimization was stable, but gold development performance fell:

```text
B8 AUC                 0.5300962807
B7.1 AUC               0.5644802945
median(B8 - B7.1)     -0.0335501423
95% paired CI         [-0.0900453633, +0.0223997827]
P(B8 > B7.1)           0.1156
```

Do not tune B8 spatial grids, anatomy priors, target-specific priors or blend weights from this result.

## 4. B9 motivation: the six-stream semantic contract was not exact

The intended streams are:

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

The historical selector attempted to populate both slots in a plane when multiple series existed. With two acquisitions of the same contrast type, one could be assigned to the opposite slot.

A label-free audit of all 4,407 training studies found:

```text
historical selected streams  21886
strict semantically valid     21334
cross-contrast substitutions    552
fraction wrong-slot            2.52%
```

Per-stream wrong-slot assignments:

```text
sagittal_fluid       251
sagittal_structural   28
coronal_fluid          2
coronal_structural    34
axial_fluid            0
axial_structural     237
```

The provided three-study test surface has one analogous false sagittal-fluid assignment. Historical routing selects 14 streams; strict routing selects 13 valid streams.

This finding uses only acquisition metadata, not target outcomes.

## 5. B9 single scientific change

B9 strict routing:

```text
fluid slot:
    choose only Fluid_Sensitive == True

structural slot:
    choose only Fluid_Sensitive == False

if the required contrast is unavailable:
    slot = None
    presence mask = False
```

Unknown contrast after metadata repair is not forced into either class.

The historical selector remains untouched so B7.1 is reproducible.

## 6. B9 keeps B7.1 otherwise fixed

Unchanged:

```text
B5 encoder initialization
B6 v1.2.1 labels and weights
KneeMILNet architecture
16 slices/stream
batch size 2
4 epochs
1560 batches/epoch
encoder LR 1e-5
head LR 1e-4
same augmentation
TTA [-1,0,1]
5000 bootstrap replicates
no gold gradients
no gold early stopping
```

The first B9 evaluation is one-shot. Primary comparison: paired B7.1 -> B9.

## 7. Validation discipline

The campaign has repeatedly used the same 58 studies, so results are model-selection/development estimates. Do not:

- tune target-specific routing from gold outcomes;
- restore individual substituted streams after seeing target AUCs;
- retune B6 rules or weak-label weights;
- select per-target winners;
- optimize ensemble weights;
- call development AUC a hidden-test guarantee.

## 8. Current next step

Test and train B9 exactly as documented in [`B9_STRICT_ROUTING.md`](B9_STRICT_ROUTING.md). Inspect `routing_audit.json`, `history.json` and `supervision_plan.json` before the first gold evaluation.
