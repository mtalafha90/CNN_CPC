# Experiment status

**Snapshot:** 2026-08-10  
**Package:** `0.13.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

This file is the canonical repository summary for measured experiment status. The 58-study set has supported repeated development decisions and should now be interpreted as a development/model-selection set rather than pristine independent validation.

## Current headline

- **Best standalone development point estimate:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`, 95% bootstrap CI `[0.5052432984, 0.6229422178]`.
- B7.1 improves the point estimate over B7-v1 (`0.5397724412`) by `+0.0247078534` and over B5 (`0.5243650851`) by `+0.0401152095`.
- Paired B7-v1 -> B7.1 bootstrap: median difference `+0.0241102714`, 95% CI `[-0.0140197876, +0.0660558004]`, `P(B7.1 > B7-v1)=0.8694`.
- Paired B5 -> B7.1 bootstrap: median difference `+0.0399233552`, 95% CI `[-0.0301354430, +0.1092349994]`, `P(B7.1 > B5)=0.8716`.
- The predeclared fixed B5+B7.1 50:50 rank ensemble scored `0.5540141184`, below B7.1, and is rejected with no blend-weight search.
- **B8 spatial-anatomy learning completed but underperformed B7.1:** macro AUC `0.5300962807`, 95% CI `[0.4723014866, 0.5867732651]`.
- Paired B7.1 -> B8 bootstrap: median difference `-0.0335501423`, 95% CI `[-0.0900453633, +0.0223997827]`, `P(B8 > B7.1)=0.1156`. The interval crosses zero, but the point estimate and paired direction favor B7.1 strongly.
- B8 improved only 3/12 target point estimates and declined on 9/12. These differences are descriptive only and are not used for target-specific model selection.
- **Decision:** reject B8-v1 as a replacement; retain B7.1 as the current main standalone development model. The B8 spatial-prior branch is closed to post-hoc tuning on these 58 labels.

## Completed measured experiments

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected as general MRI teacher |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected globally |
| B1+B3 rank | fixed 50:50 rank ensemble | `0.5048038179` | neutral |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | retained image-only ablation |
| B4.1 | shared policy per fold | `0.4847792672` | rejected |
| B4.2 | pathology-group policies | `0.4901328905` | rejected |
| B4.3 | target-wise two-way-CV selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | retained historical ensemble |
| B5 | image-report SSL representation + unchanged B4 probe | `0.5243650851` | retained representation baseline |
| B6 | multilingual structured report labels | n/a | completed; frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI model + frozen B6 weak labels, 500 batches/epoch | `0.5397724412` | retained coverage ablation |
| **B7.1** | **same B7 recipe with full 3,120-study coverage each epoch** | **`0.5644802945`** | **best standalone development point estimate** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected versus B7.1; no weight search |
| B8 | B7.1-init 2x2 within-slice spatial tokens + fixed soft pathology stream/slice priors | `0.5300962807` | rejected versus B7.1; spatial-prior branch closed |

## B6 weak supervision

B6 v1.2.1 converts multilingual reports into positive / negated / uncertain / unmentioned target states. Unmentioned is not treated as negative. The frozen training export contains:

```text
report-only studies       4349
active studies            3120
usable cells             14123
positive cells            6871
negative cells            7252
```

The completed 58-study audit showed very high sensitivity/NPV but noisier positive precision. This motivated one global asymmetric B7/B8 policy: positive soft target `0.85` with base weight `0.50`, negative soft target `0.05` with base weight `1.00`, confidence threshold `0.75`, uncertain/unmentioned ignored. The parser is frozen after the audit.

## B7-v1

B7-v1 uses the B5 competition-only encoder initialization and `KneeMILNet`: six MRI streams, 2.5D ConvNeXt slice features, slice-position/stream embeddings, cross-sequence Transformer, and 12 interacting pathology queries.

B7-v1 capped each epoch at 500 batches with batch size 2. Across four epochs it drew 4,000 studies, about 1.28 nominal passes over the 3,120 active pool.

```text
macro AUC = 0.5397724412
95% CI   = [0.4733481702, 0.6035621405]
```

Paired B5 -> B7-v1:

```text
median difference = +0.0155102430
95% paired CI     = [-0.0607472600, +0.0889531461]
P(B7 > B5)        = 0.6678
```

## B7.1 full coverage

B7.1 changed only:

```text
b7_max_batches_per_epoch: 500 -> 1560
```

With batch size 2, every epoch covered all 3,120 active studies and all 14,123 usable weak-supervision cells. Four epochs completed 12,480 study draws with no budget limiting.

Training loss:

```text
epoch 1  0.7524191749
epoch 2  0.6651707418
epoch 3  0.6391165589
epoch 4  0.6127582232
```

Gold development result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

Per-target AUC:

```text
ACL               0.5159313725
MCL               0.4693877551
Medial Meniscus   0.5841346154
Lateral Meniscus  0.5950310559
Medial OA         0.4604651163
Lateral OA        0.5764023211
PF OA             0.5817245817
Effusion          0.6484472050
Synovitis         0.6654719235
Baker's           0.5452898551
Contusion         0.5398110661
Fracture          0.5916666667
```

## B8 spatial anatomy — completed / rejected

B8 initialized from B7.1 and increased MRI memory from 96 to 384 tokens by retaining a `2x2` ConvNeXt spatial grid for each sampled slice. It added fixed gentle pathology-specific stream/slice attention priors while leaving the frozen B6 supervision, full 3,120-study coverage and four-epoch optimization recipe unchanged.

Training completed cleanly:

```text
epoch 1 loss 0.6707552306
epoch 2 loss 0.6445401128
epoch 3 loss 0.6186956850
epoch 4 loss 0.5997290100
batches/epoch 1560
studies/epoch 3120
usable cells/epoch 14123
budget limited false
```

Gold development result:

```text
macro AUC = 0.5300962807
95% CI   = [0.4723014866, 0.5867732651]
```

Per-target AUC:

```text
ACL               0.4767156863
MCL               0.4739229025
Medial Meniscus   0.6045673077
Lateral Meniscus  0.5801242236
Medial OA         0.5317829457
Lateral OA        0.4100580271
PF OA             0.5135135135
Effusion          0.5788819876
Synovitis         0.5818399044
Baker's           0.5181159420
Contusion         0.5249662618
Fracture          0.5666666667
```

Relative to B7.1, B8 improved only MCL, Medial Meniscus and Medial OA point estimates; the other nine targets declined. This is descriptive only and must not be used to create a target-specific B7.1/B8 hybrid.

Paired B7.1 -> B8:

```text
B7.1 macro AUC       0.5644802945
B8 macro AUC         0.5300962807
point difference     -0.0343840138
median difference    -0.0335501423
95% paired CI        [-0.0900453633, +0.0223997827]
P(B8 > B7.1)         0.1156
P(B7.1 > B8)         0.8844
```

Decision: reject B8 as a replacement for B7.1. Do not tune spatial-grid size, anatomy-prior strength, target-specific priors, epoch count, target-specific winners or blend weights from this result.

## Key paired comparisons

### B5 versus B4

```text
B4 macro AUC      0.5137567459
B5 macro AUC      0.5243650851
median difference +0.0105821232
95% paired CI    [-0.0408197338, +0.0622131599]
P(B5 > B4)        0.656
```

### B7-v1 versus B7.1

```text
B7-v1 macro AUC   0.5397724412
B7.1 macro AUC    0.5644802945
median difference +0.0241102714
95% paired CI    [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   0.8694
```

### B5 versus B7.1

```text
B5 macro AUC      0.5243650851
B7.1 macro AUC    0.5644802945
median difference +0.0399233552
95% paired CI    [-0.0301354430, +0.1092349994]
P(B7.1 > B5)      0.8716
```

### B7.1 versus fixed B5+B7.1 rank ensemble

```text
B7.1 macro AUC             0.5644802945
fixed rank ensemble AUC    0.5540141184
median(ensemble - B7.1)   -0.0105429030
95% paired CI             [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)         0.3054
```

### B7.1 versus B8

```text
B7.1 macro AUC       0.5644802945
B8 macro AUC         0.5300962807
median(B8 - B7.1)   -0.0335501423
95% paired CI       [-0.0900453633, +0.0223997827]
P(B8 > B7.1)         0.1156
```

## Decision policy from here

1. Keep B7.1 as the main standalone development model.
2. Close the B8 spatial-prior branch; do not retune it from target-level gold outcomes.
3. Do not tune B6 parser rules, target-specific weak-label weights, target-specific model winners, or ensemble weights from the 58 gold labels.
4. The fixed B5+B7.1 ensemble question is closed.
5. New trained variants must be explicitly named and interpreted as additional development on the same 58-study set.
6. Prefer substantive supervision/representation improvements that preserve the successful B7.1 global-token architecture over further post-hoc spatial-prior tuning.
7. Report paired bootstrap uncertainty with every comparison.
8. Actual competition leaderboard performance remains unknown until a real submission has been made.
