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
- Both paired intervals still cross zero; superiority is therefore not statistically conclusive on only 58 studies.
- The predeclared fixed B5+B7.1 50:50 rank ensemble scored `0.5540141184`, below B7.1, and is rejected with no blend-weight search.
- **B8 spatial-anatomy training is now complete.** Four full epochs covered all 3,120 active weakly labelled studies and all 14,123 usable weak-label cells per epoch, with no budget limiting and monotonic weak-training loss `0.6707552306 -> 0.5997290100`.
- **No B8 gold-development score has been recorded yet.** B7.1 remains the current leader until the predeclared one-shot B8 evaluation is completed.

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

## Trained experiment awaiting first gold evaluation

| ID | Method | Status |
|---|---|---|
| **B8** | **B7.1-init 2x2 within-slice spatial tokens + fixed soft pathology stream/slice priors; same B6/full-coverage training** | **training complete; artifacts inspected; one-shot gold evaluation pending** |

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

## B8 spatial anatomy — completed training

B8 initializes from the completed B7.1 checkpoint and changes the MRI memory before pathology-query cross-attention:

```text
B7.1: 6 streams x 16 slices x 1 globally pooled token = 96 MRI tokens
B8:   6 streams x 16 slices x 4 spatial regions       = 384 MRI tokens
```

The fixed anatomy prior is soft. Preferred streams have prior `1.0`, nonpreferred streams `0.75`; focal targets have a broad center-slice prior with floor `0.80`, while diffuse/fluid findings are slice-neutral. The four in-plane regions have no hard-coded anatomical direction because the current preprocessing does not certify canonical within-slice orientation.

B8 keeps the same frozen B6 v1.2.1 weak labels, target balancing, 3,120-study full coverage, four epochs, learning-rate schedule, augmentations, and zero gold-gradient/early-stopping use.

Completed training:

| Epoch | Loss | Batches | Study draws | Active cells | Positive | Negative | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.6707552306` | 1560 | 3120 | 14123 | 6871 | 7252 | 2239.67 |
| 2 | `0.6445401128` | 1560 | 3120 | 14123 | 6871 | 7252 | 3071.56 |
| 3 | `0.6186956850` | 1560 | 3120 | 14123 | 6871 | 7252 | 3256.41 |
| 4 | `0.5997290100` | 1560 | 3120 | 14123 | 6871 | 7252 | 2724.49 |

```text
completed epochs       4
batches              6240
study draws         12480
nominal corpus passes 4.0
training seconds     11292.13
budget limited       false
checkpoint           runs/b8_spatial_anatomy/b8_model.pt
gold evaluation      pending
benchmark to beat    B7.1 = 0.5644802945
```

Training optimization was stable and monotonic. The lower weak-training loss is not by itself evidence of improved gold AUC.

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
P(B7.1 > B5)      = 0.8716
```

### B7.1 versus fixed B5+B7.1 rank ensemble

```text
B7.1 macro AUC             0.5644802945
fixed rank ensemble AUC    0.5540141184
median(ensemble - B7.1)   -0.0105429030
95% paired CI             [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)         0.3054
```

The fixed ensemble is rejected; no alternative blend-weight search follows.

## Decision policy from here

1. Keep B7.1 as the current main standalone development model until B8 completes its frozen evaluation.
2. Run the first B8 gold evaluation exactly once with the completed `runs/b8_spatial_anatomy/b8_model.pt` checkpoint.
3. The primary B8 comparison is paired B7.1 -> B8 with 5,000 study-level bootstrap replicates.
4. Do not tune B8 spatial grid, target-specific priors, prior strength, epochs or blend weights from the 58 gold labels and still call it B8-v1.
5. Do not tune B6 parser rules, target-specific weak-label weights, target-specific model winners, or ensemble weights from the 58 gold labels.
6. New trained variants must be explicitly named and interpreted as additional development on the same 58-study set.
7. Prefer substantive model/data improvements over post-hoc gold tuning.
8. Actual competition leaderboard performance remains unknown until a real submission has been made.
