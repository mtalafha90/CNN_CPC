# Experiment status

**Snapshot:** 2026-08-10  
**Package:** `0.14.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

This file is the canonical repository summary for measured experiment status. The 58-study set has supported repeated development decisions and should now be interpreted as a development/model-selection set rather than pristine independent validation.

## Current headline

- **Best standalone development point estimate:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`, 95% bootstrap CI `[0.5052432984, 0.6229422178]`.
- Paired B7-v1 -> B7.1: median difference `+0.0241102714`, 95% CI `[-0.0140197876, +0.0660558004]`, `P(B7.1 > B7-v1)=0.8694`.
- Paired B5 -> B7.1: median difference `+0.0399233552`, 95% CI `[-0.0301354430, +0.1092349994]`, `P(B7.1 > B5)=0.8716`.
- The fixed B5+B7.1 50:50 rank ensemble scored `0.5540141184` and is rejected; no blend-weight search follows.
- **B8 spatial-anatomy learning is complete and rejected:** macro AUC `0.5300962807`, 95% CI `[0.4723014866, 0.5867732651]`; paired `P(B8 > B7.1)=0.1156`. The spatial-prior branch is closed to post-hoc tuning.
- A subsequent label-free train/test metadata audit found that the historical dual-stream selector assigns **552 / 21,886 selected training streams (2.52%)** to a semantic slot that contradicts `Fluid_Sensitive`. The three-study test surface contains one such false assignment among 14 historically selected streams.
- **B9 strict semantic routing is now implemented and predeclared before training/evaluation.** It returns to the successful B7.1 architecture/B5 initialization and changes only routing: fluid slots accept only `Fluid_Sensitive=True`, structural slots only `False`; unavailable contrasts remain masked.
- B7.1 remains the current leader until B9 completes its frozen one-shot development evaluation.

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
| B8 | B7.1-init 2x2 spatial tokens + fixed soft pathology stream/slice priors | `0.5300962807` | rejected versus B7.1; branch closed |

## Active predeclared experiment

| ID | Method | Status |
|---|---|---|
| **B9** | **B7.1 recipe with exact `Fluid_Sensitive` six-stream routing and no cross-contrast substitution** | **implemented / tests pending / real-data training pending** |

## B6 weak supervision

Frozen B6 v1.2.1 training export:

```text
report-only studies       4349
active studies            3120
usable cells             14123
positive cells            6871
negative cells            7252
```

The completed gold audit motivated one global asymmetric policy used by B7/B7.1/B8/B9: positive soft target `0.85` with base weight `0.50`, negative soft target `0.05` with base weight `1.00`, confidence threshold `0.75`, uncertain/unmentioned ignored. The parser and this policy are frozen.

## B7.1 full coverage — retained leader

B7.1 changed only the B7-v1 epoch coverage from 500 to 1,560 batches. With batch size 2, every epoch covered all 3,120 active weakly labelled studies and all 14,123 usable cells.

```text
epoch 1 loss 0.7524191749
epoch 2 loss 0.6651707418
epoch 3 loss 0.6391165589
epoch 4 loss 0.6127582232
```

Gold development result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

## B8 spatial anatomy — completed / rejected

B8 retained 2x2 within-slice ConvNeXt spatial features, increasing MRI memory from 96 to 384 tokens/study, and added fixed soft pathology stream/slice priors. Training was stable but the development result fell below B7.1:

```text
B8 macro AUC          0.5300962807
95% CI               [0.4723014866, 0.5867732651]
B7.1 macro AUC        0.5644802945
median(B8 - B7.1)    -0.0335501423
95% paired CI        [-0.0900453633, +0.0223997827]
P(B8 > B7.1)          0.1156
```

Decision: reject B8; do not tune its spatial grid, priors, target-specific winners or blend weights from this result.

## B9 strict-routing motivation — label-free metadata audit

The historical `mode="dual"` selector tries to populate both semantic slots when a plane has multiple series. This can create a false opposite-contrast assignment when only one contrast class exists.

Full training metadata audit:

| Stream | Historical selected | Strict selected | Historical semantic mismatches removed |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 4,150 | 251 |
| sagittal_structural | 4,294 | 4,266 | 28 |
| coronal_fluid | 4,250 | 4,248 | 2 |
| coronal_structural | 3,440 | 3,406 | 34 |
| axial_fluid | 4,407 | 4,407 | 0 |
| axial_structural | 1,094 | 857 | 237 |
| **Total** | **21,886** | **21,334** | **552** |

Strict routing therefore removes all 552 known cross-contrast substitutions. Because removing a false assignment can also change which valid same-class series occupies the remaining slot, 805 stream assignments differ in total between historical and strict routing.

Provided test metadata audit:

```text
historical selected streams   14
strict selected streams       13
historical semantic mismatch   1
strict semantic mismatches     0
```

The affected test study has two sagittal structural series and no sagittal fluid series. B9 leaves `sagittal_fluid` missing rather than fabricating it.

This hypothesis was derived from metadata consistency, not from target-level gold AUCs.

## B9 frozen scientific contract

The only scientific change versus B7.1 is routing:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> None / presence mask False
```

Everything else remains B7.1-equivalent:

```text
B5 encoder initialization
B6 v1.2.1 weak supervision
KneeMILNet global-token architecture
16 slices/stream
batch size 2
4 epochs
1560 batches/epoch
encoder LR 1e-5
head LR 1e-4
same augmentation
TTA [-1,0,1]
5000 bootstrap replicates
zero gold-gradient use
zero gold early stopping
```

Implementation:

```text
src/rsna_knee/strict_routing.py
src/rsna_knee/b9_strict_routing.py
src/rsna_knee/b9_gold_eval.py
configs/b9_strict_routing.yaml
tests/test_b9_strict_routing.py
docs/B9_STRICT_ROUTING.md
```

## Decision policy from here

1. Keep B7.1 as the main standalone development model until B9 completes.
2. Run B9 exactly as predeclared; do not restore selected substitutions based on target outcomes.
3. Inspect B9 `routing_audit.json`, `history.json` and `supervision_plan.json` before gold evaluation.
4. The primary B9 comparison is paired B7.1 -> B9 with 5,000 study-level bootstrap replicates.
5. Do not tune B6 parser rules, target-specific weak-label weights, target-specific routing, model winners or ensemble weights from the 58 gold labels.
6. The B8 spatial-prior branch and prior ensemble-weight questions are closed.
7. New variants must be explicitly named and treated as additional development on the same 58-study set.
8. Actual leaderboard performance remains unknown until a real competition submission is made.
