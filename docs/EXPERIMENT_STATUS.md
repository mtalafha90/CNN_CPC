# Experiment status

**Snapshot:** 2026-08-10  
**Package:** `0.15.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study set has supported repeated sequential development decisions. It is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Retained standalone champion:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`.
- B7.1 95% bootstrap CI: approximately `[0.5053, 0.6214]` in the latest 5,000-replicate rerun.
- **B8 spatial-token model rejected:** macro AUC `0.5300962807`; paired `P(B8 > B7.1)=0.1156`.
- **B9 strict semantic routing rejected:** macro AUC `0.5334962669`; paired median `(B9-B7.1)=-0.0302397961`, 95% CI `[-0.0679414819,+0.0070291202]`, `P(B9>B7.1)=0.0562`.
- B9 fixed all known routing semantic mismatches but removed informative acquisitions; B7.1 routing remains the reference for the next experiment.
- **B10 physical-scale normalization is implemented and predeclared.** It changes only in-plane physical geometry preprocessing before the unchanged `224x224` resize. A label-free geometry audit must be frozen before GPU training.

## Completed measured experiments

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected as general MRI teacher |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B1+B3 rank | fixed 50:50 rank ensemble | `0.5048038179` | neutral |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B4.1 | shared policy per fold | `0.4847792672` | rejected |
| B4.2 | pathology-group policies | `0.4901328905` | rejected |
| B4.3 | target-wise two-way-CV selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | historical reference |
| B5 | image-report SSL representation + unchanged B4 probe | `0.5243650851` | retained representation baseline |
| B6 | multilingual structured report labels | n/a | completed; frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 weak labels, 500 batches/epoch | `0.5397724412` | coverage ablation |
| **B7.1** | **same B7 recipe with full 3,120-study coverage each epoch** | **`0.5644802945`** | **retained champion** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial tokens + fixed anatomy priors | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |

## B6 frozen weak supervision

```text
report-only studies       4349
active studies            3120
usable cells             14123
positive cells            6871
negative cells            7252
```

Frozen policy used by B7/B7.1/B8/B9/B10:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

The parser and policy are closed to further tuning on the 58 gold studies.

## B7.1 retained champion

B7.1 changed only epoch coverage from B7-v1. With batch size 2 and 1,560 batches/epoch, all 3,120 active weak studies and all 14,123 usable cells are seen once per epoch.

```text
epoch 1 loss 0.7524191749
epoch 2 loss 0.6651707418
epoch 3 loss 0.6391165589
epoch 4 loss 0.6127582232
macro AUC    0.5644802945
```

## B8 completed / rejected

```text
B8 macro AUC          0.5300962807
95% CI               [0.4723014866, 0.5867732651]
median(B8-B7.1)      -0.0335501423
95% paired CI        [-0.0900453633,+0.0223997827]
P(B8>B7.1)            0.1156
```

Decision: close the spatial-prior branch; do not tune grid size, priors, target-specific winners, or blend weights from this result.

## B9 completed / rejected

B9 enforced exact `Fluid_Sensitive` semantics and no cross-contrast substitution. On the exact 3,120-study weak-training subset:

```text
legacy selected streams                 15468
strict selected streams                 15026
legacy semantic mismatches                442
strict semantic mismatches                  0
removed cross-contrast substitutions      442
changed stream assignments                 666
```

Training completed four full epochs:

```text
0.741013 -> 0.661024 -> 0.620775 -> 0.598611
```

Gold development result:

```text
B9 macro AUC           0.5334962669
95% CI                [0.4787138255,0.5891763236]
B7.1 macro AUC         0.5644802945
median(B9-B7.1)       -0.0302397961
95% paired CI         [-0.0679414819,+0.0070291202]
P(B9>B7.1)             0.0562
```

Decision: reject B9 as a replacement for B7.1. The result suggests preserving MRI information is more important than forcing rigid slot semantics in the current architecture.

## Active predeclared experiment: B10

**Hypothesis:** direct `224x224` resizing leaves anatomical scale dependent on scanner/protocol `PixelSpacing` and physical FOV. Standardizing in-plane physical geometry before the network resize may reduce acquisition-domain shift.

### Single scientific change

```text
native DICOM pixels
 -> canonical plane-specific PixelSpacing
 -> canonical plane-specific physical FOV by center crop/pad
 -> unchanged 224x224 resize
 -> unchanged B7.1 network/training/evaluation
```

B10 deliberately retains **historical B7.1 dual routing**, because B9 strict routing was negative.

### Label-free policy derivation

Before training, `rsna-knee-b10-audit` uses only the exact 3,120 active weak-training studies. For sagittal, coronal and axial planes separately it derives:

```text
target PixelSpacing = median valid PixelSpacing
target physical FOV = median valid physical FOV
minimum valid-geometry coverage = 95%
missing PixelSpacing action = legacy resize, do not discard study
```

The audit records scanner/model/field-strength metadata, slice thickness/spacing, and a SHA-256 signature of the exact B7.1 selected-series mapping. Gold labels are not used to choose spacing or FOV.

Implementation:

```text
src/rsna_knee/physical_scale.py
src/rsna_knee/b10_spacing_audit.py
src/rsna_knee/b10_physical_scale.py
src/rsna_knee/b10_gold_eval.py
configs/b10_physical_scale.yaml
tests/test_b10_physical_scale.py
docs/B10_PHYSICAL_SCALE.md
```

## B10 execution order

1. Update package and run tests.
2. Run the label-free B10 geometry audit.
3. Inspect `spacing_audit.json` and `physical_scale_policy.json` before GPU training.
4. Train B10 exactly four full epochs using the frozen policy.
5. Inspect training artifacts before gold evaluation.
6. Run one frozen gold evaluation.
7. Compare B7.1 -> B10 with 5,000 aligned study-level bootstrap replicates.
8. Do not search multiple physical spacings/FOVs on the gold set.

## Next candidates after B10

Only after B10 is evaluated independently:

- **B11:** frozen B7.1/B10 teacher with confidence- and consistency-gated pseudo-label completion for currently unsupervised report cells;
- **B12:** information-preserving variable-series set model rather than six forced one-series slots;
- stronger competition-only MRI representation learning;
- scanner/protocol domain augmentation after the physical-normalization ablation is isolated.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
