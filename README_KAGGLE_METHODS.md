# RSNA Knee Abnormality Detection — Public Code Methodology Review

**Repository:** `mtalafha90/CNN_CPC`  
**Review/status snapshot:** 2026-08-10  
**Purpose:** methodology context and repository-measured development evidence, not a leaderboard claim.

> Canonical measured results are maintained in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md). **B7.1 is the current best standalone development model at macro AUC `0.5644802945`; B8 spatial-anatomy learning is currently training and has no gold score yet.**

## 1. Problem structure

```text
4,407 training studies
58 fully gold-labelled studies
4,349 report-only studies
24,371 series rows
12 study-level targets
primary metric: macro ROC AUC
```

This is fundamentally a **weak/semi-supervised multi-sequence MRI problem with an extremely small trusted development set**.

## 2. Public implementation families reviewed

The methodology review included public RSNA-knee repositories/notebooks and MRNet-style historical work. These are used for design context only; public self-reported scores are not treated as independently verified benchmarks.

The repository's own decisions are based on controlled experiments recorded in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).

## 3. Reports should supervise training, not be required at inference

The hidden/test path is MRI-only. Reports are therefore used only during training.

The first report-teacher attempt—rules plus TF-IDF—reached only:

```text
macro OOF AUC = 0.49245
```

and was rejected as a general 12-target MRI teacher.

The report strategy then evolved in two directions:

- **B5:** use reports as semantic representation targets for image-report alignment;
- **B6/B7:** extract conservative pathology-specific report states and use them as direct weak MRI supervision.

## 4. Unmentioned is not negative

B6 uses four states:

```text
positive
negated
uncertain
unmentioned
```

`unmentioned` is never silently converted to a negative target. B7/B7.1/B8 ignore uncertain/unmentioned cells entirely.

Frozen B6 v1.2.1 training export:

```text
report-only studies             4349
active weakly labelled studies  3120
usable cells                   14123
positive cells                  6871
negative cells                  7252
```

## 5. Weak supervision must be audited before image training

The completed B6 gold audit covered 251 usable cells and showed asymmetric reliability:

```text
pooled positive precision  0.6905
pooled sensitivity         0.9748
pooled specificity         0.6061
pooled NPV                 0.9639
pooled balanced accuracy   0.7904
```

This motivated one global B7 supervision policy:

| state | soft target | base weight |
|---|---:|---:|
| positive | 0.85 | 0.50 |
| negated | 0.05 | 1.00 |
| uncertain | ignored | 0.00 |
| unmentioned | ignored | 0.00 |

The parser and global policy are frozen after that audit. The same 58-study set therefore becomes a development/model-selection set for later experiments rather than pristine independent validation.

## 6. DICOM preprocessing is model infrastructure

Implemented and audited:

- physical orientation/position ordering;
- `InstanceNumber` fallback;
- deterministic filename fallback;
- rescale slope/intercept;
- `MONOCHROME1` inversion;
- multi-frame support;
- mixed-shape crop/pad;
- percentile normalization;
- selected-series preflight and full audit.

Verified:

```text
21,886 / 21,886 selected series decoded
732,554 / 732,556 files decoded
0 selected series lost
```

## 7. Multiple sequence roles matter

The model routes up to six semantic MRI streams:

```text
sagittal fluid       sagittal structural
coronal fluid        coronal structural
axial fluid          axial structural
```

Missing roles are explicitly masked. This is important because some streams—especially axial structural—are absent in many studies.

## 8. 2.5D remains the core input representation

Distributed three-slice triplets:

```text
[z-gap, z, z+gap]
```

retain local through-plane information while keeping ConvNeXt inference practical.

## 9. Strong competition-only MRI SSL improved representation quality

The strong SSL run used only the 4,349 non-gold competition MRI studies:

```text
8 epochs
8,000 batches
24,000 study draws
~5.52 corpus passes
238,274 active 2.5D examples
```

Controlled Stage-1 result:

```text
B0 random initialization = 0.4762536432
B1 strong SSL            = 0.5030284974
```

The paired bootstrap favored B1 (`P=0.771`) but was not decisive on 58 studies.

## 10. Small optimizer/head changes were not the main lever

```text
B2 lower encoder LR = 0.4993244663
B3 pathology MIL    = 0.4944652486
```

Neither improved B1 globally. This discouraged repeated supervised-head redesign on only 58 labels.

## 11. Frozen representation probes were informative

B4 froze the strong SSL encoder and used mean/std/max stream features with target-wise PCA + balanced logistic regression:

```text
B4 macro AUC = 0.5137567459
```

B4.1/B4.2/B4.3 attempts to stabilize the tiny-fold downstream selector all reduced pooled performance. That selector branch is closed.

## 12. B5: image-report representation learning helped, but modestly

B5 aligned competition MRI with TF-IDF/SVD report semantics while excluding all 58 gold studies from representation training.

Frozen unchanged B4 probe result:

```text
B5 macro AUC = 0.5243650851
95% CI      = [0.4728108406, 0.5761619105]
```

Paired B4 -> B5:

```text
median difference = +0.0105821232
95% paired CI     = [-0.0408197338, +0.0622131599]
P(B5 > B4)        = 0.656
```

B5 became the retained report-aligned representation baseline, but the gain was not statistically conclusive.

## 13. B6: structured multilingual report labels enabled direct weak supervision

B6 v1.2.1 provides pathology-specific positive/negated/uncertain/unmentioned states with confidence. The final frozen export has 14,123 usable target cells over 3,120 report-only studies.

This was a strategic shift: instead of using reports only to shape a generic representation, the MRI model could now receive direct target-level weak supervision without calling unmentioned findings negative.

## 14. B7: direct weak supervision improved the point estimate

B7 initializes ConvNeXt from B5 and uses:

```text
6 MRI streams
-> 16 distributed 2.5D slices/stream
-> slice + stream embeddings
-> cross-sequence Transformer
-> 12 interacting pathology queries
-> 12 logits
```

B7-v1 used 500 batches/epoch and reached:

```text
macro AUC = 0.5397724412
```

The supervision audit showed that this represented only about 1.28 nominal passes over the 3,120 active weak-training studies.

## 15. B7.1: full-corpus coverage was the clearest successful change

B7.1 changed only:

```text
batches/epoch: 500 -> 1560
```

with batch size 2, so every epoch covered all 3,120 active studies and all 14,123 weak-label cells.

Result:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

Paired B7-v1 -> B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

Paired B5 -> B7.1:

```text
median difference = +0.0399233552
95% paired CI     = [-0.0301354430, +0.1092349994]
P(B7.1 > B5)      = 0.8716
```

B7.1 is therefore the current main standalone model, while superiority remains statistically inconclusive because the paired intervals cross zero.

## 16. Fixed rank ensembling did not improve B7.1

A single predeclared global B5+B7.1 50:50 percentile-rank ensemble was tested to avoid probability-scale mismatch.

```text
B7.1                       0.5644802945
fixed B5+B7.1 rank blend   0.5540141184
```

Paired B7.1 -> ensemble:

```text
median(ensemble-B7.1) = -0.0105429030
95% paired CI         = [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)     = 0.3054
```

The ensemble is rejected. No blend-weight search, raw-vs-rank search, calibration fitting, or target-specific mixture follows.

## 17. B8: preserve spatial evidence before pathology attention

B7.1 collapses each sampled slice to one global ConvNeXt vector before the MRI Transformer. B8 tests whether that loses pathology-relevant localization.

B8 changes the memory from:

```text
B7.1: 6 streams x 16 slices x 1 pooled token = 96 tokens
B8:   6 streams x 16 slices x 2x2 regions    = 384 tokens
```

B8 initializes all compatible weights from the completed B7.1 checkpoint and adds:

- a 2x2 adaptive spatial grid from the final ConvNeXt feature map;
- learned region-position embeddings;
- fixed gentle pathology-specific stream/slice attention priors;
- uniform fixed in-plane region prior because canonical medial/lateral/anterior/posterior pixel orientation is not guaranteed.

The B6 supervision policy, target balancing, 3,120-study full-corpus coverage, four epochs, and learning rates remain frozen.

**Current status: B8 training is in progress. No B8 gold score is recorded yet.**

## 18. Current repository-measured evidence

| Candidate | Macro AUC | Status |
|---|---:|---|
| B0 | `0.4763` | baseline |
| report teacher | `0.49245` | rejected |
| B1 | `0.5030` | retained reference |
| B2 | `0.4993` | rejected |
| B3 | `0.4945` | rejected |
| B4 | `0.5138` | image-only ablation |
| B5 | `0.524365` | report-aligned representation baseline |
| B7-v1 | `0.539772` | coverage ablation |
| **B7.1** | **`0.564480`** | **current leader** |
| B5+B7.1 rank | `0.554014` | rejected ensemble |
| B8 | pending | training in progress |

## 19. Tiny-gold validation requires campaign-level discipline

Nested/cross-fitted procedures prevent direct outer leakage within individual experiments, but the same 58 gold studies have informed many sequential method choices. The aggregate campaign is therefore **model-selection cross-validation**.

Do not:

- choose target-specific outer-OOF winners;
- optimize ensemble weights;
- retune B6 parser rules from the gold audit;
- tune target-specific weak-label weights from observed B7/B7.1 AUCs;
- search B8 spatial grid sizes, prior strengths, epochs, or target-specific priors after reading the first B8 result and still call it B8-v1;
- describe the best current OOF point estimate as a hidden-test guarantee.

Actual leaderboard performance remains unknown until a real competition submission is made.
