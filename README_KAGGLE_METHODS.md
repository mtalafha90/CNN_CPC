# RSNA Knee Abnormality Detection — Public Code Methodology Review

**Repository:** `mtalafha90/CNN_CPC`  
**Review snapshot:** 2026-08-08  
**Purpose:** methodological context for the implemented `CNN_CPC` baseline, not a leaderboard claim.

## Scope

This document summarizes public early-competition ideas that informed the design review and records which ideas are actually implemented in `CNN_CPC`.

It does **not** claim exhaustive coverage of every Kaggle notebook. Public notebook discovery can be incomplete, code changes quickly during an active competition, and repository-reported scores may be self-reported rather than independently reproduced.

The repository's own measured claims are restricted to data audit and smoke evidence unless a completed non-smoke OOF or leaderboard result is explicitly available.

## 1. Dataset facts that drive the methodology

The verified downloaded data contain:

```text
4,407 training studies
58 fully gold-labeled studies
4,349 report-only studies
24,371 training series rows
12 study-level targets
macro ROC AUC metric
```

This is therefore not a conventional fully supervised classification problem. The central challenge is to use report-only cases without converting missing report findings into false negatives and without contaminating the tiny trusted validation set.

## 2. Public implementation families reviewed

The earlier review examined public repositories including:

- `Chagatai404/knee-abnormality-ML` — gold-only multi-plane/2.5D baseline ideas;
- `jiaweizhong/rsna-knee` — modular DICOM audit, selection and aggregation ideas;
- `andreluizpedroso/rsna-knee-abnormality-detection` — report-derived weak labels and image student;
- `chaitanyajamble/RSNA-Knee-Abnormality-Detection` — rule/text/statistical/CNN experiments;
- `dianisay/RSNA-Knee-Abnormality-Detection` — multilingual report supervision, MIL/ranking and MRI preprocessing ideas;
- `soumic28/RSNA-knee-abnormality-predictin` — 2.5D backbone/pooling abstraction;
- `JunhaoLiXD/RSNA_Knee_Abnormality_Detection` — fold-safe teacher calibration and sequence routing;
- `Msmile-shiny/CVproject_RSNA-knee-detection` — cross-plane/2.5D/3D experimental variants;
- `Saianiruthm/rsna-knee-abnormality` — report-teacher concept;
- `tomyimkc/sophia-agi` RSNA knee work — instrumented gold validation, distillation and experiment tracking;
- MRNet-style historical knee MRI work — multi-plane slice aggregation precedent.

These are methodology references, not certified winning solutions.

## 3. Lesson: reports should supervise images, not replace them at inference

The strongest recurring principle is:

```text
radiology report
-> text/rule teacher
-> soft target + confidence
-> image-only student
-> gold-only validation
```

`CNN_CPC` follows this pattern. Final inference requires only MRI and self-describing checkpoints.

## 4. Lesson: unmentioned is not negative

A report may omit a pathology even when it is present. Treating all unmentioned targets as negatives would create systematic label noise.

`CNN_CPC` uses four report states:

```text
positive
negated
uncertain
unmentioned
```

and sets unmentioned direct weight to zero by default.

## 5. Lesson: teacher calibration must be fold-safe

If all 58 gold labels are used to calibrate the report teacher and those same studies are later evaluated, validation is optimistic.

The implemented calibration is phase/fold-local. It learns state-conditioned probabilities only from gold rows allowed in the current training phase.

Official finite cells override pseudo-labels cell-by-cell with high gold weight.

## 6. Lesson from the real audit: generic lexicons are not enough for OA

The initial implementation used narrow explicit OA phrases. The first real audit exposed a complete failure mode:

```text
Medial OA  : 4407 zero-confidence cells
Lateral OA : 4407 zero-confidence cells
PF OA      : 4407 zero-confidence cells
```

The parser was then expanded **without lowering confidence thresholds**. The current compartment-aware rules recognize OA/arthrosis plus cartilage loss, chondrosis/chondromalacia, osteophytes and related compartment-specific wording.

Verified post-fix states:

| Target | Positive | Negated | Unmentioned |
|---|---:|---:|---:|
| Medial OA | 492 | 339 | 3,576 |
| Lateral OA | 409 | 387 | 3,611 |
| PF OA | 695 | 379 | 3,333 |

This is a useful general lesson for weak supervision: **audit actual state coverage before training**, especially for targets whose clinical wording is heterogeneous.

## 7. Lesson: DICOM preprocessing must be treated as model infrastructure

Robust public implementations repeatedly emphasize correct physical slice ordering and intensity handling.

`CNN_CPC` implements:

- orientation/position geometry ordering;
- `InstanceNumber` fallback;
- deterministic filename fallback;
- slope/intercept rescaling;
- `MONOCHROME1` inversion;
- multi-frame support;
- mixed-shape crop/pad;
- percentile clipping/normalization;
- selected-series preflight and full audit.

The real audit verified:

```text
21,886 / 21,886 selected series decoded
732,554 / 732,556 files decoded
0 selected series failed
```

Only two individual files failed, one in each of two otherwise usable series.

## 8. Lesson: use multiple sequence roles, not only three planes

A one-series-per-plane baseline is simple, but different abnormalities benefit from different contrasts.

`CNN_CPC` uses six semantic roles:

```text
sagittal fluid       sagittal structural
coronal fluid        coronal structural
axial fluid          axial structural
```

The real data confirm that missing semantic roles are common, so presence masking is a required part of the architecture rather than an edge case.

## 9. Lesson: 2.5D is a strong cost/context compromise

Three-slice triplets retain local through-plane context while remaining compatible with efficient 2D backbones.

The implemented representation samples distributed centers throughout each series and uses:

```text
[z-gap, z, z+gap]
```

with mild stochastic gap/center variation during training.

## 10. Lesson: target-specific aggregation matters

ACL, menisci, OA, effusion, synovitis, Baker cyst, contusion and fracture have different preferred views and appearances.

Rather than reducing the study to one universal pooled vector, `CNN_CPC` uses:

1. ConvNeXt-Tiny triplet encoding;
2. cross-sequence Transformer context;
3. 12 learnable pathology queries;
4. pathology self-interaction;
5. cross-attention from each pathology to MRI memory;
6. target-specific readout.

## 11. Lesson: optimize for macro target balance

Weak supervision coverage differs sharply across targets. A simple mean over all weighted cells can overemphasize targets mentioned frequently in reports.

`CNN_CPC` first computes the planned epoch supervision denominator per target, then macro-averages target-specific weighted BCE contributions.

This aligns training more closely with macro-AUC than raw cell frequency.

## 12. Lesson: ranking losses require usable minibatch structure

A ranking objective can be mathematically present but operationally inactive.

The first `CNN_CPC` smoke audit showed:

```text
rank_pairs = 0 for all 12 targets
```

Root cause: with batch size 2, trusted and general rows were spread so evenly that a trusted positive and trusted negative rarely shared a batch.

The sampler was changed to pair trusted rows for even batch sizes while preserving the requested trusted-row fraction.

Verified corrected smoke:

```text
selection ranking pairs = 63
retrain ranking pairs   = 61
all 12 targets          = nonzero
```

This is an important methodological lesson: auxiliary-loss utilization should be **measured in diagnostics**, not assumed from configuration.

## 13. Lesson: tiny-gold validation requires nested discipline

The 58 gold studies are divided into three balanced folds. For outer fold `k`:

```text
remaining gold -> Phase-A trusted training
inner fold     -> epoch-count selection
outer fold     -> final OOF only
```

Phase A is discarded and Phase B starts fresh.

This avoids using the outer fold to choose its own training duration.

## 14. Lesson: SSL candidate selection must also be nested

Competition-data self-supervision is a plausible Stage-1 candidate, but choosing random versus SSL from aggregate outer OOF and then reusing that choice in downstream OOF analysis creates selection bias.

The implemented candidate selector chooses random versus SSL **independently per outer fold from that fold's inner AUC only**.

Outer AUC is ignored by the selector.

## 15. Lesson: co-training teachers must be cross-fitted

Stage 2 can use an image model to strengthen or rescue weak report supervision only if the image prediction is independent of that weak training row.

For non-gold `crossfit_fold=k`, Stage-1 fold `k` excludes those rows, predicts them after training, and writes `weak_oof.csv`.

Stage-2 fold `k` is allowed to use only the matching safe teacher. Wrong-fold or incomplete teacher files are rejected.

## 16. Lesson: TTA must be predeclared

A small outer fold can easily produce a misleading TTA-versus-center difference.

The current smoke happened to produce:

```text
outer TTA AUC    0.51396
outer center AUC 0.52285
```

That is not grounds to change TTA. The production policy was declared in advance:

```yaml
validation_tta_offsets: [-1, 0, 1]
tta_center_offsets: [-1, 0, 1]
```

Center-only output remains diagnostic.

## 17. Lesson: runtime is part of the method

A strong model that cannot complete training plus OOF/submission inference inside the runtime constraint is not a valid competition method.

The repository reserves time for:

- Phase-B retraining;
- outer OOF;
- Stage-1 weak OOF;
- bootstrap;
- loader startup;
- serialization.

Prediction is guarded batch-by-batch.

## 18. Current implemented baseline status

Completed engineering/data gates:

```text
CSV contract                         PASS
balanced nested folds               PASS
train DICOM preflight               PASS
complete local test preflight       PASS
full selected-series DICOM audit    PASS
OA weak-supervision coverage        PASS
single-GPU BF16 execution           PASS
nested selection/retrain            PASS
outer OOF plumbing                  PASS
weak OOF plumbing                   PASS
bootstrap/artifact writing          PASS
ranking auxiliary utilization       PASS
```

The next evidence tier is the completed **non-smoke Stage-1 random three-fold OOF baseline**.

## 19. What is intentionally not claimed

This review does not claim that:

- any reviewed public repository is a competition winner;
- any self-reported public score has been independently reproduced;
- the current smoke AUC predicts final performance;
- Stage 2 is better than Stage 1 before paired OOF evidence;
- SSL is better than random initialization before fold-safe comparison;
- `CNN_CPC` has a leaderboard advantage before an actual submission.

## 20. Recommended experiment order

```text
Stage-1 random production 3-fold baseline
-> evaluate macro/per-target OOF + bootstrap
-> competition-data SSL pretraining
-> Stage-1 SSL 3 folds
-> inner-AUC per-fold candidate selection
-> Stage-2 3 folds
-> paired Stage-2 vs nested-selected Stage-1 comparison
-> freeze final method
-> final checkpoint-contract inference
-> Kaggle submission
```

The central objective is not to accumulate techniques. It is to add components only when leakage-safe validation shows that they improve the image model under the actual competition constraints.