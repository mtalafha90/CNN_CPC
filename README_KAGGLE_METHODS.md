# RSNA Knee Abnormality Detection — Public Code Methodology Review

**Repository:** `mtalafha90/CNN_CPC`  
**Review/status snapshot:** 2026-08-09  
**Purpose:** methodology context, not a leaderboard claim.

> Repository-measured experiment results are maintained in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md). B5 is currently running and has no OOF score yet.

## Scope

This document summarizes public early-competition ideas that informed the design review and records what the `CNN_CPC` experiments actually supported or rejected.

Public notebooks/repositories change during an active competition and self-reported scores are not treated as independently verified benchmarks.

## 1. Dataset facts that dominate the methodology

```text
4,407 training studies
58 fully gold-labelled studies
4,349 report-only studies
24,371 series rows
12 study-level targets
macro ROC AUC
```

This is fundamentally a weak/semi-supervised representation problem with an extremely small trusted evaluation set.

## 2. Public implementation families reviewed

The review included, where available:

- `Chagatai404/knee-abnormality-ML`
- `jiaweizhong/rsna-knee`
- `andreluizpedroso/rsna-knee-abnormality-detection`
- `chaitanyajamble/RSNA-Knee-Abnormality-Detection`
- `dianisay/RSNA-Knee-Abnormality-Detection`
- `soumic28/RSNA-knee-abnormality-predictin`
- `JunhaoLiXD/RSNA_Knee_Abnormality_Detection`
- `Msmile-shiny/CVproject_RSNA-knee-detection`
- `Saianiruthm/rsna-knee-abnormality`
- `tomyimkc/sophia-agi` RSNA knee work
- `bollimunthasripavan-oss/RSNA-Knee-MRI-Abnormality-Detection`
- MRNet-style historical knee MRI work.

These are methodology references, not certified winning solutions.

## 3. Reports should supervise representation/training, not be required at inference

The test path is MRI-only. This remains a hard design constraint.

The first supervised report-teacher benchmark used rules + word/character TF-IDF and achieved only:

```text
macro OOF AUC = 0.49245
```

It was rejected as a general 12-target teacher.

This changed the report strategy: B5 uses the 4,349 reports as **semantic representation targets** rather than trying to infer strong binary pseudo-labels from only 58 labelled reports.

## 4. Unmentioned is not negative

`CNN_CPC` uses:

```text
positive
negated
uncertain
unmentioned
```

`unmentioned` receives zero direct report weight by default. Official finite labels override weak labels.

This remains important even though B5's main report path is now unsupervised semantic alignment rather than target-probability distillation.

## 5. Audit report supervision before trusting it

The original OA lexicon produced zero useful OA supervision. The real-data audit triggered a compartment-aware parser expansion without lowering confidence thresholds.

Verified states:

| Target | Positive | Negated | Unmentioned |
|---|---:|---:|---:|
| Medial OA | 492 | 339 | 3,576 |
| Lateral OA | 409 | 387 | 3,611 |
| PF OA | 695 | 379 | 3,333 |

General lesson: weak-label coverage must be measured on the actual corpus before training.

## 6. DICOM preprocessing is model infrastructure

Implemented:

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
0 selected series failed
```

## 7. Multiple sequence roles matter

The repository routes up to six semantic MRI streams:

```text
sagittal fluid       sagittal structural
coronal fluid        coronal structural
axial fluid          axial structural
```

Missing roles are explicitly masked. This matters because `axial_structural` is absent in most studies.

## 8. 2.5D remains the core image representation

Distributed three-slice triplets:

```text
[z-gap, z, z+gap]
```

retain local through-plane information while allowing efficient ConvNeXt encoding.

Strong SSL increases representation coverage by sampling multiple positions from active streams.

## 9. Strong competition-only MRI SSL helped the point estimate

The strong SSL run used only the 4,349 non-gold MRI studies:

```text
8 epochs
8,000 batches
24,000 study draws
~5.52 corpus passes
238,274 active 2.5D examples
```

Controlled Stage-1 results:

```text
B0 random initialization = 0.4762536432
B1 strong SSL            = 0.5030284974
```

Paired bootstrap gave `P(B1 > B0)=0.771`, encouraging but not decisive with only 58 gold studies.

## 10. Small optimizer/head changes did not solve the problem

B2 reduced only the encoder learning rate:

```text
B2 = 0.4993244663
```

B3 replaced the global Transformer/pathology stack with lower-capacity target-specific MIL:

```text
B3 = 0.4944652486
```

Neither improved pooled B1 performance.

General lesson: once representation quality improves, repeatedly changing the supervised head on 58 labels can add variance rather than reliable signal.

## 11. Frozen representation probes are highly informative

B4 froze the strong SSL encoder and reduced each stream to mean/std/max slice embeddings, followed by target-specific PCA + balanced logistic regression.

```text
B4 = 0.5137567459
95% CI = [0.4619827141, 0.5642366629]
```

This is the best clean standalone point estimate so far.

Because the supervised probe is low-capacity, B4 is useful evidence that the strong SSL encoder itself contains pathology-separable information.

## 12. Target-specific downstream flexibility matters, but its selector is noisy

B4 target-wise hyperparameters varied strongly across the three tiny inner folds. Three controlled stabilizers were tested:

```text
B4.1 one shared policy                 0.4847792672
B4.2 four predefined group policies   0.4901328905
B4.3 target-wise two-way CV selector  0.4966083942
```

All were below B4.

General lesson: the pathologies are heterogeneous enough that broad policy sharing hurts, but further selector redesign from the same outer labels risks meta-overfitting. The selector branch is closed.

## 13. Rank ensembling can remove probability-scale mismatch, but gains must be paired-tested

Fixed B1+B4 ensembles:

```text
raw probability 50:50 = 0.5050
rank 50:50            = 0.5167
```

The rank ensemble is numerically highest, but versus B4:

```text
median difference = +0.00276
95% CI            = [-0.03513, +0.04174]
P(ensemble > B4)  = 0.5544
```

Therefore it is treated as tied with B4. No weight search is performed.

## 14. B5: use report semantics to improve the MRI representation

The next methodological step is not another B4 classifier. B5 uses:

```text
report-only competition corpus
-> TF-IDF
-> TruncatedSVD semantic embedding

competition MRI
-> strong SSL ConvNeXt
-> image-image SSL
-> acquisition metadata objectives
-> image-report alignment
```

A report embedding queue increases semantic negatives for small MRI batches, and exact duplicate normalized reports are masked as false negatives.

All 58 gold studies are excluded from B5 representation training. No external language model or image weights are used.

**B5 is currently running; no performance result is available yet.**

## 15. B5 evaluation is deliberately fixed

The first B5 test reuses the **original B4 frozen-feature probe unchanged**.

```text
B4 image-only encoder -> B4 probe
B5 image-report encoder -> same B4 probe
```

This isolates representation improvement from downstream model-selection changes.

## 16. Tiny-gold validation requires campaign-level discipline

Nested/cross-fitted logic prevents direct outer leakage within individual experiments, but the same 58 gold studies have now informed many sequential method choices.

The aggregate campaign must therefore be described as **model-selection cross-validation**.

Do not:

- choose target-specific outer-OOF winners;
- optimize ensemble weights;
- create more B4 grouping/selector variants;
- retune B5 after reading its OOF without declaring a new experiment;
- describe the best current OOF point estimate as a hidden-test guarantee.

## 17. Runtime is part of the method

The repository uses one GPU and bounded runtime. Data audit, training, OOF generation and final inference must all fit the actual execution environment.

A method that cannot finish reliably under the competition constraints is not a valid competition method.

## 18. Current repository-measured evidence

| Candidate | Macro AUC |
|---|---:|
| B0 | `0.4763` |
| report teacher | `0.49245` |
| B1 | `0.5030` |
| B2 | `0.4993` |
| B3 | `0.4945` |
| B1+B3 rank | `0.5048` |
| **B4** | **`0.5138`** |
| B4.1 | `0.4848` |
| B4.2 | `0.4901` |
| B4.3 | `0.4966` |
| B1+B4 raw | `0.5050` |
| B1+B4 rank | `0.5167` |
| B5 | pending/running |

See [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) for exact confidence intervals and paired comparisons.

## 19. What is intentionally not claimed

This repository does not claim:

- that any reviewed public repository is a winner;
- that self-reported public scores are independently verified;
- that the B1+B4 rank ensemble is statistically superior to B4;
- that B5 improves anything before its fixed probe completes;
- that current model-selection OOF guarantees hidden-test or leaderboard performance.

The design goal remains to add components only when controlled evidence supports them under the actual competition constraints.
