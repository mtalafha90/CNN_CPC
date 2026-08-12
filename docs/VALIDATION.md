# Test and validation workflow

> **Snapshot — 2026-08-12.** B13 remains the reused-gold development champion at `0.6293565948`. B15 passed the frozen weak-v2 teacher-agreement gate (`0.7319060415` versus matched control `0.5652498118`) but its one-look reused-gold confirmation was `0.6209002783`, so it did not replace B13. Canonical results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

`CNN_CPC` now uses several distinct evaluation resources. They answer different questions and must not be mixed.

## 1. External technical fixture

`fixtures/external_validation/` is for software checks only: DICOM decoding, routing, preprocessing, missing-stream masking and inference plumbing. It is **not** a scientific benchmark.

## 2. Local released test metadata

The provided local test metadata contains 3 studies and 15 series. It has no labels and cannot measure AUC. Data-contract audits on this surface are engineering checks only.

## 3. Official 58-study expert-gold development set

The 58 fully labelled training studies are the scientific expert-labelled development surface. They have supported repeated sequential decisions and must be described as **development/model-selection data, not pristine independent validation**.

All 58 contain complete binary labels for all 12 targets. Historical nested/OOF folds remain available for early experiments, while later B7-B15 experiments use the complete 58-study set for one-shot development evaluation after training without gold gradients.

Gold labels do not enter B13/B14/B15 optimization or early stopping, but prior results on these same 58 studies have influenced the campaign. The surface is therefore repeatedly reused development evidence.

## 4. Frozen weak B6 holdout v2

The weak-v2 surface is a different resource:

```text
surface                   weak_b6_holdout_v2
active B6 studies         3120
weak-train studies        2497
holdout studies            623
holdout usable cells      2875
positive / negative    1407 / 1468
report-group overlap         0
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

It was frozen **before** B15/control training using only B6 labels and normalized report groups. It uses no gold labels and no model predictions in split selection.

Weak-v2 measures **agreement with the B6 report teacher, not expert truth**. Its role is pre-gold model ranking, not final performance estimation.

## 5. Strict weak-v2 bootstrap

Weak-v2 scoring uses study-level bootstrap with a fixed 12-target estimand:

```text
sample studies with replacement
-> compute all 12 target AUCs
-> reject replicate if any target AUC is undefined
-> macro = mean of exactly 12 defined AUCs
```

This is necessary because the rarest class, Synovitis negative, has only four holdout cells.

## 6. Historical gold development results

| Candidate | Gold macro AUC | Status |
|---|---:|---|
| B0 | `0.4762536432` | baseline |
| report teacher | `0.49245` | rejected |
| B1 | `0.5030284974` | retained reference |
| B2 | `0.4993244663` | rejected |
| B3 | `0.4944652486` | rejected |
| B4 | `0.5137567459` | retained ablation |
| B5 | `0.5243650851` | representation baseline |
| B7-v1 | `0.5397724412` | coverage ablation |
| B7.1 | `0.5644802945` | historical benchmark |
| B5+B7.1 rank | `0.5540141184` | rejected |
| B8 | `0.5300962807` | rejected |
| B9 | `0.5334962669` | rejected |
| B10 | `0.5523982721` | rejected |
| B11.1 | `0.5506902702` | rejected |
| B12 | `0.5660915179` | retained historical reference |
| **B13** | **`0.6293565948`** | **development champion** |
| B14 | `0.6197914249` | rejected globally |
| **B15** | **`0.6209002783`** | **weak-v2 gate passed; no global gold improvement** |

B11-v1 failed its pseudo-label viability gate and B12.1 was implemented but not trained.

## 7. B13 retained benchmark

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
```

Paired comparisons:

```text
B13-B12 median     +0.0638674720
95% CI            [+0.0127183837,+0.1144643292]
P(B13 > B12)       0.9920

B13-B7.1 median    +0.0652260946
95% CI            [+0.0039768779,+0.1266069220]
P(B13 > B7.1)      0.9808
```

## 8. B14 controlled successor

```text
B14 macro AUC      0.6197914249
B13 macro AUC      0.6293565948
median(B14-B13)   -0.0093726931
95% paired CI     [-0.0469823411,+0.0250137870]
P(B14 > B13)       0.2924
```

B14 is rejected globally. The paired interval crosses zero, so this is a model-selection decision rather than a claim of definitive inferiority.

## 9. B15 nested validation chain

B15 was the first experiment to use the frozen weak-v2 gate before touching the reused gold surface.

### MRI-domain SSL

B15 SSL excluded:

```text
58 expert-gold studies/images
623 weak-v2 holdout studies/images
```

It trained on 3,726 competition studies and 20,534 eligible real MRI series for four exact full passes.

### Matched downstream training

Both control and candidate trained on exactly:

```text
2497 weak-train studies
13974 series
11248 B6 supervised cells
5464 positive / 5784 negative
4 full epochs
```

The control used direct ImageNet initialization; B15 used ImageNet -> knee-MRI same-study contrastive SSL. Downstream architecture and optimization were otherwise matched.

### Weak-v2 results

```text
B13-v2 control            0.5652498118
95% CI                   [0.5361620323,0.5924683768]

B15                      0.7319060415
95% CI                   [0.6903737595,0.7675416396]

paired B15-control
raw delta                 +0.1666562297
median delta              +0.1675245839
95% CI                    [+0.1124433208,+0.2165156305]
P(B15 > control)           1.0000
valid paired replicates    4921 / 5000
```

The predeclared gate required positive raw delta, positive paired median, and `P>=0.95`; B15 passed all three.

### One-look expert-gold confirmation

```text
B15 macro AUC      0.6209002783
95% CI            [0.5706720829,0.6675892903]
B13 macro AUC      0.6293565948
raw B15-B13       -0.0084563164
```

The weak-v2 improvement therefore did not transfer to a global expert-gold improvement. B13 remains champion. No target-wise B13/B15 hybrid is allowed.

## 10. Interpretation of weak versus gold disagreement

The difference between `0.7319` weak-v2 and `0.6209` gold is not a calibration issue between the same labels; the two surfaces measure different label sources. Weak-v2 asks how well predictions rank B6 report-derived labels. Gold asks how well they rank expert binary labels.

B15 shows that a representation can improve compatibility with the weak teacher substantially without improving expert-label macro AUC. This motivates direct auditing of the supervision states before further B15-like tuning.

## 11. Paired bootstrap comparison

For aligned expert-gold prediction files, the generic evaluator reports median `B-A`, its 95% paired bootstrap interval and `P(B>A)`.

For weak-v2 B15, use the dedicated strict paired comparison because it enforces all-12-target validity on every accepted replicate.

Never compare marginal confidence-interval overlap as a substitute for an aligned paired analysis when the prediction files are available.

## 12. Current validation discipline

Do not:

- select target-specific post-hoc winners;
- optimize ensemble weights from the reused 58 studies;
- retune B6 parser rules/weak-label weights from downstream gold outcomes;
- regenerate weak-v2 based on model results;
- call weak-v2 teacher agreement expert validation;
- retune B15 SSL epochs, architecture, TTA or downstream schedule from its gold confirmation;
- use the three-study test surface or external fixture as scientific validation;
- describe the best local development AUC as a leaderboard or hidden-test guarantee.

## 13. Next evidence step

Before training another candidate, audit B6 report states against expert truth on the already-reused gold surface:

```text
positive
negated
uncertain
unmentioned
```

The audit is diagnostic; any new supervision policy must be separately named and frozen before model evaluation. In particular, unmentioned findings must not be blindly mapped to negative.

The Kaggle hidden evaluation remains the next genuinely independent model-performance signal.