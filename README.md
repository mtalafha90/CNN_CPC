# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-12:** **B13 remains the reused-gold development champion** at macro AUC `0.6293565948`. B15 decisively improved agreement with the frozen B6 weak-label teacher on weak holdout v2 (`0.7319060415` versus matched B13-v2 control `0.5652498118`), but its one-look reused-gold confirmation was `0.6209002783`, so it did **not** replace B13 globally. The next evidence-driven step is a B6 report-state audit before defining any new supervision experiment.

Canonical records:

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — complete experiment ledger.
- [`docs/B15_MRI_SSL.md`](docs/B15_MRI_SSL.md) — B15 protocol and results.
- [`docs/WEAK_HOLDOUT_V2.md`](docs/WEAK_HOLDOUT_V2.md) — frozen weak-v2 contract and paired gate.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation-surface governance.
- [`docs/RAISING_AUC.md`](docs/RAISING_AUC.md) — post-B15 improvement roadmap.

## Current software state

```text
package version          0.24.1
primary metric           12-target macro ROC AUC
development champion     B13 = 0.6293565948
B14 gold                 0.6197914249 / rejected globally
B15 weak-v2              0.7319060415 / gate passed
B15 reused gold          0.6209002783 / did not replace B13
weak holdout v2          frozen / teacher agreement only
slice undersampling      rejected as primary B13 bottleneck
next evidence step       B6 report-state audit
final inference          MRI-only
```

## Experiment ladder

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL | `0.5030284974` | retained reference |
| B2 | lower encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query model + B6 labels | `0.5397724412` | coverage ablation |
| B7.1 | B7 with full 3,120-study epoch coverage | `0.5644802945` | historical benchmark |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | failed viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| B12 | all real MRI series + full slice-token memory + B5 init | `0.5660915179` | retained historical reference |
| B12.1 | hierarchical one-token-per-series + B5 init | not run | implemented / skipped |
| **B13** | **hierarchical one-token-per-series + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / DEVELOPMENT CHAMPION** |
| B14 | full `K x 16` slice-token memory + B13 ImageNet protocol | `0.6197914249` | rejected globally |
| **B15** | **ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy** | **weak-v2 `0.7319060415`; gold `0.6209002783`** | **weak gate passed; no global gold improvement** |

## B13 retained result

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]

B13 vs B12
median delta       +0.0638674720
95% paired CI      [+0.0127183837,+0.1144643292]
P(B13 > B12)        0.9920

B13 vs B7.1
median delta       +0.0652260946
95% paired CI      [+0.0039768779,+0.1266069220]
P(B13 > B7.1)       0.9808
```

## B14 controlled successor

```text
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
raw B14-B13       -0.0095651699
paired median     -0.0093726931
95% paired CI     [-0.0469823411,+0.0250137870]
P(B14 > B13)       0.2924
```

B14 fit B6 more strongly than B13 but did not improve global gold macro AUC. Do not extend B14 or construct target-wise B13/B14 hybrids.

## Frozen weak holdout v2

The v2 surface was frozen before B15/control training using only B6 labels and report grouping.

```text
surface                  weak_b6_holdout_v2
active B6 studies        3120
weak-train studies       2497
holdout studies           623
holdout usable cells     2875
positive / negative   1407 / 1468
report-group overlap        0
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

Weak-v2 measures **agreement with the B6 report teacher, not expert truth**.

## B15 completed experiment

### MRI-domain SSL

B15 SSL excluded all 58 gold studies and all 623 weak-v2 holdout studies. It used 3,726 studies and 20,534 eligible series per full pass. All four frozen SSL epochs completed exactly, with loss decreasing:

```text
2.7094607696
2.5811344701
2.5187829415
2.4756854072
```

### Matched downstream comparison

Both B13-v2 control and B15 trained on exactly:

```text
2497 studies
13974 real MRI series
11248 usable B6 cells
5464 positive / 5784 negative
1249 batches per epoch
4 complete epochs
```

The only intended model difference was encoder initialization: direct ImageNet for the control versus ImageNet -> knee-MRI SSL for B15.

### Weak-v2 gate

```text
B13-v2 control macro AUC  0.5652498118
B15 macro AUC             0.7319060415
raw B15-control          +0.1666562297
paired median            +0.1675245839
95% paired CI            [+0.1124433208,+0.2165156305]
P(B15 > control)          1.0000
valid paired replicates   4921 / 5000
predeclared gate          PASS
```

`P=1.0` means every usable paired bootstrap replicate favored B15; it is not a claim of mathematical certainty.

### One-look reused-gold confirmation

```text
B15 macro AUC      0.6209002783
95% CI            [0.5706720829,0.6675892903]
B13 macro AUC      0.6293565948
raw B15-B13       -0.0084563164
```

The very large gain in weak-teacher agreement did **not** transfer to a global improvement on the expert-labelled development surface. B13 therefore remains the development champion. B15 is retained as an important representation/supervision diagnostic, not as the global winner.

## Completed exact B13 slice audit

```text
series audited/readable  17475 / 17475
eval unique fraction     median 100.0%
complete eval exposure   95.9%
eval max skipped run     median 0.0 slices (p95 0.0)
```

Decision: slice-count undersampling as the primary B13 bottleneck is rejected. Do not launch a gold-driven 24/32/48-slice sweep.

## Current direction

The B15 result shifts attention from simply improving MRI representation or downstream capacity toward the supervision interface. The next step is an **audit, not another training run**:

```text
B6 report state per target
positive / negated / uncertain / unmentioned
        |
        v
compare with expert truth on the already-reused gold surface
        |
        v
quantify state-specific truth rates and coverage
        |
        v
only then define a separately frozen supervision successor if justified
```

Do not blindly convert unmentioned findings to negatives. Do not construct target-wise B13/B15 hybrids, tune B15 from the gold confirmation, or regenerate weak-v2 based on model performance.

The next genuinely independent performance signal remains the hidden Kaggle evaluation.