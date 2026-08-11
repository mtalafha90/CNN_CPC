# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B13 remains the development champion**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`. **B14 is completed and rejected globally:** macro AUC `0.6197914249`, paired median `B14-B13=-0.0093726931`, 95% CI `[-0.0469823411,+0.0250137870]`, `P(B14>B13)=0.2924`. The full exact B13 slice audit is now complete on all `17,475` eligible series and rejects slice-count undersampling as a primary bottleneck. Weak holdout v1 is superseded before model training because its Synovitis holdout contained `70` positives and only `1` negative. Package `0.23.0` introduces **weak holdout v2**: report-group-safe multilabel/class stratification plus strict all-12-target bootstrap.

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B13 result/protocol: [`docs/B13_IMAGENET_INIT.md`](docs/B13_IMAGENET_INIT.md).  
B14 completed result: [`docs/B14_IMAGENET_FULL_TOKENS.md`](docs/B14_IMAGENET_FULL_TOKENS.md).  
Completed slice audit: [`docs/B13_SLICE_EXPOSURE_AUDIT.md`](docs/B13_SLICE_EXPOSURE_AUDIT.md).  
Weak holdout v2 contract: [`docs/WEAK_HOLDOUT_V2.md`](docs/WEAK_HOLDOUT_V2.md).  
AUC-improvement roadmap: [`docs/RAISING_AUC.md`](docs/RAISING_AUC.md).

## Current software state

```text
package version          0.23.0
previous benchmark       B7.1 = 0.5644802945
previous best            B12  = 0.5660915179
development champion     B13  = 0.6293565948
completed B14            0.6197914249 / rejected globally
slice undersampling      rejected as primary bottleneck
weak holdout v1          superseded before model training
weak holdout v2          stratified + strict 12-target bootstrap
next hypothesis          B15 ImageNet -> knee-MRI SSL -> B13 hierarchy
final inference           MRI-only
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
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | previous benchmark |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series with full slice-token memory + B5 init** | **`0.5660915179`** | retained / tied with B7.1 |
| B12.1 | hierarchical one-token-per-series + B5 init | not run | implemented / skipped |
| **B13** | **hierarchical one-token-per-series + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / DEVELOPMENT CHAMPION** |
| **B14** | **full `K x 16` slice-token memory + same ImageNet protocol as B13** | **`0.6197914249`** | **COMPLETED / REJECTED GLOBALLY** |
| **B15** | **ImageNet -> knee-MRI SSL -> B13 hierarchy** | not run | reserved next representation hypothesis |

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

## B14 completed result

```text
B14 final B6 loss  0.5822778610
B13 final B6 loss  0.6132239342

B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
raw B14-B13       -0.0095651699
paired median     -0.0093726931
95% paired CI     [-0.0469823411,+0.0250137870]
P(B14 > B13)       0.2924
```

B14 fit B6 better but did not improve global macro AUC. Do not extend B14 or build target-wise B13/B14 hybrids.

## Completed exact B13 slice audit

The corrected audit reproduced B13's real 2.5D sampling policy on all eligible non-gold series:

```text
series audited/readable  17475 / 17475
slices/series median     30 (p95 50, max 320)

eval unique fraction     median 100.0% (p25 100.0%)
eval max skipped run     median 0.0 slices (p95 0.0)
training expected/view   median 87.0%
complete eval exposure   95.9%
eval run >=2 slices      3.9%
eval run >=3 slices      3.8%

Axial      n=4455   eval=100.0% max-run=0.0 train/view=85.2%
Coronal    n=5815   eval=100.0% max-run=0.0 train/view=87.0%
Sagittal   n=7205   eval=100.0% max-run=0.0 train/view=87.0%
```

Decision:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

Do not launch a slice-count sweep from the reused gold surface.

## Weak holdout v1 -> v2

The first frozen report-group-safe 20% split had correct global size and zero report leakage but an unusable rare-class realization:

```text
v1 holdout studies       624
v1 usable cells         2697
v1 report overlap          0
v1 gold studies             0
v1 Synovitis             70 positive / 1 negative
v1 manifest SHA
fdbc02f88e5a4eff31783b4242890e943609d5c783bd54aca38af8a89e7e0968
```

No B15/control model was trained on v1, so package `0.23.0` supersedes it before model fitting. v2 uses only frozen B6 labels and report groups to choose a better-balanced split, with a minimum of four examples per class in train and holdout whenever globally feasible.

Freeze v2:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --min-class-count 4 \
  --search-candidates 4096 \
  --out-root runs/weak_holdout_v2
```

After v2 is frozen, its manifest must never be regenerated from model performance. Existing B13/B14 checkpoints cannot be scored on v2 as validation because they trained on all 3,120 B6-active studies.

## Current direction

```text
B13 RETAIN
B14 REJECT globally
slice-count hypothesis REJECT
weak holdout v1 SUPERSEDED
       |
       v
freeze weak holdout v2
       |
       +--> newly train matched B13-control on v2 weak-train
       `--> B15: ImageNet -> knee-MRI SSL -> B13 hierarchy
                     |
                     v
          paired strict 12-target weak bootstrap
                     |
                     v
          one development confirmation on reused gold
                     |
                     v
             Kaggle hidden signal
```

For B15, all 58 fully labelled gold studies remain excluded from SSL optimization, gradients, early stopping and checkpoint selection. The reused 58-study surface is development confirmation only; the hidden competition evaluation remains the independent signal.
