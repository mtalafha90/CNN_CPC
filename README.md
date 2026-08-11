# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B13 remains the development champion**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`. **B14 is completed and rejected globally:** macro AUC `0.6197914249`, 95% CI `[0.5706800512,0.6693542716]`; paired median `B14-B13=-0.0093726931`, 95% CI `[-0.0469823411,+0.0250137870]`, `P(B14>B13)=0.2924`. B14 fit the B6 weak labels more strongly but did not improve global macro AUC. Before B15 training, package `0.22.1` adds two corrected diagnostics: an exact B13 2.5D slice-exposure audit and a leakage-safe report-group weak holdout.

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B13 result/protocol: [`docs/B13_IMAGENET_INIT.md`](docs/B13_IMAGENET_INIT.md).  
B14 completed result: [`docs/B14_IMAGENET_FULL_TOKENS.md`](docs/B14_IMAGENET_FULL_TOKENS.md).  
AUC-improvement diagnostics: [`docs/RAISING_AUC.md`](docs/RAISING_AUC.md).

## Current software state

```text
package version          0.22.1
previous benchmark       B7.1 = 0.5644802945
previous best            B12  = 0.5660915179
development champion     B13  = 0.6293565948
completed B14            0.6197914249 / rejected globally
pre-B15 diagnostics      exact slice exposure + frozen weak holdout
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
| **B15** | **ImageNet -> knee-MRI SSL -> B13 hierarchy** | not run | next representation hypothesis |

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

B14 removed B13's one-token-per-series compression and retained every `K x 16` slice token through the study Transformer while keeping the same ImageNet encoder protocol and frozen training recipe.

Training completed cleanly:

```text
epoch 1 loss  0.7346330162
epoch 2 loss  0.6606430862
epoch 3 loss  0.6074723502
epoch 4 loss  0.5822778610
```

Gold development result:

```text
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]

raw B14-B13       -0.0095651699
paired median     -0.0093726931
95% paired CI     [-0.0469823411,+0.0250137870]
P(B14 > B13)       0.2924
```

The paired CI crosses zero, so B13 and B14 are not statistically resolved on the reused 58-study development surface. Nevertheless, B14 has the lower point estimate, lower probability of superiority, greater memory cost, and slower training. **B13 remains the retained global model.**

The training-loss contrast is important:

```text
B13 final B6 loss   0.6132239342
B14 final B6 loss   0.5822778610
```

B14 fit the weak B6 supervision better but generalized worse by the primary macro-AUC point estimate. This argues against simply increasing downstream capacity or fitting the weak labels harder.

## B14 per-target result — descriptive only

```text
ACL                0.5122549020
MCL                0.4693877551
Medial Meniscus    0.6454326923
Lateral Meniscus   0.6881987578
Medial OA          0.5116279070
Lateral OA         0.5783365571
PF OA              0.5997425997
Effusion           0.8347826087
Synovitis          0.7419354839
Baker's            0.6884057971
Contusion          0.5465587045
Fracture           0.6208333333
```

Do not use target-level differences to construct a B13/B14 hybrid.

## Corrected pre-B15 diagnostics

### 1. Exact B13 slice exposure

The old `16 / number_of_slices` proxy was incorrect because B13 feeds 16 **2.5D triplets**, not 16 isolated slices. The corrected audit reconstructs the exact non-gold 3,120-study / 17,475-series B13 surface, verifies its frozen SHA, uses orientation-aware DICOM geometry, and computes the actual unique frames touched by training gap/jitter and evaluation TTA.

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out runs/slice_audit_b13
```

### 2. Freeze the weak holdout before new training

A 20% B6 holdout is roughly 624 studies, not 3,120, and B6 cells are sparse. Its uncertainty must therefore be measured from the actual holdout bootstrap rather than assumed from `1/sqrt(n)` scaling. The split is report-group safe and is frozen before any candidate/control training.

```bash
rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --out-root runs/weak_holdout_v1
```

**Do not retrospectively score existing B13/B14 checkpoints on this weak holdout as validation.** They were trained on the full 3,120-study B6 surface. Any future weak-holdout comparison requires a new matched B13 control and candidate trained with all holdout UIDs excluded.

## Current direction

```text
B13 RETAIN
B14 REJECT globally
no B14 epoch 5
no target-wise B13/B14 mixture

run corrected slice audit
freeze weak holdout
       |
       v
B15 hypothesis:
ImageNet ConvNeXt
      -> knee-MRI self-supervised adaptation
      -> B13 hierarchical aggregation
      -> frozen B6 downstream recipe
```

For B15, the 58 fully labelled gold studies must be excluded from SSL optimization and gold labels must remain absent from gradients, early stopping and checkpoint selection. The reused 58-study surface remains development confirmation only. A real Kaggle hidden-test/leaderboard result remains the next genuinely independent signal.
