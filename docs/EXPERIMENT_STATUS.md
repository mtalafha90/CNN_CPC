# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.22.1`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than independent validation.

## Current headline

- **Development champion remains B13**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.
- **B14 is completed and rejected globally**, macro AUC `0.6197914249`, 95% CI `[0.5706800512,0.6693542716]`.
- Raw macro difference: `B14-B13 = -0.0095651699`.
- Paired B14-vs-B13 median difference: `-0.0093726931`, 95% CI `[-0.0469823411,+0.0250137870]`, `P(B14>B13)=0.2924`.
- The paired CI crosses zero, so B14 and B13 are statistically unresolved on the reused 58-study surface; model selection nevertheless retains B13 because B14 has the lower point estimate, low probability of superiority, higher token-memory cost and no global advantage.
- B14 final B6 loss was `0.5822778610` versus B13 `0.6132239342`; stronger fitting of weak supervision did not improve macro AUC.
- **Pre-B15 gate:** run the corrected exact B13 slice-exposure audit and freeze a report-group-safe B6 weak holdout before any B15/control training.
- Reserved next representation hypothesis: **B15 = ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchy**.

## Experiment ladder

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with lower encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **full 3,120-study B7 coverage** | **`0.5644802945`** | previous benchmark |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series + full slice-token memory + B5 init** | **`0.5660915179`** | retained / tied with B7.1 |
| B12.1 | one learned token per series + B5 init | not run | implemented / skipped |
| **B13** | **one learned token per series + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / DEVELOPMENT CHAMPION** |
| **B14** | **full `K x 16` slice-token memory + same ImageNet protocol** | **`0.6197914249`** | **COMPLETED / REJECTED GLOBALLY** |
| **B15** | **ImageNet -> knee-MRI SSL -> B13 hierarchy** | not run | reserved next representation hypothesis |

## Frozen B6 supervision surface

```text
report-only studies       4349
active B6 studies         3120
inactive B6 studies       1229
usable B6 cells          14123
positive cells            6871
negative cells            7252
```

Frozen policy:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

B6 audit quality values include sensitivity `0.975`, specificity `0.606`, positive precision `0.690`, balanced accuracy `0.790`, and coverage `0.361`. These establish noisy/incomplete supervision but **do not establish a numerical downstream macro-AUC ceiling**.

## Frozen all-series surface

```text
training studies        3120
eligible real series   17475
historical dual unique 15468
extra series            2007
max series / study        14
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811b7c8439bd7bcd376
```

## B13 retained result

B13 uses torchvision ConvNeXt-Tiny `IMAGENET1K_V1` plus standard ImageNet mean/std normalization and hierarchical one-token-per-series aggregation.

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
```

Training loss:

```text
epoch 1  0.7450505349
epoch 2  0.6865059846
epoch 3  0.6524747430
epoch 4  0.6132239342
```

Per-target B13 AUCs, descriptive only:

```text
ACL                0.4742647059
MCL                0.5555555556
Medial Meniscus    0.6093750000
Lateral Meniscus   0.6795031056
Medial OA          0.6279069767
Lateral OA         0.6189555126
PF OA              0.6177606178
Effusion           0.7677018634
Synovitis          0.7108721625
Baker's            0.7481884058
Contusion          0.5533063428
Fracture           0.5888888889
```

## B14 completed result

B14 preserved B13's ImageNet protocol but removed one-token-per-series compression and retained every `K x 16` slice token through the study Transformer.

```text
epoch 1 loss  0.7346330162
epoch 2 loss  0.6606430862
epoch 3 loss  0.6074723502
epoch 4 loss  0.5822778610

B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
n                  58
bootstrap          5000/5000 usable

raw macro delta         -0.0095651699
median_difference       -0.0093726931
95% paired CI           [-0.0469823411,+0.0250137870]
probability_b_better     0.2924
valid replicates         5000
```

Per-target B14 AUCs, descriptive only:

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

No target-level B13/B14 hybrid is permitted.

## Corrected pre-B15 diagnostics

### Exact B13 slice exposure

The old `16 / n_slices` proxy is retired. B13 actually uses 16 2.5D triplets, training gap choices `[1,2]`, center jitter `+/-2`, and evaluation TTA offsets `[-1,0,1]`.

`rsna-knee-slice-audit` now reconstructs the exact non-gold B13 surface, verifies the frozen 17,475-series SHA, uses orientation-projected DICOM geometry, and reports actual unique frame exposure, maximum unsampled runs, expected random-training-view exposure and the legal training-exposure envelope.

```bash
rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out runs/slice_audit_b13
```

### Frozen weak holdout

`rsna-knee-weak-holdout` freezes a report-group-safe split before new model training. A requested 20% split is roughly 624 studies, not 3,120, and B6 is sparse, so uncertainty is measured by empirical bootstrap on the actual holdout rather than assumed from a study-count scaling formula.

```bash
rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --out-root runs/weak_holdout_v1
```

Existing B13/B14 checkpoints were trained on all 3,120 active B6 studies and therefore cannot be retrospectively scored on this holdout and called validation.

For a valid future comparison:

```text
same frozen weak-train partition
    |-- B13-control: ImageNet -> B13 hierarchy
    `-- B15:        ImageNet -> MRI SSL -> B13 hierarchy

paired weak holdout -> biased teacher-agreement ranking
58 gold             -> one development confirmation only
Kaggle hidden       -> independent signal
```

## Current decision / next stage

```text
B13 RETAIN / development champion
B14 REJECT globally
run corrected diagnostics
freeze weak holdout before B15/control training
no B14 epoch extension
no B13/B14 target-wise hybrid
no ensemble-weight search on gold

B15 reserved hypothesis:
ImageNet -> competition knee-MRI SSL -> B13 hierarchy
```

For B15, all 58 gold studies must be excluded from SSL optimization. Gold labels remain forbidden from gradients, early stopping and checkpoint selection. Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
