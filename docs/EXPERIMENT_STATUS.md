# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.22.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Development champion remains B13**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.
- **B14 is completed and rejected globally**, macro AUC `0.6197914249`, 95% CI `[0.5706800512,0.6693542716]`.
- Raw macro difference: `B14-B13 = -0.0095651699`.
- Paired B14-vs-B13 median difference: `-0.0093726931`, 95% CI `[-0.0469823411,+0.0250137870]`, `P(B14>B13)=0.2924`.
- The paired CI crosses zero, so B14 and B13 are statistically unresolved on the reused 58-study surface, but B14 has the lower point estimate, lower probability of superiority, greater token-memory cost, and slower training. **Retain B13; reject B14 globally.**
- B14 reached a substantially lower final B6 training loss (`0.5822778610`) than B13 (`0.6132239342`) without improving macro AUC, indicating that stronger fitting of the weak labels is not by itself the path to a better model.
- Next major representation hypothesis: **B15 = ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchical aggregation**, with all 58 gold studies excluded from SSL optimization.

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
| **B15** | **ImageNet -> knee-MRI SSL -> B13 hierarchy** | not run | next representation hypothesis |

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

## Frozen all-series surface

```text
training studies        3120
eligible real series   17475
historical dual unique 15468
extra series            2007
max series / study        14
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
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

### Training integrity

All four epochs completed the frozen full-coverage contract:

```text
epoch 1 loss  0.7346330162
epoch 2 loss  0.6606430862
epoch 3 loss  0.6074723502
epoch 4 loss  0.5822778610

batches                         1560 each epoch
study_draws                     3120 each epoch
active_supervision_cells_seen  14123 each epoch
positive_cells_seen             6871 each epoch
negative_cells_seen             7252 each epoch
series_instances_seen          17475 each epoch
max_series_in_any_batch           14
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

### Frozen gold result

```text
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
n                  58
bootstrap          5000/5000 usable
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

### Paired B14 versus B13

```text
raw macro delta         -0.0095651699
median_difference       -0.0093726931
95% paired CI           [-0.0469823411,+0.0250137870]
probability_b_better     0.2924
valid replicates         5000
```

The paired CI crosses zero. Therefore do not claim B14 is statistically worse in a strict hypothesis-testing sense. The model-selection decision is nevertheless to retain B13 because B14 has a lower global point estimate, low bootstrap probability of superiority, higher computational cost, and no global performance advantage.

### Descriptive target deltas B14-B13

```text
ACL               +0.0379901961
MCL               -0.0861678005
Medial Meniscus   +0.0360576923
Lateral Meniscus  +0.0086956522
Medial OA         -0.1162790698
Lateral OA        -0.0406189555
PF OA             -0.0180180180
Effusion          +0.0670807453
Synovitis         +0.0310633214
Baker's           -0.0597826087
Contusion         -0.0067476383
Fracture          +0.0319444444
```

These target-level differences must not be used to create target-specific B13/B14 hybrids.

## Current decision / next stage

```text
B13 RETAIN / development champion
B14 REJECT globally
no B14 epoch extension
no B13/B14 target-wise hybrid
no ensemble-weight search on gold

next major hypothesis:
B15 = ImageNet -> competition knee-MRI SSL -> B13 hierarchy
```

For B15, the 58 gold studies must be excluded from SSL optimization. Gold labels remain forbidden from gradients, early stopping and checkpoint selection.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
