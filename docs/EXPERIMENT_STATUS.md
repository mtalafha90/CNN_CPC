# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.21.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **New development champion:** **B13**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.
- Versus B12, paired median improvement is `+0.0638674720`, 95% CI `[+0.0127183837,+0.1144643292]`, `P(B13>B12)=0.9920`.
- Versus B7.1, paired median improvement is `+0.0652260946`, 95% CI `[+0.0039768779,+0.1266069220]`, `P(B13>B7.1)=0.9808`.
- Both paired confidence intervals are above zero on the reused 58-study development surface.
- **B12.1 remains implemented but is intentionally skipped for the competition path.** Consequently, a pure causal attribution of the B13 gain to ImageNet initialization alone is not made.
- **Next priority:** freeze B13-v1 and obtain an actual Kaggle leaderboard signal before further local tuning.

## Experiment ladder

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with lower encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL representation + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **full 3,120-study B7 coverage** | **`0.5644802945`** | previous retained benchmark |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series, variable length** | **`0.5660915179`** | retained / statistically tied with B7.1 |
| **B12.1** | **learned per-series token compression + B5 init** | not run | implemented / skipped for competition path |
| **B13** | **hierarchical all-series model + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / NEW DEVELOPMENT CHAMPION** |

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

## B12 reference result

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761,0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

Frozen B12 mapping:

```text
eligible real series 17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## B13 — completed clean ImageNet encoder protocol

B13 uses the hierarchical learned series-token architecture prepared for B12.1 and replaces the B5 encoder protocol with:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
+ standard ImageNet mean/std normalization
```

The ImageNet weights and expected normalization are treated as one coherent encoder protocol.

### Frozen controls

```text
same 3120 studies
same 14123 supervised cells
same 6871 positive / 7252 negative cells
same 17475 real MRI series
same B12 series SHA-256
same hierarchical learned series-token architecture
same batch size 2
same encoder LR 1e-5
same head LR 1e-4
same augmentation
same 4 epochs
same TTA [-1,0,1]
same 5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

### Training integrity

All four epochs satisfied exact full study and series coverage:

```text
epoch 1 loss  0.7450505349
epoch 2 loss  0.6865059846
epoch 3 loss  0.6524747430
epoch 4 loss  0.6132239342

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

### Frozen gold development result

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
n                  58
bootstrap          5000/5000 usable
```

Per-target AUCs:

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

These target-level values are descriptive only and must not be used to construct target-specific model mixtures.

### Paired B13 versus B12

```text
median_difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
probability_b_better    0.9920
valid replicates        5000
```

### Paired B13 versus B7.1

```text
median_difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
probability_b_better    0.9808
valid replicates        5000
```

Both paired confidence intervals remain above zero.

## B12.1 decision

B12.1 remains implemented and reproducible but is **not being run for the competition path**. Its absence means the project cannot isolate the ImageNet encoder protocol from hierarchical aggregation in a strict B13-versus-B12.1 ablation.

This is an explicit tradeoff: preserving development budget and reducing further reuse of the same 58 gold studies is prioritized over completing that causal ablation.

## Current decision / next stage

```text
B13-v1 RETAIN
B12.1 SKIP for competition path
B12.2 DEFER
B14 stronger in-domain SSL DEFER
B15 robustness DEFER
freeze B13-v1 architecture / preprocessing / series policy / TTA
prepare competition test predictions
create Kaggle submission
use leaderboard performance as the next independent signal
```

Do not tune target-specific winners, ImageNet variants, normalization, learning rates, epoch counts, series caps, thresholds or ensemble weights on the repeatedly reused 58-study development set.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
