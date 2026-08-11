# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B13 is the new development champion**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`. Its paired improvement is resolved versus both B12 and B7.1. B12.1 remains implemented but is intentionally skipped in the competition workflow; therefore the isolated causal contribution of ImageNet initialization versus the hierarchical architecture is not claimed. The next priority is model freeze and an actual Kaggle submission for an independent signal.

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B12 result: [`docs/B12_VARIABLE_SERIES.md`](docs/B12_VARIABLE_SERIES.md).  
B12.1 archived control protocol: [`docs/B12_1_HIERARCHICAL_SERIES.md`](docs/B12_1_HIERARCHICAL_SERIES.md).  
B13 result/protocol: [`docs/B13_IMAGENET_INIT.md`](docs/B13_IMAGENET_INIT.md).

## Current software state

```text
package version         0.21.0
previous benchmark      B7.1 full-corpus weak supervision = 0.5644802945
previous best           B12 variable-series model = 0.5660915179
new development champion B13 ImageNet hierarchical model = 0.6293565948
B12.1                   implemented / skipped for competition workflow
next priority           freeze B13-v1 -> Kaggle submission
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
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | previous retained benchmark |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **variable number of real MRI series** | **`0.5660915179`** | retained / statistically tied with B7.1 |
| **B12.1** | **learned per-series token compression + B5 init** | not run | implemented / skipped |
| **B13** | **hierarchical all-series model + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / NEW DEVELOPMENT CHAMPION** |

## B13 result

B13 completed four exact full-coverage epochs on the frozen B6/B12 training surface:

```text
epoch 1 loss  0.7450505349
epoch 2 loss  0.6865059846
epoch 3 loss  0.6524747430
epoch 4 loss  0.6132239342

per epoch:
batches                         1560
study_draws                     3120
active_supervision_cells_seen  14123
positive_cells_seen             6871
negative_cells_seen             7252
series_instances_seen          17475
expected_series_instances      17475
max_series_in_any_batch           14
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

Frozen gold development evaluation:

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
58 studies
5000 / 5000 usable bootstrap replicates
```

Paired versus B12:

```text
median(B13-B12)    +0.0638674720
95% paired CI      [+0.0127183837,+0.1144643292]
P(B13 > B12)        0.9920
```

Paired versus B7.1:

```text
median(B13-B7.1)   +0.0652260946
95% paired CI      [+0.0039768779,+0.1266069220]
P(B13 > B7.1)       0.9808
```

Both paired confidence intervals are above zero. B13 is therefore retained as the strongest development model so far.

## Scientific interpretation

B13 uses the B12.1 hierarchical learned series-token architecture together with the torchvision ConvNeXt-Tiny `IMAGENET1K_V1` encoder protocol and standard ImageNet normalization. B12.1 was not trained, so the project does **not** claim that the full B13 gain is caused solely by ImageNet initialization. Relative to B12, both hierarchical aggregation and encoder protocol differ; relative to B7.1, additional representation changes differ as well.

The 58 fully labelled cases have been reused throughout sequential development. B13's `0.6294` is therefore a **development/model-selection estimate**, not an independent validation or leaderboard result.

## Frozen B12 series surface

```text
training studies        3120
supervised cells       14123
positive / negative  6871 / 7252
eligible real series   17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Clean B12.1 / B13 separation

B12.1 remains competition-only and requires B5:

```text
rsna-knee-b12-1
rsna-knee-b12-1-eval
runs/b12_1_hierarchical/b12_1_model.pt
```

B13 is a separate first-class experiment with no B5 checkpoint argument:

```text
rsna-knee-b13
rsna-knee-b13-eval
runs/b13_imagenet/b13_model.pt
```

## Current decision

For the competition workflow:

```text
B13-v1 RETAIN
B12.1 SKIP
no target-wise hybrids
no ImageNet/LR/epoch/normalization sweep on the 58-study surface
freeze B13-v1
prepare Kaggle test inference/submission
use leaderboard performance as the next independent signal
```

Further B14/B15 research experiments are deferred unless an independent competition result or a clear technical diagnostic justifies reopening development.
