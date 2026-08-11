# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** B12 variable-number-of-series has the highest development point estimate so far, macro AUC `0.5660915179`, but is statistically tied with B7.1 (`0.5644802945`). B12.1 hierarchical series aggregation remains pending. **B13 is now a clean standalone ImageNet encoder-protocol experiment** with its own trainer, evaluator, checkpoint identity and CLI.

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B12 result: [`docs/B12_VARIABLE_SERIES.md`](docs/B12_VARIABLE_SERIES.md).  
B12.1 protocol: [`docs/B12_1_HIERARCHICAL_SERIES.md`](docs/B12_1_HIERARCHICAL_SERIES.md).  
B13 protocol: [`docs/B13_IMAGENET_INIT.md`](docs/B13_IMAGENET_INIT.md).

## Current software state

```text
package version         0.21.0
retained benchmark      B7.1 full-corpus weak supervision
benchmark macro AUC     0.5644802945
highest point estimate  B12 variable-series model = 0.5660915179
B12.1                   implemented / pending
B13                     implemented / training ready
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
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | **retained benchmark** |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **variable number of real MRI series** | **`0.5660915179`** | **retained / statistically tied with B7.1** |
| **B12.1** | **learned per-series token compression** | pending | implemented |
| **B13** | **B12.1 architecture + ImageNet ConvNeXt encoder protocol** | pending | **implemented / training ready** |

## B12 result

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761, 0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

B12 retained 17,475 real MRI series versus 15,468 historical unique selected series, adding 2,007 acquisitions across 1,099 of 3,120 studies. The frozen mapping SHA-256 is:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Clean B12.1 / B13 separation

B12.1 is competition-only and requires the B5 encoder checkpoint:

```text
rsna-knee-b12-1
rsna-knee-b12-1-eval
runs/b12_1_hierarchical/b12_1_model.pt
```

B13 is a separate experiment and has **no B5 checkpoint argument**:

```text
rsna-knee-b13
rsna-knee-b13-eval
runs/b13_imagenet/b13_model.pt
```

B13 uses:

```text
same B12.1 hierarchical architecture
same B12 17,475-series surface
same B6 supervision
same optimizer / LR / augmentation
same 4 full epochs
same TTA [-1,0,1]

changed encoder protocol:
B5 competition-only encoder
    -> torchvision ConvNeXt-Tiny IMAGENET1K_V1
    -> standard ImageNet mean/std normalization
```

The full B13 contract rejects accidental changes to architecture, training schedule, augmentation, series mapping or evaluation policy.

## Install / update

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected:

```text
0.21.0
```

## Focused tests

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b12_1_hierarchical.py \
  tests/test_b13_imagenet_init.py
```

## Train B13

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b13 \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b13_imagenet
```

ImageNet weights are downloaded by torchvision on first use if they are not already cached. There is deliberately no `--b5-checkpoint` argument.

Every full epoch must preserve:

```text
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

## Evaluate B13

After four complete epochs:

```bash
rsna-knee-b13-eval \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b13_imagenet/b13_model.pt \
  --out-root runs/b13_imagenet/gold_eval
```

Primary paired comparison is B13 versus B12.1 once both prediction files exist. Secondary comparisons are B13 versus B12 and B7.1, all with aligned 5,000-replicate bootstrap.

The 58 fully labelled studies have been repeatedly reused, so all such scores remain development/model-selection estimates rather than independent validation. Do not tune target-specific winners, learning rates, normalization variants, epoch counts or ensemble weights from this surface.
