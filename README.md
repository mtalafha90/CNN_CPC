# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B12 variable-number-of-series modeling has the highest development point estimate so far**, macro AUC `0.5660915179`, but is statistically tied with B7.1 (`0.5644802945`): paired median `(B12-B7.1)=+0.0023747526`, 95% CI `[-0.0472104067,+0.0481427722]`, `P(B12>B7.1)=0.5376`. **B12.1 hierarchical learned series-token aggregation is implemented and predeclared as the active experiment.**

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B12 result/protocol: [`docs/B12_VARIABLE_SERIES.md`](docs/B12_VARIABLE_SERIES.md).  
B12.1 protocol: [`docs/B12_1_HIERARCHICAL_SERIES.md`](docs/B12_1_HIERARCHICAL_SERIES.md).  
Post-B12.1 roadmap: [`docs/ROADMAP_AFTER_B12_1.md`](docs/ROADMAP_AFTER_B12_1.md).

## Current software state

```text
package version        0.20.0
retained benchmark     B7.1 full-corpus weak supervision
benchmark macro AUC    0.5644802945
highest point estimate B12 variable-series model = 0.5660915179
active experiment      B12.1 hierarchical learned series tokens
external pretraining   disabled
final inference        MRI-only
```

## Experiment ladder

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL | `0.5030284974` | retained reference |
| B2 | 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured multilingual report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query model + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | **retained benchmark** |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| B10 | B7.1 + physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute-threshold B7.1 teacher completion | n/a | stopped at pseudo viability gate |
| B11.1 | per-target quantile teacher tails | `0.5506902702` | rejected globally |
| **B12** | **variable number of real MRI series** | **`0.5660915179`** | **retained / statistically tied with B7.1** |
| **B12.1** | **learned per-series token compression + study Transformer** | pending | **implemented / training ready** |
| B12.2 | pathology-conditioned series attention | future | **conditional on B12.1 supporting the all-series branch** |
| B13 | stronger competition-only MRI SSL | future | planned major representation experiment |
| B14 | scanner/protocol robustness augmentation | future | optional, only if justified |

## B12 result

B12 keeps every repaired Sagittal/Coronal/Axial MRI acquisition instead of selecting six fixed semantic slots. Its frozen label-free audit retained 17,475 real series versus 15,468 historical unique selected series, adding 2,007 acquisitions (+12.98%) across 1,099 of 3,120 studies (35.22%).

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761, 0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

Decision: retain B12 as the highest point estimate but **do not claim superiority**. Do not build target-wise B7.1/B12 winners from the reused 58-study development set.

Frozen B12 series mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Why B12.1

B12 sends all `K x 16` slice tokens directly into one study Transformer. B12.1 keeps the exact same 17,475 real-series surface but introduces one learned attention query per real MRI series:

```text
16 slice tokens
    -> learned 8-head per-series attention query
    -> 1 series token
K series tokens
    -> unchanged 2-layer study Transformer
    -> unchanged pathology-query heads
```

B12.1 keeps the same B5 initialization, B6 supervision, 3,120 studies, 14,123 cells, augmentation, optimizer, four epochs, TTA, legacy resize, metadata embeddings, random seed offsets and frozen B12 series policy. It has no series-rank/position embedding.

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
0.20.0
```

## Tests

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b12_variable_series.py \
  tests/test_b12_1_hierarchical.py \
  tests/test_b7_weak_supervision.py
```

## Active next step — train B12.1

Reuse the already frozen successful B12 series policy; do not regenerate or alter it.

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b12-1 \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b12_1_hierarchical
```

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

Do not run gold evaluation unless all four epochs satisfy the complete study and series coverage contract.

## Frozen B12.1 gold evaluation

```bash
rsna-knee-b12-1-eval \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_1_hierarchical/b12_1_model.pt \
  --out-root runs/b12_1_hierarchical/gold_eval
```

The primary paired comparisons are B12.1 versus B12 and B12.1 versus B7.1, each with the same aligned 5,000-replicate bootstrap. Do not tune pooling heads, target-specific winners, series caps, or ensemble weights on the repeatedly reused 58-study gold development surface.

## Planned path after B12.1

The next steps are intentionally limited so repeated use of the 58-study development surface does not turn into uncontrolled architecture search.

```text
B12.1 hierarchical series aggregation
   |
   |-- if the all-series branch remains supported:
   |      B12.2 pathology-conditioned series attention
   |
   |-- if B12.1 is clearly worse:
   |      skip B12.2
   |
   v
B13 stronger competition-only MRI self-supervised learning
   |
   v
B14 scanner/protocol robustness augmentation
   |  optional; only if a clear domain-robustness problem remains
   v
FINAL MODEL FREEZE
   |
   v
KAGGLE SUBMISSION / independent leaderboard signal
```

### B12.2 — conditional only

B12.2 would let each pathology query learn which acquired series are relevant, rather than using one generic study representation for every target. It will only be attempted if B12.1 provides evidence that the all-series branch remains worthwhile. No target-specific routing rules will be selected from the 58 gold studies.

### B13 — major remaining representation experiment

B13 is planned as stronger **competition-only MRI self-supervised learning**, with candidate objectives such as same-study cross-sequence contrastive learning, masked slice/token reconstruction, and cross-plane consistency. The best globally retained architecture from the B12 family would then be initialized from B13 instead of B5.

### B14 — optional robustness experiment

B14 is reserved for acquisition/scanner robustness if justified by diagnostics. Candidate perturbations include intensity/contrast variation, resolution/downsampling perturbation, acquisition-quality variation, and metadata dropout. This is not a return to B10-style fixed physical normalization.

After these major experiments, the model should be frozen and submitted. No target-wise winner selection, threshold tuning, series-count tuning, or ensemble-weight optimization should be performed on the reused 58-study development set.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
