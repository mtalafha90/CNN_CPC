# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.20.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Retained benchmark:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`.
- **Highest point estimate:** **B12 variable-number-of-series**, macro AUC `0.5660915179`, 95% CI `[0.5094993761,0.6244034568]`.
- B12 versus B7.1 paired median difference `+0.0023747526`, 95% CI `[-0.0472104067,+0.0481427722]`, `P(B12>B7.1)=0.5376`; therefore B12 is retained as statistically tied, not declared superior.
- **B12.1 hierarchical learned series-token aggregation is implemented, frozen and training ready.**
- Post-B12.1 development is limited to **B12.2 only if justified**, then **B13**, optional **B14**, and final model freeze/submission.

## Experiment ladder

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL representation + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | multilingual structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels, limited epoch coverage | `0.5397724412` | coverage ablation |
| **B7.1** | **same B7 recipe with full 3,120-study coverage** | **`0.5644802945`** | **retained benchmark** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial tokens + fixed anatomy priors | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | plane-specific in-plane physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute-threshold B7.1 teacher completion | n/a | stopped at viability gate |
| B11.1 | calibration-aware target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series, variable length** | **`0.5660915179`** | **retained / statistically tied with B7.1** |
| **B12.1** | **learned per-series token compression** | pending | **implemented / training ready** |
| B12.2 | pathology-conditioned series attention | future | conditional on B12.1 |
| B13 | stronger competition-only MRI SSL | future | planned major representation experiment |
| B14 | scanner/protocol robustness augmentation | future | optional only if justified |

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

## Closed branches before B12

```text
B8     macro AUC 0.5300962807   P(B8>B7.1)=0.1156
B9     macro AUC 0.5334962669   P(B9>B7.1)=0.0562
B10    macro AUC 0.5523982721   P(B10>B7.1)=0.2706
B11.1  macro AUC 0.5506902702   P(B11.1>B7.1)=0.2184
```

B11-v1 was never trained because its absolute teacher-confidence pseudo-label gate failed. The teacher-derived pseudo-label branch is closed for now.

## B12 result

B12 replaced six selected semantic slots by every repaired Sagittal/Coronal/Axial acquisition while keeping B5 initialization, B6 supervision, legacy resize, optimizer, augmentation and four full epochs unchanged.

Frozen label-free series audit:

```text
training studies                         3120
eligible real series                    17475
historical dual unique series           15468
extra real series                        2007
extra series fraction                  12.9752%
studies gaining extras                   1099
fraction studies gaining extras        35.2244%
zero-series studies                          0
historical selected series missing          0
series/study median                          5
q90 / q95 / q99 / max                 8 / 9 / 10 / 14
```

Frozen mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

Training completed four exact full-coverage epochs with 17,475 successfully loaded series per epoch. Frozen development evaluation:

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761,0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

Decision: retain B12 as the highest point estimate and as evidence that the all-series direction is viable, but do not claim a confirmed improvement or create target-wise B7.1/B12 winners.

## Active experiment: B12.1 hierarchical learned series tokens

### Hypothesis

B12 may expose redundant/noisy acquisitions too directly because all `K x 16` slice tokens enter one study Transformer. B12.1 tests whether explicit per-series compression improves use of the same real-series information.

### Single scientific change versus B12

```text
B12:
K x 16 slice tokens -> 2-layer study Transformer -> pathology queries

B12.1:
16 slice tokens -> learned 8-head attention query -> 1 series token
K series tokens -> same 2-layer study Transformer -> same pathology queries
```

There is no series-rank/position embedding.

### Frozen controls

```text
same B5 encoder initialization
same B6 v1.2.1 supervision
same 3120 active studies
same 14123 cells (6871 positive / 7252 negative)
same 17475-series B12 mapping and SHA-256
same legacy 224x224 resize
same 16 2.5D slice positions per series
same plane/fluid/fat metadata embeddings
same seed and DataLoader seed offsets as B12
same optimizer / LR / augmentation
same batch size 2
same 4 full epochs
same TTA [-1,0,1]
same 5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

Expected every epoch:

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

Primary frozen comparisons after four complete epochs are B12.1 versus B12 and B12.1 versus B7.1, each using the aligned 5,000-replicate paired bootstrap. Do not tune pooling heads, target-wise winners, series caps or ensemble weights from the reused 58-study surface.

See [`B12_1_HIERARCHICAL_SERIES.md`](B12_1_HIERARCHICAL_SERIES.md) for commands.

## Frozen post-B12.1 roadmap

The number of remaining major development experiments is intentionally small. Repeatedly using the same 58-study surface for many local architecture variants would increasingly optimize to the development set rather than to unseen competition data.

### Decision after B12.1

- If B12.1 is **clearly worse** than B12, close the B12 architecture branch and move directly to B13.
- If B12.1 is **competitive with or better than B12** and the all-series hypothesis remains supported, allow one final architecture experiment: B12.2.
- Do not create B12.2 by choosing per-target B12/B12.1 winners.

### B12.2 — pathology-conditioned series attention

Purpose: let each pathology query learn which acquired series are relevant rather than forcing all pathologies to consume the same generic study representation. This is a global architectural experiment, not a set of hand-coded target-specific routing rules.

### B13 — stronger competition-only MRI self-supervised learning

B13 is the main remaining representation experiment. Candidate objectives include:

```text
same-study cross-sequence contrastive learning
masked slice/token reconstruction
cross-plane consistency
```

The globally retained architecture from the B12 family would be initialized from B13 rather than B5.

### B14 — optional scanner/protocol robustness

Only pursue B14 if diagnostics indicate a remaining acquisition/domain robustness problem. Candidate perturbations include intensity/contrast variation, resolution/downsampling perturbation, acquisition-quality variation and metadata dropout. Do not return to target-wise or B10-style fixed normalization tuning on gold.

### Final stage

After B13 and, only if justified, B14:

```text
freeze one global model
freeze preprocessing and inference policy
create competition test predictions
create Kaggle submission
use leaderboard performance as the next independent signal
```

Do not tune target-specific winners, thresholds, series caps or ensemble weights on the repeatedly reused 58-study development set.

Full roadmap: [`ROADMAP_AFTER_B12_1.md`](ROADMAP_AFTER_B12_1.md).

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
