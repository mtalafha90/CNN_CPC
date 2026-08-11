# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.21.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Retained benchmark:** B7.1 full-corpus weak supervision, macro AUC `0.5644802945`.
- **Highest point estimate:** B12 variable-number-of-series, macro AUC `0.5660915179`, 95% CI `[0.5094993761,0.6244034568]`.
- B12 versus B7.1 paired median difference `+0.0023747526`, 95% CI `[-0.0472104067,+0.0481427722]`, `P(B12>B7.1)=0.5376`; B12 is statistically tied, not confirmed superior.
- **B12.1** hierarchical learned series-token aggregation is implemented and pending.
- **B13** is now a clean standalone ImageNet encoder-protocol experiment, implemented and training ready.

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
| **B7.1** | **full 3,120-study B7 coverage** | **`0.5644802945`** | **retained benchmark** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series, variable length** | **`0.5660915179`** | **retained / statistically tied with B7.1** |
| **B12.1** | **learned per-series token compression** | pending | implemented |
| **B13** | **B12.1 + ImageNet ConvNeXt encoder protocol** | pending | **implemented / training ready** |

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

## B12 result

Frozen all-series audit:

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
q90 / q95 / q99 / max                 8 / 9 / 10 / 14
```

Frozen mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

Frozen development result:

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761,0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

## B12.1

B12.1 keeps the complete B12 series surface but compresses each real MRI series before study aggregation:

```text
16 slice tokens
    -> learned 8-head attention pool
    -> 1 series token
K series tokens
    -> 2-layer study Transformer
    -> pathology queries
```

B12.1 remains competition-only and requires the B5 encoder checkpoint. Its trainer now explicitly rejects external pretraining so it cannot be confused with B13.

## B13 — clean standalone ImageNet encoder protocol

B13 uses the exact B12.1 hierarchical architecture and training/evaluation surface but replaces the encoder protocol:

```text
B12.1 encoder protocol
B5 competition-only SSL checkpoint

B13 encoder protocol
torchvision ConvNeXt-Tiny IMAGENET1K_V1
+ standard ImageNet mean/std normalization
```

The ImageNet weights and expected normalization are treated as one coherent encoder-initialization protocol. B13 is not described as a literal weight-only change.

### Frozen controls

```text
same 3120 studies
same 14123 supervised cells
same 6871 positive / 7252 negative cells
same 17475 real MRI series
same series SHA-256
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

B13 has its own experiment identity:

```text
trainer      rsna-knee-b13
evaluator    rsna-knee-b13-eval
checkpoint   runs/b13_imagenet/b13_model.pt
variant      b13_imagenet_init_b6_hierarchical_series_token_v1
```

There is deliberately no B5 checkpoint argument in the B13 trainer.

### Training command

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b13 \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b13_imagenet
```

Every epoch must report exact full study/series coverage:

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

Do not run gold evaluation unless all four epochs satisfy this contract.

See [`B13_IMAGENET_INIT.md`](B13_IMAGENET_INIT.md) for the full run/evaluation commands.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
