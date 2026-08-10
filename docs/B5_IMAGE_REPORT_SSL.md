# B5 — competition-only image-report representation learning

> **Status — 2026-08-10:** **COMPLETE / RETAINED REPRESENTATION BASELINE.** B5 achieved macro AUC `0.5243650851` with 95% bootstrap CI `[0.4728108406, 0.5761619105]` under the unchanged B4 frozen probe. It is no longer the campaign leader: B7.1 full-corpus weak supervision reached `0.5644802945`. B8 spatial-anatomy learning is currently training.

Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Goal

B5 changes the MRI representation while keeping the downstream gold-label probe fixed. It uses the 4,349 report-only competition studies as semantic representation supervision without converting reports into brittle 12-target hard pseudo-labels.

All 58 gold studies are excluded from B5 representation training.

## Controlled reference

```text
B4 image-only frozen representation + original B4 probe
macro AUC = 0.5137567459
95% CI   = [0.4619827141, 0.5642366629]
```

B5 is compared with B4 under the **same downstream probe**.

## Text representation

B5 uses no external clinical language model.

```text
normalized competition report
-> word TF-IDF, 1-2 grams
-> at most 20,000 features
-> min_df = 2
-> TruncatedSVD <= 256 dimensions
-> L2 normalization
```

The fitted TF-IDF/SVD objects are retained for reproducibility. The report branch is training-only and is discarded for MRI-only inference.

## MRI initialization

B5 initializes ConvNeXt from:

```text
runs/ssl_strong/ssl_encoder.pt
```

No ImageNet or other external image weights are loaded.

## Objectives

```text
loss = image_weight * image_contrast
     + metadata_weight * metadata_loss
     + report_weight * (report_NCE + cosine_weight * report_cosine)
```

Reference coefficients:

```text
image weight       1.0
metadata weight    0.25
report weight      0.5
report cosine      0.25
image temperature  0.15
report temperature 0.10
```

For each study, active 2.5D ConvNeXt features are mean-pooled to one study representation before the report projection head. A report embedding queue of 256 supplies semantic negatives for small MRI batches, with exact duplicate normalized reports masked as false negatives.

## Leakage contract

B5 representation training:

- uses competition training MRI only;
- uses competition training reports only;
- excludes all 58 gold studies;
- uses no outer-fold labels;
- uses no external image weights;
- uses no external language model.

Checkpoint metadata records `variant=b5_image_report_tfidf_svd` and competition-only provenance.

## Completed training run

Checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Training history:

| Epoch | Total loss | Image contrast | Metadata | Report NCE | Report cosine | Seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.520392 | 3.006825 | 0.447246 | 4.603128 | 0.801537 | 1403.84 |
| 2 | 5.100010 | 2.961406 | 0.399780 | 3.906748 | 0.682283 | 1441.52 |
| 3 | 4.893490 | 2.936515 | 0.380151 | 3.566160 | 0.630856 | 1539.21 |
| 4 | 4.704915 | 2.893706 | 0.368420 | 3.290113 | 0.592378 | 1434.28 |

Totals:

```text
completed epochs       4
batches              4000
study draws         16000
active 2.5D examples 158886
queue size            256
budget limited        false
```

All logged objectives decreased monotonically.

## Frozen feature audit

```text
studies                   58
feature shape             [58, 6, 2304]
pooling                   mean + std + max
encoder frozen            true
encoder trainable params  0
checkpoint                runs/b5_report_ssl/b5_encoder.pt
checkpoint source         competition_training_data
completed epochs          4
external pretrained       false
n_slices                  16
image_size                224
triplet_gap               1
metadata repairs needed   0
```

## Controlled B5 probe

The original B4 target-wise nested PCA/logistic-regression probe was reused unchanged.

Pooled result:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
n         = 58
bootstrap = 5000/5000 usable
```

Per-target AUC:

| Target | B5 AUC |
|---|---:|
| ACL | `0.6678921569` |
| MCL | `0.4058956916` |
| Medial Meniscus | `0.6658653846` |
| Lateral Meniscus | `0.6173913043` |
| Medial OA | `0.6589147287` |
| Lateral OA | `0.4042553191` |
| PF OA | `0.6061776062` |
| Effusion | `0.5167701863` |
| Synovitis | `0.5555555556` |
| Baker's | `0.3858695652` |
| Contusion | `0.3994601889` |
| Fracture | `0.4083333333` |

## Controlled B4 -> B5 comparison

```text
B4 macro AUC              0.5137567459
B5 macro AUC              0.5243650851
paired median difference +0.0105821232
paired 95% CI            [-0.0408197338, +0.0622131599]
P(B5 > B4)                0.656
valid replicates          5000
```

Interpretation: B5 improves the observed representation point estimate, but the paired interval crosses zero.

B5 is higher on 8 of 12 target point estimates versus B4. These target-level differences are descriptive only and are not used to choose post-hoc target-specific winners.

## Downstream role in the current pipeline

B5 remains important even though it is no longer the leader:

1. it is the retained report-aligned representation baseline;
2. its ConvNeXt encoder initializes B7;
3. B7/B7.1 build direct pathology-specific weak supervision on top of the B5 representation;
4. B7.1 is the current best standalone development model;
5. B8 initializes from B7.1 and tests whether coarse within-slice spatial information improves pathology evidence extraction.

The predeclared B5+B7.1 50:50 rank ensemble was evaluated once and scored `0.5540141184`, below B7.1 `0.5644802945`; that ensemble branch is closed.

## Decision

Retain B5 as the **report-aligned representation baseline and B7 initialization source**. B4 remains the image-only ablation.

Do not use the completed B5 gold result to tune:

- target-specific B4/B5 model selection;
- report-loss weights or temperatures;
- extra B5 representation-training epochs;
- new downstream B4 selector variants;
- post-hoc ensemble weights.

Current leader and active-experiment status belong in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md), not in reinterpretations of the B5 score.
