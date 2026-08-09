# B5 — competition-only image-report representation learning

> **Status — 2026-08-10:** **COMPLETE / MAIN STANDALONE BASELINE.** The B5 encoder completed all four predefined representation-training epochs and the frozen gold probe has now completed. Using the unchanged B4 downstream probe, B5 achieved macro AUC `0.5243650851` with 95% bootstrap CI `[0.4728108406, 0.5761619105]`. This is the highest standalone point estimate in the current campaign. The paired B4-vs-B5 bootstrap favors B5 but is statistically inconclusive: median difference `+0.0105821232`, 95% CI `[-0.0408197338, +0.0622131599]`, `P(B5 > B4)=0.656`.

Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

B5 changes the MRI representation while keeping the downstream gold-label probe fixed.

## Motivation

B1 strong SSL uses only MRI structure: multiple views from the same study are positives and plane/sequence metadata are auxiliary labels. B5 adds semantic supervision from radiology reports without converting the reports into brittle 12-target hard pseudo-labels.

The 58 explicitly gold-labelled studies are excluded completely from B5 representation training. The text space and MRI training both use only the 4,349 report-only competition studies.

The controlled reference is:

```text
B4 image-only frozen representation + original B4 probe
macro AUC = 0.5137567459
95% CI   = [0.4619827141, 0.5642366629]
```

B5 is compared with that representation under the **same downstream B4 probe**.

## Text representation

B5 does not use an external clinical language model.

Reports are normalized with the repository text normalizer and represented with:

- word TF-IDF, 1-2 grams;
- at most 20,000 features by default;
- `min_df=2` by default;
- TruncatedSVD to at most 256 dimensions;
- L2 normalization.

The fitted TF-IDF/SVD objects are stored for audit/reproducibility. The report branch is training-only and is discarded for MRI-only inference.

## MRI initialization

B5 initializes ConvNeXt from the completed strong competition-only SSL checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

No ImageNet or other external image weights are loaded.

Default learning rates:

- encoder: `5e-5`;
- newly initialized heads: `2e-4`.

## Objectives

B5 keeps the strong MRI SSL objectives and adds report alignment:

```text
loss = image_weight * image_contrast
     + metadata_weight * metadata_loss
     + report_weight * (report_NCE + cosine_weight * report_cosine)
```

Defaults:

- image weight: 1.0;
- metadata weight: 0.25;
- report weight: 0.5;
- report cosine coefficient: 0.25;
- image temperature: 0.15;
- report temperature: 0.10.

For each study, active 2.5D ConvNeXt features are mean-pooled to one study representation before the report projection head.

Because local MRI batches are small, report contrast uses a queue of normalized report embeddings (default 256) as additional negatives. Exact duplicate normalized report hashes are masked as negatives so duplicate reports do not become false negatives.

## Leakage contract

B5 representation training:

- uses competition training MRI only;
- uses competition training reports only;
- excludes all 58 gold studies;
- uses no outer-fold labels;
- uses no external image weights;
- uses no external language model.

The saved `b5_encoder.pt` follows the competition-only checkpoint source contract so the B4 frozen-feature extractor can consume it directly. Extra metadata records `variant=b5_image_report_tfidf_svd` and the report-supervision policy.

## Completed training run

Command:

```bash
rsna-knee-b5 \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/ssl_strong/ssl_encoder.pt \
  --out-root runs/b5_report_ssl
```

Checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Outputs:

```text
runs/b5_report_ssl/b5_encoder.pt
runs/b5_report_ssl/history.json
runs/b5_report_ssl/coverage.json
runs/b5_report_ssl/policy.json
runs/b5_report_ssl/report_semantics.json
runs/b5_report_ssl/report_semantics.npz
runs/b5_report_ssl/report_text_space.joblib
```

### Training history

| Epoch | Total loss | Image contrast | Metadata | Report NCE | Report cosine | Encoder LR | Head LR | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.520392 | 3.006825 | 0.447246 | 4.603128 | 0.801537 | 4.2824e-5 | 1.7086e-4 | 1403.84 |
| 2 | 5.100010 | 2.961406 | 0.399780 | 3.906748 | 0.682283 | 2.5500e-5 | 1.0050e-4 | 1441.52 |
| 3 | 4.893490 | 2.936515 | 0.380151 | 3.566160 | 0.630856 | 8.1759e-6 | 3.0143e-5 | 1539.21 |
| 4 | 4.704915 | 2.893706 | 0.368420 | 3.290113 | 0.592378 | 1.0000e-6 | 1.0000e-6 | 1434.28 |

All four epochs completed 1,000 batches each with `budget_limited=false`.

Totals:

```text
completed epochs       4
batches              4000
study draws         16000
active 2.5D examples 158886
queue size            256
```

The optimization trend was healthy: total loss, image contrast, metadata loss, report NCE, and report cosine loss all decreased monotonically.

## Frozen feature audit

The deterministic gold feature cache confirms that the completed B5 checkpoint was used:

```text
candidate                 B4_frozen_ssl_classical
split                     train
scope                     gold
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

The `candidate=B4_frozen_ssl_classical` label names the fixed downstream frozen-feature/classical-probe machinery; the recorded checkpoint path is the B5 encoder.

## Controlled B5 probe — completed

The original B4 target-wise nested PCA/logistic-regression probe was reused unchanged:

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000
```

The pooled OOF evaluation is:

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

## Controlled comparison with B4

Using B4 image-only representation as A and B5 image-report representation as B:

```text
B4 macro AUC              0.5137567459
B5 macro AUC              0.5243650851
paired median difference +0.0105821232
paired 95% CI            [-0.0408197338, +0.0622131599]
P(B5 > B4)                0.656
valid replicates          5000
```

B5 improves the observed point estimate, but the paired confidence interval crosses zero. The correct interpretation is therefore **positive but statistically inconclusive evidence that report-aligned representation learning improves the MRI representation** on this 58-study gold set.

Target-level descriptive changes versus B4:

| Target | B4 AUC | B5 AUC | B5 - B4 |
|---|---:|---:|---:|
| ACL | 0.585784 | 0.667892 | +0.082108 |
| MCL | 0.480726 | 0.405896 | -0.074830 |
| Medial Meniscus | 0.542067 | 0.665865 | +0.123798 |
| Lateral Meniscus | 0.604969 | 0.617391 | +0.012422 |
| Medial OA | 0.550388 | 0.658915 | +0.108527 |
| Lateral OA | 0.398453 | 0.404255 | +0.005803 |
| PF OA | 0.638353 | 0.606178 | -0.032175 |
| Effusion | 0.444720 | 0.516770 | +0.072050 |
| Synovitis | 0.445639 | 0.555556 | +0.109916 |
| Baker's | 0.375000 | 0.385870 | +0.010870 |
| Contusion | 0.558704 | 0.399460 | -0.159244 |
| Fracture | 0.540278 | 0.408333 | -0.131944 |

B5 is higher on 8 of the 12 target point estimates. These differences are descriptive and are **not** a license to create target-specific post-hoc B4/B5 winners from the same outer OOF labels.

## Decision

B5 is now the **main standalone representation baseline** because it has the highest controlled standalone point estimate in the campaign. B4 remains the critical image-only ablation.

Do not use the completed B5 outer OOF result to tune:

- target-specific B4/B5 model selection;
- report-loss weights or temperatures;
- extra representation-training epochs;
- new downstream B4 selector variants;
- post-hoc ensemble weights.

Any future representation-fusion experiment should be predefined independently of these outer target-level differences and evaluated as a new controlled experiment.
