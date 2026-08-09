# B5 — competition-only image-report representation learning

> **Status — 2026-08-09:** **REPRESENTATION TRAINING COMPLETE / FROZEN GOLD PROBE PENDING.** The B5 encoder completed all four predefined epochs cleanly. No B5 macro AUC is available yet; do not enter a B5 score in README tables, manuscript text, or model comparisons until the frozen B5 probe has completed.

Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

B5 changes the MRI representation while keeping the downstream gold-label probe fixed.

## Motivation

B1 strong SSL uses only MRI structure: multiple views from the same study are positives and plane/sequence metadata are auxiliary labels. B5 adds semantic supervision from radiology reports without converting the reports into brittle 12-target hard pseudo-labels.

The 58 explicitly gold-labelled studies are excluded completely from B5 representation training. The text space and MRI training both use only the 4,349 report-only competition studies.

The controlled reference before B5 is:

```text
B4 image-only frozen representation + original B4 probe
macro AUC = 0.5137567459
95% CI   = [0.4619827141, 0.5642366629]
```

B5 must be compared with that representation under the **same downstream B4 probe**.

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

The optimization trend is healthy: total loss, image contrast, metadata loss, report NCE, and report cosine loss all decreased monotonically. This establishes stable B5 pretraining, but it does **not** establish improved gold-label generalization.

## Completion checks

Inspect the saved artifacts before probing:

```bash
cat runs/b5_report_ssl/policy.json
cat runs/b5_report_ssl/report_semantics.json
cat runs/b5_report_ssl/coverage.json
cat runs/b5_report_ssl/history.json
```

Confirm the checkpoint is competition-only, contains no gold representation-training rows, and records the expected four completed epochs.

## Controlled B5 probe — next step

Do **not** change the B4 classifier protocol for the first B5 test.

Extract deterministic gold features:

```bash
mkdir -p runs/b5_frozen_probe

rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --split train \
  --scope gold \
  --out runs/b5_frozen_probe/gold_features.npz
```

Expected contract:

```text
study_uids = 58
features   = [58, 6, 2304]
finite     = true
```

Run the **original B4 target-wise nested PCA/logistic probe unchanged**:

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000
```

Then compare B4 image-only representation (A) with B5 image-report representation (B):

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b4_frozen_ssl/oof.csv \
  --compare-oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b4_vs_b5.json

cat runs/b4_vs_b5.json
```

Positive `median_difference` and `probability_b_better > 0.5` favor B5.

## Decision gate

The first B5 result answers only one question: **does report-aligned competition-only pretraining improve the MRI representation under the same downstream probe?**

Do not tune B5 using target-specific outer-fold winners, new B4 selector variants, extra pretraining epochs chosen after seeing gold OOF, or post-hoc ensemble weights. If B5 fails, diagnose the representation objective using non-gold training diagnostics before reopening downstream gold-label tuning.
