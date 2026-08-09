# B5 — competition-only image-report representation learning

B5 changes the MRI representation while keeping the downstream gold-label probe fixed.

## Motivation

B1 strong SSL uses only MRI structure: multiple views from the same study are positives, and plane/sequence metadata are auxiliary labels. B5 adds semantic supervision from the radiology reports without converting reports into brittle binary pseudo-labels.

The 58 explicitly gold-labelled studies are excluded completely from B5 representation training. The text space and MRI training both use only the report-only competition studies.

## Text representation

B5 does not use an external clinical language model.

Reports are normalized with the repository text normalizer, then represented with:

- word TF-IDF, 1-2 grams,
- at most 20,000 features by default,
- `min_df=2` by default,
- TruncatedSVD to at most 256 dimensions,
- L2 normalization.

The fitted TF-IDF/SVD objects are stored for audit/reproducibility, but the report branch is discarded for MRI-only test inference.

## MRI initialization

B5 initializes ConvNeXt from the completed strong competition-only SSL checkpoint. No ImageNet or other external image weights are loaded.

Default B5 learning rates are deliberately discriminative:

- encoder: `5e-5`,
- newly initialized heads: `2e-4`.

## Objectives

B5 keeps the existing strong SSL objectives and adds report alignment:

`loss = image_weight * image_contrast + metadata_weight * metadata_loss + report_weight * (report_NCE + cosine_weight * report_cosine)`

Defaults:

- image weight: 1.0,
- metadata weight: 0.25,
- report weight: 0.5,
- report cosine coefficient: 0.25,
- image temperature: 0.15,
- report temperature: 0.10.

For each study, active 2.5D ConvNeXt features are mean-pooled to one study representation before the report projection head.

Because local MRI batch sizes are small, report contrast uses a queue of normalized report embeddings (default 256) as additional negatives. Duplicate normalized report hashes are masked as negatives, preventing exact duplicate reports from becoming false negatives.

## Leakage contract

B5 representation training:

- uses competition training MRI only,
- uses competition training reports only,
- excludes all gold studies,
- uses no outer-fold labels,
- uses no external image weights,
- uses no external language model.

The saved `b5_encoder.pt` uses the existing competition-only checkpoint source contract so the B4 frozen-feature extractor can consume it directly. Extra checkpoint metadata identifies `variant=b5_image_report_tfidf_svd` and the report-supervision policy.

## Run

After pulling and installing the current package:

```bash
rsna-knee-b5 \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/ssl_strong/ssl_encoder.pt \
  --out-root runs/b5_report_ssl
```

Outputs include:

- `b5_encoder.pt`
- `history.json`
- `coverage.json`
- `policy.json`
- `report_semantics.json`
- `report_semantics.npz`
- `report_text_space.joblib`

## Controlled B5 probe

Do not change the B4 classifier protocol for the first B5 test. Extract deterministic gold features from the B5 encoder:

```bash
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --split train \
  --scope gold \
  --out runs/b5_frozen_probe/gold_features.npz
```

Then run the original B4 target-wise nested PCA/logistic probe:

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000
```

This makes the primary comparison a representation test:

- B4: strong image-only SSL encoder + B4 probe,
- B5: image-report-aligned encoder + the same B4 probe.

A paired bootstrap should then compare `runs/b4_frozen_ssl/oof.csv` against `runs/b5_frozen_probe/oof.csv`.
