# B4 frozen SSL + classical pathology classifiers

B4 is the next controlled candidate after B0 random initialization, B1 strong in-domain SSL, B2 discriminative encoder learning rate, and B3 pathology-aware low-capacity MIL.

## Motivation

The completed end-to-end experiments cluster near chance-to-modest discrimination on only 58 fully gold-labelled studies:

- B0 random-init: macro AUC `0.4762536432`
- B1 strong SSL: macro AUC `0.5030284974`
- B2 lower encoder LR: macro AUC `0.4993244663`
- B3 pathology-aware MIL: macro AUC `0.4944652486`
- fixed B1+B3 rank averaging: macro AUC `0.5048038179`

B2 and B3 show that changing fine-tuning policy or head architecture alone does not reliably solve the small-gold-set variance problem. B4 therefore changes the statistical problem: the competition-trained SSL encoder is frozen completely and gold labels are used only by low-capacity target-specific classifiers.

## Representation

B4 loads the strong SSL ConvNeXt encoder from `runs/ssl_strong/ssl_encoder.pt` and requires its checkpoint source to be exactly `competition_training_data`.

For each available MRI stream, the deterministic centre view is encoded slice-by-slice. The frozen slice embeddings are pooled using:

1. mean;
2. standard deviation;
3. maximum.

The resulting cache has shape `[study, 6 streams, 3 * encoder_dim]` plus six explicit stream-presence flags.

The initial OOF experiment extracts only the 58 gold studies. Feature extraction is label-free and the SSL encoder never receives gold-label gradients.

## Nested classical classifier

For each outer fold:

1. the outer fold is untouched;
2. the usual inner fold is used only to select target-specific hyperparameters;
3. PCA and logistic regression are fitted on the remaining gold training fold;
4. the selected recipe is refitted on all non-outer gold studies;
5. the outer fold is predicted exactly once.

PCA is always fitted only on the relevant training partition. Outer labels are never used for feature-mode, PCA, or regularization selection.

Each target chooses between two predeclared feature modes:

- `all`: all six streams;
- `prior`: a fixed anatomy/sequence subset declared before seeing B4 OOF results.

Default hyperparameter grid:

```text
PCA components: 4, 8, 12, 16
logistic C:     0.1, 1.0
feature modes:  all, prior
```

The PCA grid is capped below the smallest nested training sample count, so the candidate complexity is valid in every fold.

## Install and test

```bash
pip install -e .
pytest -q tests/test_frozen_features.py
```

## Extract frozen gold features

Use the same strong-SSL config already used for B1:

```bash
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --split train \
  --scope gold \
  --out runs/b4_frozen_ssl/gold_features.npz
```

Equivalent module invocation:

```bash
python -m rsna_knee.frozen_features extract \
  --config configs/train_local_ssl_strong.yaml \
  --split train \
  --scope gold \
  --out runs/b4_frozen_ssl/gold_features.npz
```

The extractor also writes `runs/b4_frozen_ssl/gold_features.json` with the frozen-encoder and feature contract.

## Run nested B4 OOF

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_frozen_ssl \
  --n-bootstrap 5000
```

This writes:

```text
runs/b4_frozen_ssl/fold0/oof.csv
runs/b4_frozen_ssl/fold0/selection.json
runs/b4_frozen_ssl/fold1/oof.csv
runs/b4_frozen_ssl/fold1/selection.json
runs/b4_frozen_ssl/fold2/oof.csv
runs/b4_frozen_ssl/fold2/selection.json
runs/b4_frozen_ssl/oof.csv
runs/b4_frozen_ssl/evaluation.json
runs/b4_frozen_ssl/policy.json
```

## Compare against B1

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_ssl_strong/fold0/oof.csv \
    runs/stage1_ssl_strong/fold1/oof.csv \
    runs/stage1_ssl_strong/fold2/oof.csv \
  --compare-oof runs/b4_frozen_ssl/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b1_vs_b4.json

cat runs/b1_vs_b4.json
```

The comparison orientation is A=B1 and B=B4, so a positive `median_difference` and `probability_b_better > 0.5` favor B4.

## Interpretation rule

B4 is a controlled diagnostic, not a guaranteed improvement. A substantial OOF increase would support the hypothesis that supervised end-to-end fine-tuning is the dominant variance source. A result near B1 would instead indicate that the frozen SSL representation itself does not contain enough target-separable information for these 12 abnormalities.
