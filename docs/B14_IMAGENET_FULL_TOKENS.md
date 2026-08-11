# B14 — ImageNet full slice-token aggregation

> **Status — 2026-08-11:** IMPLEMENTED / PREDECLARED / TRAINING READY. Package `0.22.0`.

## Motivation

B13 is the current development champion:

```text
B13 macro AUC       0.6293565948
95% CI             [0.5789896351,0.6775867717]

B13 vs B12
median difference  +0.0638674720
95% paired CI      [+0.0127183837,+0.1144643292]
P(B13 > B12)        0.9920

B13 vs B7.1
median difference  +0.0652260946
95% paired CI      [+0.0039768779,+0.1266069220]
P(B13 > B7.1)       0.9808
```

B13 compresses each real MRI series from 16 encoded slice tokens to one generic learned series token before the study Transformer. The pathology queries therefore see one token per acquisition rather than the individual slice-level representations.

B14 tests whether this compression is discarding focal pathology information.

## Single scientific change versus B13

```text
B13
16 slice tokens / real series
    -> learned 8-head attention pool
    -> 1 token / real series
K series tokens
    -> 2-layer study Transformer
    -> pathology-query cross-attention

B14
16 slice tokens / real series
    -> NO series compression
K x 16 slice tokens
    -> same 2-layer study Transformer
    -> same pathology-query cross-attention
```

B14 reuses the already implemented B12 full-slice-token model architecture. No new target-specific routing, pooling rule, series cap or hand-coded pathology logic is introduced.

## Frozen controls

Everything below remains the B13 recipe:

```text
ImageNet encoder protocol
  torchvision ConvNeXt-Tiny IMAGENET1K_V1
  standard ImageNet mean/std normalization

training studies        3120
B6 supervised cells    14123
positive cells          6871
negative cells          7252
eligible MRI series    17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376

16 sampled 2.5D positions / series
224x224 legacy MRI resize
plane/fluid/fat metadata embeddings
batch size 2
ConvNeXt encoder LR 1e-5
head LR 1e-4
weight decay 1e-4
same augmentation
same seed and DataLoader seed offsets
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients
zero gold early stopping
```

The checked-in B14 config matches B13 on all training/preprocessing fields. The only scientific difference is the aggregation declaration.

## Shared initialization control

B12 and B12.1 were intentionally written so all parameters shared by the full-token and hierarchical models are constructed in the same seeded order; the B13 series-pooling module is created only after the shared parameters.

B14 preserves this property. It constructs the full-token model from the same seed and then loads the exact same torchvision ImageNet encoder state used by B13. Therefore shared non-encoder initialization is controlled; the intended model difference is the presence/absence of the B13 series-compression module and the resulting memory length.

## Experiment identity

```text
experiment   B14_imagenet_full_slice_tokens
variant      b14_imagenet_b6_full_slice_tokens_v1
aggregation  all_real_series_x_16_slice_tokens_v1
trainer      rsna-knee-b14
evaluator    rsna-knee-b14-eval
checkpoint   runs/b14_imagenet_full_tokens/b14_model.pt
```

There is no B5 checkpoint argument and no B13 checkpoint warm-start. B14 begins from the same public ImageNet ConvNeXt initialization as B13.

## Install / tests

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected:

```text
0.22.0
```

Run:

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b13_imagenet_init.py \
  tests/test_b14_full_slice_tokens.py \
  tests/test_b12_variable_series.py
```

## Train B14

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b14 \
  --config configs/b14_imagenet_full_tokens.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b14_imagenet_full_tokens
```

Every complete epoch must report:

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

Do not alter the recipe during the run. Do not evaluate gold unless all four epochs satisfy the full contract.

## Frozen gold evaluation

```bash
rsna-knee-b14-eval \
  --config configs/b14_imagenet_full_tokens.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b14_imagenet_full_tokens/b14_model.pt \
  --out-root runs/b14_imagenet_full_tokens/gold_eval
```

## Primary paired comparison — B14 versus B13

This comparison is fixed before seeing B14 gold predictions:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --compare-oof runs/b14_imagenet_full_tokens/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b14_imagenet_full_tokens/gold_eval/b13_vs_b14.json
```

`probability_b_better` is the probability that B14 is better than B13 under the aligned bootstrap.

## Decision rule

B14 is evaluated globally by macro ROC AUC and the paired bootstrap versus B13.

```text
If B14 is clearly better globally:
    retain B14 as new development champion.

If B14 is statistically tied with B13:
    retain both as viable, prefer the simpler/stronger independent-signal candidate
    rather than target-wise mixing.

If B14 is clearly worse:
    reject B14 and retain B13.
```

Do not construct per-target B13/B14 winners, tune slice counts, alter pooling, change epochs or search ensemble weights from the 58-study result.

## Interpretation limitation

The 58 fully labelled studies have been reused throughout sequential development. Any B14 score remains a development/model-selection estimate, not independent validation. The next genuinely independent signal is still the Kaggle hidden test/leaderboard.
