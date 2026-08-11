# B12.1 — hierarchical learned series-token aggregation

> **Status — 2026-08-11:** IMPLEMENTED / PREDECLARED / TRAINING READY.

## Motivation

B12 retained all 17,475 eligible training MRI series and achieved the highest development point estimate so far:

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761, 0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

B12 is therefore retained as statistically tied with B7.1, not declared superior. Its result supports continuing the all-series branch without target-wise selection on the reused 58-study development surface.

## Single scientific change versus B12

B12 sends every slice token from every real series directly into one study Transformer:

```text
K real series x 16 slice tokens -> study Transformer -> pathology queries
```

B12.1 keeps the exact same real-series surface but first compresses each series through one learned attention query:

```text
16 slice tokens
    -> learned per-series attention query
    -> 1 series token
K series tokens
    -> same 2-layer study Transformer
    -> same pathology-query heads
```

The learned query uses multi-head attention with 8 heads. There is no series-rank/position embedding, so arbitrary acquisition order is not encoded.

## Frozen controls

Unchanged from B12:

```text
B5 encoder initialization
B6 v1.2.1 supervision only
3120 active training studies
14123 supervised cells
6871 positive / 7252 negative cells
17475 eligible real MRI series
B12 series mapping SHA-256:
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
legacy 224x224 resize
16 2.5D positions per real series
plane/fluid/fat metadata embeddings
no series-rank embedding
batch size 2
same B12 seed and DataLoader seed offsets
same optimizer / LR / augmentation
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients
zero gold early stopping
```

B12.1 reuses the already frozen B12 `series_policy.json`; no new series audit or selection is allowed.

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
0.20.0
```

Run:

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b12_variable_series.py \
  tests/test_b12_1_hierarchical.py \
  tests/test_b7_weak_supervision.py
```

## Train B12.1

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

Do not run gold evaluation unless all four epochs satisfy the complete study and series contracts.

## Frozen gold evaluation

```bash
rsna-knee-b12-1-eval \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_1_hierarchical/b12_1_model.pt \
  --out-root runs/b12_1_hierarchical/gold_eval
```

Primary comparisons are both predeclared:

1. B12.1 versus parent B12, to test the hierarchical aggregation change.
2. B12.1 versus retained B7.1 benchmark.

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b12_variable_series/gold_eval/gold_predictions.csv \
  --compare-oof runs/b12_1_hierarchical/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b12_1_hierarchical/gold_eval/b12_vs_b12_1.json

python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b7_1_full_coverage/gold_eval/gold_predictions.csv \
  --compare-oof runs/b12_1_hierarchical/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b12_1_hierarchical/gold_eval/b71_vs_b12_1.json
```

Interpret `probability_b_better` as the probability that B12.1 is better than the first (`--oof`) model. Do not tune pooling heads, target-specific winners, series caps, or ensemble weights from the 58-study development result.
