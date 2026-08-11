# B12.1 — hierarchical learned series-token aggregation

> **Status — 2026-08-11:** IMPLEMENTED / PREDECLARED / TRAINING READY. Package `0.21.0`.

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

B12 is retained as statistically tied with B7.1, not declared superior.

## Single scientific change versus B12

```text
B12:
K real series x 16 slice tokens -> study Transformer -> pathology queries

B12.1:
16 slice tokens -> learned 8-head per-series attention pool -> 1 series token
K series tokens -> same 2-layer study Transformer -> same pathology queries
```

There is no series-rank/position embedding.

## Frozen controls

```text
B5 competition-only encoder initialization
B6 v1.2.1 supervision only
3120 active training studies
14123 supervised cells
6871 positive / 7252 negative cells
17475 eligible real MRI series
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
legacy 224x224 resize
16 2.5D positions per real series
plane/fluid/fat metadata embeddings
batch size 2
same B12 seed/DataLoader offsets
same optimizer / LR / augmentation
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

B12.1 is explicitly competition-only. Its trainer rejects external pretrained flags; ImageNet belongs to the separate B13 experiment.

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
0.21.0
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

## Frozen gold evaluation

```bash
rsna-knee-b12-1-eval \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_1_hierarchical/b12_1_model.pt \
  --out-root runs/b12_1_hierarchical/gold_eval
```

Primary comparisons remain B12.1 versus B12 and B7.1 with aligned 5,000-replicate bootstrap.

## Relation to B13

B13 is now a separate first-class experiment, not a mode of the B12.1 trainer:

```text
B12.1
trainer      rsna-knee-b12-1
encoder      B5 competition-only SSL
checkpoint   runs/b12_1_hierarchical/b12_1_model.pt

B13
trainer      rsna-knee-b13
encoder      torchvision ConvNeXt-Tiny IMAGENET1K_V1
normalization standard ImageNet mean/std
checkpoint   runs/b13_imagenet/b13_model.pt
```

B13 keeps this exact hierarchical architecture and training surface. See [`B13_IMAGENET_INIT.md`](B13_IMAGENET_INIT.md).

## Later roadmap

B12.2 remains conditional pathology-conditioned series attention. The previously planned stronger competition-only SSL experiment has been renumbered to **B14**, and optional scanner/protocol robustness to **B15**, so B13 has one unambiguous meaning.

Full roadmap: [`ROADMAP_AFTER_B12_1.md`](ROADMAP_AFTER_B12_1.md).
