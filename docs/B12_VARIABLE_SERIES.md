# B12 — variable-number-of-series MRI model

> **Status — 2026-08-11:** **LABEL-FREE SERIES AUDIT PASSED / TRAINING READY.** B7.1 remains the retained development champion at macro AUC `0.5644802945`.

B11.1 completed its frozen four-epoch contract but scored `0.5506902702`; the paired B11.1-B7.1 median difference was `-0.0126224565` with 95% CI `[-0.0487500119,+0.0195120537]` and `P(B11.1>B7.1)=0.2184`. B11.1 is rejected globally.

## Hypothesis

B7.1 maps each study into six fixed semantic slots:

```text
sagittal_fluid
sagittal_structural
coronal_fluid
coronal_structural
axial_fluid
axial_structural
```

That selection is deliberately low-capacity, but it can discard repeated or additional acquisitions. B12 tests whether preserving every usable real MRI series improves study-level pathology discrimination.

## Single scientific change

B12 returns to the **exact original B7.1 B6 supervision surface**. It does not use B11/B11.1 pseudo-labels.

Unchanged from B7.1:

- B5 competition-only encoder initialization;
- B6 v1.2.1 supervision only;
- 3,120 active studies and 14,123 supervised cells;
- 6,871 positive and 7,252 negative cells;
- B6-derived target-balance multipliers;
- legacy direct 224x224 resize; no B10 physical normalization;
- 16 sampled 2.5D positions per series;
- ConvNeXtTiny slice encoder;
- two-layer MRI Transformer;
- pathology-token context, cross-attention and 12 output heads;
- batch size 2;
- optimizer, learning rates, augmentation and cosine schedule;
- exactly four full epochs;
- frozen gold TTA `[-1,0,1]`;
- 5,000-replicate bootstrap evaluation;
- zero gold gradients and zero gold early stopping.

Changed in B12:

- every repaired series with anatomical plane Sagittal, Coronal or Axial is retained;
- no fluid/structural winner is selected;
- repeated acquisitions remain independent series;
- each series receives categorical plane/fluid/fat metadata embeddings;
- there is **no series-position/rank embedding**;
- studies are padded only to the largest series count in the current mini-batch;
- there is no architecture-level maximum series count.

The absence of a learned series-position embedding makes the study representation insensitive to arbitrary series ordering while still preserving within-series slice position.

## Frozen label-free series audit

The predeclared audit was run on the exact 3,120 B6-active non-gold studies and **passed every viability requirement**.

```text
studies                                  3120
series rows for studies                 17475
eligible recognized-plane series       17475
excluded unknown-plane series              0
historical dual unique series           15468
extra series retained                    2007
extra fraction vs historical          12.9752%
studies with extra series                1099
fraction studies with extra series     35.2244%
studies with zero eligible series           0
historical selected series missing          0
viability_passed                         true
```

Series-count distribution:

```text
min       3
mean      5.60096
q25       5
median    5
q75       6
q90       8
q95       9
q99      10
max      14
```

The audit therefore exceeds the frozen minimums of `5%` extra series and `10%` of studies gaining extra acquisitions by a wide margin.

Frozen variable-series mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

Training reconstructs the mapping and refuses to run if this signature or the eligible-series count drifts.

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
0.19.0
```

Run:

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b12_variable_series.py \
  tests/test_b7_weak_supervision.py
```

## Train B12

Use the already frozen successful audit; do **not** regenerate or alter the policy before the first gold evaluation.

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b12 \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b12_variable_series
```

Every complete epoch must retain the B7.1 supervision counters:

```text
batches                        1560
study_draws                    3120
active_supervision_cells_seen 14123
positive_cells_seen            6871
negative_cells_seen            7252
```

B12 additionally requires the frozen series surface:

```text
series_instances_seen          17475
expected_series_instances      17475
max_series_in_any_batch        14   # expected over a complete shuffled epoch
full_coverage                  true
full_series_coverage           true
budget_limited                 false
```

`series_instances_seen` counts successfully loaded real MRI series. If it is below `17475`, do not proceed to gold evaluation; investigate missing/unreadable DICOM series first.

Gold evaluation is blocked unless all four epochs satisfy both study and series coverage contracts.

## Frozen gold evaluation

After four complete epochs:

```bash
rsna-knee-b12-eval \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_variable_series/b12_model.pt \
  --out-root runs/b12_variable_series/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Then run the aligned 5,000-replicate paired bootstrap:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b7_1_full_coverage/gold_eval/gold_predictions.csv \
  --compare-oof runs/b12_variable_series/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b12_variable_series/gold_eval/b71_vs_b12.json
```

Do not create target-wise B7.1/B12 winners, series-count caps, routing variants or ensemble weights from the reused 58-study development set.
