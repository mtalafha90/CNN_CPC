# B12 — variable-number-of-series MRI model

> **Status — 2026-08-11:** IMPLEMENTED / PREDECLARED / LABEL-FREE SERIES AUDIT PENDING.

B7.1 remains the retained development champion at macro AUC `0.5644802945`.
B11.1 completed its frozen four-epoch contract but scored `0.5506902702`; the paired B11.1-B7.1 median difference was `-0.0126224565` with 95% CI `[-0.0487500119,+0.0195120537]` and `P(B11.1>B7.1)=0.2184`. B11.1 is therefore rejected globally.

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

## Label-free series audit

Before training, B12 audits only the 3,120 B6-active non-gold studies. It compares the new all-series surface with the historical B7.1 dual routing and freezes a SHA-256 signature of the exact variable-series mapping.

The predeclared viability requirements are:

```text
zero active studies with zero eligible series
zero historical selected series missing from B12
extra eligible series >= 5% of historical unique selected-series count
>= 10% of active studies contain at least one extra retained series
```

The audit also reports:

```text
eligible recognized-plane series
excluded unknown-plane series
historical dual unique series
extra series retained
studies with extra series
series/study min, mean, median, q90, q95, q99, max
series mapping SHA-256
```

Do not train B12-v1 unless `viability_passed = true`.

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

## Step 1 — run the B12 series audit

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b12-audit \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b12_variable_series/audit
```

Inspect:

```bash
cat runs/b12_variable_series/audit/series_audit.json
cat runs/b12_variable_series/audit/series_policy.json
```

Do not train until the audit is inspected and `viability_passed` is true.

## Step 2 — training, only after audit pass

```bash
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
full_coverage                  true
budget_limited                 false
```

B12 additionally requires:

```text
series_instances_seen == expected_series_instances
full_series_coverage == true
```

Gold evaluation is blocked unless all four epochs satisfy both coverage checks.

## Step 3 — frozen gold evaluation

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

Then run the same aligned 5,000-replicate paired bootstrap. Do not create target-wise B7.1/B12 winners from the reused 58-study development set.
