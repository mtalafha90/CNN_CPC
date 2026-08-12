# B17 — frozen B16 report-aligned encoder

> **Status — 2026-08-12:** IMPLEMENTED / PREDECLARED / NOT YET RUN. Package `0.26.0`.

## Motivation

B16 is the current reused-gold development champion by the predeclared global
point-estimate rule:

```text
B16 macro AUC        0.6349770242
B13 macro AUC        0.6293565948
raw B16-B13         +0.0056204295
paired 95% CI       [-0.0395927864,+0.0519351407]
P(B16>B13)           0.5828
```

The post-B15 diagnostic also showed that a coarse B6 report-state ranking
baseline reaches `0.7024597743` on the repeatedly reused gold surface. This is
not an MRI-student ceiling, but it establishes that the report-derived
supervision contains substantial expert-ordering information beyond the current
MRI-only model output.

B17 tests one specific training hypothesis: useful ImageNet -> knee-MRI SSL ->
full-report-aligned encoder features may be partially degraded when the encoder
is subsequently updated by noisy sparse B6 gradients.

## Scientific question

```text
Does freezing the completed B16 report-aligned ConvNeXt encoder during short,
fixed B6 downstream training preserve expert-relevant MRI representation better
than B16 end-to-end fine-tuning?
```

## B17 representation path

```text
ImageNet ConvNeXt-Tiny
        -> B15 same-study knee-MRI SSL
        -> B16 full-report semantic alignment
        -> FROZEN MRI encoder
        -> B13/B16 hierarchical one-token-per-series model
        -> B6 positive/negated supervision
```

The report branch is still training-only. Test-time inference remains MRI-only.

## Frozen B17-v1 contract

### Encoder

```text
source checkpoint
runs/b16_full_report/report_ssl/b16_report_encoder.pt

requires_grad                     false for every encoder parameter
optimizer membership              false
encoder training mode             false
runtime encoder checkpointing     false
encoder LR                        0
```

B17 records a deterministic SHA-256 fingerprint of the encoder parameters and
buffers before training and after every epoch. Training fails immediately if the
fingerprint changes.

### Downstream surface

```text
B6-active studies          3120
usable B6 cells           14123
positive cells             6871
negative cells             7252
eligible real series      17475
batches / epoch            1560
max series / study           14
```

The same frozen B12/B13 all-series mapping is required:

```text
runs/b12_variable_series/audit/series_policy.json

SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

### B6 supervision

Unchanged from B7.1/B12/B13/B16:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

No gold-derived state probabilities are used.

### Architecture / augmentation

Unchanged B13/B16 hierarchy:

```text
16 sampled 2.5D positions / series
224 x 224
hierarchical learned one-token-per-series pooling
8-head series pooling
2-layer study Transformer
1 pathology-query layer
dropout 0.25
batch size 2
TTA [-1,0,1]
```

MRI augmentation is unchanged from B16.

### Optimization

```text
encoder LR              0
head LR                  1e-4
minimum LR               1e-6
weight decay             1e-4
grad clip                1.0
epochs                    5 exact full passes
```

The B17-v1 head/hierarchy construction and DataLoader reuse the B16 seed offsets
so the non-encoder initialization and first four shuffle streams follow the same
seed path as B16.

B17 deliberately does **not** add another training intervention:

```text
additional label smoothing   0
ELR / SCE / robust loss      none
gold early stopping          none
gold checkpoint selection    none
weak-v2 gate                 none
```

A later robust-loss or label-smoothing experiment must be separately versioned.

## Important interpretation caveat

B17 changes both encoder optimization (`fine-tuned -> frozen`) and the fixed
training length (`4 -> 5` epochs) relative to B16. It is therefore a frozen,
short-training protocol test rather than a mathematically pure one-variable
freezing ablation. No inference should attribute any eventual B17-B16
performance difference solely to freezing without acknowledging this.

## Run sequence

### 1. Pull / install / test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"

python -m compileall -q src tests
pytest -q \
  tests/test_b17_frozen_encoder.py \
  tests/test_b16_full_report.py \
  tests/test_b6_b15_gold_diagnostic.py
```

Expected package version:

```text
0.26.0
```

### 2. Train B17

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b17 \
  --config configs/b17_frozen_encoder.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b17_frozen_encoder
```

Every epoch must report exactly:

```text
batches                         1560 / 1560
study draws                     3120 / 3120
active supervision cells       14123 / 14123
positive cells                  6871 / 6871
negative cells                  7252 / 7252
series instances               17475 / 17475
encoder_lr                      0
encoder_frozen                  true
encoder_training_mode           false
encoder_gradients_detected      false
encoder_sha256                  identical every epoch
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

Do not run gold evaluation unless all five epochs satisfy the complete contract.

Expected checkpoint:

```text
runs/b17_frozen_encoder/b17_model.pt
```

### 3. Single reused-gold development look

Only after five exact frozen-encoder passes:

```bash
rsna-knee-b17-gold-eval \
  --config configs/b17_frozen_encoder.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b17_frozen_encoder/b17_model.pt \
  --b16-predictions runs/b16_full_report/gold_confirmation/gold_predictions.csv \
  --out-root runs/b17_frozen_encoder/gold_confirmation
```

The evaluator verifies that the supplied B16 prediction file reproduces the
frozen champion macro AUC `0.6349770242`, then performs one aligned 5000-replicate
paired B17-vs-B16 bootstrap.

## Predeclared decision rule

Primary metric remains global 12-target macro ROC AUC.

```text
current reused-gold champion B16 = 0.6349770242
```

B17 replaces B16 only if B17's global point estimate is higher. The paired
bootstrap quantifies uncertainty but does not authorize target-wise mixing.

Regardless of the result:

```text
no epoch-6 extension based on gold
no label-smoothing tuning from gold
no ELR/SCE choice from gold
no target-specific B16/B17 winner mixing
no head-LR tuning from gold
no regeneration of weak-v2
```

The 58-study gold surface remains a repeatedly reused development set, not
independent validation. The hidden competition evaluation remains the next truly
independent performance signal.
