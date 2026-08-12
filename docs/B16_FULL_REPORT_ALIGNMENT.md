# B16 — full-report semantic representation alignment

> **Status — 2026-08-12:** IMPLEMENTED / PREDECLARED / NOT YET RUN. Package `0.25.0`.

## Why B16 follows B15

B15 decisively improved frozen weak-v2 B6-teacher agreement (`0.5652498118 -> 0.7319060415`) but did not improve the reused 58-study expert-gold macro AUC (`B13=0.6293565948`, `B15=0.6209002783`).

The subsequent B6/B15 gold diagnostic added two key observations:

```text
coverage-conditioned high-confidence B6 macro AUC  0.7736374158
full-surface four-state B6 ranking baseline          0.7024597743
```

The high-confidence B6-error-alignment audit did **not** show B15 moving toward B6 mistakes. On 55 B6-wrong high-confidence gold cells, B15 moved toward expert truth more often than toward B6. Therefore B16 is not motivated as an error-correction experiment.

The state audit instead showed that ignored report states contain information but are strongly pathology-dependent. Pooled expert-positive rates were:

```text
positive       116 / 168 = 0.6905
negated          3 / 83  = 0.0361
uncertain       11 / 29  = 0.3793
unmentioned    110 / 416 = 0.2644
```

Target-specific rates vary too much to justify a universal `uncertain -> p` or `unmentioned -> p` training target. B16 therefore consumes the **full report semantics directly during representation learning** rather than converting the middle report states into new pseudo-labels.

## Scientific question

```text
Does adding full-report semantic alignment to the completed B15 knee-MRI encoder,
then returning to the unchanged full-surface B13 hierarchy/B6 recipe,
improve global 12-target expert-gold ranking?
```

## Representation path

```text
torchvision ImageNet ConvNeXt-Tiny
        |
        v
completed B15 same-study knee-MRI SSL encoder
        |
        v
B16 full-report semantic alignment
        |
        v
MRI encoder only
        |
        v
B13 hierarchical one-token-per-series downstream model
```

## Report semantics

B16 deliberately reuses the established B5 competition-only text representation:

```text
full normalized report
-> word TF-IDF, 1-2 grams
-> max 20,000 features
-> min_df = 2
-> TruncatedSVD <= 256 dimensions
-> L2-normalized report vector
```

No external clinical language model is introduced in B16-v1. This keeps the experiment focused on whether **full report content** adds value on top of B15's MRI-domain encoder.

The report projection head is training-only and discarded after alignment; final downstream inference remains MRI-only.

## B16 report-alignment data contract

```text
competition studies          4407
gold studies excluded          58
report-alignment studies     4349
uses all non-gold reports    true
gold labels                  false
B6 labels in report stage    false
weak-v2 as selection gate    false
```

Unlike B15, the old 623-study weak-v2 split is not held out from B16 representation learning. Weak-v2 was invalidated as a surrogate selector for expert-gold improvement by the B15 experiment and is no longer a B16 model-selection surface.

Every eligible repaired real MRI series for the 4,349 non-gold studies is retained.

## Frozen report-alignment protocol

```text
B15 encoder checkpoint       runs/b15_mri_ssl/b15_ssl_encoder.pt
sampled positions/series     5
used positions/series        2
study batch                  2
report dimension             256
TF-IDF max features          20000
TF-IDF min_df                2
report queue                 256
encoder LR                   5e-5
report-head LR               2e-4
minimum LR                   1e-6
weight decay                 1e-4
report temperature           0.10
cosine weight                0.25
grad clip                    1.0
epochs                       4 full passes
```

Objective:

```text
loss = image->report contrastive NCE + 0.25 * cosine alignment
```

Duplicate normalized reports are masked as false negatives by the established B5 report-group logic.

## Frozen downstream contract

After report alignment, B16 returns to the **full B13 training surface**, not the B15 weak-v2 subset:

```text
B6-active studies        3120
usable B6 cells         14123
positive cells           6871
negative cells           7252
eligible real series    17475
batches/epoch            1560
epochs                      4
```

Architecture and optimization remain B13:

```text
hierarchical learned one-token-per-series aggregation
16 sampled 2.5D positions/series
224x224 resize
8-head series pooling
2-layer study Transformer
1 pathology-query layer
batch size 2
encoder LR 1e-5
head LR 1e-4
TTA [-1,0,1]
5000 bootstrap replicates
```

Frozen B6 policy remains unchanged:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

No state probabilities from the 58-study audit are inserted into B16 training.

## Run sequence

### 1. Pull/install/test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .

pytest -q \
  tests/test_b16_full_report.py \
  tests/test_b15_mri_ssl.py \
  tests/test_b6_b15_gold_diagnostic.py
```

### 2. B16 report alignment

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b16-report-ssl \
  --config configs/b16_full_report_alignment.yaml \
  --data-root "$DATA_ROOT" \
  --b15-ssl-checkpoint runs/b15_mri_ssl/b15_ssl_encoder.pt \
  --out-root runs/b16_full_report/report_ssl
```

Before downstream training inspect:

```bash
cat runs/b16_full_report/report_ssl/history.json
cat runs/b16_full_report/report_ssl/policy.json
```

Require four complete unbudgeted passes.

### 3. B16 downstream training

Use the same frozen B12/B13 series policy used by B13:

```bash
rsna-knee-b16 \
  --config configs/b16_full_report_alignment.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b16_full_report/downstream
```

Require every downstream epoch to report:

```text
study draws                  3120
active supervision cells    14123
positive cells               6871
negative cells               7252
series instances            17475
full coverage                true
full series coverage         true
budget limited               false
```

### 4. Single reused-gold development look

Only after the four downstream epochs satisfy the frozen contract:

```bash
rsna-knee-b16-gold-eval \
  --config configs/b16_full_report_alignment.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b16_full_report/downstream/b16_model.pt \
  --b13-predictions runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --out-root runs/b16_full_report/gold_confirmation
```

The evaluator performs B16 inference and the aligned paired B16-vs-B13 bootstrap in the same one-look run.

## Predeclared decision rule

Primary selection is **global 12-target macro ROC AUC**.

```text
historical champion B13 = 0.6293565948
```

B16 replaces B13 as the development champion only if the B16 global point estimate is higher. The paired bootstrap interval and `P(B16>B13)` quantify uncertainty but do not authorize target-wise mixing.

Regardless of result:

```text
no B16 epoch extension from gold
no report-loss tuning from gold
no target-specific B13/B16 winners
no post-gold queue/temperature/LR search
no weak-v2 gate
```

A failed B16 is still informative: it would indicate that simple TF-IDF/SVD full-report alignment is insufficient and would motivate a separately defined richer image-text representation experiment rather than retroactive B16 tuning.

## Outputs

Representation stage:

```text
runs/b16_full_report/report_ssl/
├── b16_report_encoder.pt
├── history.json
├── policy.json
├── report_vectorizer.joblib
├── report_svd.joblib
└── report_semantics.npz
```

Downstream:

```text
runs/b16_full_report/downstream/
├── b16_model.pt
├── history.json
├── policy.json
└── supervision_plan.json
```

Gold development confirmation:

```text
runs/b16_full_report/gold_confirmation/
├── gold_predictions.csv
└── eval.json
```
