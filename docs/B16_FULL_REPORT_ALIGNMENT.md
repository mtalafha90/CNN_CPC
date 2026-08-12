# B16 — full-report semantic representation alignment

> **Status — 2026-08-12:** REPORT-ALIGNMENT STAGE COMPLETED SUCCESSFULLY / DOWNSTREAM NOT YET RUN. Package `0.25.0`.

## Why B16 follows B15

B15 decisively improved frozen weak-v2 B6-teacher agreement (`0.5652498118 -> 0.7319060415`) but did not improve the reused 58-study expert-gold macro AUC (`B13=0.6293565948`, `B15=0.6209002783`).

The post-B15 B6/B15 gold diagnostic found:

```text
coverage-conditioned high-confidence B6 macro AUC  0.7736374158
full-surface four-state B6 ranking baseline          0.7024597743
```

The B6-error alignment audit did not show B15 moving toward B6 mistakes. The state audit instead showed that ignored report states contain information but are strongly pathology-dependent, so B16 uses full report semantics directly rather than assigning new hard or soft targets to `uncertain` or `unmentioned` states.

## Scientific question

```text
Does adding full-report semantic alignment to the completed B15 knee-MRI encoder,
then returning to the unchanged full-surface B13 hierarchy/B6 recipe,
improve global 12-target expert-gold ranking?
```

## Representation path

```text
ImageNet ConvNeXt-Tiny
        -> completed B15 same-study knee-MRI SSL encoder
        -> B16 full-report semantic alignment
        -> MRI encoder only
        -> B13 hierarchical one-token-per-series downstream model
```

## Report semantics

B16 reuses the established B5 competition-only text representation:

```text
full normalized report
-> word TF-IDF, 1-2 grams
-> max 20,000 features
-> min_df = 2
-> TruncatedSVD <= 256 dimensions
-> L2-normalized report vector
```

No external clinical language model is introduced in B16-v1. The report projection head is training-only; final inference remains MRI-only.

## Report-alignment data contract

```text
competition studies          4407
gold studies excluded          58
report-alignment studies     4349
eligible real MRI series    24035
2.5D examples / epoch       48070
uses all non-gold reports    true
gold labels                  false
B6 labels in report stage    false
weak-v2 as selection gate    false
```

Unlike B15, the old 623-study weak-v2 split is not held out from B16 representation learning because weak-v2 is no longer used as a surrogate selector for expert-gold improvement.

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

Duplicate normalized reports are masked as false negatives using the established B5 report-group logic.

## Completed representation stage

All four epochs completed the exact full-coverage contract:

| Epoch | Total loss | Report NCE | Cosine loss | Batches | Studies | Series | 2.5D examples | Full coverage | Budget limited |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `3.8958491301` | `3.7314807830` | `0.6574733884` | 2175 | 4349 | 24035 | 48070 | true | false |
| 2 | `3.1331863229` | `3.0068265086` | `0.5054392496` | 2175 | 4349 | 24035 | 48070 | true | false |
| 3 | `2.7944439663` | `2.6814318427` | `0.4520484886` | 2175 | 4349 | 24035 | 48070 | true | false |
| 4 | `2.5218941658` | `2.4161238326` | `0.4230813365` | 2175 | 4349 | 24035 | 48070 | true | false |

Observed optimization change from epoch 1 to epoch 4:

```text
total loss    -35.27%
report NCE    -35.25%
cosine loss   -35.65%
```

Final report-alignment checkpoint:

```text
runs/b16_full_report/report_ssl/b16_report_encoder.pt
```

The representation stage is therefore accepted as complete. Loss improvement is an optimization diagnostic only; it is not a model-selection metric and does not establish expert-label improvement.

## Frozen downstream contract

B16 now returns to the full B13 training surface, not the B15 weak-v2 subset:

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

No state probabilities from the reused 58-study audit enter B16 training.

## Next command — downstream training

Use the exact frozen B12/B13 series policy:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b16 \
  --config configs/b16_full_report_alignment.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b16_full_report/downstream
```

Every downstream epoch must report:

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

Do not run the gold evaluator unless all four downstream epochs satisfy this exact contract.

## Single reused-gold development look

Only after four exact downstream epochs:

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

Primary selection is global 12-target macro ROC AUC. Historical champion:

```text
B13 = 0.6293565948
```

B16 replaces B13 only if the global B16 point estimate is higher. The paired bootstrap quantifies uncertainty but does not authorize target-wise mixing.

Regardless of result:

```text
no B16 epoch extension from gold
no report-loss tuning from gold
no target-specific B13/B16 winners
no post-gold queue/temperature/LR search
no weak-v2 gate
```
