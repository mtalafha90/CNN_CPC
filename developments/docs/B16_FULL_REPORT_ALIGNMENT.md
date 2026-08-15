# B16 — full-report semantic representation alignment

> **Status — 2026-08-12:** COMPLETED / RETAINED BY PREDECLARED GLOBAL POINT-ESTIMATE RULE. Package `0.25.0`. B16 reused-gold macro AUC is `0.6349770242`; the gain over B13 is small and statistically unresolved.

## Why B16 followed B15

B15 decisively improved frozen weak-v2 B6-teacher agreement (`0.5652498118 -> 0.7319060415`) but did not improve the reused 58-study expert-gold macro AUC (`B13=0.6293565948`, `B15=0.6209002783`).

The post-B15 B6/B15 diagnostic found:

```text
coverage-conditioned high-confidence B6 macro AUC  0.7736374158
full-surface four-state B6 ranking baseline          0.7024597743
```

The B6-error alignment audit did not show B15 moving toward B6 mistakes. The state audit instead showed that ignored report states contain information but are strongly pathology-dependent, so B16 used full report semantics directly rather than assigning new hard or soft targets to `uncertain` or `unmentioned` states.

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

No external clinical language model was introduced in B16-v1. The report projection head was training-only; final inference remains MRI-only.

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

Unlike B15, the old 623-study weak-v2 split was not held out from B16 representation learning because weak-v2 is no longer used as a surrogate selector for expert-gold improvement.

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

Duplicate normalized reports were masked as false negatives using the established B5 report-group logic.

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

Final representation checkpoint:

```text
runs/b16_full_report/report_ssl/b16_report_encoder.pt
```

The representation stage satisfied the frozen protocol. Its loss reduction is an optimization diagnostic only and was not used for model selection.

## Frozen downstream contract

B16 returned to the full B13 training surface, not the B15 weak-v2 subset:

```text
B6-active studies        3120
usable B6 cells         14123
positive cells           6871
negative cells           7252
eligible real series    17475
batches/epoch            1560
epochs                      4
```

Architecture and optimization remained B13:

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

Frozen B6 policy remained unchanged:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

No state probabilities from the reused 58-study audit entered B16 training.

## Completed downstream stage

All four downstream epochs completed the exact B13 full-surface study/series contract:

| Epoch | B6 training loss | Batches | Studies | Active cells | Positive | Negative | Series | Full coverage | Full series | Budget limited |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | `0.7379701049` | 1560 | 3120 | 14123 | 6871 | 7252 | 17475 | true | true | false |
| 2 | `0.6212521367` | 1560 | 3120 | 14123 | 6871 | 7252 | 17475 | true | true | false |
| 3 | `0.5901195104` | 1560 | 3120 | 14123 | 6871 | 7252 | 17475 | true | true | false |
| 4 | `0.5675074643` | 1560 | 3120 | 14123 | 6871 | 7252 | 17475 | true | true | false |

Final downstream checkpoint:

```text
runs/b16_full_report/downstream/b16_model.pt
```

The B6 training loss decreased by about 23.1% from epoch 1 to epoch 4. Training loss was not used as a model-selection metric.

## Single reused-gold development result

B16 completed the one frozen reused-gold look with all 12 target AUCs defined and all 5000 bootstrap replicates usable:

```text
B16 macro AUC        0.6349770242
95% CI              [0.5854729266, 0.6830266155]
B13 macro AUC        0.6293565948
raw B16-B13          +0.0056204295
paired median        +0.0050711608
95% paired CI        [-0.0395927864, +0.0519351407]
P(B16 > B13)          0.5828
n                     58 studies
bootstrap             5000 / 5000 usable
```

Per-target B16 AUCs, descriptive only:

```text
ACL                0.5012254902
MCL                0.4580498866
Medial Meniscus    0.6658653846
Lateral Meniscus   0.6683229814
Medial OA          0.6945736434
Lateral OA         0.5841392650
PF OA              0.6370656371
Effusion           0.8360248447
Synovitis          0.7514934289
Baker's            0.6757246377
Contusion          0.5708502024
Fracture           0.5763888889
```

Outputs:

```text
runs/b16_full_report/gold_confirmation/gold_predictions.csv
runs/b16_full_report/gold_confirmation/eval.json
```

The Transformer nested-tensor warnings emitted during evaluation are benign PyTorch warnings associated with `norm_first=True`; they did not interrupt evaluation.

## Decision

The predeclared rule stated that B16 replaces B13 as the reused-gold development champion if its **global macro-AUC point estimate is higher**. B16 therefore satisfies that rule:

```text
B16  0.6349770242
B13  0.6293565948
      -----------
delta +0.0056204295
```

Accordingly:

```text
B16 RETAIN by predeclared point-estimate rule
B16 = current reused-gold development champion
B13 = retained historical reference / statistically unresolved with B16
```

However, the paired interval crosses zero widely and `P(B16>B13)=0.5828`. Therefore the evidence does **not** establish that B16 is truly superior to B13. The scientifically correct interpretation is that B16 and B13 remain statistically unresolved on the repeatedly reused 58-study development set, with B16 having the slightly higher frozen point estimate.

## What B16 establishes

B16 provides modest support for the hypothesis that full-report semantic alignment can add expert-label ranking information beyond B15's MRI-domain SSL when the model is returned to the full B13 downstream surface. It does **not** establish a large effect, and it does not demonstrate that TF-IDF/SVD report alignment is the final representation solution.

The contrast between B15 and B16 is informative:

```text
B15: stronger MRI-domain SSL + reduced 2497-study downstream surface
      -> 0.6209002783 gold

B16: B15 encoder + full-report semantic alignment + restored 3120-study B13 surface
      -> 0.6349770242 gold
```

Because B16 changed both the representation sequence and restored the full downstream B6 surface relative to B15, the B16-vs-B15 difference must not be attributed solely to report alignment.

## Governance after the gold look

The frozen experiment is now closed. Do not perform any of the following based on this result:

```text
no B16 epoch extension
no report-loss / temperature / queue / LR tuning
no target-specific B13/B16 winner mixing
no per-target report-alignment tuning
no weak-v2 gate regeneration
no new soft labels derived from the 58-study state audit
```

The 58-study gold surface has now been reused many times and cannot provide clean confirmation of further small improvements. The next genuinely independent performance signal remains the hidden Kaggle evaluation. Any B17 experiment should be separately versioned and motivated before another development-set look rather than tuned as a continuation of B16.
