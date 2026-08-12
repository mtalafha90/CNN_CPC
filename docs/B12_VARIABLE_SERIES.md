# B12 — variable-number-of-series MRI model

> **Status — 2026-08-12:** **COMPLETED / RETAINED HISTORICAL REFERENCE.** B12 macro AUC is `0.5660915179`; it was statistically tied with B7.1 and has since been surpassed by B13. B15 completed afterward and did not replace B13.

## Scientific change

B7.1 maps each study into six fixed semantic slots. B12 instead retains every repaired Sagittal/Coronal/Axial MRI acquisition as a separate real series, with plane/fluid/fat metadata embeddings and no series-rank embedding. B5 initialization, B6 supervision, preprocessing, optimizer, augmentation, four-epoch schedule, TTA and bootstrap remain B7.1-equivalent.

## Frozen label-free series audit

```text
studies                                  3120
eligible recognized-plane series       17475
historical dual unique series           15468
extra series retained                    2007
extra fraction vs historical          12.9752%
studies with extra series                1099
fraction studies with extra series     35.2244%
studies with zero eligible series           0
series/study min / median / max       3 / 5 / 14
viability_passed                         true
```

Frozen mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Training integrity

All four epochs completed the exact contract:

```text
batches                         1560
study_draws                     3120
active supervision cells       14123
positive / negative          6871 / 7252
series_instances_seen          17475
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

Losses:

```text
0.7349378360
0.6693112939
0.6405184795
0.6084634456
```

## Reused-gold result

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761,0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

Decision at the time: retain B12 as a viable all-series direction but do not claim superiority over B7.1.

Per-target B12 values are descriptive only and must not be used for target-wise hybrids.

## Successor experiments

B12.1 introduced hierarchical one-token-per-series aggregation with B5 initialization; it was implemented but skipped.

B13 used the same hierarchical family with the ImageNet ConvNeXt protocol and reached:

```text
B13 gold macro AUC  0.6293565948
```

It remains the development champion.

B14 returned to full `K x 16` slice-token memory with the B13 ImageNet protocol and scored `0.6197914249`, so it was rejected globally.

B15 then tested ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy. It passed frozen weak-v2 strongly (`0.7319060415` vs control `0.5652498118`, paired median `+0.1675245839`) but reached only `0.6209002783` on the single reused-gold confirmation. Thus B13 remains retained.

## Current interpretation

B12 established that using every real MRI series is viable and supplied the all-series mapping inherited by B13-B15. The B15 weak/gold divergence now makes supervision-state quality a higher-priority diagnostic than another all-series aggregation sweep.

## Reproduction

```bash
rsna-knee-b12 \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b12_variable_series

rsna-knee-b12-eval \
  --config configs/b12_variable_series.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_variable_series/b12_model.pt \
  --out-root runs/b12_variable_series/gold_eval
```

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B15 record: [`B15_MRI_SSL.md`](B15_MRI_SSL.md).