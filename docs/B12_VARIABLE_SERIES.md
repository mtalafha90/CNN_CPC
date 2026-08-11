# B12 — variable-number-of-series MRI model

> **Status — 2026-08-11:** **COMPLETED / RETAINED HISTORICAL REFERENCE.** B12 macro AUC is `0.5660915179`; it was statistically tied with B7.1 and has since been surpassed by B13.

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
historical selected series missing          0
series/study min / median / max       3 / 5 / 14
q90 / q95 / q99                      8 / 9 / 10
viability_passed                         true
```

Frozen variable-series mapping SHA-256:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Training integrity

All four epochs completed the exact frozen contract:

```text
batches                         1560
each epoch study_draws          3120
active supervision cells       14123
positive / negative          6871 / 7252
series_instances_seen          17475
expected_series_instances      17475
max_series_in_any_batch           14
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

Losses:

```text
epoch 1  0.7349378360
epoch 2  0.6693112939
epoch 3  0.6405184795
epoch 4  0.6084634456
```

## Frozen development result

```text
B12 macro AUC         0.5660915179
95% CI               [0.5094993761, 0.6244034568]
B7.1 macro AUC        0.5644802945
median(B12-B7.1)     +0.0023747526
95% paired CI        [-0.0472104067,+0.0481427722]
P(B12 > B7.1)         0.5376
```

Decision at the time: retain B12 as a viable all-series direction but do not claim superiority over B7.1.

Per-target B12 AUCs:

```text
ACL                0.4791666667
MCL                0.5147392290
Medial Meniscus    0.6574519231
Lateral Meniscus   0.6298136646
Medial OA          0.4031007752
Lateral OA         0.4990328820
PF OA              0.6151866152
Effusion           0.6658385093
Synovitis          0.6654719235
Baker's            0.5163043478
Contusion          0.5317139001
Fracture           0.6152777778
```

These target-level values are descriptive only. They must not be used to construct target-wise hybrids.

## Reproduction

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

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

## Successor experiments

B12.1 introduced one learned token per real series with B5 initialization; it remains implemented but was skipped for the competition workflow.

B13 combined that hierarchical aggregation with the ImageNet ConvNeXt protocol and reached macro AUC `0.6293565948`, becoming the retained development champion. See [`B13_IMAGENET_INIT.md`](B13_IMAGENET_INIT.md).

B14 now returns to the **B12 full `K x 16` slice-token architecture** while retaining B13's ImageNet encoder protocol. This makes B14 a controlled test of whether B13's one-token-per-series compression is discarding useful slice-level information. See [`B14_IMAGENET_FULL_TOKENS.md`](B14_IMAGENET_FULL_TOKENS.md).

Stronger in-domain SSL, if later justified, is now reserved for B15 rather than B14.

Full roadmap: [`ROADMAP_AFTER_B12_1.md`](ROADMAP_AFTER_B12_1.md).
