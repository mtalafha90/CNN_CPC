# B12.1 — hierarchical learned series-token aggregation

> **Status — 2026-08-11:** IMPLEMENTED / PREDECLARED / **SKIPPED FOR THE COMPETITION WORKFLOW**. Package `0.21.0`.

## Original purpose

B12.1 was designed as the clean hierarchical control between B12 and B13:

```text
B12
K real series x 16 slice tokens -> study Transformer -> pathology queries

B12.1
16 slice tokens -> learned 8-head per-series attention pool -> 1 series token
K series tokens -> same 2-layer study Transformer -> same pathology queries
encoder initialization -> B5 competition-only SSL
```

Its exact implementation remains in the repository and is reproducible.

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

B12.1 is explicitly competition-only and requires the B5 encoder checkpoint. Its trainer rejects external pretrained flags; ImageNet belongs to the separate B13 experiment.

## Why it is now skipped

B13 completed before B12.1 and produced a substantially stronger development result:

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
```

Paired versus B12:

```text
median(B13-B12)    +0.0638674720
95% paired CI      [+0.0127183837,+0.1144643292]
P(B13 > B12)        0.9920
```

Paired versus B7.1:

```text
median(B13-B7.1)   +0.0652260946
95% paired CI      [+0.0039768779,+0.1266069220]
P(B13 > B7.1)       0.9808
```

Both paired confidence intervals are above zero on the repeatedly reused 58-study development set.

The competition workflow therefore prioritizes freezing B13-v1 and obtaining an independent Kaggle signal rather than spending another full training/evaluation cycle only to complete this ablation.

## Scientific consequence of skipping B12.1

The clean comparison:

```text
B12.1 = hierarchical architecture + B5 initialization
B13   = hierarchical architecture + ImageNet encoder protocol
```

will not be available.

Therefore the project must **not** claim that the entire B13 improvement is caused solely by ImageNet initialization. Relative to B12, B13 differs in both hierarchical aggregation and encoder protocol.

This limitation is explicit and intentional.

## Reproduction commands — archived, not current next step

B12.1 can still be reproduced later if a scientific ablation is required:

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

Frozen evaluation command:

```bash
rsna-knee-b12-1-eval \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_1_hierarchical/b12_1_model.pt \
  --out-root runs/b12_1_hierarchical/gold_eval
```

These commands are retained for reproducibility, **not as part of the current competition roadmap**.

## Current successor

B13 is now the retained development champion. See [`B13_IMAGENET_INIT.md`](B13_IMAGENET_INIT.md).

Current competition path:

```text
B13-v1 RETAIN
     |
     v
freeze model / preprocessing / series policy / TTA
     |
     v
Kaggle test inference and submission
     |
     v
use leaderboard as the next independent signal
```

Full updated roadmap: [`ROADMAP_AFTER_B12_1.md`](ROADMAP_AFTER_B12_1.md).
