# B12.1 — hierarchical learned series-token aggregation

> **Status — 2026-08-12:** **IMPLEMENTED / SKIPPED FOR THE COMPETITION WORKFLOW.** Its architecture became the basis of B13 and B15 downstream modeling, but the clean B5-initialized B12.1 control itself was never trained.

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
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

B12.1 is explicitly competition-only and requires the B5 encoder checkpoint. ImageNet belongs to the separate B13+ experiments.

## Why it was skipped

B13 completed first and produced a much stronger reused-gold development result:

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

The competition workflow did not spend another full run solely to complete this causal ablation.

## Scientific consequence

The clean comparison

```text
B12.1 = hierarchy + B5 initialization
B13   = hierarchy + ImageNet encoder protocol
```

is unavailable. Therefore the project does not claim that the entire B13 improvement is caused solely by ImageNet initialization.

## Successor results through B15

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249  rejected globally
B15 gold  0.6209002783  no global improvement
```

B15 reused the hierarchical one-token-per-series downstream architecture after knee-MRI same-study contrastive adaptation. On frozen weak-v2 it improved from matched-control `0.5652498118` to `0.7319060415`, with paired median `+0.1675245839` and 95% CI `[+0.1124433208,+0.2165156305]`, but the gain did not transfer to global expert-gold AUC.

This makes the B12.1 hierarchy an important architectural ancestor, while the immediate scientific priority shifts toward auditing weak-supervision states rather than training the skipped B12.1 ablation.

## Reproduction commands — archived

```bash
rsna-knee-b12-1 \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b12_1_hierarchical

rsna-knee-b12-1-eval \
  --config configs/b12_1_hierarchical.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b12_1_hierarchical/b12_1_model.pt \
  --out-root runs/b12_1_hierarchical/gold_eval
```

These commands remain for reproducibility, not the active roadmap.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).