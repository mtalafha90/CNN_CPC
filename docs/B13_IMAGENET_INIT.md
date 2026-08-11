# B13 — ImageNet encoder initialization protocol

> **Status — 2026-08-11:** **COMPLETED / RETAINED / DEVELOPMENT CHAMPION.** Package `0.22.0`.

## Scientific question

Does replacing the historical B5 competition-only encoder path with a standard publicly available ImageNet-pretrained ConvNeXt-Tiny protocol improve the hierarchical all-series model?

B13 uses:

```text
ConvNeXt-Tiny encoder <- torchvision IMAGENET1K_V1 weights
input normalization   <- standard ImageNet mean/std
```

The ImageNet weights and expected normalization are treated as one coherent encoder-initialization protocol. The repository does **not** describe this as a literal weight-only change because the historical B5 path used a different normalization policy.

## Competition-rule status

The competition rules supplied by the repository owner were checked before this experiment was finalized. Their External Data and Tools section permits external data and models when they are publicly/equally accessible or otherwise satisfy the competition reasonableness standard, unless specifically prohibited by the Host. No competition-specific prohibition on publicly available pretrained models was present in the supplied rules.

The conservative default remains `pretrained: false`; B13 opts in explicitly with:

```yaml
allow_external_pretrained: true
pretrained: true
```

## Experiment identity

```text
trainer     rsna-knee-b13
evaluator   rsna-knee-b13-eval
checkpoint  runs/b13_imagenet/b13_model.pt
variant     b13_imagenet_init_b6_hierarchical_series_token_v1
```

There is deliberately no B5 checkpoint argument in the B13 trainer.

## Frozen training surface

```text
training studies        3120
B6 supervised cells    14123
positive cells          6871
negative cells          7252
eligible MRI series    17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

Architecture/training controls:

```text
hierarchical learned series-token aggregation
16 sampled 2.5D positions per series
224x224 legacy MRI resize
plane/fluid/fat metadata embeddings
8-head per-series learned attention pool
2-layer study Transformer
pathology-query heads
batch size 2
encoder LR 1e-5
head LR 1e-4
same augmentation as B12/B12.1
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients
zero gold early stopping
```

## Completed training

```text
epoch 1 loss  0.7450505349
epoch 2 loss  0.6865059846
epoch 3 loss  0.6524747430
epoch 4 loss  0.6132239342
```

Every epoch certified:

```text
batches                         1560
study_draws                     3120
active_supervision_cells_seen  14123
positive_cells_seen             6871
negative_cells_seen             7252
series_instances_seen          17475
expected_series_instances      17475
max_series_in_any_batch           14
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

Checkpoint:

```text
runs/b13_imagenet/b13_model.pt
```

## Frozen gold development evaluation

```text
macro AUC       0.6293565948
95% CI         [0.5789896351,0.6775867717]
n studies       58
bootstrap       5000/5000 usable
```

Per-target AUC:

```text
ACL                0.4742647059
MCL                0.5555555556
Medial Meniscus    0.6093750000
Lateral Meniscus   0.6795031056
Medial OA          0.6279069767
Lateral OA         0.6189555126
PF OA              0.6177606178
Effusion           0.7677018634
Synovitis          0.7108721625
Baker's            0.7481884058
Contusion          0.5533063428
Fracture           0.5888888889
```

Target-level results are descriptive only and are not permission to build target-wise hybrids.

## Paired comparison versus B12

```text
median_difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
probability_b_better    0.9920
valid replicates        5000
```

## Paired comparison versus B7.1

```text
median_difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
probability_b_better    0.9808
valid replicates        5000
```

Both paired confidence intervals are entirely above zero.

## Retained decision

**B13 remains the development champion.** B12.1 is still skipped, so the project does not claim that the full B13 gain is caused solely by ImageNet initialization.

Development has now been reopened for one specific controlled hypothesis: B13's one-token-per-series compression may discard focal slice-level information.

## Active successor — B14

B14 keeps B13's encoder protocol and every training/evaluation control but changes aggregation:

```text
B13
16 slice tokens / real series -> one learned series token
K series tokens -> study Transformer -> pathology queries

B14
K real series x 16 slice tokens -> study Transformer -> pathology queries
```

B14 uses the already tested B12 full-token architecture, not a new target-specific architecture.

See [`B14_IMAGENET_FULL_TOKENS.md`](B14_IMAGENET_FULL_TOKENS.md).

The primary comparison is frozen as B14 versus B13 with the aligned 5,000-replicate paired bootstrap. Do not tune slice counts, epochs, LR, normalization or B13/B14 target-wise mixtures from that result.

## Interpretation policy

The 58 fully labelled studies have been reused throughout sequential development. B13's `0.6294` and any future B14 score are development/model-selection estimates, not independent validation and not leaderboard results.
