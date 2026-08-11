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

## Controlled successor result — B14

B14 tested the hypothesis that B13's one-token-per-series compression discarded useful focal slice-level information. B14 retained the full `K x 16` slice-token memory while keeping B13's ImageNet protocol and training recipe.

```text
B14 macro AUC       0.6197914249
95% CI             [0.5706800512,0.6693542716]
raw B14-B13        -0.0095651699
paired median      -0.0093726931
95% paired CI      [-0.0469823411,+0.0250137870]
P(B14 > B13)        0.2924
```

The paired CI crosses zero, but B14 has the lower global point estimate, lower probability of superiority, higher token-memory cost and slower training. B14 is therefore rejected globally and B13 remains the retained development champion.

B14 also achieved a lower final B6 training loss (`0.5822778610`) than B13 (`0.6132239342`) without improving macro AUC. This is evidence against simply increasing downstream capacity or fitting the weak labels harder.

See [`B14_IMAGENET_FULL_TOKENS.md`](B14_IMAGENET_FULL_TOKENS.md).

## Retained decision

```text
B13 RETAIN / DEVELOPMENT CHAMPION
B14 REJECT GLOBALLY
```

B12.1 is still skipped, so the project does not claim that the full B13 gain is caused solely by ImageNet initialization.

## Next representation hypothesis

The next major controlled hypothesis is B15:

```text
ImageNet ConvNeXt-Tiny
        -> competition knee-MRI self-supervised adaptation
        -> B13 hierarchical aggregation
        -> frozen B6 downstream recipe
```

All 58 gold studies must be excluded from B15 SSL optimization. No gold labels may enter gradients, early stopping or checkpoint selection.

## Interpretation policy

The 58 fully labelled studies have been reused throughout sequential development. B13's `0.6294` and B14's `0.6198` are development/model-selection estimates, not independent validation and not leaderboard results. A Kaggle hidden-test/leaderboard result remains the next genuinely independent performance signal.
