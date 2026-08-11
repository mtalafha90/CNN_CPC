# B13 — ImageNet encoder initialization protocol

> **Status — 2026-08-11:** **COMPLETED / RETAINED / NEW DEVELOPMENT CHAMPION.** Package `0.21.0`.

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

B13 is a separate first-class experiment:

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

The B13 contract rejects accidental changes to optimizer, architecture, epoch count, augmentation, series mapping, TTA or bootstrap settings.

## Shared initialization control

To keep non-encoder random initialization controlled, B13 constructs the complete hierarchical architecture from the frozen seed before replacing only the encoder state with torchvision ImageNet weights. This prevents pretrained-weight construction from shifting RNG draws used by the study Transformer, pathology tokens, target heads or learned series-pooling module.

## Completed training

B13 completed all four frozen epochs with exact full study and series coverage:

```text
epoch 1
loss                0.7450505349
epoch seconds       2529.2077
encoder LR          8.6819805e-06
head LR             8.5501786e-05

epoch 2
loss                0.6865059846
epoch seconds       2658.0064
encoder LR          5.5e-06
head LR             5.05e-05

epoch 3
loss                0.6524747430
epoch seconds       2903.9735
encoder LR          2.3180195e-06
head LR             1.5498214e-05

epoch 4
loss                0.6132239342
epoch seconds       2135.4708
encoder LR          1e-06
head LR             1e-06
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

All 12 target AUCs are defined. Target-level results are descriptive only and are not permission to build target-wise hybrids.

Outputs:

```text
runs/b13_imagenet/gold_eval/gold_predictions.csv
runs/b13_imagenet/gold_eval/eval.json
```

## Paired comparison versus B12

Baseline B12 macro AUC: `0.5660915179`.

Aligned 5,000-replicate paired bootstrap, sign convention `B13-B12`:

```text
median_difference      +0.0638674720
95% paired CI          [+0.0127183837,+0.1144643292]
probability_b_better    0.9920
valid replicates        5000
```

The paired confidence interval is entirely above zero.

Output:

```text
runs/b13_imagenet/gold_eval/b12_vs_b13.json
```

## Paired comparison versus B7.1

Baseline B7.1 macro AUC: `0.5644802945`.

Aligned 5,000-replicate paired bootstrap, sign convention `B13-B7.1`:

```text
median_difference      +0.0652260946
95% paired CI          [+0.0039768779,+0.1266069220]
probability_b_better    0.9808
valid replicates        5000
```

The paired confidence interval is also entirely above zero.

Output:

```text
runs/b13_imagenet/gold_eval/b71_vs_b13.json
```

## Decision

**B13 is retained as the new development champion.** It is the first model in the current ladder with paired confidence intervals entirely above zero versus both B12 and B7.1.

For the competition path:

```text
B13-v1 RETAIN
B12.1 SKIP
B12.2 DEFER
no target-wise hybrids
no ImageNet/normalization/LR/epoch sweeps on gold
freeze B13-v1
prepare Kaggle submission
```

## Important causal limitation

B12.1 was implemented but not trained/evaluated. Therefore the clean parent comparison:

```text
B12.1 hierarchical + B5 init
versus
B13 hierarchical + ImageNet protocol
```

is unavailable.

Accordingly, the project does **not** claim that the full B13 gain is caused solely by ImageNet initialization. Relative to B12, B13 changes both hierarchical aggregation and encoder protocol. Relative to B7.1, additional representation differences exist.

Skipping B12.1 is an explicit competition-workflow decision to preserve development budget and reduce further sequential reuse of the same 58 labelled studies.

## Interpretation policy

The 58 fully labelled studies have been reused throughout sequential development. Therefore B13's `0.6294` remains a development/model-selection estimate, not independent validation and not a leaderboard result.

The next high-value signal should come from actual competition test inference/submission. Further local experiments should be reopened only if independent leaderboard evidence or a clear technical diagnostic justifies them.
