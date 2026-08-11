# B14 — ImageNet full slice-token aggregation

> **Status — 2026-08-11:** **COMPLETED / REJECTED GLOBALLY.** Package `0.22.0`. B13 remains the retained development champion.

## Scientific question

B13 compresses each real MRI series from 16 encoded slice tokens to one generic learned series token before the study Transformer. B14 tested whether retaining the full slice-token memory would improve global macro ROC AUC.

## Single scientific change versus B13

```text
B13
16 slice tokens / real series
    -> learned 8-head attention pool
    -> 1 token / real series
K series tokens
    -> 2-layer study Transformer
    -> pathology-query cross-attention

B14
16 slice tokens / real series
    -> NO series compression
K x 16 slice tokens
    -> same 2-layer study Transformer
    -> same pathology-query cross-attention
```

B14 reuses the already implemented B12 full-slice-token architecture and the exact B13 ImageNet encoder protocol.

## Frozen controls

```text
ImageNet encoder protocol
  torchvision ConvNeXt-Tiny IMAGENET1K_V1
  standard ImageNet mean/std normalization

training studies        3120
B6 supervised cells    14123
positive cells          6871
negative cells          7252
eligible MRI series    17475
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376

16 sampled 2.5D positions / series
224x224 legacy MRI resize
plane/fluid/fat metadata embeddings
batch size 2
ConvNeXt encoder LR 1e-5
head LR 1e-4
weight decay 1e-4
same augmentation
same seed and DataLoader seed offsets
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients
zero gold early stopping
```

## Experiment identity

```text
experiment   B14_imagenet_full_slice_tokens
variant      b14_imagenet_b6_full_slice_tokens_v1
aggregation  all_real_series_x_16_slice_tokens_v1
trainer      rsna-knee-b14
evaluator    rsna-knee-b14-eval
checkpoint   runs/b14_imagenet_full_tokens/b14_model.pt
```

## Completed training

All four epochs satisfied exact full study and series coverage.

```text
epoch 1
loss                0.7346330162
epoch seconds       2830.6144
encoder LR          8.6819805e-06
head LR             8.5501786e-05

epoch 2
loss                0.6606430862
epoch seconds       3010.1086
encoder LR          5.5e-06
head LR             5.05e-05

epoch 3
loss                0.6074723502
epoch seconds       3436.0601
encoder LR          2.3180195e-06
head LR             1.5498214e-05

epoch 4
loss                0.5822778610
epoch seconds       2663.7554
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

B14 fit the frozen B6 weak supervision more strongly than B13: final training loss `0.5822778610` versus B13 `0.6132239342`. This did **not** translate into higher gold macro AUC.

## Frozen gold development evaluation

```text
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
n studies          58
bootstrap          5000/5000 usable
```

Per-target B14 AUCs:

```text
ACL                0.5122549020
MCL                0.4693877551
Medial Meniscus    0.6454326923
Lateral Meniscus   0.6881987578
Medial OA          0.5116279070
Lateral OA         0.5783365571
PF OA              0.5997425997
Effusion           0.8347826087
Synovitis          0.7419354839
Baker's            0.6884057971
Contusion          0.5465587045
Fracture           0.6208333333
```

Outputs:

```text
runs/b14_imagenet_full_tokens/gold_eval/gold_predictions.csv
runs/b14_imagenet_full_tokens/gold_eval/eval.json
```

## Primary paired comparison — B14 versus B13

B13 reference macro AUC: `0.6293565948`.

Raw macro difference:

```text
B14 - B13 = -0.0095651699
```

Aligned 5,000-replicate paired bootstrap, sign convention `B14-B13`:

```text
median_difference      -0.0093726931
95% paired CI          [-0.0469823411,+0.0250137870]
probability_b_better    0.2924
valid replicates        5000
```

The paired confidence interval crosses zero, so B14 and B13 are not statistically resolved on the repeatedly reused 58-study development surface. However, B14 has the lower point estimate, only `0.2924` bootstrap probability of outperforming B13, higher study-Transformer memory cost, and slower training. The global decision is therefore to **reject B14 and retain B13**.

Paired output:

```text
runs/b14_imagenet_full_tokens/gold_eval/b13_vs_b14.json
```

## Descriptive target deltas versus B13

These values are descriptive only and must not be used to construct target-wise winners.

```text
ACL               +0.0379901961
MCL               -0.0861678005
Medial Meniscus   +0.0360576923
Lateral Meniscus  +0.0086956522
Medial OA         -0.1162790698
Lateral OA        -0.0406189555
PF OA             -0.0180180180
Effusion          +0.0670807453
Synovitis         +0.0310633214
Baker's           -0.0597826087
Contusion         -0.0067476383
Fracture          +0.0319444444
```

The mixed target response reinforces the global interpretation: more slice-level memory helps some findings but is not a better global representation under the primary 12-target macro-AUC objective.

## Decision

```text
B13  0.6293565948   RETAIN / DEVELOPMENT CHAMPION
B14  0.6197914249   REJECT GLOBALLY
```

Do not run B14 epoch 5, tune slice count, change learning rates, construct B13/B14 target-wise hybrids, or search ensemble weights using this 58-study result.

## Next representation hypothesis

The B14 result shows that simply increasing downstream token memory/capacity is not enough. The next higher-upside global hypothesis is **B15: ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchical aggregation**, with the 58 gold studies excluded from SSL optimization and no gold labels used for gradients, early stopping or checkpoint selection.

The 58 fully labelled studies remain a repeatedly reused development/model-selection surface, not independent validation. A Kaggle hidden-test/leaderboard result remains the next genuinely independent performance signal.
