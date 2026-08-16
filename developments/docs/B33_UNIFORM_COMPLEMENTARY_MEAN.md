# B33 — exact-uniform complementary mean

> **Status — 2026-08-16:** IMPLEMENTED / FROZEN BEFORE B33 OUTCOME / READY FOR SAFETY TESTS. **B20 remains the active reference. B29 and B31 remain frozen development candidates. B32 is closed and not promoted.**

## Motivation

B29 introduced a learned complementary softmax summary and improved the reused 58-study development macro from 0.667407 (B20) to 0.676888. B31 reached 0.682280 on the same reused surface, but its prospective mechanism audit showed that the learned complementary weights were almost uniform and that the B31 context perturbation barely changed them.

B32 then added a genuinely non-redundant weighted feature-dispersion summary. Its mechanism worked as designed, but the reused expert macro was only 0.668699, below B29 and B31. The B32 dispersion formulation is closed.

B33 therefore asks the simplest remaining mechanistic question:

> Is the B29/B31 development gain primarily caused by providing B20 with a second broad **uniform mean-like series representation**, rather than by learned slice selection?

## Frozen architecture

For the 16 original B20 slice tokens `X_i` of each real series:

```text
C_mean = LN0((1/16) * sum_i X_i)
```

where `LN0` is parameter-free LayerNorm.

The historical B20 token remains:

```text
A = historical learned-attention series token
```

and the final B33 series token is:

```text
T = A + tanh(g) * (C_mean - A)
```

There is no learned complementary query.

## Parameter contract

```text
uniform mean arithmetic        0
parameter-free normalization   0
feature-wise gate g          768
                             ---
total new parameters         768
```

B33 explicitly contains no:

```text
B29 complementary query
B31 local-context Conv1d
B32 dispersion gate/statistic
additional projection
trainable normalization
target-specific routing
extra encoder pass
```

## Zero-init and RNG safety

The complete historical B20 model is constructed first. The only new parameter is created with `torch.zeros`, so B33 introduces no random parameter draw after B20 construction.

At initialization:

```text
g = 0
T = A
```

Therefore B33 is exactly the B20 function.

The historical B20 series pool executes before the new deterministic uniform-mean branch. The B33 branch contains no dropout or random operation. Safety tests require both evaluation-mode functional equivalence and training-mode RNG-path equivalence with B20 at zero gate.

## Frozen controls

```text
training studies            3120
eligible MRI series        17475
usable B6 cells            14123
positive / negative        6871 / 7252
B16 report-aligned encoder  frozen
encoder SHA                 b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
B20 crop                    90% post-resize crop only
slice sampling              16 positions / series
B20 learned series pool     unchanged
study Transformer           unchanged
pathology-query heads       unchanged
optimizer / LR              unchanged
augmentation                unchanged
loader seed                 unchanged
scheduler horizon           5 epochs
training endpoint           fixed E2
expert labels in gradients  0
expert checkpoint selection none
```

B33 is trained from the historical B20/B16 initialization recipe. It is not a fine-tune of B29, B31 or B32.

## Prospective mechanism audit

Before any B33 expert outcome is inspected, the trainer records diagnostics on the first 64 frozen-loader batches of each epoch:

```text
raw uniform-mean norm / primary-token norm
cosine(C_mean, A)
||C_mean - A|| / ||A||
maximum ||C_mean - A|| / ||A||
||tanh(g)*(C_mean-A)|| / ||A||
maximum effective residual / ||A||
gate magnitude
```

The audit does not select a checkpoint or tune any threshold. Its purpose is only to verify that the fixed uniform summary differs meaningfully from the historical B20 token and that the learned gate remains controlled.

## Safety tests

B33 tests pin:

- exactly 768 new parameters;
- no complementary query, local-context module or dispersion gate;
- shared B20 initialization equality under the same construction seed;
- exact zero gate initialization;
- exact B20 functional equivalence at zero gate;
- B20 training-mode RNG-path equivalence;
- exact arithmetic-mean plus parameter-free-LayerNorm summary;
- nonzero finite gate gradient on the first backward pass;
- bounded tanh gate;
- nonzero-gate functional effect;
- finite/bounded audit metrics;
- bf16 finite behavior;
- empty-study finite behavior.

## Runtime

```text
hard budget       <= 8.25 h
internal reserve   >= 30 min
competition limit      9 h
```

B33 adds only an arithmetic mean, parameter-free normalization and elementwise gated residual over already computed slice tokens.

## Outputs

```text
runs/b33_uniform_complementary_mean/
├── b33_model.pt
├── training_audit.json
├── uniform_gate_state.json
├── uniform_audit.json
└── history.json
```

## Evaluation governance

The historical 623-study weak-v2 partition is not a holdout because B33 trains on all 3,120 historical B20 weak-supervision studies.

After a valid training/mechanism audit, the reused 58-study expert surface may be used only descriptively. The frozen evaluator compares:

```text
B20
B29
B31
B33
```

and reports paired B33-vs-B20, B33-vs-B29 and B33-vs-B31 bootstrap differences.

This reused surface cannot independently promote B33. Do not derive a B33.1 mean definition, gate formulation, endpoint, target-specific route, or blend from that outcome.
