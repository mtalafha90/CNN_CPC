# B22 — pre-resize crop training-duration audit

> **Status — 2026-08-14:** COMPLETE. The five-epoch trajectory was trained and audited. **Epoch 2 remained the best pre-resize-crop endpoint; E3–E5 did not rescue B21. B20 remains the active working model.**

## Question

B21 tested the corrected pre-resize 90% crop at a fixed epoch-2 endpoint and failed the predeclared gold acceptance gate. B22 asked one narrower post-hoc question:

```text
Does the B21 pre-resize crop pipeline require more downstream training than B20?
```

B22 changed **training duration only** relative to the full-data B21 recipe.

## Frozen training recipe

```text
initializer                    historical B16 report-aligned encoder
encoder                        frozen
B6-active training studies     3120
usable B6 cells               14123
positive / negative            6871 / 7252
eligible MRI series           17475
crop fraction                  0.90
crop stage                     native array before resize
normalization support          cropped native field
output resolution              224 x 224
training epochs                5
cosine scheduler horizon       5
expert evaluation in training  none
checkpoint selection training  none
```

Every epoch was saved under:

```text
runs/b22_duration_audit/candidates/epoch_1.pt
runs/b22_duration_audit/candidates/epoch_2.pt
runs/b22_duration_audit/candidates/epoch_3.pt
runs/b22_duration_audit/candidates/epoch_4.pt
runs/b22_duration_audit/candidates/epoch_5.pt
```

The run required the completed B21 acceptance record and only started after that record certified that B21 had consumed its one-look gold comparison and failed promotion.

## Exact training trajectory

All five epochs had exact full coverage and the frozen encoder SHA remained unchanged:

```text
encoder SHA
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

```text
epoch   training loss    head LR           batches   studies   cells    series
E1      0.7388751291     9.054634122e-05   1560      3120      14123    17475
E2      0.6381611442     6.579634122e-05   1560      3120      14123    17475
E3      0.6087977977     3.520365878e-05   1560      3120      14123    17475
E4      0.5890809184     1.045365878e-05   1560      3120      14123    17475
E5      0.5680555741     1.000000000e-06   1560      3120      14123    17475
```

Training loss decreased monotonically through E5.

## Reproducibility safeguards

Historical B20 replayed at:

```text
canonical B20 macro AUC    0.6671593555313430
B22-audit B20 replay       0.6679590975360873
replay - canonical       +0.0007997420047443
allowed tolerance          0.005
```

The retrained B22 E2 checkpoint reproduced the prior B21 E2 expert macro AUC extremely closely:

```text
prior B21 E2 macro AUC     0.6573196516459231
B22 E2 macro AUC           0.6574269017531732
B22 E2 - prior B21 E2     +0.0001072501072501
allowed tolerance          0.005
```

Both safeguards passed, so E3–E5 are interpretable as a clean duration extension of the B21 trajectory.

## Expert-gold duration trajectory

The 58-study gold trajectory was evaluated post hoc for diagnosis only:

| Epoch | Gold macro AUC | Raw delta vs B20 replay | Paired 95% CI | P(epoch > B20) |
|---|---:|---:|---:|---:|
| E1 | 0.6135270850 | -0.0544320126 | [-0.0786018515, -0.0280350783] | 0.0000 |
| **E2** | **0.6574269018** | **-0.0105321958** | **[-0.0323859143, +0.0098214527]** | **0.1672** |
| E3 | 0.6387456622 | -0.0292134353 | [-0.0523986045, -0.0087333144] | 0.0030 |
| E4 | 0.6136783995 | -0.0542806980 | [-0.0827548184, -0.0276651497] | 0.0000 |
| E5 | 0.6282683534 | -0.0396907441 | [-0.0654472831, -0.0162928843] | 0.0000 |

The exploratory best epoch is therefore:

```text
best epoch by reused gold     E2
best macro AUC                0.6574269017531732
```

## Interpretation

B22 answers the duration question negatively:

```text
more downstream training does not rescue the pre-resize-crop formulation
```

The most important pattern is the divergence between optimization loss and expert ranking after E2:

```text
E2 -> E3
training loss   0.6382 -> 0.6088   improves
expert AUC      0.6574 -> 0.6387   worsens

E3 -> E4
training loss   0.6088 -> 0.5891   improves
expert AUC      0.6387 -> 0.6137   worsens further
```

This is consistent with the frozen downstream model increasingly fitting report-derived weak supervision after the point at which expert-pathology ranking peaks. It strengthens the practical case for epoch 2 as the downstream stopping point in this regime.

B22 does **not** establish an independent statistical claim because the 58 expert studies were already reused throughout historical development and the B22 trajectory itself is post-hoc. The duration result is a bounded diagnostic, not a promotion experiment.

## Decision

```text
B20                         ACTIVE WORKING MODEL
B21 pre-resize crop         NOT PROMOTED
B22 longer-duration rescue  NOT SUPPORTED
best observed B22 epoch     E2
next optimization priority  label / development-selection problem
```

Do not reopen B21/B22 training duration, crop fraction, target mixing, or gold-guided epoch selection from this trajectory.

## Canonical artifacts

```text
runs/b22_duration_audit/history.json
runs/b22_duration_audit/candidates/epoch_1.pt
runs/b22_duration_audit/candidates/epoch_2.pt
runs/b22_duration_audit/candidates/epoch_3.pt
runs/b22_duration_audit/candidates/epoch_4.pt
runs/b22_duration_audit/candidates/epoch_5.pt
runs/b22_duration_audit/gold_trajectory/trajectory.json
runs/b22_duration_audit/gold_trajectory/trajectory_predictions.csv
```
