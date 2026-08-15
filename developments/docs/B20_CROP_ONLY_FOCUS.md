# B20 — Crop-only knee focus

> **Status — 2026-08-13:** COMPLETED / PRIMARY KNEE-FOCUSED CANDIDATE / SELECTED EPOCH 2 / NESTED EPOCH-SELECTION AUDIT COMPLETE.

B20 is the clean corrective successor to B19. B19 used a 90% centered crop followed by a cosine/vignette mask; Grad-CAM exposed the deterministic taper boundary as an artificial shortcut. B20 retains only the centered crop and removes the synthetic boundary.

## Spatial policy

```text
B19: 90% centered crop -> resize 224x224 -> cosine/vignette mask
B20: 90% centered crop -> resize 224x224
```

Frozen B20 policy:

```text
version        joint_focus_center_crop_only_v1
crop_fraction  0.90
```

There is no multiplicative mask, black border, cosine taper, or crop jitter.

## Frozen training contract

```text
training studies                    3120
usable B6 cells                    14123
positive / negative                6871 / 7252
eligible MRI series               17475
initializer                        completed B16 report-aligned encoder
encoder                            frozen
encoder LR                         0
head LR                            1e-4
candidate epochs                   5
resolution after crop              224 x 224
sampled positions / series         16
TTA                                [-1,0,1]
selection metric                   global 12-target expert macro AUC
selection tie break                earliest epoch
expert labels in gradients         no
```

Encoder SHA remained unchanged:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

Every epoch completed exact full coverage:

```text
batches                    1560 / 1560
studies                    3120 / 3120
active supervision cells  14123 / 14123
positive cells             6871 / 6871
negative cells             7252 / 7252
series instances          17475 / 17475
max series / batch            14
encoder gradients detected    no
```

## Completed training result

```text
epoch 1  loss 0.7456469554  selection AUC 0.6177301847
epoch 2  loss 0.6459858875  selection AUC 0.6671593555  <- selected
epoch 3  loss 0.6226234155  selection AUC 0.6492154172
epoch 4  loss 0.5998095677  selection AUC 0.6570041510
epoch 5  loss 0.5828775678  selection AUC 0.6577823350
```

Canonical checkpoint:

```text
selected epoch        2
selection statistic   0.667159355531343
checkpoint            runs/b20_crop_focus/b20_model.pt
```

The 58-study score is development/checkpoint-selection evidence, not pristine independent validation.

## B20 nested epoch-selection audit

The five saved candidate checkpoints were re-scored without retraining. The statistical selection rule was unchanged: global 12-target macro AUC, earliest epoch on a numerical tie.

### Primary two-fold cross-fitted selection

For each held-out outer fold, the other two folds were combined to choose the epoch.

```text
outer fold 0 -> epoch 2
outer fold 1 -> epoch 2
outer fold 2 -> epoch 2

cross-fitted OOF macro AUC          0.667159355531343
all 12 target AUCs defined          true
estimated epoch-selection optimism  0.0
```

All three outer-fold analyses independently selected epoch 2. Therefore B20 epoch 2 is robust to this checkpoint-selection audit.

Original selected-versus-fixed endpoint:

```text
all-58 selected macro AUC           0.667159355531343
fixed epoch-5 macro AUC             0.6577823350159498
selection uplift vs epoch 5        +0.00937702051539313
```

### Strict historical-manifest sensitivity analysis

Using only the single `inner_selection` fold to choose the epoch produced:

```text
selected epochs                     [2,5,2]
strict OOF macro AUC                0.6351640998170208
estimated selection optimism        0.03199525571432216
```

This strict variant selects from only about one third of the 58-study surface and is much noisier. B20 never uses gold labels in gradient training, so the remaining `gold_train` fold has no gradient-training role here. The strict result is retained as a small-selection-set sensitivity diagnostic; the two-fold cross-fitted result is the primary checkpoint-selection estimate.

The audit measures **checkpoint-selection optimism only**. The same 58 expert studies have influenced earlier modelling decisions, so the result does not remove broader development-set reuse.

Canonical audit record: [`B20_NESTED_EPOCH_AUDIT.md`](B20_NESTED_EPOCH_AUDIT.md).

## Development comparison

```text
              B18        B19        B20
selected     0.665450    0.658131    0.667159
selected ep     2           3           2
```

```text
B20 - B18 selected statistic   +0.0017097421
B20 - B19 selected statistic   +0.0090285200
```

B20 is the preferred knee-focused formulation because it removes B19's synthetic vignette shortcut. The small B20-B18 difference still must not be presented as independent evidence that B20 is globally more accurate than B18.

## Local submission smoke test

The selected B20 checkpoint passed the local inference/schema test:

```text
test studies                   3
test series                   15
series / study                 5 / 5 / 5
TTA                            [-1,0,1]
metadata repairs               0
sample columns match           true
sample UID order match         true
cosine mask used               false
```

This is an engineering smoke test only.

## Same-source Grad-CAM comparison

A deterministic same-source comparison used one expert-positive effusion case:

```text
StudyInstanceUID
1.2.826.0.1.3680043.8.498.12801308844398614687904447633432197492

target                         Effusion
expert truth                   1
common plane                   Sagittal
common series index            1
common slice index             0
common series UID              1.2.826.0.1.3680043.8.498.51148402259712862712353546920527079297
CAM layer                      28x28
CAM threshold                  0.65
```

Canonical corrected probabilities were approximately:

```text
                         TTA probability    explained view probability
B18 full FOV                  0.917                    0.919
B19 crop + cosine             0.897                    0.895
B20 crop only                 0.815                    0.816
```

The comparison tool explicitly places models in evaluation mode and checks direct-view versus Grad-CAM-forward probability consistency. The earlier visualization bookkeeping discrepancy was confined to the visualization path and did not affect training, checkpoint selection, or submission inference.

Qualitatively:

- B19 shows strong peripheral/border activation consistent with the imposed vignette shortcut and is rejected.
- B20 removes that synthetic-boundary artifact.
- On this single effusion case, B18 is more focal than B20.
- One Grad-CAM case cannot establish global localization or predictive superiority.

Localization conclusion:

```text
B19: rejected
B18 vs B20: unresolved
```

## Current decision

```text
primary knee-focused candidate      B20
canonical epoch                     2
canonical checkpoint                runs/b20_crop_focus/b20_model.pt
cross-fitted epoch selections       [2,2,2]
measured epoch-selection optimism   0.0
```

Do not start a new modelling variant merely from the single-case Grad-CAM result. Continue B20-focused analysis by ranking the 12 targets from the cross-fitted predictions, auditing false positives/false negatives for the weakest targets, and then deciding whether the next B20 modification should address series/plane routing, slice sampling, crop behaviour, or weak-label quality.

Independent competition evaluation remains required for a genuinely external predictive-performance signal.
