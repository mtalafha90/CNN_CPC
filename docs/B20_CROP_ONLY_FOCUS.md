# B20 — Crop-only knee focus

> **Status — 2026-08-13:** COMPLETED / SELECTED EPOCH 2 / LOCAL INFERENCE PASS.

B20 follows the completed B19 joint-focus experiment. B19 retained strong
effusion classification but Grad-CAM showed that its deterministic cosine
vignette created a new artificial border shortcut: activation concentrated on
the top/bottom taper boundaries rather than only on plausible joint-fluid
regions.

B20 tests the narrowest corrective change.

## Single change versus B19

B19:

```text
90% centered crop -> resize 224x224 -> cosine/vignette mask
```

B20:

```text
90% centered crop -> resize 224x224
```

There is **no multiplicative spatial mask, no black border, no cosine taper and
no additional crop jitter** in B20-v1.

Frozen policy:

```text
version        joint_focus_center_crop_only_v1
crop_fraction  0.90
```

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

Encoder SHA remained unchanged in every epoch:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

Every epoch had exact full coverage:

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

The predeclared highest-global-AUC / earliest-tie rule selected:

```text
selected epoch        2
selection statistic   0.667159355531343
checkpoint            runs/b20_crop_focus/b20_model.pt
```

The selected 58-study score is a checkpoint-selection statistic only and is
**not independent validation evidence**.

## Development comparison

```text
              B18        B19        B20
selected     0.665450    0.658131    0.667159
selected ep     2           3           2
```

Numerically:

```text
B20 - B18 selected statistic   +0.0017097421
B20 - B19 selected statistic   +0.0090285200
```

The B20-B18 difference is too small and comes from the same repeatedly reused
58-study development/selection surface; it must **not** be presented as evidence
that B20 is more accurate than B18.

## Local submission smoke test

The selected B20 checkpoint passed the local competition-inference/schema test:

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

The three-row local output remains only a smoke test and is not hidden-test
performance evidence.

## Same-source Grad-CAM comparison

A post-selection comparison was performed on one expert-positive effusion case:

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

The same MRI series, sampled slice, and TTA view were supplied to B18, B19 and
B20.

### Visualization bookkeeping correction

The first same-source comparison exposed a visualization-only bookkeeping issue:
per-view probabilities were computed under `torch.no_grad()` without explicitly
placing the model in `eval()` mode, so dropout could remain active until the
Grad-CAM pass switched the model to evaluation mode. This could make the reported
automatic view choice disagree with the later Grad-CAM view probability.

The shared visualization helper was corrected to enforce `model.eval()` before
all per-view probability passes and the comparison tool now checks the direct
view probability against the Grad-CAM forward probability. This issue affected
visualization bookkeeping only; it did not change B18/B19/B20 training,
checkpoint selection, or submission inference.

### Canonical corrected deterministic comparison

After the `eval()` fix the comparison was rerun directly under
`runs/focus_comparison/`. The automatic B18 reference selection chose:

```text
common TTA view                 +1
common plane                    Sagittal
common series index             1
common sampled slice            0
```

Corrected probabilities shown on the canonical figure were:

```text
                         TTA probability    explained view probability
B18 full FOV                  0.917                    0.919
B19 crop + cosine             0.897                    0.895
B20 crop only                 0.815                    0.816
```

The near equality between the direct explained-view probability and the plotted
Grad-CAM view probability confirms that the deterministic visualization path is
now internally consistent. The earlier pre-fix mask fractions/probability values
should therefore be treated as non-canonical diagnostic output and should not be
used for quantitative comparison.

Qualitative interpretation of the corrected figure:

- **B19 is rejected as the spatial formulation.** It shows broad, diffuse
  activation and multiple strong peripheral/border hotspots consistent with the
  synthetic cosine-vignette shortcut.
- **B20 removes the B19 synthetic-boundary artifact.** Its activation is tied to
  real image/anatomical structures rather than a deterministic taper boundary.
- **B20 is still not clearly superior to B18 on this effusion case.** B18 is
  more focal and more confident, while B20 remains distributed over several
  peripheral and joint-adjacent structures.
- Both B18 and B20 visibly activate around plausible fluid-bearing/anatomical
  regions, but one single Grad-CAM case is insufficient to establish better
  localization.

Therefore the current scientifically defensible localization conclusion remains:

```text
B19: rejected
B18 vs B20: unresolved
```

This single-case Grad-CAM audit is diagnostic only and cannot establish global
localization quality.

## Current decision

Do **not** start B21 solely from the single effusion visualization. The next
useful analysis is a fixed multi-case CAM audit over expert-positive cases,
quantifying measures such as peripheral-border activation, central-joint
activation, CAM center of mass, disconnected-component structure, mask fraction,
and retained pathology probability.

Independent competition evaluation is still required to decide whether B18 or
B20 is the better predictive model.
