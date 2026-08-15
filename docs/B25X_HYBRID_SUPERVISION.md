# B25X — ChatGPT hybrid supervision on the current B20 development pipeline

> **Status — 2026-08-15:** COMPLETE as an exploratory weak-v2 experiment. **No gold evaluation and no promotion are allowed. B20 remains the active working model.**

## Purpose

B25X asks whether the larger ChatGPT-created hybrid report-label cache contains useful supervision for the current MRI learner, while preserving the leakage-safe weak-v2 development protocol and the B20 downstream recipe.

The source cache has mixed/unknown original LLM provenance and is therefore **not** a canonical B23 cache and is **incompatible with formal B23/B24 governance**. It is used only as an exploratory weak-supervision source.

## Fixed protocol

All three arms use:

```text
training studies/order           identical: 2497
frozen weak-v2 holdout           623 studies
train/holdout overlap            0
expert gold in gradients         0
encoder                          frozen weak-v2-safe B16-v2
crop geometry                    B20 post-resize 90% crop
endpoint                         fixed E2
checkpoint selection             none
primary development surface      frozen B6 weak-v2
```

Hybrid raw confidence is diagnostic only. Definite positive/negated states use the fixed supervision mapping; uncertain/unmentioned cells remain unsupervised.

## Training surfaces

### B6 control

```text
possible cells              29964
usable cells                11248  (37.5%)
```

### Pure Hybrid

```text
usable cells                20001  (66.8%)
added over B6                9542
dropped relative to B6        789
cells both committed        10459
disagreements                1120  (10.7%)
```

### B6 + Hybrid-fill

Fill preserves every B6 committed cell and uses Hybrid only where B6 is silent:

```text
B6 cells preserved          11248
Hybrid-only cells added      9542
final usable cells          20790  (69.4%)
B6 cells dropped                0
B6 cells overridden             0
```

This increases supervised-cell coverage by about 84.8% relative to B6 while leaving all existing B6 decisions unchanged.

## Fixed-E2 training

```text
B6 control
E1 loss  0.7601064120
E2 loss  0.6592396402
runtime  45m50s
checkpoint  runs/b25x_hybrid/control/b25x_control_model.pt

B6 + Hybrid-fill
E1 loss  0.6799390770
E2 loss  0.5913315904
runtime  46m43s
checkpoint  runs/b25x_hybrid/fill/b25x_fill_model.pt

Pure Hybrid
E1 loss  0.6557762888
E2 loss  0.5718413529
runtime  46m03s
checkpoint  runs/b25x_hybrid/hybrid/b25x_hybrid_model.pt
```

Training losses are not directly comparable across arms because the supervision masks differ.

## Frozen weak-v2 result

```text
B6 control          0.6723718048
Pure Hybrid         0.7268784872
B6 + Hybrid-fill    0.7308472686
```

Paired comparisons:

```text
Hybrid - B6
raw                 +0.0545066824
median              +0.0557034913
95% CI              [+0.0269870416,+0.0750180195]
P(>0)                1.0000

Fill - B6
raw                 +0.0584754637
median              +0.0591551676
95% CI              [+0.0301804537,+0.0814020218]
P(>0)                1.0000

Hybrid - Fill
raw                 -0.0039687813
median              -0.0039491728
95% CI              [-0.0137571379,+0.0058102163]
P(Hybrid > Fill)     0.2037
```

The full 12-target point estimate favors **Fill**, but Hybrid and Fill are not separated by the paired interval.

## Per-target weak-v2 AUC

| Target | B6 | Hybrid | Fill | Hybrid-B6 | Fill-B6 |
|---|---:|---:|---:|---:|---:|
| ACL | 0.6553 | 0.6678 | 0.6845 | +0.0126 | +0.0293 |
| MCL | 0.6658 | 0.6755 | 0.6737 | +0.0097 | +0.0078 |
| Medial Meniscus | 0.7042 | 0.7181 | 0.7228 | +0.0140 | +0.0186 |
| Lateral Meniscus | 0.6752 | 0.6715 | 0.6667 | -0.0037 | -0.0085 |
| Medial OA | 0.7326 | 0.7493 | 0.7548 | +0.0167 | +0.0222 |
| Lateral OA | 0.7195 | 0.7069 | 0.7128 | -0.0126 | -0.0068 |
| PF OA | 0.7258 | 0.7295 | 0.7235 | +0.0037 | -0.0023 |
| Effusion | 0.7380 | 0.7372 | 0.7611 | -0.0008 | +0.0231 |
| Synovitis | 0.2370 | 0.9221 | 0.9123 | +0.6851 | +0.6753 |
| Baker's | 0.8113 | 0.7537 | 0.7672 | -0.0575 | -0.0441 |
| Contusion | 0.6393 | 0.6377 | 0.6473 | -0.0016 | +0.0080 |
| Fracture | 0.7646 | 0.7534 | 0.7435 | -0.0112 | -0.0211 |

## Synovitis mechanism audit

The 12-target macro gain is dominated by Synovitis. Excluding Synovitis:

```text
11-target macro
B6                  0.7119498792
Hybrid              0.7091330840
Fill                0.7143481419

Hybrid - B6        -0.0028167951
Fill - B6          +0.0023982627
Hybrid - Fill      -0.0052150578
```

Thus the current evidence does **not** support a broad 11-target gain from Hybrid supervision.

The Synovitis training labels explain the exceptional effect:

```text
B6 Synovitis
usable                335
positive              322
negative               13

Hybrid-only additions used by Fill
usable                202
positive               66
negative              136

Final Fill Synovitis
positive              388
negative              149
```

B6 therefore supplied an extremely one-sided Synovitis training surface. Hybrid-fill repaired the negative-class scarcity without removing any B6 labels.

The frozen weak-v2 Synovitis surface is itself small and imbalanced:

```text
usable                 81
positive               77
negative                4
```

### Four-negative robustness audit

The four negative holdout scores changed from approximately positive-like scores under B6 to clearly lower scores under Hybrid/Fill:

```text
negative case        B6        Hybrid      Fill
1                    0.6504     0.1376     0.0809
2                    0.6589     0.0565     0.0634
3                    0.6208     0.6150     0.2720
4                    0.6523     0.0754     0.0430
```

Positive-score medians:

```text
B6       0.6191
Hybrid   0.7690
Fill     0.6145
```

Leave-one-negative-out Synovitis AUC ranges:

```text
B6       0.177489 -- 0.259740
Hybrid   0.900433 -- 0.978355
Fill     0.887446 -- 0.961039
```

The Synovitis rescue is therefore not controlled by one lucky negative case. It is consistent across the four available negatives, while still being limited by the very small negative holdout count.

## Scientific interpretation

B25X supports a narrow but useful conclusion:

> Hybrid report supervision can repair a severe class-coverage failure in the current weak-label training surface. In B25X, the measurable macro-AUC gain is overwhelmingly a Synovitis effect caused by recovering many missing negative Synovitis examples. Across the other eleven targets, B6+Hybrid-fill is approximately neutral overall.

The experiment also favors **fill-only supervision over replacement as the safer development mechanism**: Fill achieves the best 12-target point estimate while preserving all B6 cells. There is no paired evidence that the pure Hybrid replacements/drops improve over Fill.

This is a mechanism-development result, **not a model-promotion result**.

## Governance

```text
B20 active working model                  unchanged
B25X source provenance                    mixed/unknown
formal B23 compatible                     no
formal B24 eligible                       no
expert gold used in B25X development      no
B25X promotion allowed                    no
weak-v2 meaning                           B6 teacher agreement, not expert truth
```

No DINOv2 or soft-dense-label branch is part of the current plan. The next research phase should develop the **current B20-family working model**, using the B25X finding as a diagnostic about supervision coverage/class balance rather than replacing the architecture wholesale.

## Artifacts

```text
runs/b25x_hybrid/control/b25x_control_model.pt
runs/b25x_hybrid/hybrid/b25x_hybrid_model.pt
runs/b25x_hybrid/fill/b25x_fill_model.pt
runs/b25x_hybrid/weak_v2_eval/three_arm_predictions.csv
runs/b25x_hybrid/weak_v2_eval/comparison.json
```
