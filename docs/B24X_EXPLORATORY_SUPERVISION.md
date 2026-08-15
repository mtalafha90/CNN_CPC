# B24X — exploratory supervision-source pilot

> **Status — 2026-08-15:** B24X matched B6-vs-B23 pilot and B24X-Density are both complete on frozen weak-v2. **No gold evaluation and no promotion are allowed. B20 remains the active working model.**

## Why B24X exists

Formal B23-v1 did not pass its predeclared labeller gate because specificity (`0.5678`) was below frozen B6 (`0.6061`). No canonical B23 holdout was frozen and formal B24 remains blocked/not run.

B24X is therefore deliberately separate and exploratory. Its purpose is to ask whether denser B23 report supervision improves MRI learning under a tightly matched downstream recipe.

## Governance

```text
formal B23 gate                  FAILED
formal B23 holdout               not frozen
formal B24                       not run
B24X gold evaluation             prohibited
B24X promotion                   prohibited
active working model             B20
```

## Matched downstream recipe

```text
training studies/order           identical
MRI series exposure              identical
encoder                          frozen weak-v2-safe B16-v2
encoder learning rate            0
crop geometry                    B20 post-resize 90% crop
input resolution                 224
head learning rate               1e-4
scheduler horizon                5
fixed endpoint                   E2
TTA evaluation offsets           [-1,0,1]
seed                             2026
```

## Pilot matched surface

```text
shared studies                         692
possible cells                        8304
B6 usable cells                       3045  (36.7%)
B23 usable cells                      5697  (68.6%)
added by B23                          2844
dropped by B23                         192
cells both committed on              2853
disagreements there                    70  (2.5%)
```

Gold and frozen weak-v2 holdout studies were excluded from gradients.

## B24X full-B23 arm

Training:

```text
B6 control
E1 loss  0.8581187165
E2 loss  0.7132374823

B23/Qwen candidate
E1 loss  0.7599072829
E2 loss  0.6096711156
```

Frozen weak-v2:

```text
B6 control       0.6148488366  [0.5856757959,0.6451316589]
B23/Qwen         0.7116126450  [0.6785972089,0.7435358854]

raw B23 - B6    +0.0967638083
paired median   +0.0963512743
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
valid bootstrap  4913/5000
```

Per-target deltas:

```text
Synovitis          +0.3344
PF OA              +0.2804
Lateral Meniscus   +0.2172
ACL                +0.1678
Contusion          +0.1497
Medial Meniscus    +0.0724
MCL                +0.0427
Medial OA          +0.0091
Lateral OA         -0.0137
Effusion           -0.0142
Baker's            -0.0184
Fracture           -0.0663
```

## B24X-Density

Density preserves every B6 committed cell and adds B23 only where B6 is silent:

```text
shared studies                 692
B6 cells preserved            3045
B23-only cells added           2844
final usable cells             5889
B6 cells dropped                  0
B6 labels overridden              0
```

Training:

```text
E1 loss  0.7647414911
E2 loss  0.6197285242
checkpoint  runs/b24x_density/density/b24x_density_model.pt
```

### Final frozen weak-v2 comparison

```text
B6 control       0.6148488366
Density          0.7147994969
Full B23         0.7116126450

Density - B6       +0.0999506603
Full B23 - B6      +0.0967638083
Full B23 - Density -0.0031868519
```

Paired bootstrap:

```text
B6 -> Density
median             +0.0998800219
95% CI             [+0.0642300469,+0.1348991590]
P(Density > B6)     1.0000

B23 - Density
median             -0.0031277652
95% CI             [-0.0099855349,+0.0034718378]
P(B23 > Density)    0.1799
```

Density captured `103.3%` of the full-B23 point-estimate gain. This percentage is descriptive, not causal.

### Per-target Density comparison

| Target | B6 | Density | B23 | Density-B6 | B23-Density |
|---|---:|---:|---:|---:|---:|
| ACL | 0.4840 | 0.6540 | 0.6519 | +0.1700 | -0.0022 |
| MCL | 0.6182 | 0.6727 | 0.6610 | +0.0545 | -0.0118 |
| Medial Meniscus | 0.6399 | 0.7111 | 0.7123 | +0.0712 | +0.0011 |
| Lateral Meniscus | 0.4658 | 0.6814 | 0.6830 | +0.2156 | +0.0016 |
| Medial OA | 0.7170 | 0.7383 | 0.7260 | +0.0213 | -0.0123 |
| Lateral OA | 0.7082 | 0.6911 | 0.6946 | -0.0172 | +0.0035 |
| PF OA | 0.4492 | 0.7394 | 0.7297 | +0.2902 | -0.0098 |
| Effusion | 0.7736 | 0.7554 | 0.7594 | -0.0182 | +0.0041 |
| Synovitis | 0.5292 | 0.8831 | 0.8636 | +0.3539 | -0.0195 |
| Baker's | 0.7829 | 0.7637 | 0.7645 | -0.0192 | +0.0008 |
| Contusion | 0.4596 | 0.6051 | 0.6093 | +0.1455 | +0.0042 |
| Fracture | 0.7504 | 0.6822 | 0.6842 | -0.0683 | +0.0020 |

## Interpretation

The density experiment resolves the main B24X mechanism question:

> **Almost the entire B24X point-estimate gain comes from filling B6-silent supervision cells. There is no evidence that replacing/dropping B6 decisions is beneficial.**

The negative targets under Density cannot be blamed on B23 overrides, because Density performs none. Any harm must arise from the additional B23-only labels and/or their interaction with optimization/class balance.

This conclusion motivated the later B25X full matched hybrid/fill experiment, recorded in [`B25X_HYBRID_SUPERVISION.md`](B25X_HYBRID_SUPERVISION.md).

## What must not be done

```text
no B24X gold acceptance
no B24X promotion
no target-wise B6/B23 hybrid chosen from weak-v2 per-target tables
no reinterpretation of B24X as formal B24
no claim that weak-v2 is expert truth
```
