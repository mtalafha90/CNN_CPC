# B24X — exploratory supervision-source pilot

> **Status — 2026-08-15:** B24X matched B6-vs-B23 pilot completed; frozen weak-v2 evaluation completed. B24X-Density training completed; density evaluation pending. **No gold evaluation and no promotion are allowed.** B20 remains the active working model.

## Why B24X exists

Formal B23-v1 did not pass its predeclared labeller gate because its specificity (`0.5678`) was below frozen B6 (`0.6061`). Consequently no canonical B23 holdout was frozen and formal B24 could not legally proceed.

B24X is therefore a deliberately separate exploratory experiment. It does **not** override the failed B23 gate and is incompatible with formal B24 gold acceptance. Its only purpose is to ask whether denser B23 report supervision appears to improve MRI learning under a tightly matched downstream recipe.

## Governance

```text
formal B23 gate                  FAILED
formal B23 holdout               not frozen
formal B24                       not run
B24X gold evaluation             prohibited
B24X promotion                   prohibited
active working model             B20
independent signal               hidden competition evaluation only
```

## Matched downstream recipe

The B6 and B23 arms are matched on:

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

The only intended difference is the supervision cells/targets.

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

This surface was built only on studies available in the B23 pilot and excludes gold and frozen weak-v2 holdout studies from gradients.

## Training

### B6 control

```text
E1 loss  0.8581187165   coverage=true
E2 loss  0.7132374823   coverage=true
checkpoint  runs/b24x_pilot/b6_control/b24_b6_control_model.pt
```

### B23/Qwen candidate

```text
E1 loss  0.7599072829   coverage=true
E2 loss  0.6096711156   coverage=true
checkpoint  runs/b24x_pilot/b23_candidate/b24_b23_candidate_model.pt
```

Checkpoint verification confirmed:

```text
same 692 studies/order            true
same encoder initialization       true
same crop                         true
both fixed E2                     true
both exploratory                  true
gold acceptance allowed           false
formal B23 gate retained failed   true
```

Training losses are not directly comparable because the active supervision masks differ.

## Frozen weak-v2 evaluation

The evaluation uses the frozen 623-study B6 weak-v2 holdout. It contains no overlap with the 692 B24X training studies:

```text
training studies                  692
holdout studies                   623
overlap                             0
```

### Macro result

```text
B6 control       0.6148488366  [0.5856757959,0.6451316589]
B23/Qwen         0.7116126450  [0.6785972089,0.7435358854]

raw B23 - B6    +0.0967638083
paired median   +0.0963512743
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
valid bootstrap  4913/5000
```

This is a strict all-12-target paired study bootstrap.

### Per-target result

| Target | B6 | B23 | Delta |
|---|---:|---:|---:|
| Synovitis | 0.5292 | 0.8636 | +0.3344 |
| PF OA | 0.4492 | 0.7297 | +0.2804 |
| Lateral Meniscus | 0.4658 | 0.6830 | +0.2172 |
| ACL | 0.4840 | 0.6519 | +0.1678 |
| Contusion | 0.4596 | 0.6093 | +0.1497 |
| Medial Meniscus | 0.6399 | 0.7123 | +0.0724 |
| MCL | 0.6182 | 0.6610 | +0.0427 |
| Medial OA | 0.7170 | 0.7260 | +0.0091 |
| Lateral OA | 0.7082 | 0.6946 | -0.0137 |
| Effusion | 0.7736 | 0.7594 | -0.0142 |
| Baker's | 0.7829 | 0.7645 | -0.0184 |
| Fracture | 0.7504 | 0.6842 | -0.0663 |

## Interpretation

The point estimate and paired interval provide strong exploratory evidence that the B23 supervision produced a better MRI learner **on B6's own frozen weak surface**. This is notable because that surface is labelled by B6 and therefore structurally favors the B6-supervised control.

However, weak-v2 still measures agreement with a report teacher, not expert pathology truth. B21 already demonstrated that weak-surface improvement can fail to transfer to reused expert gold. B24X therefore cannot justify model promotion.

The per-target pattern also argues against a simplistic "more cells always helps" interpretation. Fracture, for example, received many additional B23 labels but its weak-v2 AUC decreased. The next ablation isolates density from changed/dropped B23 decisions.

## B24X-Density

### Hypothesis

B24X-Density keeps every B6 committed cell exactly unchanged and lets B23 contribute only where B6 is silent:

```text
B6 committed     -> keep B6 target/weight
B6 silent,
B23 committed    -> add B23 target/weight
otherwise        -> unsupervised
```

This removes all B23 overrides of B6 labels and keeps all B6 cells that full B23 drops.

### Frozen density surface

```text
shared studies                 692
possible cells                8304
B6 cells preserved            3045
B23-only cells added           2844
final usable cells             5889
B6 cells dropped                  0
B6 labels overridden              0
```

Per-target density cells:

```text
ACL                635
MCL                616
Medial Meniscus    665
Lateral Meniscus   633
Medial OA          429
Lateral OA         361
PF OA              457
Effusion           657
Synovitis          209
Baker's            402
Contusion          429
Fracture           396
```

### Training result

```text
E1 loss  0.7647414911   coverage=true
E2 loss  0.6197285242   coverage=true
checkpoint  runs/b24x_density/density/b24x_density_model.pt
```

**Evaluation status:** pending on the same frozen 623-study weak-v2 holdout. The existing B6 and full-B23 weak-v2 predictions should be reused rather than recomputed.

The decisive three-arm comparison is:

```text
B6       = 0.6148488366
Density  = pending
Full B23 = 0.7116126450
```

Interpretation after that evaluation:

```text
Density ~ Full B23
    most of the gain is explained by filling B6-silent cells.

Density << Full B23
    changed B23 decisions/semantics contribute materially.

Density > Full B23
    B23-only additions are useful but some full-B23 drops/overrides are harmful.
```

## What must not be done

```text
no B24X gold acceptance
no B24X promotion
no target-wise B6/B23 hybrid chosen from the weak-v2 per-target table
no prompt retuning to optimize this weak-v2 result
no reinterpretation of B24X as formal B24
no claim that weak-v2 is expert truth
```

If B23 is revised semantically, that revision must be a new version with new provenance/cache and its own labeller audit. Formal B24 can resume only after a B23 version passes the formal labeller gate and a valid development split is frozen prospectively.
