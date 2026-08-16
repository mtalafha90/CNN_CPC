# B30 reused-expert result

> **Status — 2026-08-16:** B30 is **NOT PROMOTED** and the projected-complementary formulation is **CLOSED**. B20 remains the active reference. B29 remains the frozen promising candidate.

## Frozen B30 result

B30 was frozen before its expert outcome. It completed the exact historical B20 training surface with a frozen encoder and fixed-E2 endpoint, then was evaluated on the same heavily reused 58-study expert development surface used for descriptive post-hoc comparisons.

```text
B20 macro AUC                 0.6674066371
B30 macro AUC                 0.6547034568
raw B30 - B20                -0.0127031803
paired median difference     -0.0121719969
paired 95% CI                [-0.0391192226, +0.0107218769]
P(B30 > B20)                  0.1422
bootstrap replicates          5000
```

The 58-study surface is not independent validation. B20 was historically selected using this expert surface, B30 was not. The result is therefore development evidence only.

## Per-target result

| Target | B20 AUC | B30 AUC | B30 - B20 |
|---|---:|---:|---:|
| ACL | 0.526961 | 0.504902 | -0.022059 |
| MCL | 0.462585 | 0.455782 | -0.006803 |
| Medial Meniscus | 0.677885 | 0.742788 | +0.064904 |
| Lateral Meniscus | 0.744099 | 0.703106 | -0.040994 |
| Medial OA | 0.694574 | 0.720930 | +0.026357 |
| Lateral OA | 0.671180 | 0.611219 | -0.059961 |
| PF OA | 0.674389 | 0.624196 | -0.050193 |
| Effusion | 0.864596 | 0.828571 | -0.036025 |
| Synovitis | 0.837515 | 0.755078 | -0.082437 |
| Baker's | 0.711957 | 0.762681 | +0.050725 |
| Contusion | 0.520918 | 0.524966 | +0.004049 |
| Fracture | 0.622222 | 0.622222 | 0.000000 |

## Mechanism audit

The prospectively frozen attention audit showed that the new projected query evolved from almost the same near-uniform attention distribution as the historical query at E1 to an extremely different distribution by E2:

```text
E2 primary normalized entropy       0.801805
E2 complementary normalized entropy 0.541575
E2 normalized JS divergence         0.707689
E2 top-1 slice agreement            0.000000
E2 top-3 overlap fraction           0.000457
```

The effective representation perturbation was nevertheless small:

```text
mean ||g(C2-A)|| / ||A||            0.004540
max  ||g(C2-A)|| / ||A||            0.012488
```

This supports a mechanism-level interpretation that B30 learned a very different, more selective slice-attention pattern, but that pattern did not improve the reused macro result. The audit does **not** establish a causal reason for the performance change.

## Frozen decision

```text
B20   ACTIVE REFERENCE
B29   FROZEN PROMISING CANDIDATE
B30   NOT PROMOTED / FORMULATION CLOSED
```

Do not create B30.1 by tuning attention divergence, overlap, gate size, endpoint, or target-specific behavior from this result. Do not selectively preserve only the B30 targets that improved.

The next experiment must introduce an outcome-independent mechanism rather than tuning B30 to the reused expert surface.
