# B26.2 — reused expert development result

> **Status — 2026-08-16:** fixed-E2 training COMPLETE; reused 58-study expert diagnostic COMPLETE; **NOT PROMOTED**. B20 remains the active working model.

## Governance

B26.2 was trained on the complete historical 3,120-study B20/B6 gradient surface, so the historical 623-study weak-v2 partition is not a holdout for this checkpoint and must not be used as validation.

The only available expert diagnostic is the same 58-study / 696-cell expert surface previously consumed during B20 development and epoch selection. The result below is therefore **post-hoc reused development evidence, not independent validation and not automatic promotion evidence**.

## Paired result

```text
B20 macro AUC        0.6674066371
B26.2 macro AUC      0.6662972442
raw delta           -0.0011093928
paired 95% CI       [-0.0156579503, +0.0142502372]
P(B26.2 > B20)       0.4442
```

The macro result is essentially unchanged and the paired interval includes zero comfortably.

## Per-target AUC

| Target | B20 | B26.2 | Delta |
|---|---:|---:|---:|
| ACL | 0.5270 | 0.5355 | +0.0086 |
| MCL | 0.4626 | 0.4943 | +0.0317 |
| Medial Meniscus | 0.6779 | 0.7019 | +0.0240 |
| Lateral Meniscus | 0.7441 | 0.7180 | -0.0261 |
| Medial OA | 0.6946 | 0.7023 | +0.0078 |
| Lateral OA | 0.6712 | 0.6654 | -0.0058 |
| PF OA | 0.6744 | 0.6384 | -0.0360 |
| Effusion | 0.8646 | 0.8845 | +0.0199 |
| Synovitis | 0.8375 | 0.7826 | -0.0550 |
| Baker's | 0.7120 | 0.7174 | +0.0054 |
| Contusion | 0.5209 | 0.5331 | +0.0121 |
| Fracture | 0.6222 | 0.6222 | +0.0000 |

## Decision

B26.2 was designed specifically to repair the Synovitis supervision imbalance. On the reused expert diagnostic, Synovitis AUC decreased by about 0.055 while the 12-target macro remained effectively tied with B20. Therefore the targeted repair does not provide evidence for replacing B20.

```text
B26 raw label gate        FAILED
B26.1 label gate          FAILED
B26.2 label gate          PASSED
B26.2 fixed-E2 training   PASSED
B26.2 reused-gold result  no macro improvement; Synovitis lower
B26.2 promotion           REJECTED
working model             B20
```

This does not prove that the 171 B26.2 labels are wrong. The manual report-level quality audit passed. It shows that improving report-supervision balance did not translate into better ranking on the reused expert imaging target under the frozen B20 training recipe.

## Next analysis

Before any further training variant, run a mechanism audit that uses no model-selection outcome to modify labels. The audit should quantify:

1. how the 171 fills changed the Synovitis class contribution under `target_balance_multipliers`;
2. how Synovitis co-occurs with Effusion and related targets on the B6/B26.2 supervision surface;
3. the corresponding contingency on the reused 58-study expert set, explicitly marked post-hoc;
4. whether the B26.2 intervention primarily changed class mass rather than target-level mass.

Any subsequent B26.3 experiment would be exploratory/post-hoc and cannot be promoted from this reused expert surface.
