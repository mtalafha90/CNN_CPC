# B19 — Joint-focused MRI input

> **Status — 2026-08-13:** COMPLETED. Selected epoch 3.

B19 tested whether suppressing peripheral field-of-view context would make the
B18 classifier rely more strongly on the knee joint itself. It kept the complete
B18 architecture, B6 supervision, frozen B16 encoder, optimizer and global
expert checkpoint-selection rule unchanged and modified only the spatial input.

## Frozen B19 transform

```text
center crop to 90% FOV
-> resize to 224 x 224
-> central 72% full weight
-> cosine taper
-> zero from normalized |x| or |y| >= 0.90
```

Policy:

```text
version                 joint_focus_center_crop_cosine_v1
crop_fraction           0.90
full_weight_fraction    0.72
outer_zero_fraction     0.90
```

## Completed five-epoch result

| Epoch | B6 loss | Expert-selection macro AUC |
|---:|---:|---:|
| 1 | 0.7529144474 | 0.5802164014 |
| 2 | 0.6582958376 | 0.6242721863 |
| **3** | **0.6268349332** | **0.6581308356** |
| 4 | 0.6028206218 | 0.6369926082 |
| 5 | 0.5850531640 | 0.6485687319 |

Selected checkpoint:

```text
runs/b19_joint_focus/b19_model.pt
selected_epoch = 3
selected_expert_selection_macro_auc = 0.6581308355747585
```

The selected expert score is a checkpoint-selection statistic only and is not
independent validation evidence.

All five epochs completed the exact frozen B18 surface:

```text
training studies                    3120 / 3120
usable B6 cells                    14123 / 14123
positive / negative                6871 / 7252
eligible MRI series               17475 / 17475
encoder gradients                  none
encoder SHA                        b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
budget limited                     no
```

## Comparison with B18

B18 selected epoch 2 at `0.6654496134`; B19 selected epoch 3 at
`0.6581308356`, a difference of `-0.0073187778` on the repeatedly reused expert
selection surface. This difference is not independent evidence of superiority.

B19 recovered substantially after a weak first epoch and exceeded the matching
B18 statistic at epochs 3 and 5, indicating that the focused input itself did
not simply destroy diagnostic information.

## Important Grad-CAM finding

Post-selection visualization of the same expert-positive effusion study showed:

```text
B19 TTA probability ~0.897
```

so classification remained strong. However, Grad-CAM activation concentrated
heavily along the synthetic top/bottom boundaries produced by the cosine
vignette. The transform therefore replaced one possible peripheral shortcut
with a new deterministic preprocessing shortcut.

**Conclusion:** B19 is a valid completed experiment, but the cosine/vignette
mask is not suitable as the final joint-focusing strategy.

This directly motivates B20: retain the same 90% center crop and resize, but
remove the multiplicative mask entirely.

See `docs/B20_CROP_ONLY_FOCUS.md`.
