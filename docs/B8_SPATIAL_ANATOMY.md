# B8 — pathology-aware spatial anatomy learning

> **Status — 2026-08-10:** **COMPLETE / REJECTED AS CAMPAIGN LEADER.** B8-v1 completed its frozen four-epoch real-data training recipe and its one-shot 58-study gold development evaluation. It achieved macro ROC AUC `0.5300962807`, below B7.1 (`0.5644802945`). The paired B7.1 -> B8 bootstrap favored B8 in only `11.56%` of replicates. B7.1 remains the retained main standalone model.

## Motivation

B7.1 is the current best standalone development model:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

B7.1 globally pools every sampled 2.5D slice to one vector before MRI-token attention. B8 tested whether retaining coarse within-slice spatial structure and adding gentle pathology-specific stream/slice priors would improve pathology learning while leaving the successful B7.1 weak-supervision recipe fixed.

## Architecture change

B7.1 MRI memory:

```text
6 streams x 16 slices x 1 pooled token = 96 MRI tokens
```

B8 MRI memory:

```text
6 streams x 16 slices x 2x2 regions = 384 MRI tokens
```

The B8 ConvNeXt encoder reuses the B7.1 weights. Instead of global average pooling only, B8 takes the final ConvNeXt feature map, adaptive-pools it to a `2x2` grid, applies the same learned ConvNeXt classifier normalization, and emits four spatial tokens per sampled 2.5D slice.

Each token receives the inherited slice-position embedding, inherited stream embedding, and a new learned region-position embedding. The inherited MRI Transformer contextualizes the 384-token memory and the inherited 12 pathology queries cross-attend to it.

## Soft anatomy priors

B8 applies a fixed additive attention-logit prior for each pathology query. The prior is deliberately soft:

- preferred MRI streams have prior weight `1.0`;
- non-preferred streams retain prior weight `0.75`;
- focal internal structures receive only a broad center-slice preference with floor `0.80`;
- diffuse/fluid findings are slice-neutral;
- no MRI stream or slice is hard-masked.

Predeclared stream preferences:

| Target | Preferred streams |
|---|---|
| ACL | sagittal fluid, sagittal structural, coronal fluid |
| MCL | coronal fluid, coronal structural |
| Medial Meniscus | sagittal fluid/structural, coronal fluid/structural |
| Lateral Meniscus | sagittal fluid/structural, coronal fluid/structural |
| Medial OA | coronal structural/fluid, sagittal structural |
| Lateral OA | coronal structural/fluid, sagittal structural |
| PF OA | axial fluid/structural, sagittal structural |
| Effusion | fluid-sensitive sagittal/coronal/axial |
| Synovitis | fluid-sensitive sagittal/coronal/axial |
| Baker's | sagittal fluid, axial fluid, coronal fluid |
| Contusion | fluid-sensitive sagittal/coronal/axial |
| Fracture | structural sagittal/coronal/axial |

These priors were defined from general knee MRI anatomy and sequence sensitivity, not from target-specific B5/B7/B7.1 development AUCs.

### Why no fixed in-plane quadrant prior

The preprocessing does not certify canonical left/right or anterior/posterior pixel orientation across every selected series. B8 therefore does not hard-code a quadrant as medial, lateral, anterior or posterior. The fixed prior is uniform across the four in-plane regions, while region embeddings and pathology queries learn spatial preferences from weak supervision.

## Initialization and frozen supervision

B8 initialized from:

```text
runs/b7_1_full_coverage/b7_model.pt
```

The loader enforced:

```text
implementation variant          b7_b5_init_b6_asymmetric_weak_v1
experiment name                 B7.1_full_coverage
completed epochs                4
batches per epoch               1560
training studies                3120
training usable cells           14123
gold studies in gradient        0
gold early stopping             0
```

B8 kept B6 v1.2.1 unchanged:

```text
active weakly labelled studies = 3120
usable cells                   = 14123
positive cells                  = 6871
negative cells                  = 7252
```

The asymmetric global policy remained positive target `0.85` / weight `0.50`, negative target `0.05` / weight `1.00`, with uncertain and unmentioned cells ignored.

## Completed real-data training

| Epoch | Loss | Batches | Study draws | Active cells | Positive | Negative | Seconds | Budget limited |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.6707552306` | 1560 | 3120 | 14123 | 6871 | 7252 | 2239.67 | false |
| 2 | `0.6445401128` | 1560 | 3120 | 14123 | 6871 | 7252 | 3071.56 | false |
| 3 | `0.6186956850` | 1560 | 3120 | 14123 | 6871 | 7252 | 3256.41 | false |
| 4 | `0.5997290100` | 1560 | 3120 | 14123 | 6871 | 7252 | 2724.49 | false |

Totals:

```text
completed epochs          4
batches                 6240
study draws            12480
nominal corpus passes      4.0
training seconds        11292.13
budget limited            false
checkpoint               runs/b8_spatial_anatomy/b8_model.pt
```

Optimization was stable and monotonic. The completed gold evaluation confirms that the lower weak-training loss did not translate into higher development macro AUC.

## Gold development evaluation

B8-v1 achieved:

```text
macro AUC = 0.5300962807
95% CI   = [0.4723014866, 0.5867732651]
n         = 58
bootstrap = 5000/5000 usable
```

Per-target AUC:

| Target | B8 AUC | B7.1 AUC | B8 - B7.1 |
|---|---:|---:|---:|
| ACL | `0.4767156863` | `0.5159313725` | `-0.0392156863` |
| MCL | `0.4739229025` | `0.4693877551` | `+0.0045351474` |
| Medial Meniscus | `0.6045673077` | `0.5841346154` | `+0.0204326923` |
| Lateral Meniscus | `0.5801242236` | `0.5950310559` | `-0.0149068323` |
| Medial OA | `0.5317829457` | `0.4604651163` | `+0.0713178295` |
| Lateral OA | `0.4100580271` | `0.5764023211` | `-0.1663442940` |
| PF OA | `0.5135135135` | `0.5817245817` | `-0.0682110682` |
| Effusion | `0.5788819876` | `0.6484472050` | `-0.0695652174` |
| Synovitis | `0.5818399044` | `0.6654719235` | `-0.0836320191` |
| Baker's | `0.5181159420` | `0.5452898551` | `-0.0271739130` |
| Contusion | `0.5249662618` | `0.5398110661` | `-0.0148448043` |
| Fracture | `0.5666666667` | `0.5916666667` | `-0.0250000000` |

B8 improved only 3 of the 12 target point estimates and declined on 9 of 12. These target differences are descriptive only; they must not be used to construct target-specific B7.1/B8 winners on the same 58 studies.

## Paired comparison: B7.1 -> B8

Using A=B7.1 and B=B8:

```text
B7.1 macro AUC       = 0.5644802945
B8 macro AUC         = 0.5300962807
point difference     = -0.0343840138
median difference    = -0.0335501423
95% paired CI        = [-0.0900453633, +0.0223997827]
P(B8 > B7.1)         = 0.1156
valid replicates     = 5000/5000
```

The paired interval crosses zero, so a statistically definitive inferiority claim is not made. However, the point estimate and paired bootstrap direction both strongly favor retaining B7.1: `88.44%` of paired replicates favor B7.1 over B8.

## Decision

**Reject B8-v1 as a replacement for B7.1.** Keep B7.1 as the main standalone development model.

Do not respond to this result by searching spatial grid sizes, target-specific anatomy priors, prior strengths, epoch counts, B7.1/B8 target-specific winners, or ensemble weights on the same 58 studies. The B8 spatial-prior branch is closed for this campaign unless a future experiment is independently motivated and predeclared for reasons other than these target-level gold outcomes.

## Development caveat

B8 was designed after prior development results on the same 58 studies. Its score is therefore an additional development estimate, not independent validation. Gold labels did not enter B8 optimization or early stopping.
