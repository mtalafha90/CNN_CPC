# Active working model

> **Decision — 2026-08-14:** **B20 remains the active working model.** B21-v1 passed weak-v2 but failed the predeclared full-data gold acceptance comparison. B22 then showed that extending the same pre-resize formulation from E2 to E5 does not rescue expert performance.

## Active model

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
implemented geometry   native MRI -> resize 224 -> center crop 90% -> resize 224
cosine/vignette mask   no
encoder                frozen historical B16 report-aligned encoder
canonical gold score   0.667159355531343
```

Historical B20 is preserved unchanged.

## Why B20 remains active

### B21-v1: corrected pre-resize crop

B21 changed the spatial ordering to:

```text
B20 historical: native MRI -> resize 224 -> crop 90% -> resize 224
B21-v1:         native MRI -> crop 90% -> percentile normalization -> resize 224
```

The leakage-safe weak-v2 development comparison favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21 pre-resize macro AUC        0.7410090411
raw B21 - control              +0.0111362500
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
```

But the frozen full-data expert acceptance comparison went in the opposite direction:

```text
B20 canonical macro AUC         0.6671593555
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired median                  -0.0095857726
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

Therefore:

```text
promotion_rule_passed              false
scientific_superiority_supported   false
```

B21-v1 is not promoted.

Canonical records:
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md)

### B22: five-epoch duration audit

B22 retrained the same pre-resize formulation from scratch for a coherent E1-E5 trajectory, using the same historical B16 frozen encoder, full 3,120-study B6 surface and five-epoch cosine schedule.

The training loss decreased monotonically:

```text
E1  0.7388751291
E2  0.6381611442
E3  0.6087977977
E4  0.5890809184
E5  0.5680555741
```

The expert-gold macro AUC did not follow that improvement:

```text
E1  0.6135270850
E2  0.6574269018  <- best
E3  0.6387456622
E4  0.6136783995
E5  0.6282683534
```

B22 E2 reproduced prior B21 E2 within `+0.0001072501`, well inside the predefined `0.005` tolerance. Historical B20 replay also passed its sanity tolerance.

Relative to the B22-audit B20 replay (`0.6679590975`):

```text
E2 raw delta  -0.0105321958   paired 95% CI [-0.0323859143,+0.0098214527]
E3 raw delta  -0.0292134353   paired 95% CI [-0.0523986045,-0.0087333144]
E4 raw delta  -0.0542806980   paired 95% CI [-0.0827548184,-0.0276651497]
E5 raw delta  -0.0396907441   paired 95% CI [-0.0654472831,-0.0162928843]
```

So longer downstream training does not rescue the pre-resize crop. E2 remains the observed optimum in this frozen-encoder weak-supervision regime.

Canonical record: [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md).

## Current interpretation

The combined B18/B20/B21/B22 evidence now points away from training duration as the next optimization lever.

Three observations matter:

1. **B18 and B20 both peak around E2** on the reused expert surface.
2. **B22 reproduces that E2 optimum for the pre-resize pipeline**, while E3-E5 continue lowering weak-training loss but degrade expert ranking.
3. **Weak-v2 favored B21 while expert gold did not**, showing that teacher agreement is not a sufficient surrogate for expert-pathology ranking for near-neighbor model selection.

The practical bottleneck is therefore more likely the **weak-label / development-selection problem** than insufficient training duration or the specific crop-order defect.

This is consistent with instance-dependent report supervision: continued optimization can improve agreement with report-derived targets while moving away from the expert-defined ranking objective.

## Historical B20/B18 audit context

```text
B20 cross-fitted epoch selections       [2,2,2]
B20 cross-fitted OOF macro AUC          0.6671593555313430
B20 measured epoch-selection optimism   0.0

B18 cross-fitted epoch selections       [2,2,2]
B18 replay OOF macro AUC                0.6655517376076434
B18 measured epoch-selection optimism   0.0
```

The B20-vs-B18 difference remains too small to establish predictive superiority on the repeatedly reused 58-study expert surface.

## Model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  pre-resize crop candidate; weak-v2 passed, gold acceptance failed; NOT PROMOTED
B22  five-epoch B21 duration audit; E2 best, no longer-training rescue; CLOSED
```

## Current optimization priority

Do **not** spend the next experiment on:

```text
more epochs
crop-order retry
crop-fraction sweep under the current normalization order
stronger optimization of B6 labels
target-wise B20/B21/B22 mixing
another gold-guided epoch search
```

The next campaign should address the development signal itself. Priority directions are:

```text
1. audit how well weak-v2 model deltas predict expert-gold model deltas;
2. improve the pathology-development surface / weak-label quality;
3. if expert-labelled expansion is feasible, add genuinely new expert cases;
4. otherwise predeclare a very small number of future hypotheses before any reused-gold audit;
5. only after that revisit architecture, encoder adaptation, FOV or routing changes.
```

## Governance

- Keep historical B20 unchanged as the working checkpoint.
- B21-v1 is closed and not promoted.
- B22 is closed as an exploratory duration diagnostic.
- Do not run another B21-v1 acceptance look.
- Do not choose a B22 epoch for production from the reused 58-study trajectory.
- Do not build target-wise mixtures from B20/B21/B22 expert results.
- Do not treat weak-v2 teacher agreement as a sufficient proxy for expert truth.
- Preserve B21/B22 artifacts as controlled negative/diagnostic results.
- The 58 expert studies remain a repeatedly reused development surface, not pristine independent validation.
- Hidden competition evaluation remains the independent predictive-performance signal.
