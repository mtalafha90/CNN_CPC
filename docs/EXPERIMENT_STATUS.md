# Experiment status

**Snapshot:** 2026-08-10  
**Package:** `0.18.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study set has supported repeated sequential development decisions. It is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Retained standalone champion:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`.
- B7.1 latest 5,000-replicate CI: `[0.5052996126, 0.6214295635]`.
- **B8 rejected:** macro AUC `0.5300962807`; paired `P(B8>B7.1)=0.1156`.
- **B9 rejected:** macro AUC `0.5334962669`; paired `P(B9>B7.1)=0.0562`.
- **B10 rejected globally:** macro AUC `0.5523982721`; paired median `(B10-B7.1)=-0.0121030792`, 95% CI `[-0.0507382525,+0.0250750953]`, `P(B10>B7.1)=0.2706`.
- **B11-v1 stopped before training:** pseudo viability failed because the global absolute teacher threshold produced 4,794 pseudo-cells but only 23 positives and zero coverage for two targets.
- **B11.1 pseudo audit passed:** 3,656 calibration-aware quantile-tail pseudo-cells, 3,454 active studies, all 12 targets with both tails represented. **B11.1 student training is now the active experiment.**

## Completed measured experiments

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL representation + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | multilingual structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels, limited epoch coverage | `0.5397724412` | coverage ablation |
| **B7.1** | **same B7 recipe with full 3,120-study coverage** | **`0.5644802945`** | **retained champion** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial tokens + fixed anatomy priors | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | plane-specific in-plane physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute-threshold B7.1 teacher completion | n/a | stopped at viability gate |
| B11.1 | calibration-aware target-wise teacher tails | pending | pseudo audit passed; training active |

## Frozen B6 supervision surface

```text
report-only studies       4349
active B6 studies         3120
inactive B6 studies       1229
possible target cells    52188
usable B6 cells          14123
positive cells            6871
negative cells            7252
```

Frozen policy:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

## B7.1 retained champion

```text
epoch losses  0.752419 -> 0.665171 -> 0.639117 -> 0.612758
macro AUC     0.5644802945
```

B7.1 remains the reference architecture, routing, preprocessing and initialization contract for B11.1.

## B8 / B9 / B10 decisions

### B8 spatial tokens

```text
macro AUC          0.5300962807
median(B8-B7.1)   -0.0335501423
P(B8>B7.1)         0.1156
```

Decision: close the spatial-prior branch.

### B9 strict routing

```text
macro AUC          0.5334962669
median(B9-B7.1)   -0.0302397961
95% paired CI     [-0.0679414819,+0.0070291202]
P(B9>B7.1)         0.0562
```

Decision: retain historical B7.1 dual routing.

### B10 physical scale

```text
macro AUC          0.5523982721
95% CI             [0.4935605888,0.6091548645]
median(B10-B7.1)  -0.0121030792
95% paired CI     [-0.0507382525,+0.0250750953]
P(B10>B7.1)        0.2706
```

Decision: reject B10-v1 as global replacement. Do not select B7.1/B10 winners target by target from the reused 58 gold cases.

## B11-v1 pseudo audit — failed

Frozen absolute policy:

```text
teacher mean >= 0.90 OR <= 0.10
TTA range <= 0.05
pseudo base weight 0.20
pseudo mass cap 25% of B6 mass per target
```

Audit:

```text
pseudo cells              4794
combined active studies   4000
newly activated studies    880
pseudo positive cells       23
pseudo negative cells     4771
Medial Meniscus cells        0
Synovitis cells               0
Lateral OA cells             21
viability_passed           false
```

Decision: do not train B11-v1.

## B11.1 pseudo audit — passed

B11.1 was defined after a label-free calibration diagnostic showed that teacher probabilities are target-dependent while TTA predictions are generally stable.

Frozen per-target rule:

```text
bottom 5% teacher tail + TTA range <= 0.05 -> target 0.10
top    5% teacher tail + TTA range <= 0.05 -> target 0.90
pseudo base weight -> 0.10
pseudo mass cap    -> 15% of B6 mass per target
```

Audit result:

```text
B6 cells                  14123
pseudo cells                3656
combined cells              17779
B6 active studies            3120
combined active studies      3454
newly activated studies       334
pseudo low cells             1864
pseudo high cells            1792
viability_passed             true
```

Every target exceeds 100 pseudo-cells and has at least 50 cells in both tails. Synovitis alone is mass-capped, with scale `0.8242385787`; all other targets use pseudo cell weight `0.10`.

Frozen pseudo SHA-256:

```text
94f914f3548fab17f67ae0bf1906424bac850268c09ce5febede72b2ed7246b6
```

## Active experiment: B11.1 student

Single scientific change versus B7.1: added frozen B11.1 pseudo supervision on B6-unsupervised cells.

The student starts from the same B5 encoder initialization as B7.1, not from the B7.1 teacher. Historical routing, legacy resize, architecture, optimizer, B6 policy, B6-derived target balancing, augmentation and four-epoch schedule remain fixed.

Expected per full epoch:

```text
studies                    3454
batches                    1727
B6 cells                  14123
pseudo cells               3656
combined cells            17779
pseudo low cells           1864
pseudo high cells          1792
full_coverage              true
budget_limited             false
```

Gold evaluation is allowed only after all four epochs meet the full-coverage contract. Primary benchmark remains B7.1 `0.5644802945`, followed by the aligned 5,000-replicate paired bootstrap.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
