# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.19.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Retained standalone champion:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`.
- B7.1 5,000-replicate CI: `[0.5052996126, 0.6214295635]`.
- **B8 rejected:** macro AUC `0.5300962807`; paired `P(B8>B7.1)=0.1156`.
- **B9 rejected:** macro AUC `0.5334962669`; paired `P(B9>B7.1)=0.0562`.
- **B10 rejected globally:** macro AUC `0.5523982721`; paired median `(B10-B7.1)=-0.0121030792`, 95% CI `[-0.0507382525,+0.0250750953]`, `P(B10>B7.1)=0.2706`.
- **B11-v1 stopped before training:** absolute teacher threshold failed the label-free pseudo viability gate.
- **B11.1 rejected globally:** macro AUC `0.5506902702`; paired median `(B11.1-B7.1)=-0.0126224565`, 95% CI `[-0.0487500119,+0.0195120537]`, `P(B11.1>B7.1)=0.2184`.
- **B12 variable-number-of-series modeling is implemented and predeclared.** Its label-free series audit is now the active next step.

## Experiment ladder

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
| B11.1 | calibration-aware target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **variable number of real MRI series** | pending | **implemented / series audit pending** |

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

## Closed branches

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

Decision: retain historical B7.1 dual routing when using the six-slot model.

### B10 physical scale

```text
macro AUC          0.5523982721
95% CI             [0.4935605888,0.6091548645]
median(B10-B7.1)  -0.0121030792
95% paired CI     [-0.0507382525,+0.0250750953]
P(B10>B7.1)        0.2706
```

Decision: reject B10-v1 globally.

### B11 teacher pseudo-label branch

B11-v1 failed its label-free viability gate because the absolute `0.10/0.90` rule yielded 4,794 pseudo-cells but only 23 positives, with zero accepted cells for Medial Meniscus and Synovitis.

B11.1 replaced absolute thresholds by stable target-wise 5/95% tails. Its pseudo audit passed with 3,656 pseudo-cells and 334 newly activated studies, and four full training epochs completed. The frozen gold result was:

```text
B11.1 macro AUC       0.5506902702
95% CI               [0.4917424630,0.6086153876]
B7.1 macro AUC        0.5644802945
median(B11.1-B7.1)   -0.0126224565
95% paired CI        [-0.0487500119,+0.0195120537]
P(B11.1>B7.1)         0.2184
```

Decision: reject B11.1 globally and close teacher-derived pseudo-label completion for now. Do not build target-wise B7.1/B11.1 winners on the reused 58-study set.

## Active experiment: B12 variable-number-of-series model

### Hypothesis

The six-slot B7.1 representation can discard repeated/additional acquisitions. B12 tests whether retaining every usable real series improves study-level pathology discrimination.

### Frozen controls

B12 returns to the exact original B7.1 supervision surface. It does **not** use B11/B11.1 pseudo-labels.

Unchanged:

```text
B5 encoder initialization
B6 v1.2.1 supervision only
3120 active training studies
14123 supervised cells
6871 positive / 7252 negative cells
B6-derived target balancing
legacy 224x224 resize
16 2.5D positions per series
B7.1 optimizer / LR / augmentation
batch size 2
4 full epochs
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

Single scientific change:

```text
six selected semantic slots
    -> every repaired Sagittal/Coronal/Axial series
    -> separate real series retained even when metadata are duplicated
    -> plane/fluid/fat categorical embeddings
    -> no series-rank embedding
    -> dynamic padding to batch maximum series count
```

There is no architecture-level maximum series count.

### Pre-training label-free audit

The audit is frozen on the 3,120 B6-active non-gold studies. Viability requires:

```text
zero studies without eligible series
zero historical selected series missing from B12
extra series >= 5% of historical unique selected-series count
>=10% of studies have at least one extra retained series
```

The exact variable-series mapping is SHA-256 signed. Training re-audits and refuses mapping drift.

See [`B12_VARIABLE_SERIES.md`](B12_VARIABLE_SERIES.md) for commands and integrity checks.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
