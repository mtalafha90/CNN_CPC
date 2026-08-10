# RSNA Knee Abnormality Detection — Public Code Methodology Review

**Repository:** `mtalafha90/CNN_CPC`  
**Snapshot:** 2026-08-10  
**Purpose:** methodology context and repository-measured development evidence, not a leaderboard claim.

> Canonical measured results are in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md). **B7.1 is the current leader at macro AUC `0.5644802945`; B8 spatial anatomy is rejected at `0.5300962807`; B9 strict semantic sequence routing is the active predeclared experiment.**

## Problem structure

```text
4,407 training studies
58 fully gold-labelled studies
4,349 report-only studies
24,371 series rows
12 study-level targets
primary metric: macro ROC AUC
```

This is a weak/semi-supervised multi-sequence MRI problem with an extremely small trusted development set.

## Main lessons from the repository experiments

### Reports are useful as training supervision, not inference inputs

The test path is MRI-only. The first fold-safe report teacher reached only `0.49245` macro OOF and was rejected as a general 12-target teacher.

The useful report paths became:

- **B5:** image-report representation alignment using competition reports only;
- **B6:** conservative structured positive/negated/uncertain/unmentioned states;
- **B7/B7.1/B9:** direct MRI training from frozen B6 target cells.

### Unmentioned is not negative

B6 v1.2.1 training export:

```text
active weakly labelled studies  3120
usable cells                   14123
positive cells                  6871
negative cells                  7252
```

B7-family weak supervision uses:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

### Strong competition-only MRI representation learning helped

B1 strong SSL improved the random baseline from `0.4762536432` to `0.5030284974`. B4's frozen classical probe reached `0.5137567459`, and B5 report-aligned representation reached `0.5243650851` under the same probe.

### Direct weak supervision and corpus coverage mattered most so far

B7-v1 reached:

```text
0.5397724412
```

but saw only about 1.28 nominal corpus passes over four epochs.

B7.1 changed only the batch cap so each epoch covered all 3,120 active weakly labelled studies:

```text
B7.1 = 0.5644802945
95% CI = [0.5052432984, 0.6229422178]
```

This is the current strongest standalone development point estimate.

### More spatial tokens did not help

B8 retained a 2x2 spatial ConvNeXt grid per slice and increased MRI memory from 96 to 384 tokens/study. Training was stable, but development AUC fell:

```text
B8                  0.5300962807
B7.1                0.5644802945
median(B8-B7.1)    -0.0335501423
P(B8 > B7.1)        0.1156
```

The B8 spatial-prior branch is therefore closed to post-hoc tuning.

## B9: exact sequence semantics before more architecture changes

A label-free `train_series.csv` audit found that the historical dual-stream router sometimes placed a series in the opposite semantic slot when a plane had multiple acquisitions of only one contrast class.

Full training metadata audit:

| Stream | Historical selected | Strict selected | Wrong-slot assignments removed |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 4,150 | 251 |
| sagittal_structural | 4,294 | 4,266 | 28 |
| coronal_fluid | 4,250 | 4,248 | 2 |
| coronal_structural | 3,440 | 3,406 | 34 |
| axial_fluid | 4,407 | 4,407 | 0 |
| axial_structural | 1,094 | 857 | 237 |
| **Total** | **21,886** | **21,334** | **552** |

Thus `2.52%` of historically selected training streams contradict the supplied `Fluid_Sensitive` slot meaning.

The three provided test studies contain one analogous false sagittal-fluid assignment. Historical routing selects 14 streams; strict routing selects 13 semantically valid streams.

B9 therefore uses:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> masked missing stream
```

Everything else returns to the B7.1 recipe: B5 initialization, frozen B6 supervision, global-token KneeMILNet architecture, four full corpus passes, same optimizer/augmentation/TTA, and no gold-gradient/early-stopping use.

This is a cleaner next test than another attention variant because the hypothesis comes from acquisition metadata consistency rather than target-level gold outcomes.

## Current measured ladder

| Candidate | Macro AUC |
|---|---:|
| B0 | `0.4763` |
| B1 | `0.5030` |
| B4 | `0.5138` |
| B5 | `0.524365` |
| B7-v1 | `0.539772` |
| **B7.1** | **`0.564480`** |
| B5+B7.1 rank | `0.554014` |
| B8 | `0.530096` |
| B9 | pending |

## Validation discipline

The same 58 gold studies have supported repeated method decisions. The campaign must therefore be described as development/model-selection CV.

Do not:

- select target-specific winners;
- optimize blend weights;
- retune B6 parser rules/weak-label weights from gold outcomes;
- tune B9 target-specific routing from per-target AUCs;
- describe the best development score as a hidden-test or leaderboard guarantee.

See [`docs/B9_STRICT_ROUTING.md`](docs/B9_STRICT_ROUTING.md) for the frozen B9 protocol.
