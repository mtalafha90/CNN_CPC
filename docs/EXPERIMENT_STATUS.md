# Experiment status

**Snapshot:** 2026-08-10  
**Package:** `0.16.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study set has supported repeated sequential development decisions. It is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Retained standalone champion:** **B7.1 full-corpus weak supervision**, macro AUC `0.5644802945`.
- B7.1 latest 5,000-replicate CI: `[0.5052996126, 0.6214295635]`.
- **B8 rejected:** macro AUC `0.5300962807`; paired `P(B8>B7.1)=0.1156`.
- **B9 rejected:** macro AUC `0.5334962669`; paired median `(B9-B7.1)=-0.0302397961`, `P(B9>B7.1)=0.0562`.
- **B10 rejected globally:** macro AUC `0.5523982721`, 95% CI `[0.4935605888,0.6091548645]`; paired median `(B10-B7.1)=-0.0121030792`, 95% CI `[-0.0507382525,+0.0250750953]`, `P(B10>B7.1)=0.2706`.
- B10 was much closer to B7.1 than B8/B9 and improved several individual targets, but target-specific winner selection is prohibited on the reused 58-study set.
- **B11 teacher-student completion is implemented and predeclared.** It retains B7.1 as the student recipe and uses a frozen B7.1 teacher to add conservative pseudo-supervision only to B6-unsupervised cells.

## Completed measured experiments

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL representation + unchanged B4 probe | `0.5243650851` | retained representation baseline |
| B6 | multilingual structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels, limited epoch coverage | `0.5397724412` | coverage ablation |
| **B7.1** | **same B7 recipe with full 3,120-study coverage** | **`0.5644802945`** | **retained champion** |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial tokens + fixed anatomy priors | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | plane-specific in-plane physical-scale normalization | `0.5523982721` | rejected globally |

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

B6 remains frozen:

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

B7.1 remains the reference architecture, routing and preprocessing for B11.

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

Decision: retain historical B7.1 dual routing; do not tune target-specific restoration from gold outcomes.

### B10 physical scale

The label-free audit found complete in-plane geometry for all 15,468 selected weak-training streams and froze approximately `0.3125 mm/pixel` with `160 mm` physical FOV for all planes. Training followed B7.1 closely and completed four full epochs.

```text
B10 macro AUC       0.5523982721
95% CI             [0.4935605888,0.6091548645]
B7.1 macro AUC      0.5644802945
median(B10-B7.1)   -0.0121030792
95% paired CI      [-0.0507382525,+0.0250750953]
P(B10>B7.1)         0.2706
```

Decision: reject B10-v1 as the global replacement. Do not construct per-target B7.1/B10 winners from the same 58 studies.

## Active predeclared experiment: B11

**Hypothesis:** the largest remaining unused resource is supervision. B6 uses only 14,123 of 52,188 report-only study/target cells and completely excludes 1,229 studies from direct weak training. B11 tests whether a completed B7.1 MRI teacher can conservatively supervise a subset of these currently unused cells.

### Frozen pseudo-label policy

Pseudo labels are generated only where B6 weight is zero. For the completed B7.1 teacher under TTA offsets `[-1,0,1]`, a cell is accepted only when:

```text
teacher mean >= 0.90 OR teacher mean <= 0.10
TTA probability range <= 0.05
```

The teacher mean probability is used as the soft target. Base pseudo weight is `0.20`, but total pseudo weight mass for each target is capped at 25% of that target's original B6 base-weight mass. B6 cells are never overwritten.

Pre-training viability gates:

```text
>= 500 pseudo cells overall
>= 25 pseudo cells per target
```

The student starts from the same B5 checkpoint as B7.1, not from the teacher. Historical B7.1 routing, legacy resize, architecture, optimizer, augmentations, four epochs, gold TTA and bootstrap policy remain unchanged. B6-derived target-balance multipliers remain frozen.

Implementation:

```text
src/rsna_knee/b11_pseudo_labels.py
src/rsna_knee/b11_teacher_student.py
src/rsna_knee/b11_gold_eval.py
configs/b11_teacher_student.yaml
tests/test_b11_teacher_student.py
docs/B11_TEACHER_STUDENT.md
```

## B11 execution order

1. Update/install package 0.16.0 and run tests.
2. Generate B11 teacher predictions/pseudo labels from `runs/b7_1_full_coverage/b7_model.pt`.
3. Inspect `pseudo_summary.json` and `pseudo_policy.json` before any student training.
4. If viability gates pass, train the B11 student for four full passes over the combined-supervision pool.
5. Inspect all four histories; do not evaluate gold unless every epoch reports `full_coverage=true` and `budget_limited=false`.
6. Run one frozen gold evaluation.
7. Compare B7.1 -> B11 with 5,000 aligned study-level bootstrap replicates.
8. Do not tune pseudo thresholds, weights, target-specific acceptance, model winners or ensemble weights from the reused 58 gold labels.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
