# Experiment status

**Snapshot:** 2026-08-15  
**Package:** `0.29.0`  
**Active working model:** `B20_crop_only_joint_focus`  
**Canonical B20 checkpoint:** `runs/b20_crop_focus/b20_model.pt`  
**Canonical B20 epoch:** `2`  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/model-selection surface, not independent validation**. Hidden competition evaluation remains the independent predictive-performance signal.

## Current headline

- **B20 remains the active working model.** Canonical expert macro AUC `0.6671593555` at epoch 2.
- B21 passed a leakage-safe B6 weak-v2 development comparison but failed its predeclared reused-gold acceptance look; it is closed/not promoted.
- B22 showed that extending B21 from E2 to E5 does not rescue performance; more downstream optimization is not the current priority.
- **B23-v1 has now been run and audited.** State-only macro AUC rose from `0.7024597743` (B6) to `0.8125164416`, and coverage rose from `0.3606` to `0.6365`, but specificity fell from `0.6061` to `0.5678`. The formal B23 gate therefore **FAILED**.
- Because the B23 gate failed, **no canonical B23 holdout was frozen and formal B24 remains blocked/not run**.
- **B24X exploratory matched supervision pilot completed.** On the frozen 623-study B6 weak-v2 holdout, B23/Qwen supervision scored `0.7116126450` versus `0.6148488366` for the matched B6 control, paired delta `+0.0967638083`, 95% CI `[+0.0612014772,+0.1316174812]`, `P=1.0000`.
- **B24X-Density training completed.** It preserves all 3,045 B6 cells and adds only 2,844 B23-only cells, for 5,889 supervised cells with zero B6 overrides/drops. Frozen weak-v2 evaluation is pending.
- B24X/B24X-Density are **exploratory only**. They do not use gold and cannot promote a model.

## Experiment ladder

| ID | Method | Macro AUC / evaluation | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` gold | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` gold | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` gold | historical reference |
| B2 | B1 with lower encoder LR | `0.4993244663` gold | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` gold | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` gold | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` gold | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels | `0.5397724412` gold | coverage ablation |
| B7.1 | full 3,120-study B7 coverage | `0.5644802945` gold | historical benchmark |
| B8 | spatial-token anatomy model | `0.5300962807` gold | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` gold | rejected |
| B10 | physical-scale normalization | `0.5523982721` gold | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | failed viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` gold | rejected globally |
| B12 | all real MRI series + full slice-token memory + B5 init | `0.5660915179` gold | historical reference |
| B12.1 | one learned token per series + B5 init | not run | implemented / skipped |
| **B13** | one learned token per series + ImageNet ConvNeXt | `0.6293565948` gold | historical high-performing tier |
| B14 | full `K x 16` slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| B15 | knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no global gold gain |
| **B16** | B15 encoder -> full-report semantic alignment | `0.6349770242` gold | retained representation source |
| **B17** | frozen B16 encoder; fixed five passes | E5 `0.6425890153` gold | fixed-epoch reference |
| **B18** | full-FOV comparator | replay E2 `0.6655517376` gold | frozen; nested audit complete |
| **B19** | 90% crop + cosine vignette | E3 `0.6581308356` gold | rejected: artificial border shortcut |
| **B20** | post-resize 90% crop only | E2 `0.6671593555` gold | **ACTIVE WORKING MODEL** |
| **B21 weak-v2** | pre-resize crop; leakage-safe matched development | control `0.7298727911`; B21 `0.7410090411`; delta `+0.0111362500` | weak-v2 passed |
| **B21 acceptance** | full B6 refit; fixed E2 | B21 `0.6573196516`; B20 replay `0.6674066371`; delta `-0.0100869854` | **not promoted** |
| **B22** | B21 pipeline retrained E1-E5 | best E2 `0.6574269018` gold | **closed; no duration rescue** |
| **B23-v1** | local Qwen report labeller | state-only AUC `0.8125164416`; coverage `0.6365`; specificity `0.5678` | **formal gate FAILED** |
| **B24 formal** | matched B6-vs-B23 supervision | not run | **blocked by B23 gate** |
| **B24X** | exploratory matched B6-vs-B23 pilot | weak-v2 control `0.6148488366`; B23 `0.7116126450`; delta `+0.0967638083` | completed exploratory evidence |
| **B24X-Density** | B6 preserved + B23-only missing cells | trained fixed E2 on 5,889 cells | training complete; weak-v2 pending |
| FINAL | B17-style frozen encoder + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## B18/B20 nested epoch-selection context

Primary cross-fitted audits:

```text
B18 selected epochs                 [2,2,2]
B18 OOF macro AUC                   0.6655517376076434
B18 measured selection optimism     0.0

B20 selected epochs                 [2,2,2]
B20 OOF macro AUC                   0.6671593555313430
B20 measured selection optimism     0.0
```

Strict historical-manifest sensitivity analyses showed that the broader reused-gold uncertainty is materially larger than the tiny B20-vs-B18 difference. B20 is retained as the clean historical knee-focused formulation, not because predictive superiority over B18 has been established.

## B21/B22 completed result

### B21 weak-v2 development

```text
B20-v2 control macro AUC        0.7298727911
B21-v1 macro AUC                0.7410090411
raw B21 - control              +0.0111362500
paired median                  +0.0109814529
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
```

### B21 reused-gold acceptance

```text
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired median                  -0.0095857726
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

B21 was not promoted.

### B22 duration audit

```text
Epoch   training loss   expert macro AUC
E1      0.7388751291    0.6135270850
E2      0.6381611442    0.6574269018  <- best
E3      0.6087977977    0.6387456622
E4      0.5890809184    0.6136783995
E5      0.5680555741    0.6282683534
```

Longer training lowers the weak-training objective while expert ranking worsens after E2. The campaign therefore moved to supervision/development-surface quality rather than more epochs.

## B23-v1 labeller audit — completed, formal gate failed

Pilot export:

```text
structured rows                 1290
training/non-gold rows          1232
gold studies available            58 / 58
gold rows used for training         0
usable pilot cells              9321 / 14784 = 63.0%
```

Audit summary:

```text
                         B6                 B23
state-only macro AUC     0.7024597743       0.8125164416
sensitivity              0.9748             0.9855
specificity              0.6061             0.5678
PPV                      0.6905             0.6667
NPV                      0.9639             0.9781
coverage                 0.3606             0.6365
usable gold cells        251                443
```

Paired state-only AUC difference:

```text
raw B23 - B6             +0.1100566673
paired median            +0.1095402088
paired 95% CI            [+0.0680786389,+0.1531882641]
P(B23 > B6)              1.0000
```

Predeclared gate condition violated:

```text
required specificity > 0.6061
observed specificity   0.5678
formal B23 gate        FAILED
```

The audit is descriptive/post-hoc because the 58-study expert set is not independent. The formal gate remains binding regardless of the favorable AUC/coverage result.

Consequences:

```text
B23-v1 formally adopted           no
B23 validation split frozen       no
formal B24 allowed                no
```

## B24X exploratory matched supervision — completed

B24X was created separately after the failed B23 gate. It cannot be used for formal B24 acceptance.

### Matched training surface

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

### Fixed-E2 training

```text
B6 control
E1 loss 0.8581187165
E2 loss 0.7132374823

B23 candidate
E1 loss 0.7599072829
E2 loss 0.6096711156
```

Both checkpoints passed the matched invariants: same 692-study order, same initial frozen encoder, same crop and same fixed E2 endpoint.

### Frozen weak-v2 evaluation

```text
training studies                    692
weak-v2 holdout studies             623
train/holdout overlap                 0
```

```text
B6 control       0.6148488366  [0.5856757959,0.6451316589]
B23/Qwen         0.7116126450  [0.6785972089,0.7435358854]
raw B23 - B6    +0.0967638083
paired median   +0.0963512743
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
valid bootstrap  4913/5000
```

This is strong exploratory cross-teacher evidence because the B23-supervised MRI model wins on B6's own weak surface. It remains teacher-agreement evidence only and does not promote B23/B24X.

Per-target deltas:

```text
Synovitis          +0.3344
PF OA              +0.2804
Lateral Meniscus   +0.2172
ACL                +0.1678
Contusion          +0.1497
Medial Meniscus    +0.0724
MCL                +0.0427
Medial OA          +0.0091
Lateral OA         -0.0137
Effusion           -0.0142
Baker's            -0.0184
Fracture           -0.0663
```

## B24X-Density — training complete, evaluation pending

Density keeps every B6 committed cell exactly and adds B23 only where B6 is silent.

```text
shared studies                 692
B6 cells preserved            3045
B23-only cells added           2844
final supervised cells         5889
B6 cells dropped                  0
B6 labels overridden              0
```

Training:

```text
E1 loss 0.7647414911
E2 loss 0.6197285242
checkpoint runs/b24x_density/density/b24x_density_model.pt
```

Next evaluation:

```text
same frozen weak-v2 holdout    623 studies
B6                              0.6148488366
Density                         pending
Full B23                        0.7116126450
```

The purpose is to determine whether the B24X gain is mostly caused by added supervision density or by B23's changed/dropped decisions.

## Frozen historical B6 supervision surface

```text
report-only studies       4349
active B6 studies         3120
inactive B6 studies       1229
usable B6 cells          14123
positive cells            6871
negative cells            7252
```

Frozen downstream policy:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

## Current scientific position

The current evidence supports the following narrow claims:

1. **Training duration is not the next lever.** B22 closes that path under the current recipe.
2. **Weak teacher agreement is not sufficient for formal promotion.** B21 demonstrated that directly.
3. **B23-v1 is formally rejected by its own gate**, because specificity is below B6.
4. **B23 supervision still appears to contain useful MRI-learning signal** in the exploratory matched B24X pilot.
5. **The mechanism of the B24X gain remains unresolved.** B24X-Density is the next controlled diagnostic.
6. **B20 remains active** until a future result has a valid promotion path and an independent enough evaluation signal.

## Immediate next steps

1. Evaluate `runs/b24x_density/density/b24x_density_model.pt` on the frozen 623-study weak-v2 holdout.
2. Reuse the already saved B6/full-B23 B24X predictions and compute B6-vs-Density and Density-vs-B23 paired comparisons.
3. Quantify what fraction of the `+0.0967638083` full-B23 point-estimate gain is captured by density alone.
4. Do not use gold for B24X/Density selection.
5. If B23 is revised, make it a new version with new provenance/cache and a new labeller audit; do not retune B23-v1 post hoc.
6. Resume formal B24 only after a future B23 version passes the formal gate and a valid holdout is frozen prospectively.

## Governance

```text
B20: ACTIVE WORKING MODEL; preserve checkpoint/preprocessing exactly
B21: closed; no second acceptance look
B22: closed; no longer-training rescue
B23-v1: formal gate FAILED; not adopted
B23 canonical holdout: does not exist
B24 formal: BLOCKED / NOT RUN
B24X: exploratory only; NO GOLD / NO PROMOTION
B24X-Density: exploratory only; training complete; weak-v2 pending; NO GOLD / NO PROMOTION
58-study gold surface: reused/post-hoc development surface, not independent validation
weak-v2: B6 teacher-agreement surface, not validated expert truth
no target-specific model mixing from B24X per-target results
hidden competition evaluation: independent predictive-performance signal
FINAL all-data expert-label fit: deferred
```

## Canonical records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`WORKING_MODEL.md`](WORKING_MODEL.md)
- [`B18_NESTED_EPOCH_AUDIT.md`](B18_NESTED_EPOCH_AUDIT.md)
- [`B19_JOINT_FOCUS.md`](B19_JOINT_FOCUS.md)
- [`B20_CROP_ONLY_FOCUS.md`](B20_CROP_ONLY_FOCUS.md)
- [`B20_NESTED_EPOCH_AUDIT.md`](B20_NESTED_EPOCH_AUDIT.md)
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md)
- [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md)
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md)
- [`B24_SUPERVISION_SOURCE.md`](B24_SUPERVISION_SOURCE.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`VALIDATION.md`](VALIDATION.md)
- [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md)
