# Experiment status

**Snapshot:** 2026-08-15  
**Package:** `0.29.0`  
**Active working model:** `B20_crop_only_joint_focus`  
**Canonical B20 checkpoint:** `runs/b20_crop_focus/b20_model.pt`  
**Canonical B20 epoch:** `2`  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/post-hoc surface, not independent validation**. Hidden competition evaluation remains the independent predictive-performance signal.

## Current headline

- **B20 remains the active working model** at reused-expert macro AUC `0.6671593555`.
- B21 weak-v2 improved but reused-gold acceptance failed; B22 showed no longer-training rescue.
- B23-v1 improved report-label coverage/ranking but failed its formal specificity gate (`0.5678 < 0.6061`). Formal B24 remains blocked/not run.
- B24X showed strong exploratory benefit from denser B23 supervision on the 692-study pilot.
- B24X-Density resolved the pilot mechanism: preserving B6 and filling only B6-silent cells reached `0.7147994969` versus `0.7116126450` for full B23; replacements/drops were not required.
- B25X scaled the experiment to the full leakage-safe 2,497-study weak-v2 training surface. Frozen weak-v2 scores were `B6=0.6723718048`, `Hybrid=0.7268784872`, `Fill=0.7308472686`.
- B25X diagnosis showed that the aggregate gain is overwhelmingly Synovitis-specific. Excluding Synovitis, Fill improves the other eleven targets by only `+0.0023982627` macro AUC.
- The next phase will develop the **current B20-family model**. DINOv2 replacement and soft-dense-label branches are not currently planned.

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
| B14 | full slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| B15 | knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no global gold gain |
| **B16** | B15 encoder -> full-report semantic alignment | `0.6349770242` gold | retained representation source |
| **B17** | frozen B16 encoder; fixed five passes | E5 `0.6425890153` gold | fixed-epoch reference |
| **B18** | full-FOV comparator | replay E2 `0.6655517376` gold | frozen comparator |
| **B19** | 90% crop + cosine vignette | E3 `0.6581308356` gold | rejected |
| **B20** | post-resize 90% crop only | E2 `0.6671593555` gold | **ACTIVE WORKING MODEL** |
| B21 | pre-resize crop | weak-v2 passed; gold `0.6573196516` vs B20 replay `0.6674066371` | closed / not promoted |
| B22 | B21 duration audit E1-E5 | best E2 `0.6574269018` | closed; no rescue |
| B23-v1 | local Qwen report labeller | AUC `0.8125164416`; specificity `0.5678` | **formal gate FAILED** |
| B24 formal | matched B6-vs-B23 supervision | not run | **blocked by B23 gate** |
| B24X | exploratory B6-vs-B23 pilot | `0.6148488366 -> 0.7116126450` weak-v2 | complete / no promotion |
| B24X-Density | B6 preserved + B23-only missing cells | `0.7147994969` weak-v2 | complete / no promotion |
| **B25X** | B6 vs ChatGPT Hybrid vs B6+Hybrid-fill | `0.6723718048 / 0.7268784872 / 0.7308472686` weak-v2 | complete / no promotion |
| FINAL | B17-style frozen encoder + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## B21/B22 retained conclusion

B21 weak-v2 development favored the pre-resize crop, but reused-gold acceptance did not:

```text
B20-v2 control weak-v2      0.7298727911
B21 weak-v2                 0.7410090411
weak delta                 +0.0111362500

B20 replay gold             0.6674066371
B21 gold                    0.6573196516
gold delta                 -0.0100869854
```

B22 extended the same formulation to E5; E2 remained best. More downstream epochs are therefore not the current priority.

## B23 formal status

```text
B6 state-only macro AUC     0.7024597743
B23 state-only macro AUC    0.8125164416
B6 specificity              0.6061
B23 specificity             0.5678
formal B23 gate             FAILED
canonical B23 holdout       does not exist
formal B24                  BLOCKED / NOT RUN
```

## B24X and B24X-Density

Pilot surface:

```text
shared studies                 692
B6 usable cells               3045
B23 usable cells              5697
B23-only added                2844
B6 dropped by full B23         192
```

Full B23 result:

```text
B6 control                    0.6148488366
Full B23                      0.7116126450
delta                        +0.0967638083
95% paired CI                [+0.0612014772,+0.1316174812]
```

Density preserved all 3,045 B6 cells and added only the 2,844 B23-only cells:

```text
Density                       0.7147994969
Density - B6                 +0.0999506603
95% paired CI                [+0.0642300469,+0.1348991590]
P(Density > B6)               1.0000

Full B23 - Density           -0.0031868519
95% paired CI                [-0.0099855349,+0.0034718378]
P(B23 > Density)              0.1799
```

The B24X mechanism conclusion is that **supervision recovery, not replacement of B6 decisions, explains almost the entire pilot gain**.

## B25X full hybrid experiment

### Surface

```text
training studies              2497
weak-v2 holdout studies        623
train/holdout overlap            0
possible training cells      29964

B6 usable                    11248  (37.5%)
Pure Hybrid                  20001  (66.8%)
B6 + Hybrid-fill             20790  (69.4%)
Hybrid-only additions         9542
Fill B6 drops                    0
Fill B6 overrides                0
```

Pure Hybrid also dropped 789 B6 cells and disagreed on 1,120 of 10,459 cells where both sources committed (`10.7%`).

### Fixed-E2 training

```text
Control  E1 0.7601064120  E2 0.6592396402  runtime 45m50s
Fill     E1 0.6799390770  E2 0.5913315904  runtime 46m43s
Hybrid   E1 0.6557762888  E2 0.5718413529  runtime 46m03s
```

### Frozen weak-v2

```text
B6 control          0.6723718048
Pure Hybrid         0.7268784872
B6 + Hybrid-fill    0.7308472686

Hybrid - B6         +0.0545066824
95% CI              [+0.0269870416,+0.0750180195]
P(>0)                1.0000

Fill - B6           +0.0584754637
95% CI              [+0.0301804537,+0.0814020218]
P(>0)                1.0000

Hybrid - Fill       -0.0039687813
95% CI              [-0.0137571379,+0.0058102163]
P(Hybrid > Fill)     0.2037
```

Fill is the safest of the two hybrid-supervision strategies on this development surface because it has the best point estimate and preserves every B6 decision.

## B25X target mechanism

The full macro gain is dominated by Synovitis:

```text
Synovitis AUC
B6       0.2370
Hybrid   0.9221
Fill     0.9123
```

Excluding Synovitis:

```text
11-target macro
B6       0.7119498792
Hybrid   0.7091330840
Fill     0.7143481419

Hybrid - B6   -0.0028167951
Fill - B6     +0.0023982627
```

Synovitis training supervision:

```text
B6                    322 positive / 13 negative
Hybrid-only additions  66 positive / 136 negative
Fill final             388 positive / 149 negative
```

The frozen weak-v2 Synovitis subset contains only 77 positives and 4 negatives, but leave-one-negative-out results remain strong:

```text
B6       0.177489 -- 0.259740
Hybrid   0.900433 -- 0.978355
Fill     0.887446 -- 0.961039
```

The defensible B25X conclusion is therefore **class-coverage repair for Synovitis, not broad 12-target superiority**.

## Current scientific position

1. B20 remains the active working model.
2. Longer training is not the current lever.
3. B23-v1 remains formally rejected.
4. B24X-Density shows that filling B6-silent cells is more important than replacing B6 labels.
5. B25X confirms that the effect can scale, but the measurable gain is highly target-specific and dominated by Synovitis negative-class recovery.
6. The next phase will develop the existing B20-family model with controlled one-variable experiments.
7. DINOv2 replacement and soft-dense-label branches are not currently planned.

## Governance

```text
B20: ACTIVE WORKING MODEL
B21/B22: CLOSED
B23-v1: formal gate FAILED
B24 formal: BLOCKED / NOT RUN
B24X: exploratory only; COMPLETE; NO GOLD / NO PROMOTION
B24X-Density: exploratory only; COMPLETE; NO GOLD / NO PROMOTION
B25X: exploratory only; COMPLETE; NO GOLD / NO PROMOTION
weak-v2: B6 teacher-agreement surface, not expert truth
58-study gold: reused/post-hoc development surface
hidden competition evaluation: independent predictive signal
```

## Canonical records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`WORKING_MODEL.md`](WORKING_MODEL.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`B25X_HYBRID_SUPERVISION.md`](B25X_HYBRID_SUPERVISION.md)
- [`VALIDATION.md`](VALIDATION.md)
