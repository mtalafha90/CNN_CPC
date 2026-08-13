# Experiment status

**Snapshot:** 2026-08-13  
**Package:** `0.28.0`  
**Gold development/selection set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/model-selection surface, not independent validation**. B18--B20 deliberately consume it for one global checkpoint-selection statistic. The frozen weak-v2 surface measures B6 teacher agreement and is not an expert-validation surface.

## Current headline

- **B18, B19 and B20 are completed five-epoch frozen-encoder experiments.**
- B18 full-FOV selected epoch 2 at `0.6654496134`.
- B19 90% crop + cosine vignette selected epoch 3 at `0.6581308356`, but post-hoc Grad-CAM exposed a strong artificial vignette-boundary shortcut; **B19 is rejected as the spatial formulation**.
- B20 90% crop-only selected epoch 2 at `0.6671593555` and passed the local 3-study / 15-series inference/schema smoke test.
- The B20-B18 selected-statistic difference is only `+0.0017097421` and is **not independent evidence of superiority** because both use the same reused 58-study development/selection surface.
- Same-source Grad-CAM on one expert-positive effusion case showed B20 removes B19's synthetic-border shortcut but remains more diffuse than B18; therefore **B18 vs B20 remains unresolved**.
- The next useful internal analysis is a fixed multi-case CAM audit; the next genuinely independent predictive-performance signal is competition evaluation on data not used for selection.

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
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` gold | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` gold | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` gold | rejected |
| B10 | physical-scale normalization | `0.5523982721` gold | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | failed viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` gold | rejected globally |
| B12 | all real MRI series + full slice-token memory + B5 init | `0.5660915179` gold | historical reference |
| B12.1 | one learned token per series + B5 init | not run | implemented / skipped |
| **B13** | **one learned token per series + ImageNet ConvNeXt** | **`0.6293565948` gold** | unresolved high-performing tier |
| B14 | full `K x 16` slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no global gold gain |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | unresolved high-performing tier |
| **B17** | **freeze B16 report-aligned encoder; train hierarchy/head only for five fixed full B6 passes** | **`0.6425890153` gold** | historical fixed-epoch reference; superiority unresolved |
| **B18** | **same B17 five-epoch training; expert set selects one GLOBAL epoch; full FOV** | **epoch 2; `0.6654496134` selection only** | completed; B18/B20 unresolved |
| **B19** | **B18 recipe + 90% crop + cosine vignette** | **epoch 3; `0.6581308356` selection only** | **rejected spatial formulation; artificial border shortcut** |
| **B20** | **B18 recipe + 90% crop only; no vignette** | **epoch 2; `0.6671593555` selection only** | **completed; B18/B20 unresolved** |
| FINAL | B17-style frozen encoder + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## Frozen B6 supervision surface

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

The B6 gold audit gave sensitivity `0.9748`, specificity `0.6061`, positive precision `0.6905`, NPV `0.9639`, balanced accuracy `0.7904`, and coverage `0.3606`. These values characterize noisy/incomplete report-derived supervision and do **not** define a downstream AUC ceiling.

## B6 state/noise diagnostic

```text
coverage-conditioned high-confidence B6 macro AUC  0.7736374158
coverage                                              0.360632
full-surface state-only macro AUC                    0.7024597743
95% CI                                               [0.6537393397,0.7507506766]
high-confidence cells                                251
B6-correct / B6-wrong                                196 / 55
```

On the 55 B6-wrong cells, B15 did not systematically move toward B6 errors; 63.6% moved toward expert truth. The state baseline is therefore a **supervision-information reference**, not a teacher/student ceiling or a guaranteed MRI-extraction target.

## Frozen all-series surface

```text
B6-active studies        3120
eligible real series    17475
historical dual unique  15468
extra series             2007
max series / study         14
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## B17 completed result

Canonical record: [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md).

```text
training studies                3120
training series                17475
B6 cells                       14123
positive / negative           6871 / 7252
batches / epoch                1560
epochs                            5
encoder LR                        0
head LR                         1e-4
additional label smoothing        0
robust loss                     none
```

Encoder SHA:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

B17 fixed epoch-5 reused-gold result:

```text
B17 macro AUC      0.6425890153
95% CI            [0.5935606351,0.6887356582]
B16 macro AUC      0.6349770242
raw B17-B16       +0.0076119910
paired median     +0.0074330332
95% paired CI     [-0.0188853047,+0.0332991195]
P(B17 > B16)       0.7110
```

## B18 completed expert-selection result

Canonical record: [`B18_FISHER_SELECTION.md`](B18_FISHER_SELECTION.md).

B18 changed **only checkpoint selection** relative to B17. The five completed candidate epochs were:

```text
epoch 1  loss 0.7371836930  selection AUC 0.6187157061
epoch 2  loss 0.6336947483  selection AUC 0.6654496134  <- selected
epoch 3  loss 0.6087776578  selection AUC 0.6511148368
epoch 4  loss 0.5862506992  selection AUC 0.6394162186
epoch 5  loss 0.5667051629  selection AUC 0.6425890153
```

Because the expert set selected the checkpoint, `0.6654496134` is **not independent validation evidence**.

## B19/B20 spatial-focus ablation

B19 and B20 preserved the B18 training/selection contract and changed only the spatial input policy.

```text
B19: 90% centered crop -> resize -> cosine vignette
B20: 90% centered crop -> resize
```

Completed selection histories:

```text
              E1         E2         E3         E4         E5
B18        0.618716   0.665450   0.651115   0.639416   0.642589
B19        0.580216   0.624272   0.658131   0.636993   0.648569
B20        0.617730   0.667159   0.649215   0.657004   0.657782
```

Post-hoc same-source Grad-CAM on one expert-positive effusion case showed:

```text
B18 mask fraction   0.01256
B19 mask fraction   0.05899
B20 mask fraction   0.02938
```

The B19 CAM was dominated by the imposed vignette boundaries. B20 removed that synthetic-boundary shortcut, but B18 was more focal on the inspected case. Therefore B19 is rejected and B18 versus B20 remains unresolved pending broader localization auditing and independent predictive evaluation.

The first comparison also exposed a visualization-only mode bug: direct per-view probabilities were obtained without explicitly setting `model.eval()`, allowing dropout to affect the automatic view bookkeeping. The comparison code now enforces evaluation mode and checks direct-versus-Grad-CAM view-probability consistency. Training, checkpoint selection and submission inference were not affected.

## Local selected-checkpoint smoke tests

B18 and B20 selected checkpoints passed local inference/schema validation:

```text
test studies                   3
test series                   15
series / study                 5 / 5 / 5
TTA                            [-1,0,1]
sample columns match           true
sample UID order match         true
metadata repairs               0
```

The three-row local test is only a smoke surface and must not be confused with independent competition evaluation.

## Governance

```text
B16/B17: closed to post-gold retuning
B18: completed; epoch 2 frozen as selected checkpoint
B19: completed and rejected as spatial formulation because of artificial vignette shortcut
B20: completed; epoch 2 frozen as selected checkpoint
B18 vs B20: unresolved; do not claim B20 superiority from +0.00171 reused-selection difference
B18/B19/B20: expert labels never entered gradients
B18/B19/B20: selected expert scores are not validation evidence
B18/B19/B20: no target-specific epoch choice or target mixing
weak-v2: do not regenerate from outcomes
uncertain/unmentioned: no universal gold-derived pseudo-labels
FINAL all-data fit: remain deferred until the development/competition-evaluation decision is made
```

The next useful internal diagnostic is a pre-specified multi-case CAM audit. The next genuinely independent predictive-performance signal is competition evaluation on data not used to select B18/B20.
