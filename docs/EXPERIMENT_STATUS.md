# Experiment status

**Snapshot:** 2026-08-13  
**Package:** `0.28.0`  
**Gold development/selection set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/model-selection surface, not independent validation**. With B18 it is deliberately consumed for checkpoint selection. The frozen weak-v2 surface measures B6 teacher agreement and is not an expert-validation surface.

## Current headline

- **B13--B17 remain one statistically unresolved high-performing development tier.**
- B17 (`0.6425890153`) remains the historical fixed-epoch reference checkpoint, not because superiority is established.
- **B18 is completed.** All five predeclared B17-equivalent frozen-encoder epochs were run and the global 58-study expert-selection rule chose **epoch 2**.
- B18 selection history: `0.6187157061`, **`0.6654496134`**, `0.6511148368`, `0.6394162186`, `0.6425890153` for epochs 1--5.
- The selected B18 score `0.6654496134` is **selection-only and not validation evidence**. The numerical difference from the fixed epoch-5 endpoint (`+0.0228605982`) must not be presented as an independent performance gain.
- Epoch 5 reproduces B17 (`0.6425890153`) to numerical precision, supporting the intended unchanged B17 training trajectory.
- The selected checkpoint `runs/b18_fisher_selection/b18_model.pt` passed the local 3-study / 15-series inference/schema smoke test.
- The next genuinely independent performance signal is competition evaluation on a dataset not used for B18 selection.

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
| **B18** | **same B17 five-epoch training; expert set selects one GLOBAL epoch** | **epoch 2 selected; `0.6654496134` selection statistic only** | **completed; awaiting independent evaluation** |
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

Frozen constraints all passed:

```text
expert labels in gradients            NO
expert studies                         58
expert target cells                   696
expert MRI series                     336
selection metric                      global 12-target macro AUC only
per-target epoch selection            forbidden
per-target selection values logged    no
selection bootstrap                   none
encoder                               frozen B16 report-aligned encoder
encoder SHA                           unchanged
B6 training surface                   identical B17
additional generic smoothing          0
robust loss                           none
resolution / positions                224 / 16
TTA                                   [-1,0,1]
full coverage every epoch             yes
full series coverage every epoch      yes
```

Selection decision:

```text
selected epoch                        2
selection statistic                   0.6654496134
fixed epoch-5/B17 endpoint            0.6425890153
numerical difference                  +0.0228605982
```

Because the expert set selected the checkpoint, `0.6654496134` is **not independent validation evidence** and the `+0.0228605982` difference is **not** an independently established B18 gain. Independent competition evaluation is required.

## Local selected-checkpoint smoke test

The selected checkpoint passed local inference/schema validation:

```text
checkpoint                     runs/b18_fisher_selection/b18_model.pt
selected epoch                 2
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
B13--B17: statistically unresolved development tier
B18: completed; epoch 2 frozen as selected checkpoint
B18: expert labels never entered gradients
B18: selected expert score is not validation evidence
B18: no target-specific epoch choice or target mixing
B18: no smoothing/robust-loss/LR/architecture/resolution/TTA tuning from selection curve
weak-v2: do not regenerate from outcomes
uncertain/unmentioned: no universal gold-derived pseudo-labels
FINAL all-data fit: remain deferred until the development/competition-evaluation decision is made
```

The next genuinely independent performance signal is competition evaluation on data not used to select B18.
