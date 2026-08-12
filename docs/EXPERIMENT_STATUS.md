# Experiment status

**Snapshot:** 2026-08-13  
**Package:** `0.28.0`  
**Gold development/selection set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/model-selection surface, not independent validation**. With B18 it is deliberately consumed for checkpoint selection. The frozen weak-v2 surface measures B6 teacher agreement and is not an expert-validation surface.

## Current headline

- **B13--B17 are now treated as one statistically unresolved high-performing development tier.**
- B17 (`0.6425890153`) remains the **reference checkpoint** because it has the largest reused-gold point estimate, not because superiority is established.
- B17-B16: raw `+0.0076119910`, paired median `+0.0074330332`, 95% paired CI `[-0.0188853047,+0.0332991195]`, `P(B17>B16)=0.7110`.
- B16-B13: raw `+0.0056204295`, paired 95% CI `[-0.0395927864,+0.0519351407]`, `P(B16>B13)=0.5828`.
- B6 full-state expert-ordering reference is `0.7024597743`; the numerical B6-state minus B17 difference is `0.0598707590`, but this is an information reference rather than a guaranteed MRI-extractable gap.
- **B18 is implemented / predeclared / not yet run.** It keeps the complete B17 training recipe and uses the 58 expert studies only to choose one global epoch among five fixed candidates.
- The final all-data fit is implemented but **deferred** while B18 is active.
- The next genuinely independent performance signal remains Kaggle hidden evaluation.

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
| **B17** | **freeze B16 report-aligned encoder; train hierarchy/head only for five fixed full B6 passes** | **`0.6425890153` gold** | **reference checkpoint; superiority unresolved** |
| **B18** | **same B17 five-epoch training; expert set selects one GLOBAL epoch** | **selection statistic only** | **implemented / predeclared** |
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

Pooled state truth rates remain descriptive only:

```text
positive       116 / 168 = 0.6905
negated          3 / 83  = 0.0361
uncertain       11 / 29  = 0.3793
unmentioned    110 / 416 = 0.2644
```

Target-level behavior is heterogeneous, so these pooled middle-state rates are not universal training targets.

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

The B13 slice-exposure audit showed near-complete evaluation exposure and rejected slice-count undersampling as the primary bottleneck.

## B13--B17 interpretation reset

Historical point estimates:

```text
B13  0.6293565948
B14  0.6197914249
B15  0.6209002783
B16  0.6349770242
B17  0.6425890153
```

B13 previously improved over B12/B7.1 with paired evidence. From B13 onward, repeated comparisons on the same 58 studies are too noisy and too reused to support a narrative of sequential proven gains. B16-B13 and B17-B16 both have paired intervals crossing zero. Therefore B17 is a **reference checkpoint**, not a demonstrated superior model.

## B17 completed result

Canonical record: [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md).

B17 froze the completed B16 report-aligned encoder and used exactly:

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

Encoder SHA remained unchanged:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

Training losses:

```text
epoch 1  0.7371836930
epoch 2  0.6336947483
epoch 3  0.6087776578
epoch 4  0.5862506992
epoch 5  0.5667051629
```

One-look reused-gold result:

```text
B17 macro AUC      0.6425890153
95% CI            [0.5935606351,0.6887356582]
B16 macro AUC      0.6349770242
raw B17-B16       +0.0076119910
paired median     +0.0074330332
95% paired CI     [-0.0188853047,+0.0332991195]
P(B17 > B16)       0.7110
```

## B18 predeclared expert-selection protocol

Canonical record: [`B18_FISHER_SELECTION.md`](B18_FISHER_SELECTION.md).

B18 changes **only checkpoint selection** relative to B17:

```text
train epoch 1 on B6 only -> evaluate 58 expert studies -> global macro AUC
train epoch 2 on B6 only -> evaluate 58 expert studies -> global macro AUC
train epoch 3 on B6 only -> evaluate 58 expert studies -> global macro AUC
train epoch 4 on B6 only -> evaluate 58 expert studies -> global macro AUC
train epoch 5 on B6 only -> evaluate 58 expert studies -> global macro AUC
select maximum global macro; numerical tie -> earliest epoch
```

Frozen constraints:

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
B6 training surface                   identical B17
additional generic smoothing          0
robust loss                           none
resolution / positions                224 / 16
TTA                                   [-1,0,1]
```

Because the expert set selects the checkpoint, the selected 58-study score is **not validation evidence** and must not be used to claim B18 improved over B17. Independent Kaggle evaluation is required.

Expected outputs:

```text
runs/b18_fisher_selection/candidates/epoch_1.pt ... epoch_5.pt
runs/b18_fisher_selection/selection_history.json
runs/b18_fisher_selection/selection.json
runs/b18_fisher_selection/b18_model.pt
```

## Governance

```text
B16/B17: closed to post-gold retuning
B13--B17: statistically unresolved development tier
B18: five fixed B6-only epochs; expert set selects one GLOBAL checkpoint
B18: expert labels never enter gradients
B18: selected expert score is not validation evidence
B18: no target-specific epoch choice or target mixing
B18: no smoothing/robust-loss/LR/architecture/resolution/TTA tuning from selection curve
weak-v2: do not regenerate from outcomes
uncertain/unmentioned: no universal gold-derived pseudo-labels
FINAL all-data fit: deferred until B18/development phase is closed
```

The next genuinely independent performance signal is Kaggle hidden evaluation.
