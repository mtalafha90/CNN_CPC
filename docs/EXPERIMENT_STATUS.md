# Experiment status

**Snapshot:** 2026-08-13  
**Package:** `0.26.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has supported repeated sequential development decisions and is therefore a **development/model-selection set rather than independent validation**. The frozen weak-v2 surface measures B6 teacher agreement and is not an expert-validation surface.

## Current headline

- **B17 is the current reused-gold development champion by the predeclared global point-estimate rule:** macro AUC `0.6425890153`, 95% CI `[0.5935606351,0.6887356582]`.
- B16 reference: `0.6349770242`.
- B17-B16 raw delta: `+0.0076119910`; paired median `+0.0074330332`; 95% paired CI `[-0.0188853047,+0.0332991195]`; `P(B17>B16)=0.7110`.
- **B17 is retained by the frozen rule but superiority over B16 is not established.**
- B17 froze the completed B16 report-aligned encoder for all five full downstream epochs; encoder SHA-256 remained exactly unchanged.
- B15 weak-v2 AUC was `0.7319060415`, but reused-gold AUC was `0.6209002783`; stronger B6 agreement did not transfer globally.
- The B6/B15 diagnostic found a coverage-conditioned B6 AUC of `0.7736374158` on 251/696 cells and a full-surface state-only baseline of `0.7024597743`.
- On 55 high-confidence B6-wrong gold cells, B15 did not move systematically toward B6 errors; 63.6% moved toward expert truth.
- B16 and B17 are now closed to post-gold tuning.
- The next genuinely independent performance signal remains the hidden Kaggle evaluation.

## Experiment ladder

| ID | Method | Macro AUC / evaluation | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` gold | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` gold | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` gold | retained historical reference |
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
| **B13** | **one learned token per series + ImageNet ConvNeXt** | **`0.6293565948` gold** | historical champion/reference |
| B14 | full `K x 16` slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy | `0.7319060415` weak-v2; `0.6209002783` gold | teacher gain; no global gold gain |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | historical champion/reference; unresolved with B17 |
| **B17** | **freeze B16 report-aligned encoder; train hierarchy/head only for five fixed full B6 passes** | **`0.6425890153` gold** | **current champion by frozen point-estimate rule; superiority unresolved** |

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

The B6 gold audit gave sensitivity `0.9748`, specificity `0.6061`, positive precision `0.6905`, NPV `0.9639`, balanced accuracy `0.7904`, and coverage `0.3606`. These numbers describe noisy/incomplete supervision and do **not** define a numerical downstream AUC ceiling.

## B6/B15 state/noise diagnostic

```text
coverage-conditioned high-confidence B6 macro AUC  0.7736374158
coverage                                              0.360632
full-surface state-only macro AUC                    0.7024597743
95% CI                                               [0.6537393397,0.7507506766]
high-confidence cells                                251
B6-correct / B6-wrong                                196 / 55
```

On the 55 B6-wrong cells, B15's mean movement toward B6 was negative and the point estimate moved slightly toward expert truth. `63.6%` moved toward truth. All predefined strong-evidence flags for B6-error imitation were false.

Pooled state truth rates:

```text
positive       116 / 168 = 0.6905
negated          3 / 83  = 0.0361
uncertain       11 / 29  = 0.3793
unmentioned    110 / 416 = 0.2644
```

These pooled middle-state rates are not universal training targets because target-level behavior is highly heterogeneous.

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

## B13 retained historical result

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
```

B13 previously improved over B12 and B7.1 with paired evidence, and remains an important historical reference.

## B15 completed result

```text
B13-v2 control weak-v2 AUC  0.5652498118
B15 weak-v2 AUC             0.7319060415
paired median              +0.1675245839
95% paired CI              [+0.1124433208,+0.2165156305]
P(B15 > control)            1.0000

B15 reused-gold AUC         0.6209002783
B13 reused-gold AUC         0.6293565948
raw B15-B13                -0.0084563164
```

Thus weak-v2 is retained only as a B6-teacher-agreement diagnostic, not as a surrogate selector for expert AUC.

## B16 completed result

B16 added full-report semantic alignment on all 4,349 non-gold MRI/report pairs before returning to the full B13/B6 surface.

```text
B16 macro AUC      0.6349770242
95% CI            [0.5854729266,0.6830266155]
B13 macro AUC      0.6293565948
raw B16-B13       +0.0056204295
paired median     +0.0050711608
95% paired CI     [-0.0395927864,+0.0519351407]
P(B16 > B13)       0.5828
```

B16 became champion by the predeclared global point-estimate rule, but remained statistically unresolved with B13.

## B17 completed frozen-encoder result

Canonical record: [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md).

B17 started from:

```text
runs/b16_full_report/report_ssl/b16_report_encoder.pt
```

and enforced:

```text
encoder requires_grad           false
encoder optimizer membership    false
encoder training mode           false
encoder LR                      0
runtime encoder checkpointing   false
head LR                         1e-4
epochs                          5 exact full passes
training studies                3120
training series                 17475
additional label smoothing      0
robust loss                     none
gold early stopping             none
gold checkpoint selection       none
weak-v2 gate                    none
```

The encoder fingerprint remained unchanged for all five epochs:

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

Each epoch had exactly 1,560 batches, 3,120 study draws, 14,123 active B6 cells, 6,871 positive cells, 7,252 negative cells, and 17,475 series, with full study/series coverage and no budget truncation.

One-look reused-gold result:

```text
B17 macro AUC      0.6425890153
95% CI            [0.5935606351,0.6887356582]
B16 macro AUC      0.6349770242
raw B17-B16       +0.0076119910
paired median     +0.0074330332
95% paired CI     [-0.0188853047,+0.0332991195]
P(B17 > B16)       0.7110
5000 / 5000 paired bootstrap replicates usable
```

B17 is therefore retained as the development champion by the predeclared point-estimate rule. The paired interval crosses zero, so true superiority is not established.

Descriptively, the largest B17-B16 target gains were Baker's (`+0.07065`), Synovitis (`+0.05854`), and Medial Meniscus (`+0.02885`). The largest declines were Contusion (`-0.03104`), Lateral Meniscus (`-0.02112`), and MCL (`-0.02041`). These target-level differences are descriptive only and cannot authorize target-wise winner mixing.

Because B17 changed both encoder optimization and fixed training length (`4 -> 5`) relative to B16, the result supports the frozen-short-training protocol but cannot attribute the gain solely to freezing.

## Governance

Current rules:

```text
B16: closed; no post-gold tuning
B17: closed; no epoch 6 from gold
B17: no label smoothing / ELR / SCE selected from gold
B17: no head-LR tuning from gold
B17: no target-specific B16/B17 winner mixing
no regeneration of weak-v2 from model outcomes
no universal gold-derived uncertain/unmentioned pseudo-labels
```

The hidden competition evaluation remains the next genuinely independent performance signal.
