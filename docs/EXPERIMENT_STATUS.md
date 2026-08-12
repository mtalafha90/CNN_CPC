# Experiment status

**Snapshot:** 2026-08-12  
**Package:** `0.26.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has supported repeated sequential development decisions and is therefore a **development/model-selection set rather than independent validation**. The frozen weak-v2 surface measures B6 teacher agreement and is not an expert-validation surface.

## Current headline

- **B16 is the current reused-gold development champion by the predeclared global point-estimate rule:** macro AUC `0.6349770242`, 95% CI `[0.5854729266,0.6830266155]`.
- Historical B13 reference: `0.6293565948`.
- B16-B13 raw delta: `+0.0056204295`; paired median `+0.0050711608`; 95% paired CI `[-0.0395927864,+0.0519351407]`; `P(B16>B13)=0.5828`.
- **B16 is retained by the frozen rule but superiority over B13 is not established.**
- B15 weak-v2 AUC was `0.7319060415`, but reused-gold AUC was `0.6209002783`; stronger B6 agreement did not transfer globally.
- The B6/B15 diagnostic found a coverage-conditioned B6 AUC of `0.7736374158` on 251/696 cells and a full-surface state-only baseline of `0.7024597743`.
- On 55 high-confidence B6-wrong gold cells, B15 did not move systematically toward B6 errors; 63.6% moved toward expert truth.
- **B17 is implemented and frozen before its first gold look.** It freezes the completed B16 report-aligned encoder, trains only the unchanged hierarchy/head for five exact full B6 passes, and forbids added smoothing, robust loss, gold early stopping, gold checkpoint selection, and weak-v2 gating.
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
| **B13** | **one learned token per series + ImageNet ConvNeXt** | **`0.6293565948` gold** | **retained historical champion/reference** |
| B14 | full `K x 16` slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy | `0.7319060415` weak-v2; `0.6209002783` gold | teacher gain; no global gold gain |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | **current champion by frozen point-estimate rule; superiority unresolved** |
| **B17** | **freeze B16 report-aligned encoder; train hierarchy/head only for five fixed full B6 passes** | **not run** | **implemented / predeclared** |

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

Key results:

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

These pooled middle-state rates are not universal training targets. Target-level behavior is heterogeneous; notably, Effusion unmentioned cells were more often gold-positive than explicit-positive cells in the 58-study audit. B16 therefore used full report semantics rather than converting uncertain/unmentioned to fixed probabilities.

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

The B13 slice-exposure audit showed near-complete evaluation exposure and rejected slice-count undersampling as the primary bottleneck. Do not launch a gold-driven 24/32/48-slice sweep from this surface.

## B13 retained historical result

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]

B13 vs B12
median difference +0.0638674720
95% paired CI     [+0.0127183837,+0.1144643292]
P(B13 > B12)       0.9920

B13 vs B7.1
median difference +0.0652260946
95% paired CI     [+0.0039768779,+0.1266069220]
P(B13 > B7.1)      0.9808
```

## B15 completed result

B15 used ImageNet -> same-study knee-MRI contrastive SSL. Its frozen weak-v2 comparison was:

```text
B13-v2 control macro AUC  0.5652498118
B15 macro AUC             0.7319060415
raw B15-control          +0.1666562297
paired median            +0.1675245839
95% paired CI            [+0.1124433208,+0.2165156305]
P(B15 > control)          1.0000
```

But the one-look reused-gold confirmation was:

```text
B15 macro AUC      0.6209002783
95% CI            [0.5706720829,0.6675892903]
B13 macro AUC      0.6293565948
raw B15-B13       -0.0084563164
```

Thus weak-v2 is retained only as a B6-teacher-agreement diagnostic, not as a surrogate selector for expert AUC.

## B16 completed result

B16 added full-report semantic alignment on all 4,349 non-gold MRI/report pairs before returning to the full B13/B6 surface.

Representation stage:

```text
studies / epoch          4349
eligible MRI series     24035
2.5D examples / epoch   48070
epochs                       4
loss epoch 1            3.8958491301
loss epoch 4            2.5218941658
full coverage           true for all epochs
budget limited          false for all epochs
```

Downstream stage:

```text
studies / epoch          3120
B6 cells / epoch        14123
positive / negative    6871 / 7252
series / epoch          17475
batches / epoch          1560
epochs                       4
loss epoch 1            0.7379701049
loss epoch 4            0.5675074643
```

One-look reused-gold result:

```text
B16 macro AUC      0.6349770242
95% CI            [0.5854729266,0.6830266155]
B13 macro AUC      0.6293565948
raw B16-B13       +0.0056204295
paired median     +0.0050711608
95% paired CI     [-0.0395927864,+0.0519351407]
P(B16 > B13)       0.5828
```

B16 is the current development champion only because the predeclared rule used the global point estimate. The paired evidence leaves B13 and B16 statistically unresolved.

## B17 predeclared frozen protocol

Canonical protocol: [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md).

B17 starts from:

```text
runs/b16_full_report/report_ssl/b16_report_encoder.pt
```

and enforces:

```text
encoder requires_grad           false for every parameter
encoder optimizer membership    false
encoder training mode           false
encoder LR                      0
encoder SHA-256                 unchanged after every epoch
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

The hierarchy/head construction and DataLoader reuse B16's seed offsets to keep the random start and first four shuffle streams aligned as closely as possible with B16.

B17 deliberately changes both encoder optimization and fixed training length (`4 -> 5`) relative to B16. It is therefore a frozen-short-training protocol test rather than a pure one-variable causal freezing ablation.

Gold evaluation is authorized only after all five epochs report exact full study/series coverage, no budget truncation, no encoder gradients, encoder eval mode, and an unchanged encoder fingerprint. The evaluator compares directly against the frozen B16 gold prediction file and refuses a reference that does not reproduce `0.6349770242`.

## Governance

Current rules:

```text
B16: closed; no post-gold tuning
B17: no epoch 6 from gold
B17: no label smoothing / ELR / SCE added after seeing gold
B17: no head-LR tuning from gold
B17: no target-specific B16/B17 winner mixing
no regeneration of weak-v2 from model outcomes
no universal gold-derived uncertain/unmentioned pseudo-labels
```

The hidden competition evaluation remains the next genuinely independent performance signal.
