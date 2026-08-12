# Experiment status

**Snapshot:** 2026-08-12  
**Package:** `0.24.1`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has supported repeated sequential development decisions and is therefore a **development/model-selection set rather than independent validation**. The frozen weak-v2 surface is a separate teacher-agreement ranking surface and must not be interpreted as expert truth.

## Current headline

- **B13 remains the reused-gold development champion:** macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.
- **B14 is completed and rejected globally:** macro AUC `0.6197914249`, paired median `B14-B13=-0.0093726931`, 95% CI `[-0.0469823411,+0.0250137870]`, `P(B14>B13)=0.2924`.
- **B15 is completed.** It decisively improved frozen weak-v2 teacher agreement but did not improve the reused-gold global macro AUC.
- B13-v2 control weak-v2 AUC: `0.5652498118`.
- B15 weak-v2 AUC: `0.7319060415`; paired median `+0.1675245839`, 95% paired CI `[+0.1124433208,+0.2165156305]`, `P(B15>control)=1.0`; the predeclared gate passed.
- B15 one-look reused-gold confirmation: `0.6209002783`, 95% CI `[0.5706720829,0.6675892903]`; raw B15-B13 gold delta `-0.0084563164`.
- **B13 therefore remains retained. B15 is not a global gold improvement.**
- Full B13 slice exposure audit rejects slice-count undersampling as a primary bottleneck.
- Frozen weak holdout v2 remains fixed and must not be regenerated from model performance.
- **Next evidence-driven step:** audit B6 report states (`positive`, `negated`, `uncertain`, `unmentioned`) against expert truth before defining a new supervision experiment.

## Experiment ladder

| ID | Method | Macro AUC / evaluation | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` gold | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` gold | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` gold | retained reference |
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
| B12 | all real MRI series + full slice-token memory + B5 init | `0.5660915179` gold | retained historical reference |
| B12.1 | one learned token per series + B5 init | not run | implemented / skipped |
| **B13** | **one learned token per series + ImageNet ConvNeXt protocol** | **`0.6293565948` gold** | **RETAINED / DEVELOPMENT CHAMPION** |
| B14 | full `K x 16` slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| **B15** | **ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy** | **`0.7319060415` weak-v2; `0.6209002783` gold** | **weak gate passed; no global gold improvement** |

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

The B6 gold audit gave sensitivity `0.9748`, specificity `0.6061`, positive precision `0.6905`, NPV `0.9639`, balanced accuracy `0.7904`, and coverage `0.3606`. These values establish noisy/incomplete supervision but do **not** establish a numerical downstream macro-AUC ceiling.

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

## B13 retained result

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

Per-target B13 AUCs, descriptive only:

```text
ACL                0.4742647059
MCL                0.5555555556
Medial Meniscus    0.6093750000
Lateral Meniscus   0.6795031056
Medial OA          0.6279069767
Lateral OA         0.6189555126
PF OA              0.6177606178
Effusion           0.7677018634
Synovitis          0.7108721625
Baker's            0.7481884058
Contusion          0.5533063428
Fracture           0.5888888889
```

## B14 completed result

```text
B14 final B6 loss  0.5822778610
B13 final B6 loss  0.6132239342
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
raw B14-B13       -0.0095651699
median difference -0.0093726931
95% paired CI     [-0.0469823411,+0.0250137870]
P(B14 > B13)       0.2924
```

The paired CI crosses zero. B14 and B13 are statistically unresolved on the reused gold surface, but B14 provides no global advantage and has higher token-memory cost. No target-level B13/B14 hybrid is permitted.

## Completed B13 slice-exposure audit

```text
series audited/readable  17475 / 17475
slices/series median     30
slices/series p95        50
slices/series max        320

eval unique fraction     median 100.0% (p25 100.0%)
eval max skipped run     median 0.0 slices (p95 0.0)
training expected/view   median 87.0%
complete eval exposure   95.9%
eval run >=2 slices      3.9%
eval run >=3 slices      3.8%
```

Decision: **slice-count undersampling as the primary B13 bottleneck is rejected**.

## Frozen weak holdout v2

v1 is retained only as a historical superseded split. No B15/control model was trained on it.

The actual frozen v2 realization is:

```text
surface                   weak_b6_holdout_v2
status                    FROZEN before B15/control training
seed                      2026
active studies            3120
train studies             2497
holdout studies            623
actual holdout fraction   0.1996794872
train report groups       2426
holdout report groups      613
report-group overlap         0
all usable cells         14123
holdout usable cells      2875
holdout positive cells    1407
holdout negative cells    1468
gold studies                 0
uses gold labels          false
uses model predictions    false
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The rarest holdout class is Synovitis negative with four cells. Strict evaluation uses a study bootstrap and accepts a replicate only when **all 12 target AUCs are defined**.

## B15 MRI-domain SSL

The B15 SSL pool was deliberately stricter than the downstream weak-train surface:

```text
competition studies        4407
minus gold                    58
minus frozen v2 holdout      623
SSL studies                 3726
eligible SSL series        20534
active 2.5D examples/pass 41068
```

All four frozen SSL passes completed exactly:

```text
epoch 1 loss  2.7094607696
epoch 2 loss  2.5811344701
epoch 3 loss  2.5187829415
epoch 4 loss  2.4756854072
```

Every epoch had `1863/1863` batches, `3726/3726` study draws, `20534/20534` series instances, `full_coverage=true`, and `budget_limited=false`. No gold or weak-v2 holdout images entered SSL.

## Matched B13-v2 control and B15 downstream

Both models used the exact same downstream training surface:

```text
training studies            2497
real MRI series            13974
usable B6 cells            11248
positive cells              5464
negative cells              5784
batches/epoch               1249
epochs                         4
full study coverage          true
full series coverage         true
budget limited              false
```

Control final loss: `0.6622741637`.  
B15 final loss: `0.6065262400`.

Training loss is not a model-selection metric.

## B15 frozen weak-v2 results

### Matched B13-v2 control

```text
macro AUC              0.5652498118
95% CI                [0.5361620323,0.5924683768]
valid strict bootstrap 4913 / 5000
```

### B15

```text
macro AUC              0.7319060415
95% CI                [0.6903737595,0.7675416396]
valid strict bootstrap 4913 / 5000
```

### Predeclared paired gate

```text
raw B15-control        +0.1666562297
paired median          +0.1675245839
95% paired CI          [+0.1124433208,+0.2165156305]
P(B15 > control)        1.0000
valid paired bootstrap  4921 / 5000
passes gate             true
```

The gate required raw delta > 0, paired median > 0, and `P(B15>control)>=0.95`; all three passed. The weak surface measures **B6 teacher agreement, not expert truth**.

## B15 one-look reused-gold confirmation

After passing weak-v2, B15 received the single predeclared gold-development look:

```text
B15 macro AUC      0.6209002783
95% CI            [0.5706720829,0.6675892903]
58 studies
5000 / 5000 bootstrap replicates usable
raw B15-B13       -0.0084563164
```

Per-target B15 AUCs, descriptive only:

```text
ACL                0.5661764706
MCL                0.6462585034
Medial Meniscus    0.5973557692
Lateral Meniscus   0.6658385093
Medial OA          0.5085271318
Lateral OA         0.5551257253
PF OA              0.5997425997
Effusion           0.8012422360
Synovitis          0.6845878136
Baker's            0.6739130435
Contusion          0.5492577598
Fracture           0.6027777778
```

B15 was higher than B13 for several target point estimates and lower for others, but **no target-wise B13/B15 winner mixing is allowed**. The global point estimate is lower than B13, so B15 does not replace B13.

## Scientific interpretation after B15

B15 demonstrates that MRI-domain SSL can greatly improve ranking of the frozen report-derived weak labels. The absence of a corresponding expert-gold macro-AUC improvement indicates that stronger teacher agreement is not sufficient for stronger expert-label ranking in this campaign.

This is consistent with the supervision interface becoming a leading bottleneck candidate: sparse/noisy report states, instance-dependent report semantics, and the treatment of uncertain/unmentioned findings deserve direct audit before more capacity or SSL tuning.

The result does **not** prove B15 is intrinsically worse as an MRI representation, and it does not establish a numerical label-noise ceiling.

## Current decision / next stage

```text
B13 RETAIN / development champion
B14 REJECT globally
B15 weak-v2 gate PASS
B15 gold global improvement NOT ESTABLISHED
slice-count hypothesis REJECT
weak-v2 remains FROZEN
       |
       v
B6 report-state audit on already-reused gold
positive / negated / uncertain / unmentioned
       |
       v
quantify expert truth rates and coverage
       |
       v
only if justified: separately versioned/frozen supervision successor
       |
       v
Kaggle hidden signal remains independent evaluation
```

Do not tune B15 SSL epochs, learning rates, TTA, architecture, or target-specific mixtures from the gold confirmation. Do not regenerate v2 based on model outcomes. Do not blindly map unmentioned report states to negative.