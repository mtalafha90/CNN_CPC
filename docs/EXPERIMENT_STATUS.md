# Experiment status

**Snapshot:** 2026-08-12  
**Package:** `0.25.0`  
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
- B16 therefore used full report semantics, not gold-derived uncertain/unmentioned pseudo-labels.
- **No post-gold B16 tuning is permitted. The next genuinely independent signal is the hidden Kaggle evaluation.**

## Experiment ladder

| ID | Method | Macro AUC / evaluation | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 | `0.4762536432` gold | baseline |
| Report teacher | fold-safe rules + TF-IDF | `0.49245` gold | rejected |
| B1 | competition-only MRI SSL | `0.5030284974` gold | historical reference |
| B2 | lower encoder LR | `0.4993244663` gold | rejected |
| B3 | pathology-aware MIL | `0.4944652486` gold | rejected |
| B4 | frozen SSL + PCA/LR | `0.5137567459` gold | retained ablation |
| B5 | image-report SSL + B4 probe | `0.5243650851` gold | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7.1 | full B6 weak supervision | `0.5644802945` gold | historical benchmark |
| B8 | spatial-token anatomy | `0.5300962807` gold | rejected |
| B9 | strict semantic routing | `0.5334962669` gold | rejected |
| B10 | physical-scale normalization | `0.5523982721` gold | rejected globally |
| B11.1 | target-wise teacher tails | `0.5506902702` gold | rejected globally |
| B12 | all real MRI series | `0.5660915179` gold | historical reference |
| B12.1 | hierarchical one-token-per-series + B5 init | not run | implemented / skipped |
| **B13** | **ImageNet + hierarchical one-token-per-series** | **`0.6293565948` gold** | **retained historical champion/reference** |
| B14 | full `K x 16` tokens + B13 protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | weak gate passed; no gold improvement |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | **current champion by point-estimate rule; statistically unresolved vs B13** |

## Frozen B6 supervision

```text
report-only studies       4349
active B6 studies         3120
inactive B6 studies       1229
usable B6 cells          14123
positive cells            6871
negative cells            7252
```

Policy:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

The original B6 gold audit gave sensitivity `0.9748`, specificity `0.6061`, positive precision `0.6905`, NPV `0.9639`, balanced accuracy `0.7904`, and coverage `0.3606`. These are noise/coverage diagnostics, not a downstream AUC ceiling.

## B6/B15 reused-gold diagnostic

Completed result:

```text
coverage-conditioned B6 macro AUC     0.7736374158
strict 95% bootstrap CI              [0.7213680813,0.8228450378]
eligible cells                       251 / 696
coverage                             0.3606321839

full-surface B6 state-only AUC        0.7024597743
95% CI                              [0.6537393397,0.7507506766]
```

Pooled expert-positive rates by parser state:

```text
positive       116 / 168 = 0.6905
negated          3 / 83  = 0.0361
uncertain       11 / 29  = 0.3793
unmentioned    110 / 416 = 0.2644
```

Target-specific middle-state rates were heterogeneous. Therefore no universal uncertain/unmentioned soft labels were derived from the reused gold set.

Noise alignment on high-confidence cells:

```text
B6 high-confidence cells   251
teacher correct            196
teacher wrong               55
fraction B6-wrong cells B15 moved toward truth   0.6364
fraction B6-wrong cells B15 moved toward B6      0.3636
```

The predeclared B6-error-imitation evidence flags were all false.

## B16 representation stage

B16 used the completed B15 MRI-SSL encoder as initialization and then aligned MRI study features to full competition report semantics represented by competition-only TF-IDF -> TruncatedSVD vectors.

Data contract:

```text
non-gold studies           4349
eligible real MRI series  24035
2.5D examples/pass        48070
gold labels used              0
B6 labels in report stage     0
weak-v2 selection gate    false
```

Four exact report-alignment epochs:

```text
epoch 1  total 3.8958491301  NCE 3.7314807830  cosine 0.6574733884
epoch 2  total 3.1331863229  NCE 3.0068265086  cosine 0.5054392496
epoch 3  total 2.7944439663  NCE 2.6814318427  cosine 0.4520484886
epoch 4  total 2.5218941658  NCE 2.4161238326  cosine 0.4230813365
```

Every epoch had 2175/2175 batches, 4349/4349 studies, 24035/24035 series, full coverage `true`, and budget limited `false`.

## B16 downstream stage

B16 returned to the exact full B13 surface:

```text
training studies          3120
real MRI series          17475
usable B6 cells          14123
positive cells            6871
negative cells            7252
batches/epoch             1560
epochs                       4
```

Losses:

```text
0.7379701049
0.6212521367
0.5901195104
0.5675074643
```

Every epoch completed exact study, cell, and series coverage with no budget truncation. Training loss is not a model-selection metric.

## B16 reused-gold result

```text
B16 macro AUC      0.6349770242
95% CI            [0.5854729266,0.6830266155]
B13 macro AUC      0.6293565948
raw B16-B13       +0.0056204295
paired median     +0.0050711608
95% paired CI     [-0.0395927864,+0.0519351407]
P(B16 > B13)       0.5828
5000 / 5000 paired replicates usable
```

Per-target B16 AUCs are descriptive only:

```text
ACL                0.5012254902
MCL                0.4580498866
Medial Meniscus    0.6658653846
Lateral Meniscus   0.6683229814
Medial OA          0.6945736434
Lateral OA         0.5841392650
PF OA              0.6370656371
Effusion           0.8360248447
Synovitis          0.7514934289
Baker's            0.6757246377
Contusion          0.5708502024
Fracture           0.5763888889
```

The frozen decision rule used the global point estimate only, so B16 replaces B13 as the development champion by policy. The paired CI crosses zero and `P=0.5828`, so no claim of established superiority is warranted.

## Current decision

```text
B16 RETAIN / current development champion by frozen point-estimate rule
B13 RETAIN / historical reference; statistically unresolved with B16
B14 REJECT globally
B15 teacher-agreement gain confirmed; global gold gain not established
weak-v2 is not an expert model-selection gate
58-study gold remains repeatedly reused development data
```

Do not extend or retune B16 after this gold result, do not construct target-wise B13/B16 hybrids, and do not derive new state probabilities from gold.

The next credible performance check is the **hidden Kaggle evaluation**. Any B17 experiment should be separately specified before another reused-gold look.
