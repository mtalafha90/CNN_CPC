# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.23.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than independent validation.

## Current headline

- **Development champion remains B13**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.
- **B14 is completed and rejected globally**, macro AUC `0.6197914249`, paired median `B14-B13=-0.0093726931`, 95% CI `[-0.0469823411,+0.0250137870]`, `P(B14>B13)=0.2924`.
- B14 final B6 loss was `0.5822778610` versus B13 `0.6132239342`; stronger weak-label fitting did not improve global macro AUC.
- **Full B13 slice exposure audit completed:** `17,475/17,475` readable series, median evaluation exposure `100%`, complete evaluation exposure `95.9%`, median and p95 maximum skipped run both zero. **Slice-count undersampling is rejected as a primary bottleneck.**
- **Weak holdout v1 is superseded before model training:** it had 624 studies and zero report leakage but Synovitis `70 positive / 1 negative`.
- **Weak holdout v2 is the new pre-B15 validation contract:** deterministic report-group-safe multilabel/class stratification, minimum four examples per class on both sides where globally feasible, and strict all-12-target study bootstrap.
- Reserved representation hypothesis: **B15 = ImageNet -> competition knee-MRI self-supervised adaptation -> B13 hierarchy**.

## Experiment ladder

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with lower encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **full 3,120-study B7 coverage** | **`0.5644802945`** | previous benchmark |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series + full slice-token memory + B5 init** | **`0.5660915179`** | retained / tied with B7.1 |
| B12.1 | one learned token per series + B5 init | not run | implemented / skipped |
| **B13** | **one learned token per series + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / DEVELOPMENT CHAMPION** |
| **B14** | **full `K x 16` slice-token memory + same ImageNet protocol** | **`0.6197914249`** | **COMPLETED / REJECTED GLOBALLY** |
| **B15** | **ImageNet -> knee-MRI SSL -> B13 hierarchy** | not run | reserved next representation hypothesis |

## Frozen B6 supervision surface

```text
report-only studies       4349
active B6 studies         3120
inactive B6 studies       1229
usable B6 cells          14123
positive cells            6871
negative cells            7252
```

Frozen policy:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

B6 audit quality values include sensitivity `0.975`, specificity `0.606`, positive precision `0.690`, balanced accuracy `0.790`, and coverage `0.361`. These establish noisy/incomplete supervision but do **not** establish a numerical downstream macro-AUC ceiling.

## Frozen all-series surface

```text
training studies        3120
eligible real series   17475
historical dual unique 15468
extra series            2007
max series / study        14
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## B13 retained result

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]

training loss
0.7450505349
0.6865059846
0.6524747430
0.6132239342
```

Per-target B13 AUCs are descriptive only:

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
B14 final loss     0.5822778610
B14 macro AUC      0.6197914249
95% CI            [0.5706800512,0.6693542716]
raw B14-B13       -0.0095651699
median difference -0.0093726931
95% paired CI     [-0.0469823411,+0.0250137870]
P(B14 > B13)       0.2924
```

The paired CI crosses zero. B14 and B13 are statistically unresolved on the reused gold surface, but model selection retains B13 because B14 has the lower point estimate, low probability of superiority, higher compute cost and no global advantage. No target-level B13/B14 hybrid is permitted.

## Completed B13 slice-exposure audit

The corrected audit used the actual B13 2.5D sampler on the exact frozen non-gold series surface.

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
skipped-run length       median 0.0 mm (p95 0.0 mm)

Axial      n=4455   eval=100.0% max-run=0.0 train/view=85.2%
Coronal    n=5815   eval=100.0% max-run=0.0 train/view=87.0%
Sagittal   n=7205   eval=100.0% max-run=0.0 train/view=87.0%
```

Decision:

```text
slice-count undersampling as primary B13 bottleneck -> REJECT
```

Canonical record: `docs/B13_SLICE_EXPOSURE_AUDIT.md`.

## Weak holdout validation contract

### v1 — superseded before model training

```text
active studies             3120
train studies              2496
holdout studies             624
train report groups        2430
holdout report groups       609
report-group overlap          0
holdout usable cells       2697
holdout positive cells     1257
holdout negative cells     1440
Synovitis                  70 positive / 1 negative
manifest SHA-256
fdbc02f88e5a4eff31783b4242890e943609d5c783bd54aca38af8a89e7e0968
```

No B15 or matched B13-control model was trained on v1. It is superseded because the one Synovitis negative makes its 12-target macro bootstrap unnecessarily unstable.

### v2 — freeze before training

Package `0.23.0` makes `rsna-knee-weak-holdout` produce `weak_b6_holdout_v2` by default.

Frozen split policy:

```text
holdout fraction        0.20
seed                    2026
report grouping         mandatory
minimum class count     4 per side where globally feasible
candidate splits        4096
split objective         balance all 24 target/class counts + holdout size
uses gold labels        false
uses model predictions  false
```

Freeze v2 with:

```bash
rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --min-class-count 4 \
  --search-candidates 4096 \
  --out-root runs/weak_holdout_v2
```

Once successfully frozen, the v2 manifest SHA is part of the experiment contract and must not be regenerated based on model performance.

Weak-surface scoring is strict:

```text
study bootstrap
-> compute all 12 AUCs
-> discard replicate if any target AUC is undefined
-> accepted replicate macro = mean of exactly 12 AUCs
```

Existing B13/B14 checkpoints were trained on all 3,120 active B6 studies and cannot be retrospectively scored on v2 and called validation.

Canonical record: `docs/WEAK_HOLDOUT_V2.md`.

## Current decision / next stage

```text
B13 RETAIN / development champion
B14 REJECT globally
slice-count hypothesis REJECT
weak holdout v1 SUPERSEDED
       |
       v
freeze weak holdout v2 before model training
       |
       +--> matched B13-control on v2 weak-train
       `--> B15 candidate on same downstream weak-train
                     |
                     v
          paired strict all-12-target weak bootstrap
                     |
                     v
          one reused-gold development confirmation
                     |
                     v
              Kaggle hidden signal
```

For B15, all 58 gold studies must remain excluded from SSL optimization. Gold labels remain forbidden from gradients, early stopping and checkpoint selection. Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
