# Experiment status

**Snapshot:** 2026-08-14  
**Package:** `0.29.0`  
**Active working model:** `B20_crop_only_joint_focus`  
**Canonical B20 checkpoint:** `runs/b20_crop_focus/b20_model.pt`  
**Canonical B20 epoch:** `2`  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/model-selection surface, not independent validation**. Historical B18-B20 checkpoint selection, the B21 one-look acceptance comparison and the B22 duration trajectory all touch this same expert surface in different ways. Hidden competition evaluation remains the independent predictive-performance signal.

## Current headline

- **B20 remains the active working model.**
- Historical B20 geometry is `native MRI -> resize 224 -> center crop 90% -> resize 224`; this is preserved exactly for reproducibility.
- B20 canonical expert macro AUC is `0.6671593555` at epoch 2.
- B18 and B20 nested epoch-selection audits are complete; both primary cross-fitted audits selected `[2,2,2]` with measured checkpoint-selection optimism `0.0` for that narrow epoch-choice component.
- B19 remains rejected because its cosine vignette created an artificial preprocessing-border shortcut.
- B21 corrected the crop order to `native -> crop 90% -> percentile normalization -> resize 224`. It **passed weak-v2** but **failed the predeclared full-data expert acceptance comparison**, so it was not promoted.
- B22 retrained the B21 preprocessing for five epochs. **E2 remained the best expert-gold endpoint; E3-E5 did not rescue performance despite monotonically decreasing training loss.**
- The current bottleneck is therefore treated as the **weak-label / development-selection problem**, not insufficient training duration.

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
| **B13** | **one learned token per series + ImageNet ConvNeXt** | **`0.6293565948` gold** | historical high-performing tier |
| B14 | full `K x 16` slice-token memory + same ImageNet protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no global gold gain |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | retained representation source |
| **B17** | **freeze B16 report-aligned encoder; train hierarchy/head only for five fixed full B6 passes** | **epoch 5; `0.6425890153` gold** | fixed-epoch reference |
| **B18** | **same B17 trajectory; global expert set selects epoch; full FOV** | original E2 `0.6654496134`; replay `0.6655517376` | frozen full-FOV comparator; nested audit complete |
| **B19** | **B18 recipe + 90% crop + cosine vignette** | **epoch 3; `0.6581308356`** | **rejected; artificial border shortcut** |
| **B20** | **B18 recipe + post-resize 90% crop only; no vignette** | **epoch 2; `0.6671593555`** | **ACTIVE WORKING MODEL** |
| **B21-v1 weak-v2** | **pre-resize 90% crop; leakage-safe matched development** | control `0.7298727911`; B21 `0.7410090411`; paired delta `+0.0111362500` | weak-v2 gate passed |
| **B21 full acceptance** | **same pre-resize crop; full 3,120 B6 studies; fixed E2** | B21 `0.6573196516`; B20 replay `0.6674066371`; paired delta `-0.0100869854` | **not promoted** |
| **B22** | **B21 pre-resize pipeline retrained E1-E5** | E1 `0.6135270850`; **E2 `0.6574269018`**; E3 `0.6387456622`; E4 `0.6136783995`; E5 `0.6282683534` | **E2 best; no duration rescue; closed** |
| FINAL | B17-style frozen encoder + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## B18 nested epoch-selection audit — complete

Primary two-fold cross-fitted selection:

```text
selected epochs by outer fold       [2,2,2]
OOF macro AUC                        0.6655517376076434
all 12 targets defined              true
estimated epoch-selection optimism  0.0
```

Strict historical-manifest sensitivity analysis:

```text
selected epochs                     [2,5,2]
OOF macro AUC                       0.6475369755138950
estimated selection optimism        0.0180147620937484
```

Fixed endpoint comparison:

```text
fixed epoch-5 / B17 endpoint        0.6425890152580378
cross-fit B18 - B17                +0.0229627223496056
```

The audit estimates checkpoint-selection optimism only; it does not remove broader development-set reuse.

Canonical record: [`B18_NESTED_EPOCH_AUDIT.md`](B18_NESTED_EPOCH_AUDIT.md).

## B20 completed result

```text
epoch 1  loss 0.7456469554  expert macro AUC 0.6177301847
epoch 2  loss 0.6459858875  expert macro AUC 0.6671593555  <- selected
epoch 3  loss 0.6226234155  expert macro AUC 0.6492154172
epoch 4  loss 0.5998095677  expert macro AUC 0.6570041510
epoch 5  loss 0.5828775678  expert macro AUC 0.6577823350
```

```text
selected epoch        2
canonical statistic   0.667159355531343
checkpoint            runs/b20_crop_focus/b20_model.pt
```

### B20 nested epoch-selection audit — complete

Primary two-fold cross-fitted selection:

```text
selected epochs by outer fold       [2,2,2]
OOF macro AUC                        0.667159355531343
all 12 targets defined              true
estimated epoch-selection optimism  0.0
```

Strict historical-manifest sensitivity analysis:

```text
selected epochs                     [2,5,2]
OOF macro AUC                       0.6351640998170208
estimated selection optimism        0.0319952557143222
```

Fixed endpoint comparison:

```text
all-58 selected macro AUC           0.667159355531343
fixed epoch-5 macro AUC             0.6577823350159498
selection uplift vs epoch 5        +0.0093770205153931
```

Canonical record: [`B20_NESTED_EPOCH_AUDIT.md`](B20_NESTED_EPOCH_AUDIT.md).

## B18/B19/B20 spatial-focus history

```text
              E1         E2         E3         E4         E5
B18        0.618716   0.665450   0.651115   0.639416   0.642589
B19        0.580216   0.624272   0.658131   0.636993   0.648569
B20        0.617730   0.667159   0.649215   0.657004   0.657782
```

B19's cosine/vignette formulation is rejected. B20 removes the synthetic-boundary shortcut and remains the preferred historical knee-focused formulation. B18 versus B20 global predictive superiority remains unresolved because their difference is very small on the same reused expert surface.

## B21 pre-resize crop campaign — complete

### Structural correction

Historical B20 actually performs:

```text
native MRI -> percentile normalization -> resize 224 -> crop 90% -> resize 224
```

B21 tested:

```text
native MRI -> crop 90% -> percentile normalization -> resize 224
```

The change therefore includes both crop order and the support used to derive the percentile-normalization window.

### Leakage-safe weak-v2 development

Historical B16 could not be used directly for weak-v2 model ranking because it was trained on all 4,349 non-gold report pairs, including the 623 weak-v2 holdout studies. A weak-v2-safe B16 representation was therefore rebuilt from B15 MRI SSL while excluding both weak-v2 holdout and gold studies.

Matched development arms:

```text
weak-v2 training studies    2497
weak-v2 holdout studies      623
training cells             11248
training series            13974
fixed endpoint                E2
scheduler horizon               5
same safe encoder              yes
gold development use             0
```

Result:

```text
B20-v2 control macro AUC        0.7298727911
B21-v1 macro AUC                0.7410090411
raw B21 - control              +0.0111362500
paired median                  +0.0109814529
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
```

This measured agreement with frozen B6 report supervision, not expert truth.

### Full-data one-look acceptance

B21 was frozen, refit on all 3,120 B6-active studies with the historical B16 encoder, and evaluated once against B20.

```text
B20 canonical macro AUC         0.6671593555
B20 replay macro AUC            0.6674066371
B20 replay - canonical         +0.0002472815
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired median                  -0.0095857726
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

```text
promotion_rule_passed              false
scientific_superiority_supported   false
```

B21 is closed and not promoted.

Canonical records:
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md)

## B22 training-duration audit — complete

B22 retrained the B21 preprocessing from scratch for five epochs with no gold evaluation during training. All epochs had exact full coverage and an unchanged frozen encoder SHA.

Training trajectory:

```text
E1 loss 0.7388751291
E2 loss 0.6381611442
E3 loss 0.6087977977
E4 loss 0.5890809184
E5 loss 0.5680555741
```

Expert trajectory:

```text
E1 macro AUC  0.6135270850
E2 macro AUC  0.6574269018  <- best
E3 macro AUC  0.6387456622
E4 macro AUC  0.6136783995
E5 macro AUC  0.6282683534
```

Reproducibility checks:

```text
B20 replay - canonical        +0.0007997420   tolerance 0.005
B22 E2 - prior B21 E2        +0.0001072501   tolerance 0.005
```

Relative to the B22-audit B20 replay (`0.6679590975`):

```text
E2 raw delta  -0.0105321958   paired 95% CI [-0.0323859143,+0.0098214527]
E3 raw delta  -0.0292134353   paired 95% CI [-0.0523986045,-0.0087333144]
E4 raw delta  -0.0542806980   paired 95% CI [-0.0827548184,-0.0276651497]
E5 raw delta  -0.0396907441   paired 95% CI [-0.0654472831,-0.0162928843]
```

Conclusion: **longer training does not rescue the pre-resize crop.** The training loss keeps falling after E2 while expert ranking deteriorates, consistent with increasing fit to the report-derived weak target rather than improved expert-pathology discrimination.

Canonical record: [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md).

## Frozen supervision surface

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

Historical B16 encoder SHA used by B18-B22 full-data runs:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

## Current model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator; nested audit complete
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  pre-resize crop candidate; weak-v2 passed, gold acceptance failed; CLOSED
B22  five-epoch B21 duration audit; E2 best, no rescue; CLOSED
```

## Current scientific position

The current evidence does **not** support spending another experiment on more downstream epochs. It also does not support treating weak-v2 teacher agreement as a sufficient model-selection surrogate for expert truth.

The B21/B22 sequence provides the clearest current diagnostic:

```text
weak-v2 teacher agreement improves for B21
                 but
expert-gold ranking worsens

and after E2:
training loss continues to improve
                 while
expert-gold ranking deteriorates
```

The next bottleneck is therefore the **label / development-selection signal**.

## Immediate next steps

1. Keep B20 unchanged as the active working model.
2. Freeze B21 and B22 as completed negative/diagnostic experiments; do not retune them from reused gold.
3. Audit the relationship between available weak-v2 model deltas and expert-gold model deltas across frozen historical models where both measurements exist.
4. Use that audit to decide whether weak-v2 can play any role beyond teacher-agreement diagnostics.
5. Prioritize improved pathology supervision / development-surface quality before large architecture or optimization changes.
6. If genuinely new expert-labelled studies can be obtained, reserve them prospectively rather than folding them immediately into the reused 58-study surface.
7. Only after the selection problem is addressed revisit encoder adaptation, FOV/routing, resolution, or other model changes.

## Governance

```text
B20: active working model; preserve checkpoint and preprocessing exactly
B21: closed; no second acceptance look
B22: closed exploratory duration audit; no production epoch selection from reused gold
B18/B20 nested audits: completed
B18/B19/B20/B21/B22 expert labels: never entered the weak-supervision gradients
58-study gold surface: development/reused, not independent validation
weak-v2: teacher-agreement surface, not a validated expert-truth surrogate
no target-specific epoch choice or B20/B21/B22 target mixing
no crop-fraction sweep under the current B21 normalization-order implementation
FINAL all-data expert-label fit: deferred
```

## Canonical records

- [`WORKING_MODEL.md`](WORKING_MODEL.md)
- [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md)
- [`B18_FISHER_SELECTION.md`](B18_FISHER_SELECTION.md)
- [`B18_NESTED_EPOCH_AUDIT.md`](B18_NESTED_EPOCH_AUDIT.md)
- [`B19_JOINT_FOCUS.md`](B19_JOINT_FOCUS.md)
- [`B20_CROP_ONLY_FOCUS.md`](B20_CROP_ONLY_FOCUS.md)
- [`B20_NESTED_EPOCH_AUDIT.md`](B20_NESTED_EPOCH_AUDIT.md)
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md)
- [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md)
- [`VALIDATION.md`](VALIDATION.md)
- [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md)
