# Experiment status

**Snapshot:** 2026-08-13  
**Package:** `0.28.0`  
**Gold development/selection set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study expert-labelled surface has been reused repeatedly and is therefore a **development/model-selection surface, not independent validation**. B18--B20 use it for global checkpoint selection. Nested audits below estimate checkpoint-selection optimism only; they do not erase broader development-set reuse.

## Current headline

- **B20 is the primary knee-focused candidate.** It uses a centered 90% crop with no vignette, black border, cosine taper, or crop jitter.
- B20 selected epoch 2 at `0.6671593555` and passed the local 3-study / 15-series inference/schema smoke test.
- B20's primary three-fold cross-fitted epoch-selection audit selected epoch 2 in every outer fold (`[2,2,2]`), producing OOF macro AUC `0.6671593555` and measured epoch-selection optimism `0.0`.
- The strict one-inner-fold sensitivity analysis selected `[2,5,2]`, produced OOF macro AUC `0.6351640998`, and estimated optimism `0.0319952557`; this reflects the much smaller and noisier selection subset.
- B19 remains rejected because its cosine vignette created an artificial border shortcut.
- B18 full-FOV remains a retained comparison baseline at selected epoch 2 / `0.6654496134`.
- The B20-B18 selected-statistic difference is only `+0.0017097421`; **predictive superiority remains unresolved** because both models were developed on the same reused 58-study surface.
- The B18 nested epoch-selection audit is now implemented and should be run before interpreting the B17 -> B18 jump as robust progress.

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
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | retained high-performing tier |
| **B17** | **freeze B16 report-aligned encoder; train hierarchy/head only for five fixed full B6 passes** | **epoch 5; `0.6425890153` gold** | fixed-epoch reference |
| **B18** | **same B17 trajectory; global expert set selects epoch; full FOV** | **epoch 2; `0.6654496134` selection only** | retained; nested audit pending |
| **B19** | **B18 recipe + 90% crop + cosine vignette** | **epoch 3; `0.6581308356` selection only** | **rejected; artificial border shortcut** |
| **B20** | **B18 recipe + 90% crop only; no vignette** | **epoch 2; `0.6671593555`; cross-fit optimism `0.0`** | **primary knee-focused candidate** |
| FINAL | B17-style frozen encoder + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## B20 completed result

```text
epoch 1  loss 0.7456469554  selection AUC 0.6177301847
epoch 2  loss 0.6459858875  selection AUC 0.6671593555  <- selected
epoch 3  loss 0.6226234155  selection AUC 0.6492154172
epoch 4  loss 0.5998095677  selection AUC 0.6570041510
epoch 5  loss 0.5828775678  selection AUC 0.6577823350
```

```text
selected epoch        2
selection statistic   0.667159355531343
checkpoint            runs/b20_crop_focus/b20_model.pt
```

## B20 nested epoch-selection audit

The five saved B20 candidate checkpoints were rescored without retraining.

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
estimated selection optimism        0.03199525571432216
```

Fixed endpoint comparison:

```text
all-58 selected macro AUC           0.667159355531343
fixed epoch-5 macro AUC             0.6577823350159498
selection uplift vs epoch 5        +0.00937702051539313
```

Interpretation: the primary cross-fitted audit finds no measured optimism attributable specifically to B20 epoch choice. The strict result is retained as a small-selection-set sensitivity diagnostic. Neither result is pristine independent validation because the 58 studies have influenced the broader development campaign.

Canonical record: [`B20_NESTED_EPOCH_AUDIT.md`](B20_NESTED_EPOCH_AUDIT.md).

## B18/B19/B20 spatial-focus comparison

```text
              E1         E2         E3         E4         E5
B18        0.618716   0.665450   0.651115   0.639416   0.642589
B19        0.580216   0.624272   0.658131   0.636993   0.648569
B20        0.617730   0.667159   0.649215   0.657004   0.657782
```

B19's cosine/vignette formulation is rejected. B20 removes the synthetic-boundary shortcut and is the preferred knee-focused formulation. B18 versus B20 global predictive superiority remains unresolved.

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

Encoder SHA used by B18--B20:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

## Local inference smoke surface

```text
test studies                   3
test series                   15
series / study                 5 / 5 / 5
TTA                            [-1,0,1]
metadata repairs               0
```

This surface checks engineering/inference compatibility only; it has no labels and cannot measure AUC.

## Current governance

```text
B16/B17: closed to post-gold retuning
B18: completed; epoch 2 retained; nested audit implemented and pending execution
B19: completed and rejected because of artificial vignette shortcut
B20: primary knee-focused candidate; epoch 2 retained
B20 nested audit: cross-fit epochs [2,2,2], measured epoch-selection optimism 0.0
B18 vs B20: predictive superiority unresolved
B18/B19/B20: expert labels never entered gradients
B18/B19/B20: no target-specific epoch choice or target mixing
selected expert scores: development/checkpoint-selection evidence, not independent validation
weak-v2: do not regenerate from outcomes
uncertain/unmentioned: no universal gold-derived pseudo-labels
FINAL all-data fit: deferred pending the independent-evaluation decision
```

## Immediate next steps

1. Run the **B18 nested epoch-selection audit** using the same folds and saved checkpoints, so the B17 -> B18 selection gain can be quantified directly.
2. Keep B20 as the primary knee-focused candidate.
3. Use B20 cross-fitted predictions to rank all 12 targets, then inspect false positives/false negatives for the weakest targets.
4. Only after those diagnostics decide whether the next B20 change should address series/plane routing, slice sampling, crop behaviour, or weak-label quality.

## Canonical records

- [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md)
- [`B18_FISHER_SELECTION.md`](B18_FISHER_SELECTION.md)
- [`B19_JOINT_FOCUS.md`](B19_JOINT_FOCUS.md)
- [`B20_CROP_ONLY_FOCUS.md`](B20_CROP_ONLY_FOCUS.md)
- [`B20_NESTED_EPOCH_AUDIT.md`](B20_NESTED_EPOCH_AUDIT.md)
- [`VALIDATION.md`](VALIDATION.md)
- [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md)
