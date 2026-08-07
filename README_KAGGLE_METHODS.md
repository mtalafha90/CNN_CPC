# RSNA Knee Abnormality Detection — Public Code Methodology Review

**Repository:** `mtalafha90/CNN_CPC`  
**Review snapshot:** 2026-08-07  
**Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## Scope and an important limitation

This document is a technical review of the **publicly accessible code and code mirrors that could be discovered for the active 2026 RSNA Knee Abnormality Detection competition** as of the date above. The Kaggle Code tab is dynamically rendered and, at this early stage of the competition, its notebook inventory is not completely exposed to normal web indexing. Therefore, it would be inaccurate to claim that this document contains literally every public Kaggle notebook ever created.

The review instead covers all substantive public competition implementations discovered during the audit, plus the main architecture ideas inherited from MRNet-style knee MRI work. Empty repositories and repositories containing only a title were excluded from the methodological synthesis. Performance numbers are included only when a repository explicitly reports them and are marked **self-reported** unless independently verifiable.

The purpose is not to copy competitors' code. It is to understand the recurring technical ideas, identify methodological mistakes, and define a stronger experimental strategy for `CNN_CPC`.

---

# 1. Competition facts that dictate the methodology

The public data structure makes this challenge very different from a normal supervised image-classification competition.

- **4,407 training studies**.
- Only **58 studies** contain complete official binary labels for the 12 targets.
- The remaining **4,349 studies** have radiology reports but no complete structured target vector.
- Each study contains multiple MRI series.
- `train_series.csv` supplies useful sequence metadata including anatomical plane, fluid sensitivity, and fat suppression.
- The 12 study-level targets are:
  - ACL
  - MCL
  - Medial Meniscus
  - Lateral Meniscus
  - Medial OA
  - Lateral OA
  - PF OA
  - Effusion
  - Synovitis
  - Baker's
  - Contusion
  - Fracture
- The competition metric is **macro-averaged ROC AUC across the 12 targets**.
- The reports are useful for generating training supervision, but the hidden test workflow should be assumed to require an **image-only inference model** unless the official test files explicitly provide otherwise.
- Final competition execution is notebook-based and runtime constrained, so preprocessing cost and inference efficiency matter.

The central consequence is:

> **This is primarily a weakly/semi-supervised, multi-series MRI problem with a very small trusted validation set.**

A CNN trained only on 58 studies is useful as a sanity baseline, but it should not be the final strategy. Conversely, treating the other 4,349 studies as negative examples would be a serious labeling error.

---

# 2. Public implementations reviewed

| Source | Main approach | Most useful idea | Status/caution |
|---|---|---|---|
| [Chagatai404/knee-abnormality-ML](https://github.com/Chagatai404/knee-abnormality-ML) | Expert-only 3-plane 2.5D EfficientNetV2 baseline | Honest gold-only reference, DICOM preflight, physical slice ordering | Baseline intentionally omits weak supervision |
| [jiaweizhong/rsna-knee](https://github.com/jiaweizhong/rsna-knee) | Modular efficiency-oriented 2.5D framework | Sharded DICOM audit, learned Top-K selectors, per-label query aggregation | Mostly framework/scaffold; not a measured competition solution yet |
| [andreluizpedroso/rsna-knee-abnormality-detection](https://github.com/andreluizpedroso/rsna-knee-abnormality-detection) | Rule-derived weak labels + image CNN | Report-as-teacher only, confidence-weight pseudo-labels, multilabel stratification | Sound baseline methodology; full-scale result not established |
| [chaitanyajamble/RSNA-Knee-Abnormality-Detection](https://github.com/chaitanyajamble/RSNA-Knee-Abnormality-Detection) | Several notebook experiments: rules, TF-IDF/SVD, DICOM statistics, LightGBM, small CNN | Cheap classical baselines and bounded-cost feature extraction | Some experiments mix text/image features in ways unsuitable for report-free inference |
| [dianisay/RSNA-Knee-Abnormality-Detection](https://github.com/dianisay/RSNA-Knee-Abnormality-Detection) | Rule + optional LLM labels, frozen DINOv2, per-target MIL | DINOv2 CLS/mean/max features, target-specific attention, ranking loss, report-hash folds | One of the richest methodology sources; still must validate each component on real CV |
| [soumic28/RSNA-knee-abnormality-predictin](https://github.com/soumic28/RSNA-knee-abnormality-predictin) | Generic timm 2.5D MIL classifier | Backbone/pooling abstraction: ConvNeXt/EfficientNet/Swin + attention pooling | Treat published performance claims cautiously unless reproduced |
| [JunhaoLiXD/RSNA_Knee_Abnormality_Detection](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection) | 3-plane EfficientNet-B0 + calibrated report weak labels | Fold-safe teacher calibration, robust DICOM preprocessing, sequence routing | V01 score 0.613 is repository self-report; V02 not yet fully trained at review time |
| [Msmile-shiny/CVproject_RSNA-knee-detection](https://github.com/Msmile-shiny/CVproject_RSNA-knee-detection) | ConvNeXtV2 2.5D/triplane experimental framework | Cross-plane fusion, top-K study aggregation, controlled 2D/2.5D/3D ablations | Several components are experimental/scaffolded |
| [Saianiruthm/rsna-knee-abnormality](https://github.com/Saianiruthm/rsna-knee-abnormality) | LLM report teacher followed by planned image student | Strong multilingual probabilistic report labeling | Report-label AUC is self-reported on 58 gold cases, not a competition leaderboard score |
| [tomyimkc/sophia-agi — RSNA knee subproject](https://github.com/tomyimkc/sophia-agi/tree/main/agi-proof/contest-submissions/rsna-knee-abnormality) | Carefully instrumented 2D MRNet + 3D arm + distillation + ensemble | Gold-only validation, unlabeled distill-only loss, 2D/3D complementarity, experiment discipline | Explicitly a candidate scaffold with no claimed competition result at review time |
| [bollimunthasripavan-oss/RSNA-Knee-MRI-Abnormality-Detection](https://github.com/bollimunthasripavan-oss/RSNA-Knee-MRI-Abnormality-Detection) | Legacy MRNet-like DenseNet multi-view model | Historical multi-plane max-pooling + late fusion + Grad-CAM | Mainly a 3-target MRNet-style project, not a direct 12-target competition solution |

---

# 3. Technique family A — generating labels from radiology reports

## 3.1 Why report supervision is essential

With 58 trusted labels and 4,349 otherwise unlabeled studies, the report is the obvious source of scalable training supervision. Public solutions converge on the same conceptual pattern:

```text
Radiology report
       |
       v
text teacher / rules / LLM
       |
       v
12 soft probabilities + confidence
       |
       v
train image-only MRI model
       |
       v
validate only on trusted gold labels
```

The report should generally be considered a **teacher**, not an input feature required by the final image model.

## 3.2 Rule-based multilingual extraction

Several implementations use clinical dictionaries with:

- anatomy terms for ACL, MCL, menisci, OA compartments, effusion, synovitis, Baker cyst, contusion and fracture;
- negation detection such as `no`, `without`, `absence`, `sin`, `sans`, etc.;
- normality phrases such as `intact`, `preserved`, `normal`;
- uncertainty phrases such as `possible`, `suspected`, `cannot exclude`;
- severity phrases to create a soft probability rather than a hard 0/1 target;
- Unicode/diacritic normalization for multilingual reports.

A useful design is to return both:

```text
score_target       = estimated probability
confidence_target  = reliability of that estimate
```

rather than pretending every rule-derived target is equally trustworthy.

## 3.3 LLM-based report teacher

A stronger approach seen in the public code is to ask a multilingual medical-capable LLM to act as a report labeler and output **12 calibrated probabilities**. The best implementation pattern is:

1. give a precise clinical definition of every target;
2. specify calibration ranges for definite, probable, possible, absent and unmentioned findings;
3. require a strict JSON output;
4. use temperature 0 or another deterministic setting;
5. cache one result per study so the process is resumable;
6. validate the teacher against the 58 official gold studies;
7. keep the resulting probabilities as **soft labels**.

One public project self-reports approximately **0.89 macro AUC for its report teacher on the 58 gold cases**. This is encouraging but is not a competition leaderboard score and should be independently reproduced before relying on it.

## 3.4 Ensemble multiple text teachers

A useful idea from the public code is **rank averaging** rule-derived and LLM-derived target probabilities. Rank averaging is particularly sensible for an AUC competition because AUC depends on ranking rather than probability calibration.

For target `c`:

```text
r_rule = rank(rule_score_c)
r_llm  = rank(llm_score_c)
soft_target_c = mean(r_rule, r_llm)
```

Confidence can be retained separately.

## 3.5 Fold-safe teacher calibration

This is one of the most important methodological points in the entire review.

If report rules or a text teacher are calibrated using all 58 gold labels and those same 58 cases later appear in validation folds, validation becomes optimistic. The safer design is:

```text
For each CV fold:
    validation gold cases = fold k
    calibration gold cases = all gold cases except fold k
    calibrate text states / thresholds only on calibration cases
    create soft labels
    train image model
    score only on validation gold cases
```

This is especially important for empirical-Bayes mappings such as:

```text
explicit positive -> P(y=1 | positive state)
explicit negative -> P(y=1 | negative state)
unmentioned       -> P(y=1 | unmentioned state)
```

The mapping must be learned **inside the training fold**.

## 3.6 Gold labels should dominate pseudo-labels

Public implementations use gold multipliers such as 3 or 8. The exact value needs ablation, but the principle is sound:

```text
L = gold_weight * L_gold + confidence * L_pseudo
```

or per target:

```text
L = mean_i,c  w_i,c * BCE(logit_i,c, target_i,c)
```

where gold examples receive the highest `w` and uncertain/unmentioned report labels receive small weights.

### Recommendation for `CNN_CPC`

Implement three report modes:

1. `rules` — deterministic, free baseline;
2. `llm` — stronger teacher generated outside the submission notebook;
3. `ensemble` — rank-average rules + LLM, preserving confidence.

All three must be evaluated against exactly the same gold-only fold protocol.

---

# 4. Technique family B — DICOM preprocessing

Public implementations repeatedly show that DICOM handling is not a trivial preprocessing detail. Incorrect slice ordering or intensity handling can erase much of the benefit of a stronger network.

## 4.1 Correct spatial slice ordering

The strongest implementations sort slices by physical geometry:

1. read `ImageOrientationPatient`;
2. compute the slice-plane normal from the row/column direction vectors;
3. project `ImagePositionPatient` onto that normal;
4. sort by the projected coordinate.

Fallback hierarchy:

```text
ImagePositionPatient geometry
    -> InstanceNumber
    -> deterministic filename order
```

This is preferable to lexicographic filename sorting.

## 4.2 Intensity handling

Robust pipelines include:

- `RescaleSlope` and `RescaleIntercept`;
- signed pixel data support;
- `MONOCHROME1` inversion where required;
- multi-frame fallback handling;
- percentile clipping to suppress extreme outliers;
- normalization after clipping;
- consistent resize/interpolation.

A useful 2.5D trick is **joint normalization of the three neighboring slices**, rather than normalizing each channel independently. That preserves local intensity relationships between adjacent slices.

## 4.3 DICOM preflight

A very practical idea is to decode a few real studies **before launching training**. The preflight should verify:

- study paths resolve;
- selected series exist;
- DICOM headers are readable;
- compressed pixel data can be decoded;
- at least one valid triplet/volume is produced;
- tensor shapes and value ranges are sane.

This prevents wasting a long Kaggle GPU session on a path or JPEG-2000 codec error.

## 4.4 Cache what is expensive but deterministic

Useful cache candidates:

- selected-series manifests;
- ordered slice paths;
- preprocessed 2.5D arrays;
- frozen encoder embeddings;
- report labels;
- fold assignments.

A manifest is often preferable to copying the entire DICOM collection because it stores paths and geometry while leaving the original competition data mounted in place.

---

# 5. Technique family C — MRI series selection and routing

A knee study usually contains several sequences, and not all are equally useful for all targets.

## 5.1 Basic 3-plane routing

A strong minimum baseline selects one diagnostic series for each:

- Sagittal
- Coronal
- Axial

and processes each plane independently before study-level fusion.

## 5.2 Use competition-provided metadata

Do not infer the plane from pixels if `train_series.csv` already provides:

- `Anatomical_Plane`
- `Fluid_Sensitive`
- `Fat_Suppression`

A common priority is:

```text
fluid-sensitive AND fat-suppressed
    > fluid-sensitive OR fat-suppressed
    > any remaining series in that plane
```

Some code also prefers a typical slice count to avoid unusually long or unusual sequences.

## 5.3 A better six-slot protocol

One-series-per-plane is simple but leaves information unused. A stronger `CNN_CPC` routing scheme should test up to two streams per plane:

```text
Sagittal fluid-sensitive/fat-suppressed
Sagittal structural/non-fat-suppressed
Coronal fluid-sensitive/fat-suppressed
Coronal structural/non-fat-suppressed
Axial fluid-sensitive/fat-suppressed
Axial structural/non-fat-suppressed
```

Missing slots should use a mask, not fabricated image data.

## 5.4 Target-dependent diagnostic relevance

Different targets may prefer different views/sequences:

- ACL: sagittal is especially important;
- menisci: sagittal + coronal;
- tibiofemoral OA: coronal structural information;
- patellofemoral OA: axial/sagittal information;
- effusion/synovitis: fluid-sensitive sequences;
- Baker cyst: often sagittal/axial fluid-sensitive;
- contusion: fat-suppressed fluid-sensitive series;
- fracture: structural + edema-sensitive information.

This motivates **target-specific attention** rather than one universal pooled study vector.

---

# 6. Technique family D — 2.5D MRI representation

## 6.1 Why 2.5D appears so often

Pure 2D models ignore inter-slice continuity, while full 3D networks are expensive. The common compromise is a three-slice triplet:

```text
[z-gap, z, z+gap] -> 3 input channels
```

Advantages:

- directly compatible with RGB-pretrained ImageNet backbones;
- contains local depth context;
- much cheaper than full 3D convolution;
- allows many candidate positions per study;
- works naturally with multiple-instance learning.

Public code uses gaps such as 1 or 2 slices.

## 6.2 Spatially distributed sampling

Do not always take only the exact center of a series. Useful approaches include:

- uniform positions across the series;
- several central/distributed centers;
- fixed number of positions per plane;
- learned Top-K selection after a cheap scorer.

The key question is not simply `how many slices?`, but:

> How do we maximize the probability that at least one sampled window contains the focal abnormality under a fixed runtime budget?

---

# 7. Technique family E — 2D backbones

Backbones found in the reviewed code include:

- EfficientNet-B0 / B3
- EfficientNetV2-B0
- ConvNeXt / ConvNeXtV2
- Swin Transformer
- ResNet
- DINOv2 ViT
- DenseNet121 in MRNet-style background implementations

There is no reason to assume the largest backbone will win. With noisy pseudo-labels and a runtime-constrained code competition, a medium pretrained encoder plus strong aggregation may outperform a very large model.

## Recommended controlled backbone order

1. EfficientNet-B0 or ResNet18 as a cheap reference.
2. ConvNeXt-Tiny / ConvNeXtV2-Tiny.
3. DINOv2 ViT-S/14 frozen feature extractor.
4. Selective DINOv2 fine-tuning if the frozen representation is useful.
5. Larger models only after the pipeline and CV are trustworthy.

---

# 8. Technique family F — DINOv2 frozen feature extraction

One of the most interesting public approaches uses DINOv2 as a frozen per-slice encoder.

For each slice, the implementation combines:

```text
CLS token
+ mean of patch tokens
+ max of patch tokens
```

into one feature vector. This gives three complementary representations:

- global semantic token;
- average spatial content;
- strongest/localized feature response.

Advantages:

- no image-backbone training initially;
- embeddings can be cached once;
- small downstream MIL heads are cheap to train repeatedly;
- excellent for rapid ablations of fold strategy, pooling and losses.

A useful development strategy is therefore:

```text
DICOM -> frozen DINOv2 -> cached slice embeddings
                         -> many cheap MIL experiments
```

Only after finding a strong head should the last few transformer blocks be unfrozen.

---

# 9. Technique family G — study-level aggregation

Study-level aggregation is arguably as important as backbone choice because abnormalities may be visible in only a few slices.

## 9.1 Mean pooling

```text
study_feature = mean(slice_features)
```

Stable but may dilute a small focal lesion.

## 9.2 Max pooling

Classic MRNet idea:

```text
study_feature[d] = max_slice feature[slice,d]
```

Good for sparse strong evidence, but can be noisy and throws away distribution/context.

## 9.3 Attention MIL

Learn slice importance:

```text
attention_s = softmax(g(feature_s))
study_feature = sum_s attention_s * feature_s
```

This is a natural fit for knee MRI.

## 9.4 Target-specific attention

A stronger variant learns separate attention for every target:

```text
ACL query        -> ACL-relevant slices
MCL query        -> MCL-relevant slices
Meniscus query   -> meniscus-relevant slices
...
Fracture query   -> fracture-relevant slices
```

This is preferable to forcing all 12 diagnoses to share exactly the same pooled representation.

## 9.5 Mean/max statistics as virtual tokens

One public model appends per-slot mean and max feature vectors as additional tokens before target attention. This cheaply supplies both distributed and peak evidence.

## 9.6 Top-K prediction aggregation

For slice/triplet-level predictions:

```text
keep highest-scoring q fraction
study probability = mean(top-K predictions)
```

For focal abnormalities, top 20-30% may outperform averaging every window. This must be tuned only through fold-safe OOF evaluation.

## 9.7 Learned or recall-safe Top-K selector

Efficiency-oriented public code goes further:

```text
all candidate windows
   -> cheap selector/scorer
   -> keep K most informative windows
   -> expensive backbone only on those K
```

The ideal selector maximizes pathology coverage while obeying a fixed compute budget.

---

# 10. Technique family H — multi-plane fusion

## 10.1 Late concatenation

Classic MRNet pattern:

```text
Sagittal encoder -> pooled sag feature
Coronal encoder  -> pooled cor feature
Axial encoder    -> pooled ax feature

concat(sag, cor, ax) -> classifier
```

Simple and strong as a baseline.

## 10.2 Shared versus separate encoders

Two competing design choices:

**Shared backbone**
- fewer parameters;
- regularizes tiny data;
- efficient;
- assumes transferable features across planes.

**Plane-specific backbones**
- more capacity;
- each plane may learn specialized texture/anatomy;
- much more expensive and easier to overfit.

Start shared, then ablate separate branches.

## 10.3 Cross-attention fusion

Experimental public code proposes cross-attention between sagittal, coronal and axial representations. This is plausible but should be added only after a reliable late-fusion baseline exists.

---

# 11. Technique family I — 3D models

A 3D model has one major advantage: it preserves inter-slice morphology instead of independently encoding 2D slices.

A practical public design uses a small 3D branch for each plane:

```text
volume
 -> Conv3D -> BN -> ReLU -> pool
 -> Conv3D -> BN -> ReLU -> pool
 -> Conv3D -> BN -> ReLU
 -> global average pool
```

Then:

```text
concat(sag3D, cor3D, ax3D) -> 12 logits
```

The important strategy is not necessarily to replace the 2.5D model. It is to test whether the 3D model provides **orthogonal ranking information** and improves an ensemble.

### Recommended rule

Do not adopt a 3D arm merely because it is more sophisticated. Keep it only if OOF evidence shows a reproducible gain over the best 2.5D system or improves the 2D+3D ensemble.

---

# 12. Technique family J — losses aligned with the competition

## 12.1 Confidence-weighted BCE

Default starting point:

```text
BCEWithLogits(logit, soft_target) * confidence
```

Gold examples receive a larger multiplier.

## 12.2 Pairwise ranking loss

Because the leaderboard metric is AUC, one public implementation adds an AUC-surrogate ranking term. For a positive/negative pair:

```text
L_rank = max(0, margin - (score_positive - score_negative))
```

Combined objective:

```text
L = BCE + lambda_rank * L_rank
```

This directly encourages correct ordering rather than only probability calibration.

## 12.3 Asymmetric loss / focal-style variants

These can help severe imbalance, but noisy pseudo-labels complicate their use. They should be ablated against plain confidence-weighted BCE.

## 12.4 Class `pos_weight`: use cautiously

This review found conflicting implementations:

- several models use `negatives / positives` as `pos_weight`;
- another public implementation explicitly removed it after observing overprediction with weak labels.

With soft noisy labels, large class weights can amplify pseudo-label mistakes. Therefore:

> **Do not enable automatic `pos_weight` by default. Treat it as an ablation.**

## 12.5 Label smoothing

Some code uses small label smoothing, especially for tiny fully supervised experiments. For already-soft pseudo-labels, additional smoothing may be redundant.

---

# 13. Technique family K — cross-validation and leakage control

This is the most fragile part of the competition because the trusted set contains only 58 studies.

## 13.1 Gold-only validation

The validation metric must be computed only where official labels exist.

Never evaluate the image model against its own report-derived labels and call that competition CV.

## 13.2 Group duplicate/similar reports

One good public implementation hashes normalized report text and forces identical reports into the same fold. This helps protect against duplicated templated reports appearing across train and validation.

## 13.3 Stratify language when useful

If report patterns are language-dependent, `StratifiedGroupKFold` can preserve language distribution while grouping report hashes.

## 13.4 Iterative multilabel stratification

With 12 highly imbalanced targets, ordinary random K-fold can create validation folds with no positive examples for a rare target. Iterative multilabel stratification is preferable when feasible.

## 13.5 Search for fold assignments with valid AUC cells

Another clever baseline searches multiple deterministic K-fold seeds and chooses the split with:

1. the largest number of target/fold combinations containing both positive and negative cases;
2. the smallest target prevalence drift.

This is reasonable for a tiny gold set, provided the selection criterion uses labels only to make the folds—not model predictions—to avoid performance cherry-picking.

## 13.6 Three folds versus five folds

With 58 total gold studies:

- 5 folds give roughly 11-12 validation studies/fold and can make rare-target AUC undefined;
- 3 folds give roughly 19 validation studies/fold but fewer independent trained models.

Recommended policy:

```text
Use 5 folds only if target coverage is acceptable.
Otherwise use repeated 3-fold grouped/multilabel CV.
Report which target AUCs are undefined rather than fabricating values.
```

## 13.7 Bootstrap uncertainty

A single macro-AUC number from 58 studies is noisy. Store OOF predictions and bootstrap the gold studies to estimate uncertainty. A small apparent gain may be meaningless.

---

# 14. Technique family L — classical low-cost baselines

Not every useful benchmark needs a deep model. Public notebook code uses:

### Text features

- TF-IDF up to approximately 20k terms;
- Truncated SVD to 64-96 components;
- rule-derived clinical scores.

### DICOM/study metadata

- number of series;
- total slice count;
- T1/T2/PD/fat-suppressed fractions;
- pixel mean/std/min/max from a bounded slice sample;
- pixel spacing;
- slice thickness.

### Classifier

- one LightGBM model per target;
- stratified CV;
- class-prior fallback when a fold is degenerate.

These baselines are useful for three reasons:

1. detect leakage or unexpectedly predictive metadata;
2. create an inexpensive benchmark;
3. potentially provide weak ensemble diversity.

They should not replace the MRI image model.

---

# 15. Technique family M — augmentation

Augmentations appearing in public code include:

- small rotations;
- translation;
- scale;
- brightness/contrast variation;
- gamma variation;
- coarse dropout;
- horizontal flip.

For MRI, intensity transforms can help scanner/protocol robustness, but they should be modest.

Horizontal flipping should be tested rather than assumed harmless. The targets are not explicitly left/right-specific, but image orientation and anatomical conventions still warrant validation.

For 2.5D triplets, the same spatial transformation must be applied to all three channels.

---

# 16. Technique family N — efficiency engineering

The competition has practical runtime constraints, and public code contains several good engineering ideas.

## 16.1 Bound per-study work

Set explicit maxima for:

- series per study;
- slices/windows per series;
- image resolution;
- encoder batch size.

This makes worst-case runtime predictable.

## 16.2 Cache frozen embeddings

With DINOv2 or another frozen encoder:

```text
expensive image encoding once
    -> compressed embedding cache
    -> dozens of cheap head/loss/fold experiments
```

## 16.3 Runtime budget tracker

A public DINOv2 implementation monitors elapsed time and can downshift image resolution when the remaining budget becomes tight. This is a useful safety feature for notebook execution.

## 16.4 AMP and gradient accumulation

Use mixed precision and gradient accumulation to fit useful resolutions/backbones into T4/P100-class memory.

## 16.5 Checkpoint/resume

Training should save:

- model;
- optimizer;
- scheduler;
- epoch;
- best score;
- scaler/AMP state where relevant.

Long runs should be resumable across notebook sessions.

## 16.6 EMA

If exponential moving average weights are used, update EMA **per optimizer step**, not merely once per epoch.

---

# 17. What the public code suggests we should NOT do

## Do not treat unlabeled rows as negative

Empty structured labels mean **unknown**, not zero.

## Do not use report text as a mandatory test-time feature

The report is valuable supervision. The final model should remain image-capable by itself.

## Do not calibrate the report teacher on validation labels

This creates circular validation.

## Do not trust a large `pos_weight` automatically

It may amplify weak-label noise and cause systematic overprediction.

## Do not rely on filename DICOM ordering

Use physical geometry wherever possible.

## Do not train a very large 3D network first

Establish a trustworthy 2.5D baseline and only add 3D if it contributes new signal.

## Do not tune dozens of decisions against 58 gold studies

With such a small validation surface, repeated hyperparameter selection can overfit CV even without explicit data leakage.

## Do not accept README leaderboard claims without reproduction

Several public projects are very new. Distinguish:

- actual generated OOF metrics;
- self-reported Kaggle score;
- report-teacher AUC;
- synthetic/demo metrics;
- aspirational target scores.

They are not interchangeable.

---

# 18. Recommended `CNN_CPC` methodology after this review

The following is the strongest synthesis of the useful public ideas.

## Stage 0 — data audit

Before model work:

1. validate all CSV schemas;
2. count studies and series;
3. audit plane/sequence coverage;
4. sample DICOM codecs;
5. verify physical slice ordering;
6. detect duplicate reports;
7. freeze fold assignments.

## Stage 1 — report teacher

Create three label sets:

```text
A. multilingual rules
B. multilingual LLM soft labels
C. rank-ensemble(A,B)
```

For each target store:

```text
probability
confidence
mentioned / unmentioned state
teacher source
```

Evaluate each teacher against the gold set with the same CV discipline.

## Stage 2 — honest image baseline

Train an intentionally simple 3-plane model on gold only:

```text
3 adjacent slices per window
one primary series / plane
EfficientNet-B0 or ResNet18
attention or max pooling
12 logits
```

Purpose: establish an end-to-end DICOM reference and catch pipeline bugs.

## Stage 3 — weakly supervised image student

Train on all 4,407 studies:

```text
Gold studies:
    high-weight BCE against official labels

Report-only studies:
    confidence-weighted BCE/distillation against soft teacher

All useful batches:
    optional pairwise ranking term
```

Validation remains gold-only.

## Stage 4 — six-stream MRI routing

Move from one to up to two sequences per plane using fluid/fat-suppression metadata.

Use a missing-stream mask.

## Stage 5 — target-specific MIL

Replace one universal study attention with 12 target queries/heads.

Recommended representation:

```text
slice/window embedding
+ stream embedding
+ plane embedding
+ sequence metadata embedding
```

Then target query attention chooses diagnostic evidence for each pathology.

## Stage 6 — backbone experiments

Run controlled experiments with identical folds and sampling:

1. ResNet18 / EfficientNet-B0 reference;
2. ConvNeXt-Tiny;
3. DINOv2 ViT-S frozen;
4. DINOv2 partial fine-tune.

Do not change five components at once.

## Stage 7 — learned window selection

If encoding every window is too slow, add a cheap selector:

```text
candidate windows -> cheap scorer -> K windows -> expensive encoder
```

Measure both macro-AUC and inference runtime.

## Stage 8 — 3D complementary arm

Train a small per-plane 3D model. Keep it only if:

```text
AUC(2.5D + 3D ensemble) > AUC(2.5D)
```

by more than expected CV noise and with acceptable runtime.

## Stage 9 — ensemble

Preferred diversity sources:

- folds;
- ConvNeXt vs DINOv2;
- attention vs 3D;
- possibly 2.5D gaps/resolutions.

Rank averaging is particularly attractive because the metric is AUC.

---

# 19. Proposed upgraded architecture

A high-value architecture to test is:

```text
                   TRAINING ONLY
             Radiology Report
                    |
      rules --------+-------- LLM
                    |
                 rank ensemble
                    |
          soft targets + confidence
                    |
                    v

MRI study
  |
  +-- Sag FS/FSup ----- windows --+
  +-- Sag structural -- windows --+
  +-- Cor FS/FSup ----- windows --+
  +-- Cor structural -- windows --+--> shared image encoder
  +-- Ax FS/FSup ------ windows --+         |
  +-- Ax structural --- windows --+         v
                                      window tokens
                                           +
                                      plane/slot tokens
                                           |
                              12 target-specific queries
                                           |
                                  target MIL attention
                                           |
                                       12 logits
                                           |
          +--------------------------------+----------------------+
          |                                                       |
  gold weighted BCE                                 pseudo-label distillation
          |                                                       |
          +------------------ ranking loss ------------------------+
```

At inference, the report branch disappears completely.

---

# 20. Experiment matrix to run before increasing model complexity

| ID | Experiment | Single change | Decision criterion |
|---|---|---|---|
| E00 | Constant/prior baseline | None | Submission plumbing only |
| E01 | Gold-only ResNet18 | Image pipeline | Honest reference |
| E02 | Gold-only EfficientNet/ConvNeXt | Backbone | OOF macro-AUC + runtime |
| E03 | Rules weak labels | Add pseudo labels | Gold-only OOF improvement |
| E04 | LLM weak labels | Better teacher | Gold-only OOF improvement |
| E05 | Rule+LLM rank ensemble | Teacher ensemble | Gold-only OOF improvement |
| E06 | Gold weight sweep | 2, 4, 8 | Robust improvement, not one-fold spike |
| E07 | No `pos_weight` vs weighted | Loss | Per-class AUC + prediction distribution |
| E08 | Mean vs max vs attention MIL | Aggregator only | OOF macro-AUC |
| E09 | Target-specific attention | Aggregator | OOF macro-AUC, rare targets |
| E10 | One stream vs six streams | Sequence routing | OOF macro-AUC / runtime |
| E11 | 2.5D gap 1 vs 2 | Local context | OOF macro-AUC |
| E12 | Frozen DINOv2 | Representation | OOF macro-AUC / cache cost |
| E13 | DINOv2 partial fine-tune | Representation | Gain large enough for runtime cost |
| E14 | Pairwise rank loss | Metric alignment | Macro-AUC gain without instability |
| E15 | Top-K aggregation | Focal evidence | Macro-AUC + runtime |
| E16 | Learned Top-K selector | Efficiency | Pareto improvement AUC/runtime |
| E17 | Small 3D arm | Orthogonal model | Ensemble gain vs 2.5D |
| E18 | Multi-backbone ensemble | Diversity | OOF ensemble gain |

Every experiment should retain:

- exact code commit;
- config;
- fold assignment;
- OOF predictions;
- per-target AUC;
- macro AUC;
- bootstrap interval;
- runtime;
- GPU type;
- peak memory;
- checkpoint path.

---

# 21. Immediate changes worth bringing into `CNN_CPC`

The current repository already has several correct foundations: report pseudo-labeling, gold-only validation, multi-series attention and exact submission schema. Based on this review, the most valuable next upgrades are:

### Priority 0 — reliability

1. strengthen physical DICOM sorting using full orientation/position projection;
2. add DICOM codec/preflight checks;
3. persist selected-series manifests and ordered-path caches;
4. freeze grouped folds before serious experiments;
5. add bootstrap uncertainty around gold OOF macro-AUC.

### Priority 1 — supervision

6. add an optional stronger multilingual LLM teacher;
7. add fold-safe teacher calibration;
8. support teacher rank ensembling;
9. preserve per-target pseudo-label confidence;
10. make gold weight configurable and log it.

### Priority 2 — image representation

11. add 2.5D neighboring-slice triplets;
12. add ConvNeXt-Tiny;
13. add frozen DINOv2 + embedding cache;
14. add target-specific attention MIL;
15. support six sequence slots with masks.

### Priority 3 — metric/efficiency

16. add pairwise ranking loss as an optional auxiliary term;
17. add top-K window aggregation;
18. benchmark fixed-budget learned window selection;
19. add complete checkpoint/resume + time guard;
20. keep an experiment ledger with OOF artifacts.

### Priority 4 — ensemble research

21. small 3D per-plane arm;
22. rank-average heterogeneous models;
23. TTA only after measuring its runtime benefit.

---

# 22. Overall conclusion

The public code does **not** point to one magic CNN architecture. It points to a methodology hierarchy:

1. **Get supervision right.** The reports contain most of the usable training signal.
2. **Protect validation.** Fifty-eight gold studies are too valuable to contaminate through teacher calibration or duplicate leakage.
3. **Respect MRI structure.** Correct DICOM geometry, sequence routing and multi-plane information matter.
4. **Use local depth context.** 2.5D triplets are an excellent accuracy/compute compromise.
5. **Aggregate intelligently.** A lesion may exist in only a few slices; target-specific MIL or Top-K aggregation is more appropriate than blind averaging.
6. **Align training with AUC.** Confidence-weighted BCE plus a carefully tested ranking term is attractive.
7. **Cache and bound compute.** This is a code competition with a large DICOM dataset; engineering is part of model quality.
8. **Add complexity only when OOF supports it.** DINOv2, cross-attention and 3D models are promising, but none should be accepted by architectural intuition alone.

The strongest direction for `CNN_CPC` is therefore a **weakly supervised, multi-sequence, 2.5D image student with target-specific attention**, trained from a validated multilingual report teacher and evaluated through leakage-safe gold-only OOF prediction.

---

# 23. Reviewed public sources

- Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- https://github.com/Chagatai404/knee-abnormality-ML
- https://github.com/jiaweizhong/rsna-knee
- https://github.com/andreluizpedroso/rsna-knee-abnormality-detection
- https://github.com/chaitanyajamble/RSNA-Knee-Abnormality-Detection
- https://github.com/dianisay/RSNA-Knee-Abnormality-Detection
- https://github.com/soumic28/RSNA-knee-abnormality-predictin
- https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection
- https://github.com/Msmile-shiny/CVproject_RSNA-knee-detection
- https://github.com/Saianiruthm/rsna-knee-abnormality
- https://github.com/tomyimkc/sophia-agi/tree/main/agi-proof/contest-submissions/rsna-knee-abnormality
- https://github.com/bollimunthasripavan-oss/RSNA-Knee-MRI-Abnormality-Detection

This file should be updated as the competition matures and new public notebooks become accessible. Techniques should be promoted into the production pipeline only after they are reproduced under the frozen `CNN_CPC` validation protocol.