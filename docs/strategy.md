# Modeling strategy

## Core principle

The competition is a weakly supervised multi-sequence MRI problem with a very small trusted gold set. The production strategy therefore prioritizes **supervision quality and validation discipline before model scale**.

## 1. Weak supervision is positive-unlabeled aware

Reports are a training teacher, not a required inference modality. Each target/report pair is classified as positive, negated, uncertain, or unmentioned. Fold-safe calibration estimates `P(y=1 | state)` using only gold studies allowed in the current training phase.

Confidence is not simply `n/(n+alpha)`. It also measures how far the state-specific probability lies from the target prevalence. A state can therefore be frequent but receive little weight when it is not diagnostically informative.

`unmentioned` receives zero direct BCE weight by default. Report silence is treated as unlabeled rather than a weak negative. Official target cells always override the teacher cell-by-cell.

## 2. Nested gold validation

Each outer fold has three roles:

```text
outer gold fold       -> final untouched OOF evaluation
inner gold fold       -> choose training duration only
remaining gold fold   -> phase-A trusted training
```

After the best epoch is selected, the phase-A model is discarded. A fresh model is initialized and retrained for exactly the selected number of epochs using **all non-outer gold studies**. Only then is the outer fold evaluated.

This avoids using the outer fold for early stopping while recovering the inner gold cases for final outer-fold training.

## 3. Metric-aligned objective

The competition metric is macro ROC-AUC, where each pathology has equal weight. BCE is therefore normalized separately within each target and only then averaged across the 12 targets. Targets with more confident pseudo-label cells cannot dominate the loss simply because they have more supervision mass.

A confidence-gated pairwise ranking term complements BCE. With DDP, logits/targets/weights are gathered into the global batch so rare targets have a realistic chance of containing trusted positive-negative pairs.

## 4. Global trusted/general sampling

Training uses two study pools:

- trusted: gold plus unusually reliable pseudo-labeled studies;
- general: the remaining weakly supervised studies.

A deterministic global batch is constructed first and then sharded across DDP ranks. This preserves the requested trusted fraction and avoids independent ranks drawing the same sample in the same step when the pools are sufficiently large.

## 5. In-domain MRI self-supervision

Optional SSL uses only non-gold MRI studies by default. Different sequences from the same knee are positive pairs for an anatomy contrastive objective. Auxiliary heads preserve acquisition information by predicting plane and fluid/structural sequence type.

The saved ConvNeXt encoder can initialize supervised training with:

```yaml
ssl_encoder_checkpoint: runs/ssl/ssl_encoder.pt
```

This uses the released MRI distribution itself rather than relying solely on natural-image pretraining.

## 6. MRI representation and fusion

The production input is up to six semantic streams:

- sagittal fluid-sensitive
- sagittal structural
- coronal fluid-sensitive
- coronal structural
- axial fluid-sensitive
- axial structural

Each selected series is represented by distributed 2.5D triplets. During training, triplet gap and center positions vary mildly. ConvNeXt encodes each active triplet; missing streams are never sent through the backbone.

All active MRI tokens then pass through a Transformer before classification. This permits direct cross-plane and cross-sequence interaction rather than only late pooling.

## 7. Interacting pathology queries

The 12 abnormalities are represented by learned pathology query tokens. They first self-attend, allowing label relationships to be learned, and then cross-attend to the contextualized MRI tokens. Each pathology can retrieve evidence from any plane/sequence while preserving target identity.

## 8. MRI-specific augmentation and TTA

Training uses mild acquisition-compatible perturbations: center jitter, triplet gaps 1-2, small rotation/translation/scale, gamma variation, low-frequency bias field, noise, and slice dropout.

Inference is deterministic and averages slice-center offsets `[-1, 0, +1]` by default.

## 9. Cross-fitted co-training

Every non-gold normalized report group receives a deterministic cross-fit fold. During stage 1, fold `k` excludes its own weak cross-fit subset and writes image predictions to `weak_oof.csv`.

Across all three folds, these files cover non-gold studies with predictions from models that never trained on their report group. Stage 2 combines those independent image probabilities with the current fold's calibrated report teacher:

- strong agreement -> higher confidence;
- direct disagreement -> very low confidence;
- gold cells -> official labels, always.

This avoids trivial self-confirmation.

## 10. Statistical reporting

Outer OOF uses raw official cells with NaNs preserved. Report:

- per-target AUC;
- macro AUC;
- study-level bootstrap interval;
- paired bootstrap delta for controlled comparisons.

No methodology change should be called an improvement until it is measured on the same outer OOF studies.

## Recommended execution order

```text
inspect + preflight
  -> optional non-gold SSL
  -> stage-1 3-fold nested/DDP training
  -> collect weak_oof.csv from all folds
  -> stage-2 co-training 3-fold run
  -> combined outer OOF evaluation
  -> paired bootstrap stage2 vs stage1
  -> MRI-only TTA submission
```

Avoid treating missing labels/report silence as negatives, selecting epochs on outer OOF, mixing non-cross-fitted image predictions into pseudo-labels, or claiming gains before real OOF measurements exist.
