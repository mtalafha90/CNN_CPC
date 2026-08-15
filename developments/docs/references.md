# References and reviewed public work

This file separates foundational technical references from public competition implementations used for methodology context. Public repositories are engineering/research references, not verified competition winners.

> **Current repository-measured status — 2026-08-12:** B13 is the reused-gold development champion at macro AUC `0.6293565948`. B15 passed the frozen weak-v2 teacher-agreement gate but scored `0.6209002783` on the one-look reused-gold confirmation and did not replace B13. Canonical results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Competition and standards

- **RSNA Knee Abnormality Detection**, Kaggle competition, 2026.
- Radiological Society of North America (RSNA), competition data/challenge materials.
- **DICOM Standard**, National Electrical Manufacturers Association. Relevant to orientation, position, rescale metadata, photometric interpretation and multi-frame handling.

## Knee MRI deep-learning background

- Bien, N. et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet.** *PLOS Medicine*, 2018. Historical precedent for multi-plane knee MRI slice aggregation and study-level prediction.

## Neural-network architecture references

- Liu, Z. et al. **A ConvNet for the 2020s.** CVPR, 2022. ConvNeXt; the repository uses ConvNeXt-Tiny as its 2.5D MRI encoder.
- Vaswani, A. et al. **Attention Is All You Need.** NeurIPS, 2017. Transformer attention used by the multi-series/pathology-query architectures.
- Paszke, A. et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS, 2019. Primary deep-learning framework.

B15 describes its same-study knee-MRI adaptation as **MICLe-style** because it uses multiple examples from the same knee study as positives. The repository does not claim to reproduce any published MICLe implementation exactly.

## Statistical evaluation

- Hanley, J. A. and McNeil, B. J. **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology*, 1982.
- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.** *Biometrics*, 1988.
- Efron, B. and Tibshirani, R. J. **An Introduction to the Bootstrap.** Chapman & Hall/CRC, 1993.

The repository uses rank-based ROC AUC and study-level bootstrap intervals/paired comparisons. With only 58 expert-gold studies, uncertainty must accompany point estimates.

Weak-v2 uses a stricter bootstrap: a replicate is accepted only when all 12 target AUCs are defined.

## Classical representation/probe methods

B4/B5 use low-capacity classical tools after freezing the MRI encoder:

- PCA dimensionality reduction;
- balanced logistic regression;
- fixed anatomy/sequence feature subsets;
- rank averaging for fixed heterogeneous ensembles.

These are implemented with scikit-learn and are used diagnostically to test representation separability without a high-capacity supervised neural head.

## Report representation and weak supervision

B5 uses competition-report TF-IDF/TruncatedSVD semantic embeddings for image-report alignment without an external clinical language model.

B6 v1.2.1 uses structured states:

```text
positive
negated
uncertain
unmentioned
```

Frozen B6 scope:

```text
active studies  3120
usable cells   14123
positive        6871
negative        7252
```

Frozen B7-B15 downstream treatment:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

The current post-B15 diagnostic is to measure how all four states relate to expert truth before any new supervision policy is defined. Report silence is not assumed to be a negative.

## Image-side experiment lineage

The repository has tested:

- competition-only MRI SSL;
- image-report representation alignment;
- full weak-corpus training coverage;
- spatial tokens;
- strict semantic routing;
- physical-scale normalization;
- pseudo-label completion;
- all-real-series aggregation;
- hierarchical one-token-per-series aggregation;
- ImageNet initialization;
- full slice-token memory;
- ImageNet -> knee-MRI same-study contrastive adaptation.

The exact B13 slice audit found median evaluation exposure `100%` and complete evaluation exposure for `95.9%` of 17,475 eligible real series, rejecting slice-count undersampling as the primary B13 bottleneck.

## Current measured evidence

```text
B0                   0.4762536432
B1                   0.5030284974
B2                   0.4993244663
B3                   0.4944652486
B4                   0.5137567459
B5                   0.5243650851
B7-v1                0.5397724412
B7.1                 0.5644802945
B5+B7.1 rank         0.5540141184
B8                   0.5300962807
B9                   0.5334962669
B10                  0.5523982721
B11.1                0.5506902702
B12                  0.5660915179
B13                  0.6293565948  retained champion
B14                  0.6197914249
B15                  0.6209002783
```

B11-v1 failed viability; B12.1 was implemented but skipped.

## B15 weak-v2 evidence

```text
B13-v2 control       0.5652498118
B15                 0.7319060415
raw delta           +0.1666562297
paired median       +0.1675245839
95% paired CI       [+0.1124433208,+0.2165156305]
P(B15 > control)     1.0000
```

The predeclared weak gate passed, but the one-look expert-gold result was:

```text
B15 gold            0.6209002783
B13 gold            0.6293565948
```

This discrepancy is now a central scientific finding: stronger agreement with the report-derived teacher did not produce a stronger global expert-label ranking.

## Early public 2026 competition repositories reviewed

The methodology review examined public code from projects including:

- `Chagatai404/knee-abnormality-ML`
- `jiaweizhong/rsna-knee`
- `andreluizpedroso/rsna-knee-abnormality-detection`
- `chaitanyajamble/RSNA-Knee-Abnormality-Detection`
- `dianisay/RSNA-Knee-Abnormality-Detection`
- `soumic28/RSNA-knee-abnormality-predictin`
- `JunhaoLiXD/RSNA_Knee_Abnormality_Detection`
- `Msmile-shiny/CVproject_RSNA-knee-detection`
- `Saianiruthm/rsna-knee-abnormality`
- `tomyimkc/sophia-agi` RSNA knee subproject
- `bollimunthasripavan-oss/RSNA-Knee-MRI-Abnormality-Detection`

See [`../README_KAGGLE_METHODS.md`](../README_KAGGLE_METHODS.md) for the synthesis.

## How public work is used

Public implementations are used to cross-check data structure, DICOM failure modes, multi-plane/2.5D ideas, weak report supervision, representation learning, leakage risks and engineering patterns. A public idea is not treated as an improvement until tested under this repository's own controlled protocol.

## Manuscript/reporting policy

Use primary literature for architecture/statistical methods. Cite public competition repositories only when they contribute relevant software/methodology context.

Do not:

- present a public repository's self-reported score as an established benchmark unless independently reproduced or clearly labelled;
- present preflight/technical-fixture results as model performance;
- present reused-gold development results as pristine independent validation;
- present weak-v2 teacher agreement as expert truth;
- select target-specific winners or gold-tuned ensemble weights;
- claim B15's weak-v2 result is a hidden-test result.

The hidden Kaggle evaluation remains the next genuinely independent model-performance signal.