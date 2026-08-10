# References and reviewed public work

This file separates foundational technical references from public competition implementations used for methodology context. Public repositories are engineering/research references, not verified competition winners.

> Current repository-measured results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). **B7.1 is the current best standalone development model at macro AUC `0.5644802945`; B8 spatial-anatomy learning is currently training and has no gold score yet.**

## Competition and standards

- **RSNA Knee Abnormality Detection**, Kaggle competition, 2026.
- Radiological Society of North America (RSNA), competition data/challenge materials.
- **DICOM Standard**, National Electrical Manufacturers Association. Relevant to orientation, position, rescale metadata, photometric interpretation and multi-frame handling.

## Knee MRI deep-learning background

- Bien, N. et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet.** *PLOS Medicine*, 2018. Historical precedent for multi-plane knee MRI slice aggregation and study-level prediction.

## Neural-network architecture references

- Liu, Z. et al. **A ConvNet for the 2020s.** CVPR, 2022. ConvNeXt; `CNN_CPC` uses a ConvNeXt-Tiny 2.5D slice/triplet encoder.
- Vaswani, A. et al. **Attention Is All You Need.** NeurIPS, 2017. Transformer attention used by the multi-sequence MRI/pathology-query architecture.
- Paszke, A. et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS, 2019. Primary deep-learning framework.

## Statistical evaluation

- Hanley, J. A. and McNeil, B. J. **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology*, 1982.
- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.** *Biometrics*, 1988.
- Efron, B. and Tibshirani, R. J. **An Introduction to the Bootstrap.** Chapman & Hall/CRC, 1993.

The repository uses rank-based ROC AUC and study-level bootstrap intervals/paired comparisons. With only 58 gold studies, uncertainty must accompany point estimates.

## Classical representation/probe methods

B4/B5 use low-capacity classical tools after freezing the MRI encoder:

- PCA dimensionality reduction;
- balanced logistic regression;
- fixed anatomy/sequence feature subsets;
- rank averaging for fixed heterogeneous ensembles.

These are implemented with scikit-learn. Their role is diagnostic: test separability of a frozen representation without adding a high-capacity supervised neural head.

## B5 text-representation methods

B5 does **not** use an external clinical language model. It fits the text space only from competition reports using:

- word TF-IDF with 1-2 grams;
- TruncatedSVD to a compact semantic space;
- L2-normalized report embeddings;
- image-report contrastive alignment plus cosine alignment;
- an embedding queue for additional report negatives in small MRI batches;
- duplicate-report-hash masking to avoid false negatives.

The text branch is training-only; the saved downstream artifact is an MRI encoder.

B5 result under the unchanged B4 probe:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
```

B5 remains the report-aligned representation baseline and the encoder source for B7.

## Structured weak supervision context

B6 v1.2.1 uses:

- positive / negated / uncertain / unmentioned states;
- zero training weight for uncertain/unmentioned cells;
- no conversion of report silence to negative;
- confidence separated from target probability;
- compartment-aware OA parsing;
- no external language model/resource;
- zero gold rows in the weak-training export.

The final frozen report-only export contains:

```text
active studies  3120
usable cells   14123
positive        6871
negative        7252
```

The completed gold audit showed asymmetric reliability, motivating the global B7/B7.1/B8 policy:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

Because that policy was informed by the same 58-study audit set, later B7/B7.1/B8 scores are development/model-selection estimates.

## Strong competition-only MRI SSL

The strong SSL encoder was trained only on the 4,349 non-gold competition MRI studies and excludes all 58 gold studies. The completed run covered about 5.52 effective corpus passes.

This encoder supports B1/B4 and initializes B5; B5 then initializes B7.

## B7/B7.1 direct weak supervision

B7 combines:

```text
B5-initialized ConvNeXt
+ six MRI streams
+ slice/stream embeddings
+ cross-sequence Transformer
+ 12 pathology queries
+ frozen B6 target-level weak labels
```

B7-v1:

```text
macro AUC = 0.5397724412
```

B7.1 changes only training coverage to one complete 3,120-study pass per epoch and reaches:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

B7.1 is the current retained standalone development leader.

## Spatial anatomy context for B8

B8 tests a different representation question: whether global pooling of every sampled slice discards useful pathology-localization information.

```text
B7.1 MRI memory = 6 x 16 x 1   = 96 tokens/study
B8 MRI memory   = 6 x 16 x 2x2 = 384 tokens/study
```

B8 preserves a 2x2 spatial grid from the final ConvNeXt feature map, adds learned region-position embeddings, and applies fixed gentle pathology-specific stream/slice attention priors. No fixed medial/lateral/anterior/posterior quadrant is assumed because the preprocessing contract does not certify canonical in-plane orientation.

B8 initializes from the completed B7.1 checkpoint and keeps the B6 weak-label policy/full-corpus training recipe fixed.

**Current status: B8 real-data training is in progress; no B8 gold result exists yet.**

## Early public 2026 competition repositories reviewed

The methodology review examined public code from these projects where available:

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

Public implementations are used to:

- cross-check released data structure;
- identify DICOM failure modes;
- compare multi-plane/2.5D representation ideas;
- motivate weak report supervision and MRI representation learning;
- identify leakage risks in tiny-gold validation;
- identify engineering/runtime patterns worth testing.

A public idea is not treated as an improvement until tested under this repository's own validation protocol.

## Repository-specific verified evidence — 2026-08-10

Data/engineering:

- 4,407 training studies;
- 58 gold and 4,349 report-only;
- 24,371 series rows;
- 21,886 selected series audited;
- 732,556 candidate DICOM files checked;
- two failed individual files and zero lost selected series;
- B6 v1.2.1 frozen at 14,123 usable weak-label cells;
- strong SSL/B5/B7/B7.1 completed on competition-only data.

Measured model evidence:

```text
B0                   0.4762536432
B1                   0.5030284974
B2                   0.4993244663
B3                   0.4944652486
B4                   0.5137567459
B5                   0.5243650851
B7-v1                0.5397724412
B7.1                 0.5644802945  current leader
B5+B7.1 fixed rank   0.5540141184  rejected
B8                   pending         training in progress
```

See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for exact intervals and paired comparisons.

## Manuscript citation/reporting policy

Use primary literature for architecture/statistical methods. Cite public competition repositories only when they contribute relevant software/methodology context.

Do not:

- present a public repository's self-reported score as an established benchmark unless independently reproduced or clearly labelled;
- present smoke/preflight results as model performance;
- present current gold/OOF development results as pristine independent test estimates after they have informed multiple method choices;
- report a B8 performance value before its frozen training run and first gold evaluation complete;
- tune B8 spatial/prior hyperparameters from the first B8 result and then describe the re-evaluation as untouched.
