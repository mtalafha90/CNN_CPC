# References and reviewed public work

This file separates foundational technical references from public competition implementations used for methodology context. Public repositories are engineering/research references, not verified competition winners.

> Current repository-measured results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B5 is currently running and has no OOF score yet.

## Competition and standards

- **RSNA Knee Abnormality Detection**, Kaggle competition, 2026.
- Radiological Society of North America (RSNA), competition data/challenge materials.
- **DICOM Standard**, National Electrical Manufacturers Association. Relevant to orientation, position, rescale metadata, photometric interpretation and multi-frame handling.

## Knee MRI deep-learning background

- Bien, N. et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet.** *PLOS Medicine*, 2018. Historical precedent for multi-plane knee MRI slice aggregation and study-level prediction.

## Neural-network architecture references

- Liu, Z. et al. **A ConvNet for the 2020s.** CVPR, 2022. ConvNeXt; `CNN_CPC` uses a ConvNeXt-Tiny slice/triplet encoder.
- Vaswani, A. et al. **Attention Is All You Need.** NeurIPS, 2017. Transformer attention used by the neural Stage-1 architecture.
- Paszke, A. et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS, 2019. Primary deep-learning framework.

## Statistical evaluation

- Hanley, J. A. and McNeil, B. J. **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology*, 1982.
- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.** *Biometrics*, 1988.
- Efron, B. and Tibshirani, R. J. **An Introduction to the Bootstrap.** Chapman & Hall/CRC, 1993.

The current repository uses rank-based ROC AUC and study-level bootstrap intervals/paired comparisons. With only 58 gold studies, uncertainty must accompany point estimates.

## Classical representation/probe methods

B4 intentionally uses low-capacity classical tools after freezing the MRI encoder:

- PCA dimensionality reduction;
- balanced logistic regression;
- fixed anatomy/sequence feature subsets;
- rank averaging for fixed heterogeneous ensembles.

These are implemented with scikit-learn. Their role is diagnostic: test separability of a frozen representation without adding a high-capacity supervised neural head.

## B5 text-representation methods

B5 does **not** use an external clinical language model. It fits the text space only from the competition reports using:

- word TF-IDF with 1-2 grams;
- TruncatedSVD to a compact semantic space;
- L2-normalized report embeddings;
- image-report contrastive alignment plus cosine alignment;
- an embedding queue for additional report negatives in small MRI batches;
- duplicate-report-hash masking to avoid false negatives.

The text branch is training-only; the saved downstream artifact is an MRI encoder.

## Weak supervision context

The conservative report path uses:

- positive / negated / uncertain / unmentioned states;
- fold-safe calibration when gold labels are involved;
- official-label override;
- zero direct weight for report silence by default;
- confidence separated from target probability;
- compartment-aware OA parsing.

The supervised fold-safe report-teacher benchmark reached only `0.49245` macro OOF and was rejected as a general Stage-1 teacher. B5 therefore uses report **semantics for representation learning**, not the failed 12-target teacher probabilities.

## Strong competition-only MRI SSL

The strong SSL encoder was trained only on the 4,349 non-gold competition MRI studies and excludes all 58 gold studies. The completed run covered about 5.52 effective corpus passes.

This encoder supports B1, B2, B3, B4 and initializes B5.

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

## Repository-specific verified evidence — 2026-08-09

Data/engineering:

- 4,407 training studies;
- 58 gold and 4,349 report-only;
- 24,371 series rows;
- 21,886 selected series audited;
- 732,556 candidate DICOM files checked;
- two failed individual files and zero lost selected series;
- OA weak-supervision parsing verified;
- strong SSL completed on competition-only MRI.

Measured model evidence:

```text
B0    0.4763
B1    0.5030
B2    0.4993
B3    0.4945
B4    0.5138  best clean standalone point estimate
B4.1  0.4848
B4.2  0.4901
B4.3  0.4966
B1+B4 fixed rank  0.5167, statistically tied with B4
B5    running / pending
```

See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) for exact intervals and paired comparisons.

## Manuscript citation/reporting policy

Use primary literature for architecture/statistical methods. Cite public competition repositories only when they contribute relevant software/methodology context.

Do not:

- present a public repository's self-reported score as an established benchmark unless independently reproduced or clearly labelled;
- present smoke/preflight results as model performance;
- present current OOF as a pristine independent test estimate after it has informed multiple method choices;
- enter a B5 performance value before its frozen probe completes.
