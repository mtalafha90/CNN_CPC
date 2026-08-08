# References and reviewed public work

This file separates foundational technical references from early public competition implementations used for methodology review. Public repositories are cited as engineering/context sources, not as verified competition winners.

## Competition and standards

- **RSNA Knee Abnormality Detection**, Kaggle competition, 2026.
- Radiological Society of North America (RSNA), competition data and challenge materials.
- **DICOM Standard**, National Electrical Manufacturers Association. Relevant to image orientation, position, rescale metadata, photometric interpretation and multi-frame image handling.

## Knee MRI deep learning background

- Bien, N. et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet.** *PLOS Medicine*, 2018. Historical precedent for multi-plane knee MRI slice aggregation and study-level prediction.

## Neural-network architecture references

- Liu, Z. et al. **A ConvNet for the 2020s.** CVPR, 2022. ConvNeXt architecture; `CNN_CPC` uses a ConvNeXt-Tiny slice/triplet encoder.
- Vaswani, A. et al. **Attention Is All You Need.** NeurIPS, 2017. Transformer attention used conceptually by the cross-sequence and pathology-context modules.
- Paszke, A. et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** NeurIPS, 2019. Primary deep-learning framework.

## ROC-AUC and uncertainty

- Hanley, J. A. and McNeil, B. J. **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology*, 1982.
- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.** *Biometrics*, 1988.
- Efron, B. and Tibshirani, R. J. **An Introduction to the Bootstrap.** Chapman & Hall/CRC, 1993.

The current code uses rank-based AUC calculation and study-level bootstrap intervals/paired comparisons rather than claiming asymptotic certainty from the very small 58-study gold set.

## Weak supervision and calibration context

The repository's report teacher is deliberately conservative:

- four states: positive, negated, uncertain, unmentioned;
- fold-safe state calibration;
- official labels override pseudo-labels;
- unmentioned direct weight is zero by default;
- pseudo-label confidence is separated from target probability.

The OA parser was expanded only after a real-data audit showed that the original narrow lexicon produced zero usable OA report supervision.

## Early public 2026 competition repositories reviewed

The methodology review examined public code from the following projects where available:

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

See `../README_KAGGLE_METHODS.md` for the methodological synthesis.

## How public work is used here

Public implementations are used to:

- cross-check the released data structure;
- identify common DICOM failure modes;
- compare multi-plane and 2.5D representations;
- motivate weak report supervision;
- motivate target-specific aggregation and ranking objectives;
- identify leakage risks in tiny-gold validation;
- identify runtime/engineering patterns worth testing.

They are **not** used as evidence that a component improves `CNN_CPC` until that component is tested in the repository's own leakage-aware validation protocol.

## Repository-specific verified evidence

As of 2026-08-08, the project's own reproducible evidence includes:

- 4,407 training studies, 58 gold, 4,349 report-only;
- 24,371 series metadata rows;
- 21,886 selected training series audited;
- 732,556 candidate DICOM files checked;
- only two failed individual DICOM files, with zero failed selected series;
- compartment-aware OA report supervision verified on the real reports;
- pair-friendly trusted sampling verified to activate ranking loss for all 12 targets in a fold-0 smoke run;
- complete Stage-1 smoke artifact generation on an NVIDIA RTX A4500 Laptop GPU using BF16.

These are engineering/data-validation facts. They are not a substitute for the pending three-fold non-smoke OOF and actual leaderboard evaluation.

## Citation policy for the manuscript

The CPC manuscript should cite primary literature for the architectural/statistical methods and use public competition repositories only where they provide genuinely relevant software/methodology context.

Do not cite an early public repository's self-reported score as an established benchmark unless independently reproduced or clearly labeled as self-reported.