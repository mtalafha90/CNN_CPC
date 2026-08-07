# References and reviewed public work

## Competition
- Kaggle: **RSNA Knee Abnormality Detection**, 2026.
- Radiological Society of North America (RSNA), knee MRI AI challenge materials.

## Knee MRI deep learning background
- Bien, N. et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet.** PLOS Medicine, 2018. MRNet is the main architectural precedent for processing knee MRI slices and aggregating information across sagittal, coronal and axial series.

## Public early 2026 competition repositories reviewed
These are early community implementations, **not competition winners**:
- `dianisay/RSNA-Knee-Abnormality-Detection` — multilingual report labeling, DICOM preprocessing and image-model workflow.
- `soumic28/RSNA-knee-abnormality-predictin` — 2.5D slice attention, multi-plane routing ideas and Kaggle submission tooling.
- `tomyimkc/sophia-agi` RSNA knee work — detailed audits of the released schema, gold/unlabeled counts, report distillation, MRNet/3D scaffolding and an explicit gap tracker.

Because the challenge is still active at this snapshot, performance claims in public repositories may be unverified. This project uses them to cross-check data structure and identify engineering ideas; final data counts should be rechecked against the downloaded Kaggle files.
