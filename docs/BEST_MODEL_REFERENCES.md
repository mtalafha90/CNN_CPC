# Scientific references for the current best-model lineage (B17/B18)

> **Scope — 2026-08-13.** This is the canonical bibliography for the model lineage that produced B17 and the currently running B18 Fisher-style checkpoint-selection experiment. B17 remains the reference checkpoint from the statistically unresolved B13--B17 development tier; B18 changes checkpoint selection, not the underlying MRI architecture or B6 gradient supervision.
>
> This file distinguishes **direct methodological references** from **supporting/context references**. A paper being listed here does not mean its implementation was reproduced exactly. Where the repository uses only the underlying idea, that is stated explicitly.

## 1. Model lineage at a glance

```text
ImageNet supervised initialization
        ↓
ConvNeXt-Tiny 2.5D MRI encoder
        ↓
B15 same-study knee-MRI contrastive adaptation
        ↓
B16 full-report semantic alignment
        ↓
B13/B16 hierarchical one-token-per-series aggregation
        ↓
B17 frozen encoder + B6 weak pathology supervision
        ↓
B18 same frozen model + expert-guided global epoch selection
```

### Reference-to-component map

| Model component | Main reference(s) | How it is used in this repository |
|---|---|---|
| Knee MRI study-level deep learning | Bien et al. (2018), MRNet | Domain precedent for aggregating multiple MRI slices/series into study-level knee predictions. Our model is not MRNet and uses a different encoder/aggregation strategy. |
| Generic visual pretraining | Deng et al. (2009), ImageNet | B13 onward initialize the ConvNeXt-Tiny encoder with torchvision ImageNet-1K weights. |
| ConvNeXt encoder | Liu et al. (2022) | ConvNeXt-Tiny is the image encoder used by the B13--B18 lineage. |
| Transformer context modeling | Vaswani et al. (2017) | Transformer encoder blocks are used after per-series compression and in pathology-context modeling. |
| Attention-based multiple-instance aggregation | Ilse et al. (2018) | Conceptual basis for learned permutation-tolerant instance aggregation. The repository implements its own learned series attention pooling. |
| Generic contrastive SSL | Chen et al. (2020), SimCLR | Contrastive representation-learning foundation for the B15 same-study MRI adaptation. |
| Medical multi-instance contrastive SSL | Azizi et al. (2021), MICLe | Closest published analogue to B15: different images/instances from the same medical case are used as positive pairs. B15 is MICLe-style, not a reproduction of MICLe. |
| Image/text contrastive alignment | Radford et al. (2021), CLIP | Conceptual support for aligning visual features with report-derived semantic vectors. B16 does **not** use CLIP weights or a CLIP text encoder. |
| TF-IDF report representation | Salton & Buckley (1988) | B16 report semantics begin from word TF-IDF features. |
| Truncated-SVD / latent semantic representation | Deerwester et al. (1990) | B16 compresses report TF-IDF into a low-dimensional semantic space with TruncatedSVD before image-report alignment. |
| Report-derived weak labels with uncertainty states | Irvin et al. (2019), CheXpert | Important precedent for deriving structured positive/negative/uncertain observations from radiology reports and treating uncertainty explicitly. |
| Negation and uncertainty extraction | Peng et al. (2018), NegBio | Methodological precedent for distinguishing positive, negated and uncertain radiology-report findings. B6 uses a custom frozen parser, not NegBio itself. |
| Report-derived imaging labels at scale | Johnson et al. (2019), MIMIC-CXR-JPG | Supporting evidence for large-scale report-derived image supervision and the distinction between report labels and expert labels. |
| Frozen-backbone / short-training / expert-compass protocol | Fisher (2026) | Direct motivation for B18: frozen encoder, short candidate training window, expert-labelled data used as a checkpoint-selection compass rather than gradient supervision. This is a **preprint**, not peer-reviewed evidence. |
| PyTorch implementation | Paszke et al. (2019) | Core deep-learning framework. |
| scikit-learn classical text/probe tools | Pedregosa et al. (2011) | TF-IDF/SVD/logistic-regression utilities used in representation and diagnostic stages. |
| ROC-AUC interpretation | Hanley & McNeil (1982) | Statistical background for ROC-AUC as the primary ranking metric. |
| Bootstrap uncertainty | Efron & Tibshirani (1993) | Basis for the study-level bootstrap uncertainty and paired bootstrap comparisons used throughout development. |

---

## 2. Competition / dataset source

### RSNA 2026 Knee MRI AI Challenge

Radiological Society of North America. **RSNA Knee MRI AI Challenge (2026).**

- Official challenge page: https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- Role in project: source task/data definition and clinical target context.
- Repository rule: the **actual released CSV/DICOM schema** is the operational source of truth for model inputs; challenge-description wording is not used to invent unavailable test-time fields.

---

## 3. Knee MRI deep-learning foundation

### Bien et al. — MRNet

Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., Bereket, M., Patel, B. N., Yeom, K. W., Shpanskaya, K., Halabi, S., Zucker, E., Fanton, G., Amanatullah, D. F., Beaulieu, C. F., Riley, G. M., Stewart, R. J., Blankenberg, F. G., Larson, D. B., Jones, R. H., Langlotz, C. P., Ng, A. Y., & Lungren, M. P. (2018). **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.** *PLOS Medicine, 15*(11), e1002699.

- DOI: https://doi.org/10.1371/journal.pmed.1002699
- PubMed: https://pubmed.ncbi.nlm.nih.gov/30481176/
- Relevance: established study-level knee-MRI deep learning using multiple MRI series and slice aggregation.
- Difference from B17/B18: our pipeline uses 2.5D ConvNeXt features, all recognized real series, learned per-series attention pooling, a study Transformer, pathology queries, report-driven representation learning and multi-label outputs.

---

## 4. Encoder and generic visual pretraining

### Deng et al. — ImageNet

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). **ImageNet: A large-scale hierarchical image database.** *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 248--255.

- DOI: https://doi.org/10.1109/CVPR.2009.5206848
- Relevance: source dataset for the supervised initialization used by torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` in B13 onward.

### Liu et al. — ConvNeXt

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). **A ConvNet for the 2020s.** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

- arXiv: https://arxiv.org/abs/2201.03545
- Relevance: architecture family for the ConvNeXt-Tiny MRI encoder used by B13--B18.

---

## 5. Hierarchical study aggregation

### Vaswani et al. — Transformer

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need.** *Advances in Neural Information Processing Systems, 30*.

- Paper: https://papers.nips.cc/paper/7181-attention-is-all-you-need
- Relevance: self-attention/Transformer mechanism underlying the study-level and pathology-context blocks.

### Ilse et al. — Attention-based MIL

Ilse, M., Tomczak, J. M., & Welling, M. (2018). **Attention-based Deep Multiple Instance Learning.** *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2127--2136.

- Paper: https://proceedings.mlr.press/v80/ilse18a.html
- Relevance: methodological foundation for learnable, permutation-invariant attention aggregation of multiple instances into a bag/study representation.
- Repository-specific implementation: each MRI series is represented by 16 slice tokens, compressed to one learned series token, then all real series are contextualized by the study Transformer.

---

## 6. Self-supervised MRI representation learning

### Chen et al. — SimCLR

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). **A Simple Framework for Contrastive Learning of Visual Representations.** *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 1597--1607.

- Paper: https://proceedings.mlr.press/v119/chen20j.html
- Relevance: general contrastive-learning framework underlying B15's representation-learning logic.

### Azizi et al. — MICLe

Azizi, S., Mustafa, B., Ryan, F., Beaver, Z., Freyberg, J., Deaton, J., Loh, A., Karthikesalingam, A., Kornblith, S., Chen, T., Natarajan, V., & Norouzi, M. (2021). **Big Self-Supervised Models Advance Medical Image Classification.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 3478--3488.

- DOI: https://doi.org/10.1109/ICCV48922.2021.00346
- CVF: https://openaccess.thecvf.com/content/ICCV2021/html/Azizi_Big_Self-Supervised_Models_Advance_Medical_Image_Classification_ICCV_2021_paper.html
- Relevance: introduces Multi-Instance Contrastive Learning (MICLe), which forms positive pairs from multiple images belonging to the same medical case.
- Repository relationship: B15's same-study knee-MRI contrastive adaptation is **MICLe-style** because different MRI instances from the same knee study form positives. It is not claimed to reproduce the original MICLe implementation exactly.

---

## 7. Full-report semantic alignment

### Radford et al. — CLIP / image-text alignment

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). **Learning Transferable Visual Models From Natural Language Supervision.** *Proceedings of the 38th International Conference on Machine Learning*.

- arXiv: https://arxiv.org/abs/2103.00020
- Relevance: conceptual precedent for contrastively aligning images with text-derived semantic representations.
- Important difference: B16 does **not** use CLIP, a transformer language model, or external clinical text embeddings. Its report branch is competition-only TF-IDF + TruncatedSVD and is discarded before downstream MRI-only inference.

### Salton & Buckley — TF-IDF / term weighting

Salton, G., & Buckley, C. (1988). **Term-weighting approaches in automatic text retrieval.** *Information Processing & Management, 24*(5), 513--523.

- DOI: https://doi.org/10.1016/0306-4573(88)90021-0
- Relevance: classical term-weighting foundation for the TF-IDF report representation used in B16.

### Deerwester et al. — Latent Semantic Analysis / SVD

Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). **Indexing by latent semantic analysis.** *Journal of the American Society for Information Science, 41*(6), 391--407.

- DOI: https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9
- Relevance: singular-value decomposition of term-document representations into a lower-dimensional semantic space; conceptually matches B16's TF-IDF -> TruncatedSVD report representation.

---

## 8. Weak supervision from radiology reports

### Irvin et al. — CheXpert

Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., Seekins, J., Mong, D. A., Halabi, S. S., Sandberg, J. K., Jones, R., Larson, D. B., Langlotz, C. P., Patel, B. N., Lungren, M. P., & Ng, A. Y. (2019). **CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.** *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(01), 590--597.

- DOI: https://doi.org/10.1609/aaai.v33i01.3301590
- Relevance: major precedent for radiology-report-derived labels that explicitly distinguish uncertainty instead of collapsing every unmentioned/uncertain finding to a hard negative.
- Repository relationship: B6 uses four states (`positive`, `negated`, `uncertain`, `unmentioned`) and only high-confidence positive/negated cells contribute to the B17/B18 downstream pathology loss.

### Peng et al. — NegBio

Peng, Y., Wang, X., Lu, L., Bagheri, M., Summers, R. M., & Lu, Z. (2018). **NegBio: a high-performance tool for negation and uncertainty detection in radiology reports.** *AMIA Joint Summits on Translational Science Proceedings*, 2017, 188--196.

- PubMed: https://pubmed.ncbi.nlm.nih.gov/29888070/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5961822/
- Relevance: radiology-specific methodology for separating positive, negated and uncertain findings.
- Important difference: B6 is a custom frozen parser audited on the 58-study development set; the repository does not run NegBio itself.

### Johnson et al. — MIMIC-CXR-JPG

Johnson, A. E. W., Pollard, T. J., Greenbaum, N. R., Lungren, M. P., Deng, C.-Y., Peng, Y., Lu, Z., Mark, R. G., Berkowitz, S. J., & Horng, S. (2019/2020). **MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs.**

- arXiv: https://arxiv.org/abs/1901.07042
- Dataset background: https://physionet.org/content/mimic-cxr-jpg/
- Relevance: large-scale example of deriving image labels from free-text reports with NLP tools; supports the project's distinction between weak report-derived supervision and expert-reference labels.

---

## 9. B18 expert-guided short-training rationale

### Fisher — NLP-to-Expert gap

Fisher, G. (2026). **The NLP-to-Expert Gap in Chest X-ray AI.** *medRxiv* preprint 2026.02.27.26347261.

- DOI: https://doi.org/10.64898/2026.02.27.26347261
- Full text: https://www.medrxiv.org/content/10.64898/2026.02.27.26347261v1.full
- Status: **preprint; not peer reviewed at the time this protocol was defined.**
- Relevance to B18: motivates testing whether a frozen visual backbone and very short downstream training can reduce overfitting to report/NLP-derived labels, while using a small expert-labelled set as a checkpoint-selection compass rather than gradient supervision.
- Repository-specific B18 protocol:
  - frozen B16 report-aligned encoder;
  - B6-only gradients on the same 3,120 studies / 14,123 usable cells / 17,475 MRI series;
  - five fixed candidate epochs;
  - one global 12-target macro AUC on the 58 expert studies after each epoch;
  - global epoch selection only, earliest exact tie wins;
  - expert labels never enter gradients;
  - the selected expert score is **not** reported as independent validation performance.
- Important difference: no new generic label smoothing is added in B18 because the B6 positive/negative targets are already soft (`0.85/0.05`). This isolates expert-guided short checkpoint selection as the intervention.

---

## 10. Software/framework references

### Paszke et al. — PyTorch

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** *Advances in Neural Information Processing Systems, 32*.

- Paper: https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library
- Relevance: deep-learning implementation framework.

### Pedregosa et al. — scikit-learn

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). **Scikit-learn: Machine Learning in Python.** *Journal of Machine Learning Research, 12*, 2825--2830.

- Paper: https://jmlr.org/papers/v12/pedregosa11a.html
- Relevance: TF-IDF, TruncatedSVD and classical diagnostic/probe utilities.

### DICOM Standard

National Electrical Manufacturers Association (NEMA). **Digital Imaging and Communications in Medicine (DICOM) Standard.**

- Standard: https://www.dicomstandard.org/current
- Relevance: DICOM decoding, orientation/position handling, rescale metadata, photometric interpretation and multi-frame handling in the MRI preprocessing pipeline.

---

## 11. Evaluation references

### Hanley & McNeil — ROC AUC

Hanley, J. A., & McNeil, B. J. (1982). **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology, 143*(1), 29--36.

- DOI: https://doi.org/10.1148/radiology.143.1.7063747
- Relevance: classical interpretation of ROC AUC, the competition/development ranking metric.

### Efron & Tibshirani — Bootstrap

Efron, B., & Tibshirani, R. J. (1993). **An Introduction to the Bootstrap.** Chapman & Hall/CRC.

- DOI: https://doi.org/10.1201/9780429246593
- Relevance: conceptual basis for the repository's study-level bootstrap confidence intervals and paired bootstrap comparisons.

### DeLong et al. — Correlated ROC curves (background only)

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.** *Biometrics, 44*(3), 837--845.

- DOI: https://doi.org/10.2307/2531595
- Role: statistical background for correlated ROC comparisons. The current repository's principal uncertainty procedure is the paired **study-level bootstrap**, not a DeLong test.

---

## 12. What is *not* a direct model reference

Several papers and public competition repositories have been reviewed for context, failure analysis, possible headroom and engineering ideas. They should **not** be cited as if the B17/B18 implementation reproduces them unless a later experiment explicitly adopts their method.

In particular:

- public Kaggle/GitHub implementations are methodology/engineering context unless independently reproduced;
- external knee-MRI papers used only to discuss achievable performance are benchmarking context, not implementation references;
- the B6 state-only expert AUC (`0.7024597743`) is a repository measurement, not a literature result and not a proven MRI ceiling;
- B13--B17 are statistically unresolved on the repeatedly reused 58-study development surface; B17 is the current reference checkpoint, not a statistically proven superior model;
- B18's 58-study epoch-selection scores are selection statistics, not independent validation results.

See also:

- [`B18_FISHER_SELECTION.md`](B18_FISHER_SELECTION.md)
- [`B17_FROZEN_ENCODER.md`](B17_FROZEN_ENCODER.md)
- [`B16_FULL_REPORT_ALIGNMENT.md`](B16_FULL_REPORT_ALIGNMENT.md)
- [`B15_MRI_SSL.md`](B15_MRI_SSL.md)
- [`B6_B15_GOLD_DIAGNOSTIC.md`](B6_B15_GOLD_DIAGNOSTIC.md)
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
- [`references.md`](references.md) for the broader historical/public-work bibliography.

---

## 13. Short citation set for a manuscript Methods section

If only the papers directly needed to describe the present B17/B18 model are cited, the minimal set is:

1. Bien et al. (2018) — knee-MRI deep-learning precedent.
2. Deng et al. (2009) — ImageNet initialization.
3. Liu et al. (2022) — ConvNeXt.
4. Vaswani et al. (2017) — Transformer attention.
5. Ilse et al. (2018) — attention-based multiple-instance aggregation.
6. Chen et al. (2020) — contrastive self-supervised learning.
7. Azizi et al. (2021) — MICLe / same-case medical contrastive learning.
8. Salton & Buckley (1988) + Deerwester et al. (1990) — TF-IDF/SVD report semantic representation.
9. Irvin et al. (2019) + Peng et al. (2018) — report-derived positive/negated/uncertain supervision.
10. Fisher (2026, preprint) — B18 frozen-backbone/short-training/expert-compass motivation.
11. Efron & Tibshirani (1993) — bootstrap uncertainty.

This short list is sufficient to explain the major scientific ingredients without implying that every contextual paper was directly implemented.
