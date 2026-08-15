# Scientific references for the current working-model lineage (B20)

> **Scope — 2026-08-15.** This is the canonical bibliography for the **current active working-model lineage**, whose checkpoint remains **B20 (`B20_crop_only_joint_focus`)** at fixed epoch 2. The file also records which references remain relevant to the current B20-development path after B21--B25X.
>
> **Important status distinction:** B20 is the working model. B21/B22 are closed negative experiments; B23-v1 failed its formal labeller gate; formal B24 remains blocked; B24X/B24X-Density and B25X are exploratory supervision experiments and do **not** replace B20. Their main current value is to identify supervision coverage and class-balance weaknesses that can guide development of the existing B20 family.
>
> This file distinguishes **direct methodological references** from **supporting/context references**. A paper being listed here does not mean its implementation was reproduced exactly. Repository-specific choices with no direct external-method claim are labelled explicitly.

## 1. Current model lineage at a glance

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
B18 full-FOV comparator / historical checkpoint-selection audit
        ↓
B20 post-resize 90% center crop + fixed E2
        ↓
ACTIVE WORKING MODEL
```

Current B20 definition:

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
encoder                frozen historical B16 report-aligned encoder
architecture           hierarchical learned-series-token pathology-query model
implemented geometry   native MRI -> resize 224 -> center crop 90% -> resize 224
cosine/vignette mask   no
canonical expert AUC   0.667159355531343
status                 ACTIVE WORKING MODEL
```

### Current development branches

```text
B21  pre-resize crop correction        -> weak-v2 passed, reused-gold acceptance failed
B22  B21 duration audit E1-E5          -> E2 best; longer training did not rescue
B23  local LLM report labeller         -> formal gate failed on specificity
B24  formal supervision comparison     -> blocked / not run
B24X pilot                             -> exploratory denser-supervision signal
B24X-Density                           -> exploratory fill-only density signal
B25X ChatGPT hybrid supervision        -> exploratory three-arm full matched study
```

None of these branches changes the active B20 checkpoint.

---

## 2. Reference-to-component map

| Current component | Main reference(s) | How it is used in this repository |
|---|---|---|
| Knee MRI study-level deep learning | Bien et al. (2018), MRNet | Domain precedent for study-level knee-MRI diagnosis from multiple slices/series. B20 is not MRNet. |
| Generic visual pretraining | Deng et al. (2009), ImageNet | The B13-to-B20 lineage begins from torchvision ImageNet-1K ConvNeXt initialization before competition-specific representation adaptation. |
| ConvNeXt encoder | Liu et al. (2022) | ConvNeXt-Tiny is the 2.5D slice encoder in the current lineage. |
| Transformer context modeling | Vaswani et al. (2017) | Transformer blocks contextualize learned per-series tokens and pathology-query representations. |
| Attention-based MIL / learned aggregation | Ilse et al. (2018) | Conceptual basis for learned attention pooling across MRI instances. The repository uses its own learned series-query pooling implementation. |
| Generic contrastive SSL | Chen et al. (2020), SimCLR | General contrastive-learning foundation for B15 same-study MRI adaptation. |
| Medical multi-instance contrastive SSL | Azizi et al. (2021), MICLe | Closest published analogue to B15: different MRI instances from the same medical case are used as positives. B15 is MICLe-style, not a reproduction. |
| Image/text contrastive alignment | Radford et al. (2021), CLIP | Conceptual precedent for B16 image-report alignment. B16 does not use CLIP weights or a CLIP text encoder. |
| TF-IDF report representation | Salton & Buckley (1988) | B16 report semantics start from word TF-IDF. |
| Truncated-SVD / latent semantics | Deerwester et al. (1990) | B16 compresses TF-IDF report vectors with TruncatedSVD before image-report alignment. |
| Report-derived weak labels with uncertainty states | Irvin et al. (2019), CheXpert | Major precedent for positive/negative/uncertain report-derived supervision and explicit uncertainty handling. |
| Negation and uncertainty extraction | Peng et al. (2018), NegBio | Methodological precedent for separating positive, negated and uncertain radiology findings. B6 uses a custom frozen parser. |
| Large-scale report-derived imaging labels | Johnson et al. (2019/2020), MIMIC-CXR-JPG | Supporting precedent for large-scale image supervision derived from radiology reports. |
| DICOM handling | DICOM Standard | Operational reference for image decoding, orientation/position metadata, rescale handling and photometric interpretation. |
| PyTorch implementation | Paszke et al. (2019) | Core deep-learning framework. |
| scikit-learn text / diagnostic tools | Pedregosa et al. (2011) | TF-IDF, TruncatedSVD and classical diagnostic utilities. |
| ROC-AUC interpretation | Hanley & McNeil (1982) | Background for ROC AUC, the primary ranking metric. |
| Study-level bootstrap uncertainty | Efron & Tibshirani (1993) | Basis for paired study bootstrap confidence intervals and model comparisons. |
| Correlated ROC comparison | DeLong et al. (1988) | Background only; the repository's principal comparison procedure is the paired study bootstrap. |

### Repository-specific components without a direct literature claim

The following are **current repository design decisions**, not claimed reproductions of a particular published method:

- the B20 `resize 224 -> 90% center crop -> resize 224` geometry;
- the exact 16-slice-per-series sampling contract;
- the precise learned per-series pooling implementation and 12 pathology-query heads;
- the target-balanced weak-BCE implementation and asymmetric B6 soft targets/weights;
- the fixed E2 endpoint adopted after the project's own B17--B22 empirical trajectory;
- the frozen weak-v2 split and strict all-12-target bootstrap ranking protocol;
- the B24X/B25X fill-only supervision rule that preserves B6 cells and fills B6-silent cells.

These should be described as repository methods, with the closest conceptual references cited separately rather than presented as exact reproductions.

---

## 3. Competition / dataset source

### RSNA 2026 Knee MRI AI Challenge

Radiological Society of North America. **RSNA Knee MRI AI Challenge (2026).**

- Official challenge page: https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- Role in project: source task/data definition and clinical target context.
- Repository rule: the **actual released CSV/DICOM schema** is the operational source of truth for model inputs; challenge-description wording is not used to invent unavailable test-time fields.

---

## 4. Knee MRI deep-learning foundation

### Bien et al. — MRNet

Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., Bereket, M., Patel, B. N., Yeom, K. W., Shpanskaya, K., Halabi, S., Zucker, E., Fanton, G., Amanatullah, D. F., Beaulieu, C. F., Riley, G. M., Stewart, R. J., Blankenberg, F. G., Larson, D. B., Jones, R. H., Langlotz, C. P., Ng, A. Y., & Lungren, M. P. (2018). **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.** *PLOS Medicine, 15*(11), e1002699.

- DOI: https://doi.org/10.1371/journal.pmed.1002699
- PubMed: https://pubmed.ncbi.nlm.nih.gov/30481176/
- Relevance: established study-level knee-MRI deep learning using multiple MRI series and slice aggregation.
- Difference from B20: B20 uses 2.5D ConvNeXt features, all recognized real series, learned per-series attention pooling, a study Transformer, pathology queries, competition-specific representation learning and 12 multi-label outputs.

---

## 5. Encoder and generic visual pretraining

### Deng et al. — ImageNet

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). **ImageNet: A large-scale hierarchical image database.** *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 248--255.

- DOI: https://doi.org/10.1109/CVPR.2009.5206848
- Relevance: source dataset for the supervised initialization used by torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` before the B15/B16 competition-specific adaptation sequence.

### Liu et al. — ConvNeXt

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). **A ConvNet for the 2020s.** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

- arXiv: https://arxiv.org/abs/2201.03545
- Relevance: architecture family for the ConvNeXt-Tiny MRI encoder used in the B13-to-B20 lineage.

---

## 6. Hierarchical study aggregation

### Vaswani et al. — Transformer

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, L., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need.** *Advances in Neural Information Processing Systems, 30*.

- Paper: https://papers.nips.cc/paper/7181-attention-is-all-you-need
- Relevance: self-attention/Transformer mechanism underlying the study-level and pathology-context blocks.

### Ilse et al. — Attention-based MIL

Ilse, M., Tomczak, J. M., & Welling, M. (2018). **Attention-based Deep Multiple Instance Learning.** *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2127--2136.

- Paper: https://proceedings.mlr.press/v80/ilse18a.html
- Relevance: methodological foundation for learnable attention aggregation of multiple instances into a bag/study representation.
- Repository-specific implementation: each real MRI series contributes slice features that are compressed into a learned series token; the series tokens are then contextualized at the study level before 12 pathology-query outputs.

---

## 7. Same-study MRI representation learning

### Chen et al. — SimCLR

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). **A Simple Framework for Contrastive Learning of Visual Representations.** *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 1597--1607.

- Paper: https://proceedings.mlr.press/v119/chen20j.html
- Relevance: general contrastive-learning framework underlying B15's representation-learning logic.

### Azizi et al. — MICLe

Azizi, S., Mustafa, B., Ryan, F., Beaver, Z., Freyberg, J., Deaton, J., Loh, A., Karthikesalingam, A., Kornblith, S., Chen, T., Natarajan, V., & Norouzi, M. (2021). **Big Self-Supervised Models Advance Medical Image Classification.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 3478--3488.

- DOI: https://doi.org/10.1109/ICCV48922.2021.00346
- CVF: https://openaccess.thecvf.com/content/ICCV2021/html/Azizi_Big_Self-Supervised_Models_Advance_Medical_Image_Classification_ICCV_2021_paper.html
- Relevance: introduces Multi-Instance Contrastive Learning (MICLe), where multiple images from the same medical case can form positive pairs.
- Repository relationship: B15's same-study knee-MRI contrastive adaptation is **MICLe-style**. It is not claimed to reproduce MICLe exactly.

---

## 8. Full-report semantic alignment

### Radford et al. — CLIP / image-text alignment

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). **Learning Transferable Visual Models From Natural Language Supervision.** *Proceedings of the 38th International Conference on Machine Learning*.

- arXiv: https://arxiv.org/abs/2103.00020
- Relevance: conceptual precedent for contrastively aligning images with text-derived semantic representations.
- Important difference: B16 does **not** use CLIP weights, a transformer language model, or external clinical text embeddings. Its report branch is competition-only TF-IDF + TruncatedSVD and is discarded before downstream MRI-only inference.

### Salton & Buckley — TF-IDF

Salton, G., & Buckley, C. (1988). **Term-weighting approaches in automatic text retrieval.** *Information Processing & Management, 24*(5), 513--523.

- DOI: https://doi.org/10.1016/0306-4573(88)90021-0
- Relevance: classical term-weighting foundation for the TF-IDF report representation used in B16.

### Deerwester et al. — Latent Semantic Analysis / SVD

Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). **Indexing by latent semantic analysis.** *Journal of the American Society for Information Science, 41*(6), 391--407.

- DOI: https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9
- Relevance: singular-value decomposition of term-document representations into a lower-dimensional semantic space; conceptually matches B16's TF-IDF -> TruncatedSVD branch.

---

## 9. Weak supervision from radiology reports

### Irvin et al. — CheXpert

Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., Seekins, J., Mong, D. A., Halabi, S. S., Sandberg, J. K., Jones, R., Larson, D. B., Langlotz, C. P., Patel, B. N., Lungren, M. P., & Ng, A. Y. (2019). **CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.** *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(01), 590--597.

- DOI: https://doi.org/10.1609/aaai.v33i01.3301590
- Relevance: major precedent for radiology-report-derived labels that explicitly distinguish uncertainty rather than collapsing uncertain findings to hard negatives.
- Repository relationship: B6 uses four states (`positive`, `negated`, `uncertain`, `unmentioned`); only sufficiently confident positive/negated cells contribute to the historical B20 pathology loss.

### Peng et al. — NegBio

Peng, Y., Wang, X., Lu, L., Bagheri, M., Summers, R. M., & Lu, Z. (2018). **NegBio: a high-performance tool for negation and uncertainty detection in radiology reports.** *AMIA Joint Summits on Translational Science Proceedings*, 2017, 188--196.

- PubMed: https://pubmed.ncbi.nlm.nih.gov/29888070/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5961822/
- Relevance: radiology-specific methodology for separating positive, negated and uncertain findings.
- Important difference: B6 is a custom frozen parser; the repository does not run NegBio itself.

### Johnson et al. — MIMIC-CXR-JPG

Johnson, A. E. W., Pollard, T. J., Greenbaum, N. R., Lungren, M. P., Deng, C.-Y., Peng, Y., Lu, Z., Mark, R. G., Berkowitz, S. J., & Horng, S. (2019/2020). **MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs.**

- arXiv: https://arxiv.org/abs/1901.07042
- Dataset background: https://physionet.org/content/mimic-cxr-jpg/
- Relevance: large-scale example of deriving image labels from free-text reports and keeping a distinction between report-derived and expert-reference labels.

### Current B24X/B25X relationship to the literature

The B24X/B25X experiments do **not** introduce a new published model architecture. They use the existing B20-family MRI learner while changing only the report-supervision surface.

Current evidence supports the narrow development insight that **coverage of definite report-derived labels, especially missing negative examples, can materially affect the downstream learner**. The strongest B25X example is Synovitis, where the B6 training surface was severely deficient in negatives and the fill-only hybrid surface supplied many additional negative examples. This is a repository finding, not a claim imported from CheXpert, NegBio or another published benchmark.

The ChatGPT-created hybrid cache used by B25X has **mixed/unknown original LLM provenance**. Therefore:

- it must not be cited as a reproduction of a particular LLM paper;
- it is not a canonical B23/Qwen export;
- its B25X results are exploratory weak-supervision evidence only;
- its value for the current project is diagnostic: it identifies where the B20 training supervision is sparse or imbalanced.

---

## 10. Current B20 preprocessing and training decisions

### B20 crop geometry

The current working geometry is:

```text
native MRI -> resize 224 -> center crop 90% -> resize 224
```

This should be described as a **repository-specific preprocessing intervention**. The project does not currently claim that this exact crop sequence is prescribed by a published knee-MRI method.

B21 tested moving the crop before resizing. Although that formulation improved the frozen weak-v2 development comparison, it failed the predeclared reused-gold acceptance comparison and was not promoted. Therefore the historical B20 geometry remains the active implementation.

### Fixed epoch 2

B20 uses the fixed epoch-2 endpoint. This is now supported primarily by the project's own trajectory rather than by an external short-training prescription:

- B17/B18/B20 development repeatedly favored early epochs;
- B21 fixed E2 before acceptance;
- B22 explicitly trained E1--E5 and showed that lower training loss after E2 did not produce better expert ranking.

Thus **fixed E2 is an empirical repository decision**. It should not be cited as though a paper uniquely specifies two epochs for this task.

### Fisher (2026) — historical B18 context only

Fisher, G. (2026). **The NLP-to-Expert Gap in Chest X-ray AI.** *medRxiv* preprint 2026.02.27.26347261.

- DOI: https://doi.org/10.64898/2026.02.27.26347261
- Full text: https://www.medrxiv.org/content/10.64898/2026.02.27.26347261v1.full
- Status at protocol definition: preprint / not peer reviewed.
- Historical role: motivated the B18 frozen-backbone / short-training / expert-compass experiment.
- **Current role:** supporting historical context only. Fisher is **not** the defining reference for B20, and it is no longer part of the minimal direct citation set for the current working model.

---

## 11. Software and imaging standards

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

## 12. Evaluation references and current validation governance

### Hanley & McNeil — ROC AUC

Hanley, J. A., & McNeil, B. J. (1982). **The meaning and use of the area under a receiver operating characteristic (ROC) curve.** *Radiology, 143*(1), 29--36.

- DOI: https://doi.org/10.1148/radiology.143.1.7063747
- Relevance: classical interpretation of ROC AUC, the primary ranking metric.

### Efron & Tibshirani — Bootstrap

Efron, B., & Tibshirani, R. J. (1993). **An Introduction to the Bootstrap.** Chapman & Hall/CRC.

- DOI: https://doi.org/10.1201/9780429246593
- Relevance: conceptual basis for study-level bootstrap confidence intervals and paired bootstrap comparisons.

### DeLong et al. — correlated ROC curves (background only)

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). **Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.** *Biometrics, 44*(3), 837--845.

- DOI: https://doi.org/10.2307/2531595
- Role: statistical background only. The repository's principal uncertainty procedure is the paired **study-level bootstrap**, not a DeLong test.

### Current governance

```text
B20 expert score                  historical/reused development evidence
58-study expert surface           reused/post-hoc; not pristine independent validation
weak-v2                           frozen B6 teacher-agreement development surface
B24X/B25X gold evaluation         prohibited
B24X/B25X promotion               prohibited
formal B24                        blocked unless a future labeller passes its formal gate
hidden competition evaluation     independent predictive signal
```

The B25X frozen weak-v2 result is scientifically useful but must be interpreted correctly:

```text
B6 matched control       0.6723718048
Hybrid                   0.7268784872
B6 + Hybrid-fill         0.7308472686
```

The all-12-target gain is dominated by Synovitis. Excluding Synovitis, the 11-target macro difference for Fill versus B6 is only approximately `+0.0024`. Therefore B25X currently supports a **supervision-coverage/class-balance diagnosis**, not a claim that a new model broadly supersedes B20.

---

## 13. What is *not* a direct current-model reference

The following should **not** be cited as though they define the present B20 model unless a future controlled experiment explicitly adopts them:

- public Kaggle/GitHub competition notebooks and ensembles;
- DINOv2-based knee pipelines reviewed for comparison;
- soft-dense label strategies reviewed from public notebooks;
- target-specific ensemble weights chosen from public leaderboard behavior;
- external models used only to discuss possible performance headroom;
- Fisher (2026) as though it defines B20; it is historical B18 context only;
- the mixed-provenance ChatGPT hybrid cache as though it were a published LLM method;
- B24X/B25X weak-v2 results as though they were independent expert validation.

Current project direction is to **develop the existing B20-family working model**, using B24X/B25X findings as controlled diagnostics of supervision quality rather than replacing the architecture with an unrelated model family.

---

## 14. Current status notes for citation discipline

The following are repository measurements and should not be presented as literature results:

- B20 canonical expert macro AUC `0.6671593555`;
- B6 state-only expert AUC `0.7024597743`;
- B21 weak-v2 improvement followed by failed reused-gold acceptance;
- B22's finding that E2 remained the best expert-ranked endpoint across E1--E5;
- B23-v1's improved state-only AUC/coverage but failed specificity gate;
- B24X/B24X-Density density-recovery findings;
- B25X's `0.67237 -> 0.73085` weak-v2 improvement and its Synovitis-dominated mechanism;
- the Synovitis class-coverage audit showing that B6 had very few negative training examples compared with the fill-only hybrid surface.

These results may be reported as **our experiments**, with the relevant repository protocol and validation limitations stated explicitly.

---

## 15. Short citation set for a manuscript Methods section

For the **current B20 model itself**, the minimal direct citation set is:

1. **Bien et al. (2018)** — study-level knee-MRI deep-learning precedent.
2. **Deng et al. (2009)** — ImageNet initialization.
3. **Liu et al. (2022)** — ConvNeXt encoder.
4. **Vaswani et al. (2017)** — Transformer attention.
5. **Ilse et al. (2018)** — attention-based multiple-instance aggregation.
6. **Chen et al. (2020)** — generic contrastive representation learning.
7. **Azizi et al. (2021)** — MICLe / same-case medical contrastive learning.
8. **Radford et al. (2021)** — conceptual image-text contrastive alignment for the B16 representation stage.
9. **Salton & Buckley (1988)** and **Deerwester et al. (1990)** — TF-IDF/SVD report semantic representation.
10. **Irvin et al. (2019)** and **Peng et al. (2018)** — report-derived positive/negated/uncertain weak supervision.
11. **Efron & Tibshirani (1993)** — bootstrap uncertainty and paired comparisons.

Optional implementation references:

- Paszke et al. (2019) — PyTorch;
- Pedregosa et al. (2011) — scikit-learn;
- DICOM Standard — imaging data handling.

**Fisher (2026) is no longer in the minimal direct B20 citation set.** It should be cited only when describing the historical B18 expert-compass experiment.

---

## 16. Canonical repository records

Use these files for the current experimental status rather than reconstructing model state from the bibliography alone:

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — current project snapshot.
- [`WORKING_MODEL.md`](WORKING_MODEL.md) — active B20 checkpoint and governance.
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) — experiment ledger.
- [`B20_CROP_ONLY_FOCUS.md`](B20_CROP_ONLY_FOCUS.md) — active B20 preprocessing/model record.
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md) — failed pre-resize candidate.
- [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md) — E1--E5 duration audit.
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md) — B23-v1 report-labeller protocol/status.
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md) — B24X/B24X-Density exploratory record.
- [`B25X_HYBRID_SUPERVISION.md`](B25X_HYBRID_SUPERVISION.md) — current hybrid/fill supervision experiment and Synovitis diagnosis.
- [`VALIDATION.md`](VALIDATION.md) — validation governance.
- [`references.md`](references.md) — broader historical/public-work bibliography.
