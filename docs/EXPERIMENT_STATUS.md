# Experiment status

**Snapshot:** 2026-08-10  
**Package:** `0.10.0`  
**Gold evaluation set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

This file is the canonical repository summary for measured experiment status. `docs/competition.md` is intentionally preserved separately and is not modified by experiment updates.

## Current headline

- **Best standalone point estimate:** B5 image-report representation learning evaluated with the unchanged B4 frozen target-wise PCA/logistic-regression probe, macro AUC `0.5243650851`, 95% bootstrap CI `[0.4728108406, 0.5761619105]`.
- B5 improves the observed point estimate over the image-only B4 representation (`0.5137567459`) by about `+0.0106` macro AUC under the same downstream probe.
- The paired B4-vs-B5 bootstrap is positive but statistically inconclusive: median difference `+0.0105821232`, 95% CI `[-0.0408197338, +0.0622131599]`, `P(B5 > B4)=0.656`.
- **B5 is now the main standalone representation baseline. B4 is retained as the critical image-only ablation.**
- The previously retained fixed equal-weight B1+B4 rank ensemble has macro AUC `0.5167`; B5 is numerically higher, but no new ensemble tuning is performed on the same 58 gold studies.

## Completed experiments

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected as general MRI teacher |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected globally |
| B1+B3 rank | fixed 50:50 rank ensemble | `0.5048038179` | effectively neutral |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | retained image-only ablation |
| B4.1 | one shared downstream policy per fold | `0.4847792672` | rejected; too rigid |
| B4.2 | four predefined pathology-group policies | `0.4901328905` | rejected |
| B4.3 | target-wise two-way-CV policy selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | retained fixed ensemble; no weight tuning |
| **B5** | **image-report SSL representation + unchanged B4 probe** | **`0.5243650851`** | **main standalone baseline; best point estimate** |

## Key paired comparisons

### B1 versus B0

B1 improved the point estimate by about `+0.0268`. Paired bootstrap median difference was `+0.02646`, 95% CI `[-0.04464, +0.09870]`, with `P(B1 > B0)=0.771`.

### B4 versus B1

Using A=B1 and B=B4:

- paired median difference: `+0.0102107449`
- 95% CI: `[-0.0514266147, +0.0709432872]`
- `P(B4 > B1)=0.6378`

B4 improved the point estimate over B1, but the evidence was not statistically decisive.

### B4.1, B4.2 and B4.3 versus B4

All attempts to reduce B4 policy-selection variance lowered pooled OOF performance:

- B4.1: `P(B4.1 > B4)=0.1084`
- B4.2: `P(B4.2 > B4)=0.0724`
- B4.3: `P(B4.3 > B4)=0.2182`

Decision: stop redesigning B4 policy selection on the same 58 gold studies.

### Fixed B1+B4 rank ensemble versus B4

Using A=B4 and B=fixed equal-rank ensemble:

- paired median difference: `+0.0027575757`
- 95% CI: `[-0.0351268280, +0.0417415623]`
- `P(ensemble > B4)=0.5544`

Decision: retain the ensemble as a fixed historical candidate, but do not tune weights and do not claim it improves B4.

### B5 versus B4

This is the controlled representation test. B4 and B5 use the same frozen-feature contract and the same original target-wise nested PCA/logistic-regression probe; only the encoder representation changes.

Using A=B4 and B=B5:

- B4 macro AUC: `0.5137567459`
- B5 macro AUC: `0.5243650851`
- B5 95% bootstrap CI: `[0.4728108406, 0.5761619105]`
- paired median difference: `+0.0105821232`
- paired 95% CI: `[-0.0408197338, +0.0622131599]`
- `P(B5 > B4)=0.656`
- valid paired bootstrap replicates: `5000/5000`

Interpretation: report-aligned competition-only representation learning improves the observed point estimate, but the 58-study gold set does not establish a statistically decisive superiority over B4. B5 becomes the main standalone baseline because it has the highest controlled standalone point estimate; B4 remains the image-only ablation.

### B5 target-level changes versus B4

| Target | B4 AUC | B5 AUC | B5 - B4 |
|---|---:|---:|---:|
| ACL | `0.585784` | `0.667892` | `+0.082108` |
| MCL | `0.480726` | `0.405896` | `-0.074830` |
| Medial Meniscus | `0.542067` | `0.665865` | `+0.123798` |
| Lateral Meniscus | `0.604969` | `0.617391` | `+0.012422` |
| Medial OA | `0.550388` | `0.658915` | `+0.108527` |
| Lateral OA | `0.398453` | `0.404255` | `+0.005803` |
| PF OA | `0.638353` | `0.606178` | `-0.032175` |
| Effusion | `0.444720` | `0.516770` | `+0.072050` |
| Synovitis | `0.445639` | `0.555556` | `+0.109916` |
| Baker's | `0.375000` | `0.385870` | `+0.010870` |
| Contusion | `0.558704` | `0.399460` | `-0.159244` |
| Fracture | `0.540278` | `0.408333` | `-0.131944` |

B5 is higher on 8 of 12 target point estimates. The largest gains are Medial Meniscus, Synovitis, Medial OA, ACL and Effusion. The largest losses are Contusion and Fracture. These target-level differences are descriptive only and must not be used to choose post-hoc target-specific B4/B5 winners on the same outer OOF labels.

## Strong SSL representation

The completed strong competition-only MRI SSL run used all 4,349 non-gold studies and excluded the 58 gold studies. It completed 8 epochs, 8,000 batches, about 24,000 study draws (`~5.52` corpus passes), and 238,274 active 2.5D examples. The training loss decreased monotonically from about `3.434` to `2.862`.

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

No external pretrained image weights were used.

## B4 representation probe

B4 caches deterministic frozen encoder features for the 58 gold studies:

```text
shape = [58, 6, 2304]
pooling = mean + standard deviation + maximum
encoder = frozen competition-only strong SSL ConvNeXt
```

The target-wise B4 selector is visibly unstable because the inner folds contain only 18-20 studies. B4.1-B4.3 showed that forcing more shared or cross-validated policies did not improve outer OOF.

B4 is now retained primarily as the image-only representation ablation against B5.

## B5 — image-report representation learning

B5 changes the representation rather than the downstream gold-label classifier.

Training scope:

- 4,349 report-only competition studies;
- all 58 gold studies excluded from B5 representation training;
- no external image weights;
- no external language model;
- text representation fitted only on competition reports using TF-IDF -> TruncatedSVD;
- MRI encoder initialized from the completed strong SSL checkpoint;
- joint image-image, acquisition-metadata and image-report objectives;
- report embedding queue of 256 semantic negatives with duplicate-report masking;
- saved downstream artifact is an MRI encoder; final inference remains MRI-only.

### B5 training result

Checkpoint:

```text
runs/b5_report_ssl/b5_encoder.pt
```

All four predefined epochs completed without runtime-budget limiting:

| Epoch | Loss | Image contrast | Metadata | Report NCE | Report cosine | Seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.520392` | `3.006825` | `0.447246` | `4.603128` | `0.801537` | `1403.84` |
| 2 | `5.100010` | `2.961406` | `0.399780` | `3.906748` | `0.682283` | `1441.52` |
| 3 | `4.893490` | `2.936515` | `0.380151` | `3.566160` | `0.630856` | `1539.21` |
| 4 | `4.704915` | `2.893706` | `0.368420` | `3.290113` | `0.592378` | `1434.28` |

Totals:

- completed epochs: 4
- batches: 4,000
- study draws: 16,000
- active 2.5D examples: 158,886
- queue size: 256 throughout
- final encoder LR: `1e-6`
- final head LR: `1e-6`
- budget limited: false for every epoch

Every logged objective improved monotonically, establishing stable image-report alignment training.

### B5 frozen gold probe

The feature-cache audit confirms:

```text
checkpoint = runs/b5_report_ssl/b5_encoder.pt
studies = 58
feature shape = [58, 6, 2304]
pooling = mean + std + max
encoder frozen = true
completed representation epochs = 4
external pretrained = false
```

The unchanged original B4 nested probe produced:

```text
macro AUC = 0.5243650851
95% CI   = [0.4728108406, 0.5761619105]
```

Per-target B5 AUCs:

```text
ACL               0.6678921569
MCL               0.4058956916
Medial Meniscus   0.6658653846
Lateral Meniscus  0.6173913043
Medial OA         0.6589147287
Lateral OA        0.4042553191
PF OA             0.6061776062
Effusion          0.5167701863
Synovitis         0.5555555556
Baker's           0.3858695652
Contusion         0.3994601889
Fracture          0.4083333333
```

Decision: B5 is the main standalone representation baseline. Do not use these outer results to tune target-specific B4/B5 selection, B5 report-loss weights, extra epochs, or ensemble weights.

## Validation caveat

The same 58 official gold studies have now supported multiple controlled experiments. Every individual OOF run is leakage-aware according to its own predefined procedure, but repeated methodological decisions based on those OOF results mean the campaign as a whole is increasingly **model-selection cross-validation**, not a pristine independent estimate of future hidden-test performance.

Therefore:

- do not optimize ensemble weights on these 58 labels;
- do not create further B4 selector variants from observed outer results;
- do not choose target-specific post-hoc winners from B0-B5 outer OOF;
- do not tune B5 hyperparameters or extra epochs from the completed outer B5 result;
- report paired bootstrap uncertainty alongside point estimates;
- use actual competition leaderboard results only when a real submission has been made.
