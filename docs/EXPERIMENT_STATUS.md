# Experiment status

**Snapshot:** 2026-08-09  
**Package:** `0.10.0`  
**Gold evaluation set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

This file is the canonical repository summary for measured experiment status. `docs/competition.md` is intentionally preserved separately and is not modified by experiment updates.

## Current headline

- **Best clean standalone point estimate:** B4 frozen strong-SSL features + target-wise PCA/logistic regression, macro AUC `0.5137567459`, 95% bootstrap CI `[0.4619827141, 0.5642366629]`.
- **Best fixed ensemble point estimate:** equal-weight B1+B4 rank average, macro AUC `0.5167`, 95% bootstrap CI `[0.4629, 0.5723]`.
- The fixed ensemble is statistically tied with B4: paired median difference `+0.00276`, 95% CI `[-0.03513, +0.04174]`, `P(ensemble > B4)=0.5544`.
- **B5 status:** representation training is running; no B5 OOF macro AUC is available yet. Do not report a B5 performance result until the frozen B5 probe is completed.

## Completed experiments

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected as general MRI teacher |
| B1 | strong competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected globally |
| B1+B3 rank | fixed 50:50 rank ensemble | `0.5048038179` | effectively neutral |
| B4 | frozen strong-SSL features + target-wise PCA/LR | `0.5137567459` | best standalone point estimate |
| B4.1 | one shared downstream policy per fold | `0.4847792672` | rejected; too rigid |
| B4.2 | four predefined pathology-group policies | `0.4901328905` | rejected |
| B4.3 | target-wise two-way-CV policy selector | `0.4966083942` | rejected |
| B1+B4 raw | fixed 50:50 probability average | `0.5050` | rejected |
| B1+B4 rank | fixed 50:50 rank average | `0.5167` | highest numerical score; tied with B4 |
| B5 | image-report representation learning | pending | running / not yet evaluated |

## Key paired comparisons

### B1 versus B0

B1 improved the point estimate by about `+0.0268`. Paired bootstrap median difference was `+0.02646`, 95% CI `[-0.04464, +0.09870]`, with `P(B1 > B0)=0.771`.

Interpretation: strong in-domain SSL is promising, but the 58-study gold set is too small for a confident superiority claim.

### B4 versus B1

Using A=B1 and B=B4:

- paired median difference: `+0.0102107449`
- 95% CI: `[-0.0514266147, +0.0709432872]`
- `P(B4 > B1)=0.6378`

Interpretation: B4 has the best clean standalone point estimate, but the evidence is not statistically decisive.

### B4.1, B4.2 and B4.3 versus B4

All attempts to reduce B4 policy-selection variance lowered pooled OOF performance:

- B4.1: `P(B4.1 > B4)=0.1084`
- B4.2: `P(B4.2 > B4)=0.0724`
- B4.3: `P(B4.3 > B4)=0.2182`

Decision: stop redesigning B4 policy selection on the same 58 gold studies. Additional selector variants would increasingly meta-fit this validation set.

### Fixed B1+B4 rank ensemble versus B4

Using A=B4 and B=fixed equal-rank ensemble:

- paired median difference: `+0.0027575757`
- 95% CI: `[-0.0351268280, +0.0417415623]`
- `P(ensemble > B4)=0.5544`

Decision: keep the ensemble as a fixed candidate because it has the highest numerical score, but do not tune weights and do not claim it improves B4.

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

All cached values were finite. Explicit stream-presence indicators are appended to each target design matrix.

The target-wise B4 selector is visibly unstable because the inner folds contain only 18-20 studies. Across 36 target/fold selections, `prior` versus `all`, PCA dimension and logistic `C` were all split across alternatives. The follow-up B4.1-B4.3 experiments showed that forcing more shared or cross-validated policies did not improve outer OOF.

## B5 — current running experiment

B5 changes the representation rather than the downstream gold-label classifier.

Training scope:

- 4,349 report-only competition studies;
- all 58 gold studies excluded from B5 representation training;
- no external image weights;
- no external language model;
- text representation fitted only on competition reports using TF-IDF -> TruncatedSVD;
- MRI encoder initialized from the completed strong SSL checkpoint;
- joint image-image, acquisition-metadata and image-report objectives;
- report embedding queue for additional semantic negatives;
- exact duplicate normalized report hashes masked as false negatives;
- saved downstream artifact is an MRI encoder; final inference remains MRI-only.

Run:

```bash
rsna-knee-b5 \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/ssl_strong/ssl_encoder.pt \
  --out-root runs/b5_report_ssl
```

**Current status:** running. No B5 macro AUC should be entered in tables or manuscript text yet.

## B5 evaluation plan

After B5 finishes, inspect:

```bash
cat runs/b5_report_ssl/policy.json
cat runs/b5_report_ssl/report_semantics.json
cat runs/b5_report_ssl/coverage.json
cat runs/b5_report_ssl/history.json
```

Then extract frozen B5 features with the existing B4 extractor:

```bash
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --split train \
  --scope gold \
  --out runs/b5_frozen_probe/gold_features.npz
```

Run the **unchanged original B4 target-wise nested probe**:

```bash
rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b5_frozen_probe/gold_features.npz \
  --out-root runs/b5_frozen_probe \
  --n-bootstrap 5000
```

Finally compare B4 image-only representation (A) with B5 image-report representation (B):

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b4_frozen_ssl/oof.csv \
  --compare-oof runs/b5_frozen_probe/oof.csv \
  --n-bootstrap 5000 \
  --out runs/b4_vs_b5.json
```

This comparison is intentionally controlled: the representation changes, while the downstream B4 probe remains fixed.

## Validation caveat

The same 58 official gold studies have now supported multiple controlled experiments. Every individual OOF run is leakage-aware according to its own predefined procedure, but repeated methodological decisions based on those OOF results mean the campaign as a whole is increasingly **model-selection cross-validation**, not a pristine independent estimate of future hidden-test performance.

Therefore:

- do not optimize ensemble weights on these 58 labels;
- do not create further B4 selector variants from observed outer results;
- do not choose target-specific post-hoc winners from B0-B5 outer OOF;
- report paired bootstrap uncertainty alongside point estimates;
- use actual competition leaderboard results only when a real submission has been made.
