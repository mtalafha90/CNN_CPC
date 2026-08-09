# Test and validation workflow

> **Snapshot: 2026-08-09.** The experiment campaign has completed B0-B4.3 plus fixed ensembles; B5 representation training is complete and its frozen gold probe is pending. Canonical scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

`CNN_CPC` uses three distinct validation resources. They answer different questions and must not be mixed.

## 1. External four-study technical fixture

`fixtures/external_validation/` contains four openly licensed knee MRI examples converted into a competition-like DICOM contract.

Use it for:

- DICOM decoding;
- directory discovery;
- series routing;
- 2.5D preprocessing;
- missing-stream masking;
- model/inference plumbing;
- strict preflight testing.

It is **not** a scientific benchmark.

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

## 2. Real downloaded local test surface

The current local test metadata contains three studies. All were preflighted:

```text
selected streams  14 / 18 possible
selected decoded  14 / 14
candidate files   533
file failures     0
```

The local test set has no gold labels and therefore cannot measure AUC.

## 3. Official gold set

Scientific model comparison uses the 58 fully labelled training studies.

Three-fold allocation:

| Outer fold | Gold train | Inner selection | Outer validation |
|---|---:|---:|---:|
| 0 | 20 | 20 | 18 |
| 1 | 18 | 20 | 20 |
| 2 | 20 | 18 | 20 |

Every target has at least one positive and one negative in every outer fold.

## 4. Original Stage-1 nested protocol

For neural B0/B1/B2/B3 folds:

```text
outer gold       -> final fold OOF only
inner gold       -> Phase-A epoch selection
remaining gold   -> Phase-A training
Phase A          -> discarded
fresh Phase B    -> all non-outer gold for selected duration
```

The outer fold never chooses its own epoch count.

`oof.csv` uses the predeclared production TTA. `oof_center.csv` is diagnostic only.

## 5. Frozen B4 protocol

B4 representation pretraining excludes all gold studies. After the encoder is frozen, the 58 gold rows are used in nested target-specific PCA/logistic classifiers.

For each outer fold, candidate feature mode/PCA/C are selected only from the designated inner fold and then refitted on all non-outer gold before one outer prediction.

B4 reached:

```text
macro AUC = 0.5137567459
95% CI   = [0.4619827141, 0.5642366629]
```

## 6. B4 follow-up validation protocols

Three attempts to reduce downstream selection variance were predefined and leakage-aware within each run:

- B4.1: one shared policy per outer fold -> `0.4847792672`;
- B4.2: four predefined pathology-group policies -> `0.4901328905`;
- B4.3: target-wise two-way CV on the two non-outer folds -> `0.4966083942`.

All were below B4. This branch is now closed. Further selector redesign based on these outer outcomes would risk campaign-level meta-overfitting.

## 7. Fixed ensemble validation

Only fixed 50:50 combinations were tested after B4:

```text
B1+B4 raw probability average = 0.5050
B1+B4 rank average            = 0.5167
```

B4 versus fixed rank ensemble:

```text
median ensemble-B4 difference = +0.00276
95% CI                        = [-0.03513, +0.04174]
P(ensemble > B4)              = 0.5544
```

The ensemble is therefore treated as statistically tied with B4. No weight search is allowed on the 58 gold rows.

## 8. B5 validation contract

B5 representation training excluded all 58 gold studies and used only the 4,349 report-only competition studies.

The completed pretraining produced:

```text
checkpoint              runs/b5_report_ssl/b5_encoder.pt
epochs                  4
batches               4000
study draws          16000
active 2.5D examples 158886
loss          5.5204 -> 4.7049
report NCE    4.6031 -> 3.2901
report cosine 0.8015 -> 0.5924
budget limited          false
```

The first B5 evaluation remains fixed exactly as planned:

1. freeze the completed B5 encoder;
2. extract the same deterministic six-stream mean/std/max features used by B4;
3. run the **original B4 target-wise nested classifier protocol unchanged**;
4. compare `runs/b5_frozen_probe/oof.csv` with `runs/b4_frozen_ssl/oof.csv` by paired bootstrap.

This makes B5 primarily a representation test rather than another downstream hyperparameter experiment.

**Current B5 status:** representation training complete; frozen probe / macro-AUC pending.

## 9. Bootstrap comparisons

Use study-level paired bootstrap when comparing aligned OOF files:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof <A.csv> \
  --compare-oof <B.csv> \
  --n-bootstrap 5000 \
  --out <comparison.json>
```

The evaluator reports the median macro-AUC difference, a 95% interval and the bootstrap probability that B is better.

A point-estimate win is not sufficient when the interval is wide.

## 10. Current measured ranking

| Candidate | Macro AUC |
|---|---:|
| B0 | `0.4763` |
| B3 | `0.4945` |
| B4.1 | `0.4848` |
| B4.2 | `0.4901` |
| B4.3 | `0.4966` |
| B2 | `0.4993` |
| B1 | `0.5030` |
| B1+B3 rank | `0.5048` |
| B4 | `0.5138` |
| B1+B4 rank | `0.5167` |
| B5 | pending frozen probe |

## 11. Campaign-level interpretation

A crucial distinction now applies:

- each individual OOF procedure was designed to avoid direct outer-label leakage;
- however, the same 58 gold studies have now informed multiple sequential method decisions.

Therefore the campaign as a whole is **model-selection cross-validation** rather than a pristine independent generalization estimate.

Do not:

- select target-specific post-hoc winners from outer OOF;
- optimize ensemble weights;
- invent more B4 selector/grouping variants from observed results;
- tune B5 representation hyperparameters after reading B5 outer OOF without declaring a new experiment;
- choose extra B5 epochs from the same outer OOF and call the re-evaluation independent;
- treat the local test surface or external fixture as scientific validation.

Actual Kaggle leaderboard results, when available, are a separate evidence source and must be labelled as such.
