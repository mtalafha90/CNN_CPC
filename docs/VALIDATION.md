# Test and validation workflow

> **Snapshot: 2026-08-10.** B7.1 is the current leader at `0.5644802945`; B8 is rejected at `0.5300962807`; B9 strict semantic routing is implemented/predeclared and has not yet been gold-evaluated. Canonical scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

`CNN_CPC` uses several distinct validation resources. They answer different questions and must not be mixed.

## 1. External technical fixture

`fixtures/external_validation/` is for software checks only: DICOM decoding, routing, preprocessing, missing-stream masking and inference plumbing. It is **not** a scientific benchmark.

## 2. Local test surface

The provided local test metadata contains 3 studies and 15 series. It has no labels and cannot measure AUC.

Historical routing selected 14 streams; the label-free strict-routing audit finds one false sagittal-fluid assignment, so B9 strict routing selects 13 semantically valid streams.

This test-surface audit is a data-contract check, not model validation.

## 3. Official 58-study gold development set

The 58 fully labelled training studies are the scientific development set. They have supported repeated sequential decisions and must now be described as **development/model-selection data**, not pristine independent validation.

Original three-fold allocation:

| Outer fold | Gold train | Inner selection | Outer validation |
|---|---:|---:|---:|
| 0 | 20 | 20 | 18 |
| 1 | 18 | 20 | 20 |
| 2 | 20 | 18 | 20 |

Every target has positives and negatives in every outer fold.

## 4. Historical protocols

B0-B3 used nested neural OOF logic. B4/B5 used a nested frozen-feature PCA/logistic probe. B6 gold labels were used once to audit parser reliability but excluded from B6 weak-training exports.

Because the B6 gold audit informed the global asymmetric weak-label policy, B7/B7.1/B8/B9 gold scores are development estimates even though gold labels do not enter their gradients or early stopping.

## 5. Current retained benchmark: B7.1

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

Paired B7-v1 -> B7.1:

```text
median difference = +0.0241102714
95% paired CI     = [-0.0140197876, +0.0660558004]
P(B7.1 > B7-v1)   = 0.8694
```

## 6. B8 is completed and rejected

```text
B8 AUC                 0.5300962807
B7.1 AUC               0.5644802945
median(B8 - B7.1)     -0.0335501423
95% paired CI         [-0.0900453633, +0.0223997827]
P(B8 > B7.1)           0.1156
```

The B8 spatial-prior branch is closed. Do not tune spatial grids, priors, per-target winners or blend weights from this result.

## 7. B9 validation contract

B9 was motivated by a **label-free series-metadata audit**, not by target-level B7.1/B8 differences.

Full training metadata audit:

```text
historical selected streams  21886
strict selected streams      21334
wrong-slot substitutions       552
wrong-slot fraction            2.52%
strict semantic mismatches        0
```

B9 changes only routing:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> None / masked
```

The first B9 evaluation must remain one-shot:

1. train with `configs/b9_strict_routing.yaml`;
2. inspect `routing_audit.json`, `history.json`, and `supervision_plan.json`;
3. require `strict_semantic_mismatches == 0`;
4. evaluate once with fixed TTA `[-1,0,1]`;
5. compare B7.1 -> B9 with 5,000 study-level paired bootstrap replicates;
6. do not restore individual substituted streams, modify routing rules, tune target-specific routing or blend weights after seeing the result and still call it B9-v1.

## 8. Paired bootstrap comparison

For aligned prediction files:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof <A.csv> \
  --compare-oof <B.csv> \
  --n-bootstrap 5000 \
  --out <comparison.json>
```

The comparison reports median `B-A`, its 95% paired bootstrap interval and `P(B>A)`.

For B7.1 -> B9:

```text
A = runs/b7_1_full_coverage/gold_eval/gold_predictions.csv
B = runs/b9_strict_routing/gold_eval/gold_predictions.csv
```

Positive `median_difference` favors B9.

## 9. Current measured ranking

| Candidate | Macro AUC | Status |
|---|---:|---|
| B0 | `0.4763` | baseline |
| B1 | `0.5030` | retained reference |
| B4 | `0.5138` | image-only ablation |
| B5 | `0.524365` | representation baseline |
| B7-v1 | `0.539772` | coverage ablation |
| **B7.1** | **`0.564480`** | **current leader** |
| B5+B7.1 rank | `0.554014` | rejected |
| B8 | `0.530096` | rejected |
| B9 | pending | strict-routing experiment |

## 10. Campaign-level rules

Do not:

- select target-specific post-hoc winners;
- optimize ensemble weights;
- retune B6 parser rules/weak-label weights from the 58 studies;
- tune B9 routing based on per-target gold AUCs;
- use the three-study test surface as scientific validation;
- describe the best development AUC as a leaderboard or hidden-test guarantee.

Actual competition leaderboard results are a separate evidence source and must be labelled as such.
