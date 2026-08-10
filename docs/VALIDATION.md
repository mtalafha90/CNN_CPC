# Test and validation workflow

> **Snapshot: 2026-08-10.** The campaign now includes completed B0-B7.1 experiments, a rejected fixed B5+B7.1 rank ensemble, and B8 spatial-anatomy training in progress. Canonical scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

`CNN_CPC` uses several distinct validation resources. They answer different questions and must not be mixed.

## 1. External four-study technical fixture

`fixtures/external_validation/` contains four openly licensed knee MRI examples converted into a competition-like DICOM contract.

Use it for software/plumbing checks only:

- DICOM decoding;
- directory discovery;
- series routing;
- 2.5D preprocessing;
- missing-stream masking;
- model/inference plumbing;
- strict preflight testing.

It is **not** a scientific benchmark.

## 2. Real downloaded local test surface

The current local test metadata contains three studies. All selected streams decoded successfully:

```text
selected streams  14 / 18 possible
selected decoded  14 / 14
candidate files   533
file failures     0
```

The local test set has no gold labels and therefore cannot measure AUC.

## 3. Official 58-study gold development set

Scientific development comparisons use the 58 fully labelled training studies.

Original three-fold allocation:

| Outer fold | Gold train | Inner selection | Outer validation |
|---|---:|---:|---:|
| 0 | 20 | 20 | 18 |
| 1 | 18 | 20 | 20 |
| 2 | 20 | 18 | 20 |

Every target has positives and negatives in each outer fold.

The crucial campaign-level caveat is that these same 58 studies have now informed many sequential decisions. They are therefore a **development/model-selection set**, not pristine independent validation.

## 4. Original Stage-1 nested protocol

For B0/B1/B2/B3:

```text
outer gold       -> final fold OOF only
inner gold       -> Phase-A epoch selection
remaining gold   -> Phase-A training
Phase A          -> discarded
fresh Phase B    -> all non-outer gold for selected duration
```

The outer fold never chooses its own epoch count.

## 5. B4/B5 nested frozen-probe protocol

B4 representation pretraining excludes gold studies. The 58 gold rows are then used in nested target-specific PCA/logistic classifiers.

B5 reuses exactly the same frozen-probe protocol, changing only the encoder representation.

Results:

```text
B4 = 0.5137567459
B5 = 0.5243650851
```

Paired B4 -> B5:

```text
median difference = +0.0105821232
95% paired CI     = [-0.0408197338, +0.0622131599]
P(B5 > B4)        = 0.656
```

B4.1/B4.2/B4.3 selector-stabilization attempts all underperformed B4. Further B4 selector redesign is closed.

## 6. B6 gold audit

B6 v1.2.1 converts reports into positive/negated/uncertain/unmentioned states. The completed gold audit used the 58 studies only to characterize parser reliability; those gold rows are excluded from `training_targets.csv`.

Audit summary over 251 usable cells:

```text
TP 116
TN 80
FP 52
FN 3
positive precision 0.6905
sensitivity        0.9748
specificity        0.6061
NPV                0.9639
balanced accuracy  0.7904
```

This audit informed the single global B7 weak-label reliability policy. Therefore B7/B7.1/B8 scores on the same 58 studies are development estimates by construction.

## 7. B7-v1 development evaluation

B7-v1 uses no gold labels in gradient or early stopping. It is trained on frozen B6 weak labels and initialized from B5.

Result:

```text
macro AUC = 0.5397724412
95% CI   = [0.4733481702, 0.6035621405]
```

The supervision audit showed only about 1.28 nominal corpus passes because of the 500-batch epoch cap.

## 8. B7.1 full-coverage evaluation

B7.1 was a separately named follow-up that changed only the pre-identified coverage limitation:

```text
500 -> 1560 batches/epoch
batch size = 2
3120 study draws/epoch
4 complete nominal corpus passes
```

Result:

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

Paired B5 -> B7.1:

```text
median difference = +0.0399233552
95% paired CI     = [-0.0301354430, +0.1092349994]
P(B7.1 > B5)      = 0.8716
```

B7.1 is the current best standalone development point estimate. Neither paired 95% interval excludes zero.

## 9. Fixed ensemble validation

Only one B5+B7.1 ensemble rule was predeclared after B7.1:

```text
per-target percentile rank
0.5 * B5 rank + 0.5 * B7.1 rank
same rule for all 12 targets
```

Result:

```text
ensemble macro AUC = 0.5540141184
B7.1 macro AUC     = 0.5644802945
```

Paired B7.1 -> ensemble:

```text
median(ensemble-B7.1) = -0.0105429030
95% paired CI         = [-0.0523218181, +0.0333886570]
P(ensemble > B7.1)     = 0.3054
```

Decision: reject the ensemble as the leader and close the blend-search branch. No 60:40, 70:30, raw-probability, target-specific or calibrated alternatives are allowed from this result.

## 10. B8 validation contract — current experiment

B8 was designed after the B7.1 result and is therefore another development experiment.

The frozen architecture change is:

```text
B7.1 memory: 6 x 16 x 1    = 96 MRI tokens
B8 memory:   6 x 16 x 2x2  = 384 MRI tokens
```

B8 initializes from the completed B7.1 checkpoint and preserves the frozen B6 supervision, target balancing, 3,120-study full coverage, four epochs and learning rates.

The first B8 gold evaluation must remain one-shot:

1. complete the frozen B8 training run;
2. inspect `history.json` and `supervision_plan.json` before gold evaluation;
3. evaluate once with the fixed TTA `[-1,0,1]`;
4. compare B7.1 -> B8 with study-level paired bootstrap;
5. do not tune spatial grid, anatomy-prior strength, target-specific priors, epochs or blend weights from that result and still call the run B8-v1.

**Current status:** B8 training is in progress; no B8 gold AUC has been recorded.

## 11. Bootstrap comparisons

Use study-level paired bootstrap for aligned prediction files:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof <A.csv> \
  --compare-oof <B.csv> \
  --n-bootstrap 5000 \
  --out <comparison.json>
```

The evaluator reports:

- point macro AUC for A;
- median paired macro-AUC difference `B-A`;
- 95% paired bootstrap interval;
- bootstrap probability that B is better.

A point-estimate win is not sufficient when the interval is wide.

## 12. Current measured ranking

| Candidate | Macro AUC | Status |
|---|---:|---|
| B0 | `0.4763` | baseline |
| B1 | `0.5030` | retained reference |
| B2 | `0.4993` | rejected |
| B3 | `0.4945` | rejected |
| B4 | `0.5138` | image-only ablation |
| B5 | `0.524365` | representation baseline |
| B7-v1 | `0.539772` | coverage ablation |
| **B7.1** | **`0.564480`** | **current leader** |
| B5+B7.1 rank | `0.554014` | rejected ensemble |
| B8 | pending | training in progress |

## 13. Campaign-level interpretation

Do not:

- select target-specific post-hoc winners from gold predictions;
- optimize ensemble weights;
- invent further B4 grouping/selector variants from observed results;
- retune B6 parser rules or target-specific B7/B8 weak-label weights from these 58 labels;
- search B8 spatial grids, prior strengths, extra epochs or target-specific priors after the first B8 result and treat the re-evaluation as untouched;
- treat the local test surface or external fixture as scientific validation.

Actual Kaggle leaderboard results, when available, are a separate evidence source and must be labelled as such.
