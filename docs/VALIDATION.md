# Test and validation workflow

`CNN_CPC` uses three deliberately different validation resources. They answer different questions and must not be mixed.

## 1. External four-study technical fixture

`fixtures/external_validation/` contains four openly licensed knee MRI examples converted into a competition-like DICOM contract.

Purpose:

- DICOM decoding;
- directory discovery;
- series routing;
- 2.5D preprocessing;
- missing-stream masking;
- model/inference plumbing;
- strict preflight testing.

It is **not** a scientific benchmark and must not be used for macro-AUC model selection.

Strict fixture command:

```bash
python -m rsna_knee.cli preflight \
  --data-root fixtures/external_validation \
  --split test \
  --sample-size 4 \
  --max-decode-failure-rate 0 \
  --max-file-decode-failure-rate 0 \
  --out runs/external_test_preflight.json
```

The sparse `validation.csv` copy contains only source-supported positive cells; unspecified cells remain `NaN`.

## 2. Real downloaded local test set

The current downloaded test metadata contains three studies. All three were preflighted on 2026-08-08:

```text
studies sampled        3
selected streams      14 / 18 possible
selected decoded      14 / 14
candidate files      533
file failures           0
stream failures         0
```

This verifies the complete locally supplied test surface available in the current download, but it does not provide gold labels and therefore cannot measure AUC.

## 3. Official gold nested validation

Scientific/competition validation uses the 58 official gold studies in `train.csv`.

The deterministic three-fold allocator produced:

| Outer fold | Gold train | Inner selection | Outer validation |
|---|---:|---:|---:|
| 0 | 20 | 20 | 18 |
| 1 | 18 | 20 | 20 |
| 2 | 20 | 18 | 20 |

Every target has both positive and negative examples in each outer fold, so every per-target outer AUC is defined.

Generate manifests with:

```bash
mkdir -p runs/validation
for f in 0 1 2; do
  python -m rsna_knee.cli validation-manifest \
    --config configs/train_local.yaml \
    --fold "$f" \
    --out "runs/validation/fold${f}.csv"
done
```

For each outer fold `k`:

- `outer_validation` is used only for final fold OOF evaluation;
- `inner_selection` selects the Phase-A epoch count;
- `gold_train` supplies trusted Phase-A gold training examples.

Phase A is discarded. Phase B starts from a fresh model and may use all non-outer gold studies.

## 4. Weak-image cross-fitting is separate from gold validation

Each non-gold report group receives a deterministic `crossfit_fold`.

For Stage-1 fold `k`, `crossfit_fold=k` non-gold rows are excluded from training and later predicted into:

```text
runs/<stage1_root>/fold{k}/weak_oof.csv
```

Those predictions are not official gold validation. They exist only to provide leakage-safe image teachers for Stage 2.

Stage-2 fold `k` may consume only the safe Stage-1 fold-`k` weak teacher.

## 5. Primary TTA versus center-only diagnostic

The predeclared production validation policy is:

```yaml
validation_tta_offsets: [-1, 0, 1]
```

Therefore:

- `oof.csv` = primary three-view TTA OOF;
- `oof_center.csv` = center-only diagnostic.

The center-only diagnostic must not be used to retroactively change the submission policy after inspecting outer labels.

## 6. Verified paired-sampler smoke result

The current fold-0 smoke run after the trusted-pair sampler fix produced:

```text
selected epoch              2
inner macro AUC       0.5513549
outer TTA macro AUC   0.5139555
outer center AUC      0.5228523
selection gold train        20
final gold train            40
budget limited           false
```

Ranking utilization:

```text
selection ranking pairs = 63
retrain ranking pairs   = 61
```

Every target had nonzero ranking pairs.

These are **smoke-test diagnostics only**. The smoke run uses very few batches and a tiny validation sample; it must not be presented as the production model's expected performance.

## 7. Why inner and outer smoke scores can differ strongly

With only 18–20 gold studies in a fold, macro-AUC variance is large. A smoke model is additionally undertrained. Differences such as:

```text
inner AUC  > outer AUC
center AUC > TTA AUC
```

are therefore observations to record, not reasons to tune the algorithm on the outer fold.

The non-smoke three-fold experiment is required before any performance conclusion.

## 8. Production validation sequence

For Stage-1 random initialization:

```bash
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local.yaml --fold 2
```

Combine the three primary OOF files only after all folds finish:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_random/fold0/oof.csv \
    runs/stage1_random/fold1/oof.csv \
    runs/stage1_random/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_random/evaluation.json
```

Use paired evaluation for controlled comparisons such as:

- TTA versus center-only;
- random initialization versus nested-selected SSL;
- Stage 2 versus its nested-selected Stage-1 teacher configuration.

## 9. Candidate selection rule

If Stage-1 random and Stage-1 SSL both exist, candidate choice for outer fold `k` uses **only `inner_macro_auc` from fold `k`**.

`outer_macro_auc` is deliberately excluded from that choice.

This is necessary so the outer fold remains an evaluation fold for the candidate actually used downstream.

## 10. Interpreting final cross-validation

There are two distinct reporting regimes:

1. **Before outer OOF is used for a method decision:** outer OOF is a leakage-controlled estimate for the predeclared method.
2. **After outer OOF is used to choose the final competition method:** the same result becomes **model-selection cross-validation** and should not be described as a pristine independent generalization estimate.

## Important separation

Never:

- append the external Wikimedia fixture to competition training;
- use the three local test studies for model selection;
- use weak OOF rows as official validation;
- tune Stage-1 candidate choice from outer AUC;
- tune submission TTA from `oof_center.csv` after seeing outer labels.

The authoritative performance result for the current baseline will be the completed non-smoke three-fold OOF evaluation, not the technical fixtures or smoke runs.