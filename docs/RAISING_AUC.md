# Raising macro AUC beyond B13

> **Status — 2026-08-11.** B13 remains the development champion at macro AUC
> `0.6293565948`. B14 completed at `0.6197914249` and was rejected globally.
> Package `0.22.1` adds corrected diagnostic tooling only; no B13/B14 training
> recipe or checkpoint is changed.

## Current evidence

```text
B13 macro AUC        0.6293565948
B14 macro AUC        0.6197914249

paired B14-B13
median difference    -0.0093726931
95% paired CI        [-0.0469823411,+0.0250137870]
P(B14 > B13)          0.2924
```

B14 fit the B6 weak labels more strongly than B13 (`0.5822778610` versus
`0.6132239342` final training loss) but did not improve global macro AUC. This is
important evidence against simply increasing downstream token capacity or fitting
B6 harder.

## What the B6 audit does and does not imply

The frozen B6 gold audit measured:

```text
sensitivity          0.975
specificity          0.606
positive precision   0.690
balanced accuracy    0.790
coverage             0.361
```

These values establish that the weak supervision is noisy and incomplete. They do
**not** mathematically imply a macro-AUC ceiling such as `0.75-0.80`. The defensible
conclusion is narrower: supervision quality may now be an important limiting
factor, especially because report-derived errors are not guaranteed to be random
class-conditional noise.

Therefore:

- do not quote a numerical supervision ceiling;
- treat specificity/coverage improvement as a plausible way to improve the
  learning signal globally;
- preserve the frozen B6 experiment when comparing historical B7-B14 results.

## Corrected diagnostic 1 — actual B13 slice exposure

The original slice audit used `16 / n_slices` as the fraction of a series seen.
That was not the B13 input pipeline. Each of the 16 center positions is a **2.5D
triplet**, training uses gaps `[1,2]` plus center jitter `+/-2`, and evaluation uses
TTA offsets `[-1,0,1]`.

The corrected `slice_audit.py` now:

1. reconstructs the exact 3,120-study B13 B6 surface;
2. verifies the exact 17,475-series all-series mapping and frozen SHA-256;
3. excludes gold studies by construction through `prepare_b7_supervision`;
4. reads only DICOM headers;
5. derives through-plane spacing by projecting `ImagePositionPatient` onto the
   normal from `ImageOrientationPatient`, rather than assuming patient Z;
6. computes the actual unique frame indices touched by the frozen B13 triplets;
7. reports the evaluation TTA union, longest unsampled run, expected unique
   training exposure per random view, and the full legal training-exposure
   envelope.

Run the exact surface audit with:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out runs/slice_audit_b13
```

A quick smoke test may use `--limit`, but only the full 17,475-series run may be
used for the diagnostic conclusion.

Interpretation is global:

- near-complete evaluation exposure with no multi-slice gaps -> slice-count
  undersampling is not supported as the primary bottleneck;
- material multi-slice gaps across a substantial fraction of series -> slice
  exposure remains a plausible global hypothesis;
- neither case authorizes target-wise slice counts or target-specific routing.

## Corrected diagnostic 2 — frozen weak-label holdout

The original note incorrectly treated a 20% holdout as though all 3,120 active B6
studies were validation studies and therefore quoted an expected CI near `0.015`.
A 20% split contains roughly 624 studies, and B6 is sparse: there are only 14,123
usable cells across all targets. The actual interval is target- and sparsity-
dependent and must be measured empirically.

`weak_validation.py` now freezes a report-group-safe split **before candidate
training**. Report groups are derived from the normalized report hashes already
used elsewhere in the repository; grouping is mandatory. The output manifest is
hashed and records actual study/cell counts and per-target positive/negative
counts.

Freeze it with:

```bash
rsna-knee-weak-holdout \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --holdout-fraction 0.20 \
  --seed 2026 \
  --out-root runs/weak_holdout_v1
```

Outputs:

```text
runs/weak_holdout_v1/weak_holdout_manifest.csv
runs/weak_holdout_v1/weak_holdout.json
```

The critical rule is:

> Any model scored on the weak holdout must have been trained with every holdout
> `StudyInstanceUID` excluded.

Existing B13/B14 checkpoints were trained on all 3,120 active B6 studies, so they
must **not** be retrospectively scored on this new holdout and called validation.
A new B13-control and every candidate compared to it must train on the same weak-
train partition.

The weak surface measures teacher agreement, not expert truth. Use:

```text
frozen weak holdout -> rank predeclared candidate structures with paired bootstrap
58-study gold       -> one development confirmation only
Kaggle hidden test  -> independent competition signal
```

The 58-study surface is not made unbiased again by reducing future use; it has
already been repeatedly reused.

## B15 remains the next representation hypothesis

The current reserved B15 hypothesis remains:

```text
ImageNet ConvNeXt-Tiny
        |
        v
competition knee-MRI self-supervised adaptation
        |
        v
B13 one-token-per-series hierarchy
        |
        v
frozen downstream B6 recipe
```

Before implementing B15 training, run the corrected slice audit and freeze the
weak holdout. Those diagnostics may change how B15 is evaluated or motivate a
later global slice/resolution experiment, but they do not retroactively alter B13
or B14.

For a clean B15 weak-surface comparison, train two models from scratch on the
same weak-train partition:

```text
control:   ImageNet -> B13 hierarchy
candidate: ImageNet -> MRI SSL -> B13 hierarchy
```

Then compare them on the frozen weak holdout with aligned bootstrap. Only one
predeclared winner should be taken to the repeatedly reused 58-study gold surface.

## Priority order

1. **Run corrected B13 slice-exposure audit.** Cheap, label-free, exact frozen
   series surface.
2. **Freeze weak holdout.** Do this before any B15/control training.
3. **Implement B15 plus a matched B13 weak-split control.** Same downstream
   architecture/training surface, only the pretraining path changes.
4. **Investigate supervision quality** as a separate future experiment if B15
   stalls. Improve report parsing only under a new frozen B6 version and audit it
   before downstream training.
5. **Resolution/slice-budget experiment** only if the corrected exposure audit
   shows a material global gap.
6. **Multi-seed/global ensembling** only after model structure is settled and
   without gold-selected target weights.

## Still prohibited

```text
target-wise B13/B14 winners
gold-selected slice counts
gold-selected thresholds
gold-selected ensemble weights
retrospective use of the new weak holdout for checkpoints trained on its studies
claiming the weak holdout is expert truth
claiming the reused 58 studies are independent validation
claiming a 0.75-0.80 supervision ceiling from B6 balanced accuracy
```

The objective remains a higher **global macro ROC AUC** through controlled,
reproducible representation or supervision improvements rather than increasingly
fine tuning to the 58 repeatedly reused labelled studies.
