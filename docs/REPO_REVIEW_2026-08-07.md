# CNN_CPC Current Repository Technical Review

**Repository:** `mtalafha90/CNN_CPC`  
**Review date:** 2026-08-07  
**Reviewed branch:** `main`  
**Latest reviewed commit:** `fc2da2be87bcfcf9ffaea0459339d7c03bc2c635`

## Executive assessment

The repository has improved substantially since the initial competition baseline and methodology review. The strongest recent change is structural: the duplicate second pipeline was removed, leaving one authoritative implementation under `src/rsna_knee/`. That avoids conflicting assumptions, duplicated logic, import ambiguity, and divergent training behavior.

Current assessment:

| Area | Assessment |
|---|---:|
| Architecture/design | 8/10 |
| Validation methodology | 9/10 |
| DICOM robustness | 7/10 |
| Training sophistication | 6/10 |
| Ready for first real baseline run | **Yes** |
| Ready as final competition approach | **No** |

The repository is now methodologically strong enough that a first complete cross-validation run would produce a meaningful baseline. The priority should therefore shift temporarily from adding speculative architectures to validating the existing pipeline on real data.

---

# 1. Major improvements now present

## 1.1 One authoritative pipeline

The old duplicate `rsna_knee/` package was removed and useful functionality was consolidated into `src/rsna_knee/`.

This is important because the removed pipeline assumed a fully labelled training set in places, whereas the competition contains only a very small gold-labelled subset. A single package now reduces the risk of accidentally running incompatible implementations.

## 1.2 Fold-safe teacher calibration

`src/rsna_knee/calibration.py` now learns empirical probabilities of the form

```text
P(y = 1 | target, report-rule state)
```

from gold studies outside the current validation fold.

The intended protocol is:

```text
for each validation fold k:
    use gold studies not in k to calibrate report states
    generate soft labels for all training studies
    train the MRI model
    evaluate only on official gold studies in k
```

This directly prevents circular validation. Calibrating the report teacher on all gold studies and then validating on a subset of those same studies would allow the teacher to indirectly see validation answers.

The calibration implementation also uses empirical-Bayes shrinkage toward target prevalence so very small rule-state cells do not become hard 0 or 1 estimates.

## 1.3 Gold labels dominate pseudo-labels

The training pipeline combines official labels and pseudo-labels, with the gold studies receiving a larger configurable weight.

The current local GPU configuration uses:

```yaml
gold_weight: 8.0
```

This is a sensible default for experimentation, although the value should ultimately be treated as an ablation rather than a fixed truth.

## 1.4 Better uncertainty handling for the tiny validation set

`src/rsna_knee/evaluation.py` adds:

- macro ROC-AUC;
- per-target ROC-AUC;
- explicit undefined-target reporting when only one class is present;
- study-level bootstrap confidence intervals;
- paired bootstrap comparison between two OOF runs.

This is particularly valuable because the trusted validation surface contains only 58 studies. A raw macro-AUC reported to three decimal places can imply far more precision than the data supports.

The paired comparison is especially useful for model development:

```text
same bootstrap studies
    -> score model A
    -> score model B
    -> compute B - A
```

Because both models are scored on the same resampled cases, much of the study-selection noise cancels.

## 1.5 DICOM handling is significantly more robust

The DICOM loader now includes several important protections:

- physical slice ordering using `ImageOrientationPatient` and `ImagePositionPatient`;
- `InstanceNumber` fallback;
- support for files without a `.dcm` suffix;
- enhanced multi-frame DICOM support;
- `RescaleSlope` and `RescaleIntercept`;
- `MONOCHROME1` inversion;
- normalization of mixed in-plane shapes;
- percentile clipping and resize.

This is a large improvement over filename-based slice ordering.

## 1.6 Better sequence metadata handling

`src/rsna_knee/data.py` now avoids naïve `astype(bool)` conversion on series flags.

That matters because:

```python
bool("False") == True
```

and missing values can also behave unexpectedly when converted directly. Incorrect conversion could silently mark structural sequences as fluid-sensitive and corrupt series routing.

The repository also now includes `dicom_meta.py`, which can infer anatomical plane and some sequence characteristics from DICOM metadata when the CSV information is missing.

## 1.7 Better workstation/GPU runtime configuration

`configs/local_gpu.yaml` and `runtime.py` now support:

- automatic CUDA/CPU selection;
- BF16 on suitable newer GPUs;
- FP16 fallback on older GPUs;
- configurable DataLoader workers;
- persistent workers;
- prefetching;
- TF32/CuDNN runtime optimization;
- optional multi-GPU `DataParallel`;
- AMP during validation;
- checkpoint unwrapping so DataParallel and single-GPU checkpoints load consistently.

This is a meaningful practical improvement because DICOM decoding and loading can easily starve a modern GPU.

---

# 2. Important issues to fix before a serious long training run

## 2.1 DICOM failures can still silently become zero MRI streams

This is the most important remaining engineering risk.

Inside `KneeStudyDataset._load()`, failures are currently handled conceptually as:

```python
try:
    volume = preprocess_volume(read_dicom_series(...))
except Exception:
    return zero_volume, 0.0
```

Similarly, individual unreadable DICOM instances are skipped inside the series reader.

This behavior keeps training from crashing, but it can hide major dataset or codec failures. For example, a missing compressed-DICOM decoder could potentially cause many studies to become zero-valued streams while training continues.

### Recommended fix: strict preflight command

Add a command such as:

```bash
rsna-knee preflight --data-root ...
```

It should inspect a representative sample of studies and report:

- selected series UID for each stream;
- whether each series directory exists;
- number of DICOM instances discovered;
- number of instances successfully decoded;
- decoded volume shape;
- tensor min/max/mean/std;
- percentage of missing streams;
- percentage of zero streams;
- DICOM decoder failures;
- anatomical plane agreement between CSV and header inference.

Training should refuse to start when failure rates exceed a configurable threshold.

Suggested default:

```text
fail if >5% of sampled expected streams cannot be decoded
```

The exact threshold should be adjusted after auditing the real dataset.

---

## 2.2 Metadata backfilling exists but is not integrated into normal training/inference

`backfill_series_metadata()` is implemented, but the main paths still currently do essentially:

```python
series = load_series_csv(...)
series_index = build_series_index(...)
```

without first invoking the backfill step.

This means blank `Anatomical_Plane` entries can still lead to an invisible stream and ultimately a zero-valued tensor.

### Recommended change

Training:

```python
series = load_series_csv(...)
series, metadata_stats = backfill_series_metadata(
    series,
    root,
    split="train",
)
print(metadata_stats)
series_index = build_series_index(...)
```

Inference should perform the equivalent operation with `split="test"`.

The repair statistics should be written into the run metadata so that experiments are reproducible.

---

## 2.3 Test inference should be image-only by default

The current inference code still blends image predictions with report predictions:

```text
final = alpha * image_probability + (1 - alpha) * report_probability
```

with the CLI default historically set near `alpha=0.70`.

When the test data has no report, `load_test_csv()` creates an empty report string. The rule engine then produces approximately neutral values for all targets.

This means inference can become approximately:

```text
p_final = 0.7 * p_MRI + 0.3 * 0.5
```

For ROC-AUC, a fixed affine transform of every prediction preserves ranking, but it is unnecessary, compresses probabilities, and obscures the intended methodology.

### Recommended policy

The report should be:

```text
TRAINING TEACHER ONLY
```

unless the official hidden test data explicitly supplies usable reports.

Use:

```yaml
fusion_alpha: 1.0
```

or remove report fusion from the default test path entirely.

Report/image fusion can remain as an optional diagnostic experiment when both modalities genuinely exist.

---

# 3. Main scientific limitation of the current image model

The validation framework has advanced faster than the image architecture.

The current model remains intentionally simple:

```text
MRI slices
   -> shared 1-channel ResNet18
   -> slice attention
   -> stream embeddings
   -> common stream attention
   -> one pooled study vector
   -> 12 outputs
```

This is a good baseline, but it does not yet include the strongest techniques identified in the public-methodology review.

Still missing:

- 2.5D neighboring-slice triplets;
- ConvNeXt / ConvNeXtV2;
- frozen or fine-tuned DINOv2;
- target-specific MIL attention;
- ranking loss aligned with ROC-AUC;
- learned Top-K slice/window selection;
- six-stream routing as the default research configuration;
- small complementary 3D branch;
- heterogeneous model ensembling.

The correct response is not to add all of these simultaneously. Each should be introduced as a controlled experiment against frozen OOF folds.

---

# 4. Recommended immediate experiment sequence

## E01 — current baseline

Before changing the architecture, run the current model on all frozen folds.

Example:

```bash
rsna-knee train --config configs/local_gpu.yaml --fold 0
rsna-knee train --config configs/local_gpu.yaml --fold 1
rsna-knee train --config configs/local_gpu.yaml --fold 2
```

Then evaluate all OOF predictions together:

```bash
rsna-knee evaluate \
  --train-csv /path/to/train.csv \
  --oof runs/local/fold0/oof.csv \
        runs/local/fold1/oof.csv \
        runs/local/fold2/oof.csv
```

This becomes the reference experiment **E01**.

Do not change several model components before recording this number.

### Required E01 artifacts

Store:

```text
config
commit SHA
fold assignment
OOF predictions
per-target AUC
macro AUC
bootstrap interval
training history
runtime
GPU model
peak GPU memory if possible
checkpoints
```

---

## E02 — 2.5D ResNet18

This should be the first architectural change.

Current representation:

```text
slice[z] -> 1-channel ResNet18
```

Proposed:

```text
slice[z-1]
slice[z]
slice[z+1]
     |
     v
three-channel 2.5D image
     |
     v
ResNet18
```

Advantages:

- incorporates local through-plane context;
- remains computationally cheap;
- is compatible with RGB-pretrained backbones without modifying the first convolution;
- provides a clean test of whether inter-slice information matters before moving to expensive 3D networks.

Compare E02 against E01 with the paired bootstrap evaluator.

Adopt 2.5D only if the improvement is reproducible and larger than expected validation noise.

---

## E03 — target-specific attention

The current architecture uses one shared stream-attention mechanism for all 12 targets.

That is biologically and diagnostically restrictive because the most useful MRI evidence differs by abnormality.

Examples:

```text
ACL            -> sagittal ligament evidence
meniscus       -> sagittal + coronal
medial OA      -> coronal structural series
PF OA          -> axial/sagittal patellofemoral evidence
effusion       -> fluid-sensitive series
Baker cyst     -> posterior fluid-sensitive evidence
contusion      -> fat-suppressed marrow signal
fracture       -> structural + edema evidence
```

### Proposed architecture

Use one query/head per target:

```text
MRI stream/window tokens
        |
        +--> ACL query
        +--> MCL query
        +--> Medial Meniscus query
        +--> Lateral Meniscus query
        +--> Medial OA query
        +--> Lateral OA query
        +--> PF OA query
        +--> Effusion query
        +--> Synovitis query
        +--> Baker's query
        +--> Contusion query
        +--> Fracture query
```

Each target should be allowed to assign its own attention weights across streams/slices.

This may be more valuable than immediately increasing the backbone size.

---

## E04 — six-stream routing

The repository already contains a `dual` routing mode.

Test:

```text
sagittal_fluid
sagittal_structural
coronal_fluid
coronal_structural
axial_fluid
axial_structural
```

against the current best three-stream configuration.

Use missing-stream masks instead of fabricated data.

The comparison should be made with identical folds, teacher labels, architecture and optimizer settings.

---

# 5. Later model-development order

Once E01-E04 are complete, the recommended progression is:

```text
E05  ConvNeXt-Tiny / ConvNeXtV2-Tiny
E06  frozen DINOv2 slice/window embeddings
E07  partial DINOv2 fine-tuning
E08  confidence-weighted BCE + pairwise ranking loss
E09  top-K window aggregation
E10  learned fixed-budget window selector
E11  small 3D per-plane complementary model
E12  heterogeneous fold/backbone/2D-3D ensemble
```

Every experiment should modify as few components as possible.

---

# 6. README/documentation drift

The main README is now somewhat behind the implementation.

The codebase now contains important modules such as:

```text
src/rsna_knee/calibration.py
src/rsna_knee/evaluation.py
src/rsna_knee/runtime.py
src/rsna_knee/dicom_meta.py
configs/local_gpu.yaml
```

but the main README still describes the earlier package layout and workflow.

The README should be updated after the next integration fixes to document:

- fold-safe calibration;
- bootstrap evaluation;
- paired experiment comparison;
- runtime command;
- local GPU configuration;
- DICOM metadata backfill;
- strict preflight command once added;
- image-only inference default;
- current experiment naming/ledger protocol.

---

# 7. Automated testing / CI

The latest commit message reports that the test suite increased to **75 tests and all passed**. That is encouraging, but the latest GitHub commit currently has no attached automated status checks.

A GitHub Actions workflow should be added:

```text
.github/workflows/tests.yml
```

with at minimum:

```bash
pip install -e .
pip install pytest
pytest -q
```

Preferably test on one supported Python version first, then expand to a small version matrix only if useful.

The purpose is not merely cosmetic CI. With a DICOM pipeline, weak-supervision logic, fold calibration and multiple runtime paths, small refactors can create silent regressions. Automated tests should gate future merges.

---

# 8. Current priority list

Before additional architecture development, the repository should address the following in order:

1. **Add strict DICOM preflight and failure counters.**
2. **Integrate `backfill_series_metadata()` into both training and inference.**
3. **Make submission inference image-only by default.**
4. **Update the main README to match the current codebase.**
5. **Add GitHub Actions test execution.**
6. **Run the complete current three-fold baseline and preserve all OOF artifacts.**
7. **Implement 2.5D triplets as the first controlled model improvement.**
8. **Compare E02 vs E01 with paired bootstrap.**
9. **Then test target-specific MIL attention.**
10. **Then test six-stream routing.**
11. **Only after those experiments move to ConvNeXt/DINOv2/ranking loss/3D ensembles.**

---

# 9. Final assessment

The project has reached an important transition point.

Previously, the main risk was that the competition methodology itself was not yet trustworthy because of weak labels, duplicate pipelines, tiny gold validation, and fragile DICOM assumptions.

Those foundations are now substantially improved:

```text
report teacher
    -> fold-safe calibration
    -> weighted gold + pseudo supervision
    -> gold-only validation
    -> bootstrap uncertainty
    -> paired OOF comparison
```

and:

```text
DICOM
    -> physical slice ordering
    -> robust file discovery
    -> multi-frame support
    -> sequence metadata handling
    -> multi-stream CNN
```

The remaining immediate weaknesses are mostly integration and experiment-discipline issues rather than conceptual failures.

Therefore the correct next move is:

> **stabilize the data path, run the current full baseline, freeze its OOF result, and then improve the architecture one controlled experiment at a time.**

The first architectural improvement should be **2.5D local slice context**, followed by **target-specific attention** and **six-stream MRI routing**. Larger backbones, DINOv2, ranking losses and 3D models should come only after those simpler changes are measured against the frozen baseline.
