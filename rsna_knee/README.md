# RSNA Knee Abnormality Detection — solution pipeline

A complete, local-first training and inference pipeline for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
challenge: multi-label detection of knee findings from MRI, with multilingual
radiology reports available at training time.

Everything here is designed to run on your own machine, since the data is far
too large to move around.

---

## An important caveat about the competition specifics

The machine this code was written on cannot reach `kaggle.com` or `rsna.org` —
both are blocked by the network policy — so the exact label names, the exact
evaluation metric and the exact submission layout could not be read from the
official pages. Rather than guessing and hard-coding values that might be
wrong, the pipeline **discovers them at run time**:

| Unknown | How the code handles it |
| --- | --- |
| Label column names | Read from `train.csv`; any binary column that is not obvious metadata becomes a target |
| Number of findings | Follows from the above — nothing assumes twelve |
| Submission layout | Copied from `sample_submission.csv`, wide or long, including row-id format and row order |
| Exam identifier | Detected from the usual names (`StudyInstanceUID`, `exam_id`, …) |
| Evaluation metric | Per-label AUC, macro AUC and weighted AUC are all reported; set `train.metric_name` to whichever is official |

The discovered schema is written to `runs/<name>/schema.json` on the first run.
**Read that file once before you commit to a long training run** and correct it
by hand if anything looks wrong — everything downstream follows from it.

What is confirmed from public reporting: over 5,000 knee MRI exams with reports
in 12 languages from 16 sites across five continents, expert-annotated
evaluation data, a $77,000 prize pool that includes a first-ever award for the
most efficient models, and a deadline of **22 October 2026**.

---

## Method

### 1. Cache the DICOM data once

Decoding DICOM dominates epoch time. `rsnaknee.preprocess` walks the raw tree,
groups files into series, and writes each series as a single uint8 `.npy`
volume plus one manifest row.

Two details make this robust across sixteen sites:

* **The imaging plane is derived geometrically**, from the cross product of the
  direction cosines in `ImageOrientationPatient` — never from
  `SeriesDescription`, which in this dataset may be in any of twelve languages.
* **Contrast weighting comes from `EchoTime` / `RepetitionTime` /
  `InversionTime`**, so T1, PD, T2 and STIR are separated without reading any
  free text.

### 2. Model: 2.5D encoder, attention over slices, attention over series

* Knee MRI is anisotropic — roughly 0.3 mm in plane against 3-4 mm between
  slices — so a symmetric 3D kernel spends capacity on an axis with almost no
  resolution. Instead a 2D ImageNet-pre-trained backbone sees three
  neighbouring slices as its three input channels. With only ~5,000 exams,
  keeping pre-trained weights matters more than anything else.
* A **transformer over the slice axis** then pools each series. A tear appears
  on a handful of slices; mean pooling dilutes that evidence, attention finds it.
* **Attention fusion across series**, with a learned embedding per
  (plane, weighting, fat-saturation) type. Different findings live in different
  sequences — menisci on sagittal PD, marrow oedema on fat-saturated — and
  sixteen sites means no fixed protocol to rely on. The model learns what each
  series type is worth instead of assuming one.
* **Per-series auxiliary heads** shorten the gradient path and stabilise early
  training.

### 3. The reports: a text teacher, distilled

This is the part that exploits what makes this challenge unusual.

A multilingual transformer (XLM-RoBERTa) is fine-tuned to predict the findings
**from the report alone** — an easy task, since the report often names the
finding. Its *out-of-fold* probabilities then become a second training target
for the image model, alongside the hard labels.

Why it helps: the binary labels throw away the radiologist's hedging. "Possible
small radial tear of the posterior horn" and "large displaced bucket-handle
tear" both become a hard `1`. The teacher's soft output preserves that gradient
of certainty, and a borderline image *should* predict a borderline value.
Using out-of-fold predictions is essential — a teacher scoring its own training
data returns near-perfect 0/1 values that carry no more information than the
labels.

The teacher is a training-time device. If the hidden test set supplies reports
too, you can blend both modalities; if it does not, the image model keeps the
benefit.

### 4. Validation

Multi-label stratified **group** k-fold. Both constraints matter: a patient
appearing in training and validation inflates the score, and a fold with no
positives for a rare finding makes its AUC undefined and the mean noisy. No
`scikit-learn` splitter does both, so `rsnaknee.folds` implements the greedy
balanced assignment directly.

Tune blending and thresholds on the pooled out-of-fold file (`oof.csv`), never
on a single fold's validation score.

### 5. A note on augmentation

Horizontal flipping is **off by default, deliberately**. Flipping a coronal
knee series swaps medial and lateral, turning a medial meniscal tear into a
lateral one — it corrupts the label. The flag exists, but leave it off unless
your label set has no side-specific findings.

---

## Installation

```bash
cd rsna_knee
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your driver; the pinned
`torch>=2.1` will otherwise fetch a CPU-only wheel.

## Usage

**1 — Cache the training DICOMs** (once; expect an hour or two, and roughly
2-4 GB of cache per thousand exams):

```bash
python -m rsnaknee.preprocess \
  --dicom-dir /path/to/train_images \
  --out-dir cache/train \
  --size 256 --max-slices 48 --workers 12
```

Check the series mix it prints at the end — it tells you which planes and
weightings the dataset actually contains, which is worth knowing before you
tune `data.max_series`.

**2 — Train the text teacher** (optional but recommended, ~20 minutes on one GPU):

```bash
python -m rsnaknee.text --config configs/base.yaml \
  --set paths.reports_csv=/path/to/train_reports.csv
```

**3 — Train the image model:**

```bash
python -m rsnaknee.train --config configs/base.yaml \
  --set paths.cache_dir=cache/train train.distil_weight=1.0
```

Train one fold first (`--folds 0`) and check `report_fold0.json` before
committing to all five.

**4 — Predict:**

```bash
python -m rsnaknee.infer --config configs/base.yaml \
  --set paths.test_dicom_dir=/path/to/test_images paths.cache_dir=cache/test \
  --output submission.csv
```

### Fitting your GPU

| VRAM | Suggested settings |
| --- | --- |
| 8 GB | `data.image_size=192 data.depth=12 data.max_series=3 train.batch_size=2 train.accumulate=4 model.grad_checkpoint=true` |
| 12 GB | `data.depth=16 data.max_series=4 train.batch_size=2 train.accumulate=4` |
| 16 GB | defaults, `train.batch_size=4` |
| 24 GB+ | `model.backbone=convnext_small data.image_size=288 train.batch_size=6` |

Memory scales with `batch_size × max_series × depth`, since that product is how
many slices reach the backbone per step. Reduce `depth` before `image_size`:
losing in-plane resolution costs more than losing slices.

`train.amp_dtype=bf16` needs an Ampere card or newer; use `fp16` on older GPUs.

### The efficiency track

The competition awards a separate prize for efficient models. `configs/efficiency.yaml`
trains a single small backbone at reduced resolution with distillation carrying
more of the load — the teacher matters more at low capacity. For submission,
use one fold and disable test-time augmentation:

```bash
python -m rsnaknee.infer --config configs/efficiency.yaml --folds 0 \
  --set inference.tta_slice_shift=false
```

Inference time per exam is logged at the end of every run.

---

## Layout

```
configs/          base and efficiency configurations
rsnaknee/
  schema.py       run-time discovery of labels and submission format
  dicom_io.py     DICOM reading, geometric plane detection, normalisation
  preprocess.py   builds the uint8 volume cache
  folds.py        multi-label stratified group k-fold
  dataset.py      series routing, slice sampling, augmentation
  models.py       2.5D backbone, slice transformer, series fusion
  losses.py       balanced multi-label loss, distillation loss
  metrics.py      per-label / macro / weighted AUC
  train.py        cross-validated training
  text.py         multilingual report teacher
  infer.py        ensembling, TTA, submission writing
tests/            unit tests, no data required
```

## Tests

```bash
python -m pytest tests/ -q          # 37 unit tests, no data required
python scripts/smoke_test.py        # full pipeline on synthetic DICOMs, ~1 minute
```

The unit tests cover schema discovery, submission formatting, fold balance, the
AUC implementation (checked against `scikit-learn`), plane and weighting
detection, slice sampling and ragged-batch collation.

The smoke test is the more useful one before a real run: it fabricates a small
DICOM dataset with realistic geometry and metadata, then exercises caching,
schema discovery, folds, training, out-of-fold scoring and submission writing
end to end. If it passes, any failure on the real data is about the data rather
than the code.

## Suggested order of work

1. Cache the data, read `schema.json`, confirm the labels are right.
2. Train fold 0 of `configs/base.yaml`. This is your reference score.
3. Train the text teacher, set `train.distil_weight=1.0`, retrain fold 0.
   Distillation is the highest-value single change in this pipeline.
4. Train all five folds, check `report_oof.json` for weak labels.
5. Add a second backbone (for example `tf_efficientnetv2_s`) and average the
   out-of-fold predictions to confirm the blend helps before submitting.
