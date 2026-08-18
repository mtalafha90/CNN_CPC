# Training, validation and testing workflow

Three ordinary stages, one model. Paths below are examples; the dataset and run
artefacts are not stored in Git.

## 1. Environment

```bash
conda activate rsna-knee
pip install -e .
```

## 2. Training

Choose which report-label surface enters the gradient. `original` uses the
frozen rule-parser labels; `merged` adds the cells recovered by translating the
non-Latin-script reports first.

```bash
python -m training.train \
  --supervision merged \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --report-labels /path/to/report_labels \
  --translated-labels /path/to/translated_labels \
  --series-policy /path/to/series_policy.json \
  --encoder /path/to/report_aligned_encoder.pt \
  --out-root runs/working_model
```

Training runs to a fixed epoch and writes the checkpoint, a training audit and
the loss history. Nothing is selected by score.

To compare the two label surfaces, run both and change nothing else — the arms
already share architecture, study population, series exposure, optimiser and
seeds, so the supervision is the only thing that differs.

## 3. Validation

```bash
python -m validation.validate \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --checkpoint runs/working_model/candidate/model.pt \
  --out runs/working_model/validation.json
```

This scores the 58 expert-annotated studies and reports macro and per-target
AUC.

**Read the result carefully.** Those studies were reused throughout
development. At this sample size a paired difference below roughly 0.03 macro
AUC is not resolvable, so the number is a plausibility check — is the model
behaving sensibly — and not a basis for ranking two models or promoting one.

## 4. Test-set prediction

```bash
python -m testing.test \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --checkpoint runs/working_model/candidate/model.pt \
  --out submission.csv
```

Every study is scored at slice offsets `[-1, 0, 1]` and the probabilities are
averaged. Alongside the CSV this writes `submission.csv.manifest.json`
recording the checkpoint hash, the submission hash, the encoder fingerprint and
the test-split series coverage, so any submitted file can be traced back to the
run that produced it.

## 5. Model information

```bash
python -m model.architecture
```

Prints the twelve findings and the architecture and training contract.

## Historical work

Nothing from the research history was deleted. The complete experiment lineage
is preserved under `developments/`, and a safety branch exists at:

```text
archive/pre-clean-structure-2026-08-15
```
