# CNN_CPC — knee MRI abnormality detection

Twelve binary findings per knee MRI study, scored as macro ROC AUC.

The top level of this repository is one model and the three ordinary stages
that surround it. The complete research lineage that produced the model is
preserved under [`developments/`](developments/README.md) and is not part of
the working interface.

## The model

```text
MRI study
  -> every eligible real MRI series
  -> 16 sampled slice positions per series
  -> adjacent-slice 2.5D triplets
  -> 224 x 224 tensors
  -> deterministic centred 90% crop, resized back to 224
  -> frozen ConvNeXt-Tiny encoder
  -> attention pooling to one token per series, plus a complementary summary
  -> Transformer over the study's series
  -> 12 pathology queries
  -> 12 probabilities
```

See [`docs/WORKING_MODEL.md`](docs/WORKING_MODEL.md) for the design decisions
and their reasons, and [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for the commands.

## Structure

```text
CNN_CPC/
├── config/          model configuration
├── model/           architecture, preprocessing, implementation bridge
├── data/            studies, series and batching
├── training/        training entry point
├── validation/      scoring against the expert-annotated studies
├── testing/         competition test-set prediction
├── tests/           contract tests for the interface above
├── docs/            model and workflow documentation
├── developments/    the complete research archive
├── requirements.txt
└── pyproject.toml
```

Historical experiment names appear in exactly one file,
`model/_implementation.py`, which binds each preserved component to a plain
name. A test enforces that boundary, so the public interface cannot drift back
into experiment vocabulary.

## Installation

```bash
conda activate rsna-knee
pip install -e .
```

The dataset and run artefacts are deliberately not stored in Git.

## Usage

```bash
# train on a chosen report-label surface
python -m training.train --supervision all-script --data-root ... --out-root runs/working_model

# score the expert-annotated studies (a diagnostic, not a test)
python -m validation.validate --data-root ... --checkpoint runs/working_model/candidate/model.pt

# predict the competition test set
python -m testing.test --data-root ... --checkpoint runs/working_model/candidate/model.pt --out submission.csv

# print the architecture and training contract
python -m model.architecture
```

Full argument lists are in [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Two things to keep in mind

**The expert-annotated studies are not a test set.** All 58 were reused
throughout development. At that sample size a paired difference below roughly
0.03 macro AUC cannot be resolved, so `validation` answers "is this model
behaving sensibly", not "is this model better".

**Labels come from the reports, and the reports are multilingual.** The rule
parser reads Latin-script vocabulary, so a large share of studies yielded no
usable labels — the parser could not read them, rather than the reports being
silent. Translating before parsing recovers most of that population, which is
what `--supervision all-script` selects. Whether it produces a better model is
still open; the audit records are under `developments/docs/`.

## Tests

```bash
python -m pytest
```

Covers both the working interface and the preserved research implementation.
