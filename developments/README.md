# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B28 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

## Layout

```text
developments/
  configs/                  historical experiment configurations
  docs/                     experiment records and scientific notes
  policies/                 frozen development-quality approvals
  src/rsna_knee/            complete historical implementation
  scripts/                  exploratory/run scripts
  tests/                    historical unit/regression tests
  fixtures/                 validation fixtures
  kaggle/                   Kaggle support material
  .github/                  historical workflows
  README_LEGACY.md          previous repository README
  README_KAGGLE_METHODS.md  previous Kaggle methods README
  main.tex                  manuscript source
  pyproject_legacy.toml     previous all-experiments CLI/package definition
  requirements_legacy.txt   previous requirements snapshot
```

The files are preserved for reproducibility. New work on the active model should use the clean top-level `model/`, `training/`, `validation/`, `testing/`, `data/`, `config/` and `docs/` interface unless the experiment is explicitly kept in this development archive.

## Current development status

B26 supervision repair is closed and not promoted. B20 remains the active working model.

B27 was structurally superseded before outcome inspection after a routing audit found perfect fluid/fat metadata collinearity on all 17,475 training series. B27.1 corrected that defect with a 60-parameter `plane + paired_sequence` route, completed a valid fixed-E2 run, and was then evaluated on the reused 58-study expert development surface. Its macro AUC was 0.659923 versus B20 0.667407 (delta -0.007483; paired 95% CI [-0.034725, +0.019182]; P[B27.1>B20]=0.2918), so B27.1 was not promoted and the routing family is closed for that formulation.

Canonical routing records:

```text
docs/B27_PATHOLOGY_SERIES_ROUTING.md
docs/B27_1_COLLINEARITY_SAFE_ROUTING.md
docs/B27_1_REUSED_GOLD_RESULT.md
```

The next frozen imaging-side experiment is **B28**:

```text
docs/B28_MAX_EVIDENCE_RESIDUAL.md
src/rsna_knee/b28_max_evidence_residual.py
src/rsna_knee/b28_training.py
src/rsna_knee/b28_gold_eval.py
tests/test_b28_max_evidence_residual.py
```

B28 leaves the B20 learned attention-pooled series token intact and adds a feature-wise, zero-initialised tanh-gated residual from an element-wise max over encoder image-content slice embeddings. Position and acquisition-metadata embeddings are removed before the max branch. The gate has 768 parameters and starts at exactly zero, making B28 functionally identical to B20 before optimization while adding no extra image-encoder pass.

B28 keeps B6 supervision, the frozen B16 encoder, B20 crop geometry, all 3,120 studies, all 17,475 eligible series, the same optimizer/augmentation/loader seed and the fixed-E2 endpoint. B20 remains active until B28 earns promotion.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
