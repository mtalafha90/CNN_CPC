# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B29 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

B20 remains the active working model.

B26 supervision repair is closed and not promoted. B27 was structurally superseded before outcome inspection after a routing audit found perfect fluid/fat metadata collinearity on all 17,475 training series. B27.1 corrected that defect with a 60-parameter `plane + paired_sequence` route, completed a valid fixed-E2 run, and was then evaluated on the reused 58-study expert development surface. Its macro AUC was 0.659923 versus B20 0.667407 (delta -0.007483; paired 95% CI [-0.034725, +0.019182]; P[B27.1>B20]=0.2918), so B27.1 was not promoted and the routing family is closed for that formulation.

B28 then tested a zero-gated element-wise max-evidence residual over encoder image-content slice embeddings. The fixed-E2 run was valid and the 768-dimensional gate learned small finite values, but the reused expert macro fell to 0.638346 versus B20 0.667407 (delta -0.029061; paired 95% CI [-0.065624, +0.007122]; P[B28>B20]=0.0586). B28 is not promoted and the max-evidence residual formulation is closed.

Canonical closed-family records include:

```text
docs/B27_1_REUSED_GOLD_RESULT.md
docs/B28_MAX_EVIDENCE_RESIDUAL.md
docs/B28_REUSED_GOLD_RESULT.md
```

The next frozen imaging-side experiment is **B29**:

```text
docs/B29_COMPLEMENTARY_SERIES_POOL.md
src/rsna_knee/b29_complementary_series_pool.py
src/rsna_knee/b29_training.py
src/rsna_knee/b29_gold_eval.py
tests/test_b29_complementary_series_pool.py
```

B29 keeps the complete B20 learned attention-pooled series token `A` and adds a second deterministic learned softmax summary `C` of the same 16 B20 slice tokens. The mixed token is:

```text
A + tanh(g) * (C - A)
```

The new query `q` has 768 parameters and the feature-wise gate `g` has 768 parameters, for 1,536 new parameters total. The gate starts at exactly zero, so B29 starts as the exact B20 function. The complementary branch contains no dropout or other random operation, and safety tests pin training-mode RNG-path equivalence to B20 at zero gate.

B29 keeps B6 supervision, the frozen B16 encoder, B20 crop geometry, all 3,120 studies, all 17,475 eligible series, the same optimizer/augmentation/loader seed and the fixed-E2 endpoint. B20 remains active until independent evidence justifies a change.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
