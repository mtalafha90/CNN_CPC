# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B27 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

B26 supervision repair is closed and not promoted. Its final deterministic B26.2 labels passed manual semantic audit, but the fixed-E2 model did not improve the reused expert macro and Synovitis AUC decreased. The mechanism audit showed a large within-target class-mass and co-occurrence shift. Canonical records:

```text
docs/B26_TARGETED_FILL.md
docs/B26_2_DETERMINISTIC_GATE.md
docs/B26_2_REUSED_GOLD_RESULT.md
docs/B26_CLOSURE.md
policies/b26_2_quality_approval.json
```

B20 therefore remains the active working model.

The next imaging-side experiment is B27:

```text
docs/B27_PATHOLOGY_SERIES_ROUTING.md
src/rsna_knee/b27_pathology_routing.py
src/rsna_knee/b27_training.py
src/rsna_knee/b27_gold_eval.py
scripts/review_b27_routes_with_ollama.py
tests/test_b27_pathology_routing.py
```

B27 adds only 84 zero-initialised pathology-specific metadata attention-bias parameters to the existing B20 pathology-query cross-attention. It keeps B6 supervision, the frozen B16 encoder, B20 crop geometry, all 3,120 studies, all 17,475 eligible series and the fixed-E2 training endpoint unchanged. Ollama is audit-only and is not part of B27 training or competition inference.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
