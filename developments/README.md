# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B27.1 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

B27 completed a valid fixed-E2 run on the exact B20 surface, but before any expert/gold evaluation a routing audit found that its learned fluid and fat tables were exactly identical. Direct metadata inspection confirmed perfect collinearity on all 17,475 training series:

```text
fluid_id == fat_id   17475 / 17475
(1,1)                 7459
(2,2)                10016
other                     0
```

B27 is therefore structurally superseded before outcome inspection. It was not evaluated on the reused 58-study expert surface.

The frozen successor is B27.1:

```text
docs/B27_PATHOLOGY_SERIES_ROUTING.md
docs/B27_1_COLLINEARITY_SAFE_ROUTING.md
src/rsna_knee/b27_1_pathology_routing.py
src/rsna_knee/b27_1_training.py
src/rsna_knee/b27_1_gold_eval.py
scripts/review_b27_1_routes_with_ollama.py
tests/test_b27_1_pathology_routing.py
```

B27.1 replaces the redundant 84-parameter `plane + fluid + fat` route with a 60-parameter `plane + paired_sequence` route. The paired sequence categories are exactly the two combinations observed on the frozen training surface. Unknown or discordant future/test combinations receive zero paired-sequence routing bias. All routing values start at zero, so B27.1 is B20-equivalent before optimization.

B27.1 keeps B6 supervision, the frozen B16 encoder, B20 crop geometry, all 3,120 studies, all 17,475 eligible series, the same optimizer/augmentation/loader seed and the fixed-E2 endpoint. The training code refuses metadata drift from the frozen 7,459 / 10,016 pair counts.

Ollama remains audit-only and outside training/competition inference.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
