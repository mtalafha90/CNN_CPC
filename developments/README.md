# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B26.2 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

## Layout

```text
developments/
  configs/                  historical experiment configurations
  docs/                     experiment records and scientific notes
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

The files are preserved for reproducibility. New work on the active model should use the clean top-level `model/`, `training/`, `validation/`, `testing/`, `data/`, `config/` and `docs/` interface.

The current targeted-supervision records are:

```text
docs/B26_TARGETED_FILL.md
docs/B26_2_DETERMINISTIC_GATE.md
```

B26-v1 raw extraction is complete but failed its first manual negative-label quality audit. B26.1 reduced the raw 631 proposed Synovitis fill cells to 281 same-polarity candidates, but its fresh audit still achieved only 60% accepted negation precision (36/60), so B26.1 is also not approved for training. B26.2 is now the next frozen step: a deterministic precision-first evidence whitelist over the completed B26.1 output. It can only remove proposals and remains blocked from training until a new fresh manual audit passes. B20 remains the active working model throughout.

For old commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
