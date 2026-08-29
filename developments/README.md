# Developments lineage and active B42 implementation

This directory preserves the complete research/development history that
previously occupied the repository root. It also currently contains the
maintained B42--B49 implementation because those experiments were developed
after the repository was reorganized.

It contains the B0--B34 experiment lineage, the prospective weak-validation splits, the nine-phase dataset contract audit, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

The top-level B34 interface is retained for compatibility. The maintained
operational B42 endpoint instead uses `src/rsna_knee/` with
`config/b42_constant_area_aspect_sparse.yaml`. See
[`../docs/ACTIVE_ENDPOINTS.md`](../docs/ACTIVE_ENDPOINTS.md) before training or
submitting anything.

## Current development status

`docs/CURRENT_STATUS.md` is the authoritative snapshot. In brief:

**B42 is the maintained operational reference, not a scientific promotion.**
B37, B41, and B42 each obtained a displayed hidden Kaggle macro AUC of `0.714`.
B46, B48, and B49 completed without support for their proposed mechanisms;
B49's exploratory candidate-only hidden endpoint scored `0.707`.

**The B37--B49 architecture ladder is flat around the established baseline.**
Do not use B46/B48/B49 results to tune gold weight, cross-series query details,
tile geometry, crop, resolution, calibration, blend, or seeds.

**The 58-study expert surface is retired as a design surface.** It was reused
throughout development, and a paired difference below roughly 0.03 macro AUC
is not resolvable at that sample size. The prospective weak-validation splits
(PV1, PV2) replace it for architecture selection: 499-624 validation studies,
membership assigned by UID hash without reference to labels, predictions or
prior scores. PV2 produced a 95% interval entirely below zero (`P = 0.9998`),
which the expert surface has never achieved. Both PV surfaces rank on soft
BCE; macro AUC is recorded there without an interval and is not used to rank.

**The dataset contract audit found the largest single issue in the project.**
The reports are multilingual and the frozen parser reads Latin-script
vocabulary only, so 1,229 of 4,349 studies produced no usable labels at all —
not because the reports were silent, but because the parser could not read
them. Translating before running the unchanged parser raised coverage from
`71.74%` to `95.95%` and usable cells from `14123` to `18024` (`+27.62%`).
Whether that improves the model is unresolved: Phase 9 v2 tested it under a
proper holdout and came back inconclusive in aggregate, with only Contusion
surviving correction for 12 comparisons and removing Contusion flipping the
macro sign.

Independent hidden evidence exists and is recorded in
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). It is not a reason for
leaderboard-driven tuning.

Canonical result/design records include:

```text
docs/B27_1_REUSED_GOLD_RESULT.md
docs/B28_MAX_EVIDENCE_RESIDUAL.md
docs/B28_REUSED_GOLD_RESULT.md
docs/B29_COMPLEMENTARY_SERIES_POOL.md
docs/B29_REUSED_GOLD_RESULT.md
docs/B30_PROJECTED_COMPLEMENTARY_SERIES_POOL.md
docs/B30_REUSED_GOLD_RESULT.md
docs/B31_LOCAL_CONTEXT_COMPLEMENTARY_POOL.md
docs/B32_DISPERSION_COMPLEMENTARY_POOL.md
docs/B32_REUSED_GOLD_RESULT.md
docs/B33_UNIFORM_COMPLEMENTARY_MEAN.md
docs/B33_REUSED_GOLD_RESULT.md
docs/PROSPECTIVE_WEAK_V1.md
```

Current B29--B33 and prospective-validation implementation support:

```text
src/rsna_knee/b29_complementary_series_pool.py
src/rsna_knee/b29_training.py
src/rsna_knee/b29_gold_eval.py
src/rsna_knee/b29_submission.py
tests/test_b29_complementary_series_pool.py

src/rsna_knee/b30_projected_complementary_series_pool.py
src/rsna_knee/b30_training.py
src/rsna_knee/b30_gold_eval.py
tests/test_b30_projected_complementary_series_pool.py

src/rsna_knee/b31_local_context_complementary_pool.py
src/rsna_knee/b31_training.py
src/rsna_knee/b31_gold_eval.py
tests/test_b31_local_context_complementary_pool.py

src/rsna_knee/b32_dispersion_complementary_pool.py
src/rsna_knee/b32_training.py
src/rsna_knee/b32_gold_eval.py
tests/test_b32_dispersion_complementary_pool.py

src/rsna_knee/b33_uniform_complementary_mean.py
src/rsna_knee/b33_training.py
src/rsna_knee/b33_gold_eval.py
tests/test_b33_uniform_complementary_mean.py

src/rsna_knee/prospective_weak_v1.py
src/rsna_knee/prospective_weak_v1_training.py
src/rsna_knee/prospective_weak_v1_eval.py
tests/test_prospective_weak_v1.py
```

The historical B20-family controls remain B6 supervision, frozen B16 encoder, 90% post-resize crop, all-series B12/B13 policy, historical optimizer/augmentation/loader seed, five-epoch scheduler horizon, and fixed-E2 endpoint for recent candidates. The 623-study weak-v2 partition is not a holdout.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
