# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B33 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

**B20 remains the active working model.**

B26 supervision repair is closed and not promoted. B27.1, B28 and B30 were valid experiments but were not promoted. B30's projected complementary-attention formulation is closed.

B29 tested a zero-gated second learned softmax summary of the same 16 B20 slice tokens. Its fixed-E2 run completed the exact B20 surface. On the reused 58-study expert development surface, B29 reached macro AUC 0.676888 versus B20 0.667407 (delta +0.009481; paired 95% CI [-0.003749, +0.024188]; P[B29>B20]=0.9188). This is encouraging but not independent validation. B29 remains a **frozen promising candidate, not promoted**.

B31 kept B29's simple query and added a zero-init depthwise Conv1d(k=3) local-context perturbation to the attention scorer only. Its fixed-E2 run was valid and reached macro AUC 0.682280 on the reused expert surface, versus B29 0.676888 and B20 0.667407. The raw B31-B29 delta was +0.005392 and raw B31-B20 delta +0.014873. However, B31's prospective mechanism audit showed almost no attention redistribution at E2 (normalized JS divergence 3.56e-9, top-1 agreement 98.9%, top-3 overlap 99.63%). B31 is therefore a **frozen leading development candidate, not independently validated**. No B31.1 is permitted from the reused result.

B32 added a same-weight second-order feature standard-deviation summary alongside B29's mean-like residual. The training/mechanism run was valid and the dispersion residual was non-redundant, but the reused expert macro was only 0.668699, versus B29 0.676888 and B31 0.682280. B32-B31 was -0.013580 raw with P[B32>B31]=0.0946. B32 is **not promoted and its weighted-dispersion formulation is closed**. See `docs/B32_REUSED_GOLD_RESULT.md`.

B33 removed the learned complementary query entirely and used the exact arithmetic mean of the 16 B20 slice tokens behind one zero-init 768-D gate. On the reused 58-study expert surface B33 reached 0.676446, essentially matching B29 0.676888 (raw delta -0.000442; paired 95% CI [-0.023312, +0.022036]; P[B33>B29]=0.4800). B31 remained numerically highest at 0.682280. B33 is therefore a **successful frozen simplification, not independently validated and not promoted**. Its result supports the interpretation that most of B29's development gain may come from a second broad global series representation rather than learned slice selection. See `docs/B33_REUSED_GOLD_RESULT.md`.

The repeatedly reused 58-study expert surface is now retired as the primary architecture-design surface. Before any B34 experiment, the repository freezes **prospective weak-validation v1**: a StudyInstanceUID-hash 80/20 split of the 3,120 active B6 studies, with 2,496 studies for matched fixed-E2 training and 624 untouched studies for weak-label validation. Split membership uses UID hashes only and does not use B6 labels, expert labels, model predictions or previous scores. B20, B31 and B33 are retrained as matched controls on the 80% subset before B34. This is architecture-selection evidence only, not independent clinical validation. See `docs/PROSPECTIVE_WEAK_V1.md`.

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
