# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B31 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

B26 supervision repair is closed and not promoted. B27 was structurally superseded before outcome inspection after a routing audit found perfect fluid/fat metadata collinearity on all 17,475 training series. B27.1 corrected that defect with a 60-parameter `plane + paired_sequence` route, completed a valid fixed-E2 run, and was then evaluated on the reused 58-study expert development surface. Its macro AUC was 0.659923 versus B20 0.667407 (delta -0.007483; paired 95% CI [-0.034725, +0.019182]; P[B27.1>B20]=0.2918), so B27.1 was not promoted and the routing family is closed for that formulation.

B28 tested a zero-gated element-wise max-evidence residual over encoder image-content slice embeddings. The valid fixed-E2 run fell to macro 0.638346 versus B20 0.667407 on the reused expert surface (delta -0.029061; P[B28>B20]=0.0586). B28 is closed and not promoted.

B29 tested a zero-gated second learned softmax summary of the same 16 B20 slice tokens. Its fixed-E2 run completed the exact B20 surface. On the reused 58-study expert development surface, B29 reached macro AUC 0.676888 versus B20 0.667407 (delta +0.009481; paired 95% CI [-0.003749, +0.024188]; P[B29>B20]=0.9188). This is encouraging but not independent validation. B29 is therefore a **frozen promising candidate, not promoted**, and must not be retuned from the reused expert result.

B30 replaced B29's raw dot-product complementary scorer with a projected complementary query using detached current B20 Q/K/V/out/LayerNorm operators. Its training/mechanism audit was valid, but the reused expert macro declined to 0.654703 versus B20 0.667407 (delta -0.012703; paired 95% CI [-0.039119, +0.010722]; P[B30>B20]=0.1422). B30 is **not promoted and its formulation is closed**. See `docs/B30_REUSED_GOLD_RESULT.md`.

The next prospectively frozen experiment is **B31 local-context complementary pooling**. B31 keeps B29's simple learned query and original-value weighted sum, but scores each slice after a zero-initialized depthwise Conv1d(k=3) local through-plane context residual. Context changes attention scores only; the weighted values remain the original B20 slice tokens. B31 adds 2,304 context parameters to B29's 1,536 query+gate parameters, for 3,840 new parameters versus B20. Both the context convolution and outer B29 gate start at exact zero, so B31 starts as the exact B20 function and as exact B29 complementary scoring when the context branch is zero.

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
```

Current B29--B31 implementation support:

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
```

B31 keeps the historical B20/B29 controls: B6 supervision, frozen B16 encoder, 90% crop, all 3,120 studies, all 17,475 eligible series, 14,123 supervision cells, historical optimizer/augmentation/loader seed, five-epoch scheduler horizon, and fixed-E2 endpoint. The 623-study weak-v2 partition is not a holdout.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
