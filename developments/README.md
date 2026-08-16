# Developments archive

This directory preserves the complete research/development history that previously occupied the repository root.

It contains the B0--B30 experiment lineage, historical configurations, documentation, scripts, source modules, tests, Kaggle methodology notes, fixtures, manuscript material and the previous GitHub workflows.

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

B28 then tested a zero-gated element-wise max-evidence residual over encoder image-content slice embeddings. The fixed-E2 run was valid and the 768-dimensional gate learned small finite values, but the reused expert macro fell to 0.638346 versus B20 0.667407 (delta -0.029061; paired 95% CI [-0.065624, +0.007122]; P[B28>B20]=0.0586). B28 is not promoted and the max-evidence residual formulation is closed.

B29 tested a zero-gated second learned softmax summary of the same 16 B20 slice tokens. Its fixed-E2 training run completed the exact B20 surface with a frozen B16 encoder. On the reused 58-study expert development surface, B29 reached macro AUC 0.676888 versus B20 0.667407 (raw delta +0.009481; paired 95% CI [-0.003749, +0.024188]; P[B29>B20]=0.9188). This is encouraging but not independent validation. B29 is therefore a **frozen promising candidate, not promoted**. Its hidden competition submission has been frozen byte-for-byte and B29 must not be retuned from the reused 58-study result.

The next prospectively frozen experiment is **B30**. B30 keeps B20's historical learned-attention series token `A`, but replaces B29's raw dot-product complementary summary with a new query operating through the current B20 Q/K/V, output-projection and LayerNorm affine parameters as **detached deterministic operators**. The B20 `A` branch remains unchanged. B30 adds the same 1,536 trainable parameters as B29 (768 query + 768 zero-init gate), starts as the exact B20 function, uses fixed E2, and records a prospective attention-complementarity mechanism audit before any B30 expert outcome is inspected.

Canonical result/design records include:

```text
docs/B27_1_REUSED_GOLD_RESULT.md
docs/B28_MAX_EVIDENCE_RESIDUAL.md
docs/B28_REUSED_GOLD_RESULT.md
docs/B29_COMPLEMENTARY_SERIES_POOL.md
docs/B29_REUSED_GOLD_RESULT.md
docs/B30_PROJECTED_COMPLEMENTARY_SERIES_POOL.md
```

Current B29/B30 implementation support:

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
```

B30 frozen controls remain the B20/B29 historical recipe: B6 supervision, frozen B16 encoder, 90% crop, all 3,120 studies, all 17,475 eligible series, 14,123 supervision cells, the historical optimizer/augmentation/loader seed, five-epoch scheduler horizon, and fixed-E2 endpoint. The historical 623-study weak-v2 partition is not a holdout for B30.

For commands from this archive, the implementation remains importable with:

```bash
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"
```

A branch containing the exact pre-restructure tree is also retained as:

```text
archive/pre-clean-structure-2026-08-15
```
