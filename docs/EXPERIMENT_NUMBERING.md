# Experiment numbering and local run organization

## Scope and rule

This ledger was reconstructed from all **1,143 commits** reachable from `main`,
from the first commit (`934e78b`, 2026-08-07) through `17a98fe`
(2026-08-22). The numbering follows the canonical scientific sequence declared
by the experiment ledgers and result records across that history—not raw commit
timestamp order. That distinction matters because some implementation work was
committed before the preceding experiment's outcome was recorded.

- Matched controls and arms declared by one protocol stay under that experiment.
- A failed, blocked, superseded, deferred, or never-run experiment keeps its
  number. Historical numbers must not be reused.
- Operational comparisons, preflight checks, loose logs, and older pipeline
  folders are shared artifacts, not extra experiments.
- Number order records the governed scientific sequence. It does not imply that
  every experiment completed or that supporting code was committed in the same
  order.

The early `E01`-`E09` files are deliberately not numbered. Their own historical
README states that the actual campaign could not be completed without mounted
data/GPU access; they were then removed in commits explicitly calling them
historical. They are unexecuted design prototypes, not measured experiment
records. Similarly, loose B1/B3/B4 comparison JSONs are shared comparisons;
only the later predeclared B5+B7.1 ensemble is a separate governed experiment.

The machine-readable source of truth is
[`config/experiment_registry.json`](../config/experiment_registry.json). New
experiments must be appended to it; existing numbers must not be reordered.

## Numbered ledger

| Number | Code | Experiment | Repository status | Evidence commit |
|---:|---|---|---|---|
| 001 | `B0` | Random-initialized Stage-1 baseline | `completed` | `27d9c65` |
| 002 | `REPORT_TEACHER` | Fold-safe report-teacher benchmark | `completed` | `c4d0af0` |
| 003 | `B1` | Competition-only MRI SSL plus Stage-1 | `completed` | `dfcf0b7` |
| 004 | `B2` | Discriminative SSL fine-tuning | `completed` | `1d154f9` |
| 005 | `B3` | Pathology-aware low-capacity MIL | `completed` | `db2e984` |
| 006 | `B4` | Frozen SSL features with target-wise classical classifiers | `completed` | `80e244e` |
| 007 | `B4.1` | Shared-policy frozen-SSL classifier | `completed` | `2de14bb` |
| 008 | `B4.2` | Grouped-policy frozen-SSL classifier | `completed` | `e70fed9` |
| 009 | `B4.3` | Two-way cross-validated frozen-SSL classifier | `completed` | `8a9d37d` |
| 010 | `B5` | Competition-only image-report representation learning | `completed` | `3361fd0` |
| 011 | `B6` | Structured multilingual report-label supervision | `completed` | `c8be937` |
| 012 | `B7` | B5-initialized pathology-query model with B6 weak labels | `completed` | `eaf69f9` |
| 013 | `B7.1` | Full-corpus B7 weak-supervision coverage | `completed` | `0f3a0ca` |
| 014 | `B5_B7.1_ENSEMBLE` | Fixed B5 plus B7.1 rank ensemble | `completed` | `61ef16a` |
| 015 | `B8` | Pathology-aware spatial anatomy model | `completed` | `e965359` |
| 016 | `B9` | Strict exact-contrast semantic routing | `completed` | `583c614` |
| 017 | `B10` | Physical-scale normalization | `completed` | `041760b` |
| 018 | `B11` | Conservative teacher-student label completion | `viability_gate_failed` | `1bd7cd8` |
| 019 | `B11.1` | Calibration-aware target-wise quantile teacher tails | `completed` | `7837e12` |
| 020 | `B12` | All-real-series variable-series model | `completed` | `16b1ae0` |
| 021 | `B12.1` | Hierarchical learned series-token aggregation | `implemented_not_run` | `da73981` |
| 022 | `B13` | ImageNet ConvNeXt hierarchical series model | `completed` | `b7c1a24` |
| 023 | `B14` | ImageNet full slice-token memory | `completed` | `61dfc58` |
| 024 | `B15` | ImageNet to knee-MRI SSL to B13 hierarchy | `completed` | `e2d9bec` |
| 025 | `B16` | Full-report semantic representation alignment | `completed` | `88a0313` |
| 026 | `B17` | Frozen B16 encoder with fixed downstream training | `completed` | `138b090` |
| 027 | `FINAL` | All-data production model definition | `implemented_deferred` | `efaa325` |
| 028 | `B18` | Fisher-style expert-guided epoch selection | `completed` | `677cc6f` |
| 029 | `B19` | Ninety-percent crop with cosine vignette | `completed` | `68b4bec` |
| 030 | `B20` | Post-resize ninety-percent crop-only focus | `completed_promoted` | `d1e9212` |
| 031 | `B21` | Pre-resize crop with matched B20-v2 control | `completed` | `19fd6e4` |
| 032 | `B22` | B21 training-duration audit | `completed` | `2284b70` |
| 033 | `B23` | Local-Qwen report labeller and formal gate | `completed_gate_failed` | `3c7d6ac` |
| 034 | `B24` | Formal matched B6-versus-B23 supervision experiment | `blocked_not_run` | `b542609` |
| 035 | `B24X` | Exploratory B6-versus-B23 supervision pilot | `completed` | `4fac398` |
| 036 | `B24X_DENSITY` | B6-preserved B23 missing-cell density ablation | `completed` | `4fac398` |
| 037 | `B25X` | Three-arm ChatGPT hybrid-supervision experiment | `completed` | `cf0b477` |
| 038 | `B26` | Targeted supervision fill for balance-flagged targets | `quality_gate_failed` | `7ecf246` |
| 039 | `B26.1` | Strict LLM evidence-adjudication gate | `quality_gate_failed` | `a84fc03` |
| 040 | `B26.2` | Deterministic evidence gate and fixed-E2 training | `completed` | `15499c4` |
| 041 | `B27` | Pathology-specific acquisition routing | `completed_superseded` | `4d5600e` |
| 042 | `B27.1` | Collinearity-safe pathology routing | `completed` | `ce13892` |
| 043 | `B28` | Zero-gated max-evidence series residual | `completed` | `cb052ef` |
| 044 | `B29` | Zero-gated complementary softmax series pool | `completed` | `2cb7d00` |
| 045 | `B30` | Projected complementary attention | `completed` | `b226195` |
| 046 | `B31` | Complementary pooling with local slice context | `completed` | `24da028` |
| 047 | `B32` | Weighted-dispersion complementary summary | `completed` | `2fcc865` |
| 048 | `B33` | Exact-uniform complementary mean | `completed` | `2d74336` |
| 049 | `PV1` | Prospective weak-validation v1 matched controls | `completed` | `bc29ff3` |
| 050 | `B34_PV2` | B34 training-only context scaffold on PV2 | `completed` | `c7ad1f1` |
| 051 | `AUDIT_P1_P2` | Dataset-contract population and physical slice-count audits | `completed` | `dac726f` |
| 052 | `AUDIT_P3` | DICOM scanner and header heterogeneity audit | `completed` | `ad3d95e` |
| 053 | `AUDIT_P4` | B6 supervision and acquisition-domain intersection audit | `completed` | `1ed404e` |
| 054 | `AUDIT_P5` | Report-supervision failure-mode audit | `completed` | `5bf1d97` |
| 055 | `AUDIT_P6` | Frozen-parser translation-rescue feasibility pilot | `completed` | `205e87a` |
| 056 | `AUDIT_P7` | Full B6-inactive translation-rescue audit | `completed` | `8402351` |
| 057 | `AUDIT_P8` | Frozen global translation-rescue supervision merge | `completed` | `b4ba034` |
| 058 | `PHASE9_V1` | Phase-9 matched supervision on the full population | `superseded_before_result` | `3a54540` |
| 059 | `PHASE9_V2` | Phase-9 v2 matched supervision with frozen PV2 holdout | `completed` | `08db2b2` |
| 060 | `WORKING_CONTROL` | Clean working-model frozen-encoder control | `completed_hidden_0.688` | `7b9baf6` |
| 061 | `DINOV3_TINY` | DINOv3 ConvNeXt-Tiny encoder replacement | `local_artifacts_result_not_in_repo` | `7b9baf6` |
| 062 | `DINOV3_WIDE` | Wide DINOv3 base/large experimental variant | `local_artifacts_result_not_in_repo` | `f7792bb` |
| 063 | `ENCODER_PROBE` | Frozen-encoder representation probe | `local_artifacts_result_not_in_repo` | `bd076c9` |
| 064 | `FINETUNE_1STAGE` | One-stage encoder fine-tuning | `completed_hidden_0.694` | `d299b12` |
| 065 | `SEED_ENSEMBLE` | Seed-varied model and prediction ensemble | `local_artifacts_result_not_in_repo` | `f908237` |
| 066 | `TARGET_070` | Positive soft-target retargeting to 0.70 | `local_artifacts_result_not_in_repo` | `a8bb500` |
| 067 | `LLM_FILL_ALL` | B6-preserved LLM fill across all targets | `completed` | `9055d47` |
| 068 | `LLM_FILL_NO_SYNOVITIS` | LLM-fill ablation excluding Synovitis | `local_artifacts_result_not_in_repo` | `9055d47` |
| 069 | `B35` | Target-conditioned dense spatial residual | `completed` | `63d6565` |
| 070 | `B36` | Pathology-specific sparse top-k spatial MIL | `completed` | `8412e51` |
| 071 | `B37` | High-resolution 448 pathology-specific sparse-MIL test | `implemented_not_run` | `cb4198f` |
| 072 | `NATIVE_RESOLUTION_AUDIT` | Dataset-wide native DICOM geometry audit | `implemented_not_run` | `0cc80eb` |

## Organize the local archive safely

The organizer creates a symlink view at `runs/by_experiment/`. It does **not**
rename, move, copy, or delete anything in the existing archive. This avoids
breaking hard-coded checkpoint and result paths already recorded in the repo.

First review the dry-run:

```bash
cd /media/talafha/Disk_1/CNN_CPC
python tools/organize_runs.py \
  --runs-root /media/talafha/Disk_1/CNN_CPC/runs
```

Then create the index:

```bash
python tools/organize_runs.py \
  --runs-root /media/talafha/Disk_1/CNN_CPC/runs \
  --apply
```

The result has this shape:

```text
runs/by_experiment/
├── 001_Experiment_B0_random_init_stage1/
├── ...
├── 071_Experiment_B37_highres_448_sparse_mil/
├── 072_Experiment_NATIVE_RESOLUTION_AUDIT_native_dicom_geometry/
├── _Shared/
│   ├── Comparisons/
│   ├── Legacy_pipeline/
│   ├── Loose_logs_and_processes/
│   └── Preflight_and_checks/
├── _Unclassified/
├── INDEX.csv
├── INDEX.json
└── README.md
```

Each existing top-level run item appears as a relative symlink under its
numbered experiment or shared bucket. `INDEX.csv` and `INDEX.json` record the
full mapping. Empty numbered directories are still created so the ledger also
shows experiments that were blocked or never run. Anything without an approved
rule is retained in `_Unclassified` for review instead of being guessed.

The command is idempotent. A second run recognizes correct existing links. If
any destination is occupied by the wrong link or by a real file, the command
stops before creating or overwriting links.

B37 remains number 071 because that family was introduced before the native
resolution audit. Its original 288 design was never run and is now superseded;
the active implemented protocol is the frozen 448-resolution sparse-MIL design
at commit `9395665`. Both possible run names map to the same permanent B37
number so historical numbering does not shift.

## Physically migrate run directories

The symlink index above is the safest default. If the physical archive itself
must use the numbered layout, use `tools/migrate_runs.py`. It moves only real
top-level directories classified as experiments; loose logs, CSV/JSON files,
shared audits, and unclassified folders stay where they are.

Multiple runs belonging to one experiment remain distinct inside one permanent
container:

```text
runs/010_Experiment_B5_image_report_ssl/
├── b5_frozen_probe/
└── b5_report_ssl/
```

By default the old paths become relative compatibility symlinks, so commands
that still reference `runs/b5_report_ssl/...` continue to work. Every applied
migration records each completed move in `runs/_migration/manifests/`.

Dry-run first:

```bash
python tools/migrate_runs.py \
  --runs-root /media/talafha/Disk_1/CNN_CPC/runs
```

Apply after reviewing the complete plan:

```bash
python tools/migrate_runs.py \
  --runs-root /media/talafha/Disk_1/CNN_CPC/runs \
  --apply
```

The apply command prints the exact manifest path. Roll back with:

```bash
python tools/migrate_runs.py \
  --runs-root /media/talafha/Disk_1/CNN_CPC/runs \
  --rollback /absolute/path/to/migration_YYYYMMDDTHHMMSSZ.json
```

Rollback performs a complete preflight before changing anything. It refuses to
proceed if a moved directory or compatibility alias has been replaced or
changed unexpectedly.

## Adding future experiments

1. Append the next permanent number to `config/experiment_registry.json`.
2. Add exact run-folder aliases or tightly scoped glob patterns.
3. Run `pytest -q tools/tests/test_organize_runs.py`.
4. Dry-run the organizer and inspect `_Unclassified` before applying it.

Never insert or recycle a historical number. If one protocol contains several
matched arms, list all of their folder patterns under the same registry entry.
