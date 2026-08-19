# Phase 7 — full B6-inactive translation-rescue population audit

## Status

**FROZEN BEFORE FULL-POPULATION RESULTS. READY TO RUN AFTER PHASE-6 PASS.**

Phase 6 passed every predeclared coverage-mechanism gate. Phase 7 now scales the exact same mechanism to the complete frozen B6-inactive report-only population. This stage remains descriptive supervision auditing: no MRI model is trained, B6 v1.2.1 is unchanged, and B35 remains undefined.

## Frozen population

From the completed dataset contract audit:

```text
report-only studies             4349
original B6-active studies      3120
original B6-inactive studies    1229
original B6 usable cells       14123
```

Only the 1,229 studies with **zero original usable B6 cells** are eligible for translation rescue.

No B6-active study is translated for rescue. Partially silent cells inside an otherwise active study remain untouched.

## Exact translator freeze

Phase 7 must use the same translator that passed Phase 6:

```text
backend             Ollama local
model               qwen3:14b
Ollama digest        bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
quantisation         Q4_K_M
decoding             greedy
seed                 2026
max_new_tokens       4096
prompt SHA-256       086e1daae2843c70712a29662a589dee629d32d7f014a9a51613be496a95ee1a
```

The implementation aborts before generation if this provenance differs.

## Frozen rescue mechanism

```text
original report-only study
    |
    +-- original B6 has >=1 usable cell
    |       -> original B6 remains untouched
    |       -> no translation rescue
    |
    +-- original B6 has 0 usable cells
            -> deterministic English translation
            -> unchanged B6 v1.2.1
            -> record definite translated-B6 cells
```

The full stage does not introduce target-specific thresholds, script-specific acceptance rules, B6 replacements, or new clinical semantics.

## Why this stage is descriptive rather than a new acceptance test

Phase 6 established feasibility on a frozen 36-report inactive sample. Phase 7 asks what the exact mechanism does on all 1,229 eligible studies:

- how many studies become B6-active;
- how many positive and negative cells are recovered;
- which targets receive additional supervision;
- how rescue varies by report script;
- whether the acquisition-domain gap documented in Phase 4 is reduced;
- how many reports remain unrecovered even after successful translation.

No target-level Phase-7 result may be used to tune the translation prompt, alter B6, or selectively retain/discard rescued cells.

## Resumability

A 1,229-report local LLM run can be long. The generator therefore writes a local append-only cache after each successful translation:

```text
runs/report_translation_rescue_full/translation_cache.jsonl
```

Re-running the exact command resumes from that cache. The cache is validated for duplicate or non-eligible UIDs.

The cache contains translated competition report content and is **local-only**. Do not commit it.

## Outputs

```text
runs/report_translation_rescue_full/
├── full_population_summary.json
├── full_population_rescue_audit.csv
├── recovered_cells.csv
├── translation_cache.jsonl              # local-only raw translations
└── translation_failures.csv              # only if failures occur
```

`recovered_cells.csv` records only the newly definite translated-B6 cells from the originally zero-cell studies. It is an audit artifact, not yet an authorized MRI training target file.

## Run

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"

PYTHONPATH=developments/src \
python -m rsna_knee.report_translation_rescue_full \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --domain-study-csv runs/dataset_domain_intersection_audit/study_domain_table.csv \
  --out-root runs/report_translation_rescue_full \
  --model qwen3:14b \
  --num-ctx 8192 \
  --max-new-tokens 4096 \
  --seed 2026
```

The command is intentionally resumable. If interrupted, run the same command again; completed translations are reused from the local cache.

## Governance after Phase 7

```text
modify frozen B6 v1.2.1                         NO-GO
fill partially silent B6-active studies          NO-GO
target-specific translation rescue rules         NO-GO
script-specific rescue thresholds                NO-GO
define B35                                        NO-GO
MRI training from rescued cells                  NOT YET AUTHORIZED
inspect full-population rescue/domain balance    GO
```

After Phase 7, a downstream MRI experiment may be defined only if the aggregate full-population result remains scientifically coherent. Any such experiment must be separately frozen before training and must compare a global supervision policy, not target-wise choices made from Phase-7 outcomes.
