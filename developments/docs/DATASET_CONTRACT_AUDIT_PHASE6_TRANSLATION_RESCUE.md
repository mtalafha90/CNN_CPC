# Phase 6 — translation to frozen-B6 rescue feasibility pilot

## Status

**Frozen before running the translation pilot.**

This is a supervision-only feasibility experiment motivated by the completed Phase-5 report inspection. It does not train an MRI model, does not modify B6 v1.2.1, and does not define B35.

## Hypothesis

Phase 5 showed that many B6-inactive reports contain clear target information but use terminology/scripts not covered by the current rule lexicon.

The constrained hypothesis is:

> A faithful deterministic English translation can act as a language-normalization layer, allowing the unchanged frozen B6 parser to recover useful positive and negative cells from studies that currently have zero usable B6 supervision.

The candidate is intentionally narrower than direct LLM target extraction.

```text
report
  -> pinned local deterministic translation
  -> unchanged B6 v1.2.1
  -> rescue cells only if ORIGINAL study has zero usable B6 cells
```

## Why translation instead of another direct B23 extraction pass

Historical B23 demonstrated dense multilingual structured extraction but failed its predeclared reused-gold specificity gate. B24X/B25X suggested that fill-only use is safer than replacing B6 decisions.

Translation rescue keeps the B6 target/state logic fixed. The language model is asked only to translate, not to classify the 12 targets. This separates the Phase-5 language-coverage mechanism from a wholesale change in labelling semantics.

## Frozen Phase-5 input

Input:

```text
runs/report_supervision_gap_audit/report_text_sample.jsonl
```

Expected sample version:

```text
report_supervision_gap_sample_v1
```

The three primary B6-inactive feasibility strata are fixed:

```text
latin_b6_inactive       12
greek_b6_inactive       12
cyrillic_b6_inactive    12
total                   36
```

Active controls and reused gold examples are translated for diagnostics, but only the 36 originally B6-inactive report-only studies are eligible for translated-B6 rescue.

## Translation contract

Default backend uses the existing reproducible local-LLM infrastructure:

```text
backend              Ollama local
default model        qwen3:14b
decoding             greedy / temperature 0
thinking             disabled
model digest         recorded from Ollama
prompt SHA-256       recorded
context              8192
output JSON schema   {"translation": "..."}
```

The translation prompt requires preservation of assertions, negations, uncertainty, grade, anatomy, laterality, measurements and report content. It forbids target classification, summarization and diagnostic inference.

The run aborts before interpretation if the local model provenance is not reproducibly pinned.

## Candidate merge contract

The merge rule is fixed:

```text
if original B6 study has >=1 usable cell:
    use original B6 only
    translated cells are NOT applied

if original B6 study has 0 usable cells:
    translated-B6 definite cells may fill silent cells
```

An original usable B6 cell can therefore never be overridden.

This pilot does not authorize later expansion to partially silent cells in otherwise B6-active studies.

## Predeclared feasibility criteria

The Phase-6 pilot passes the **coverage-mechanism feasibility** gate only if all conditions hold:

```text
translation failures                                      0
overall rescue rate across 36 inactive reports          >=75%
rescue rate in each of Latin/Greek/Cyrillic strata      >=50%
each inactive script stratum recovers:
    at least one positive translated-B6 cell              yes
    at least one negative translated-B6 cell              yes
all original B6-active control cells preserved            yes
```

A study is rescued when it changes from zero original usable B6 cells to at least one usable candidate cell.

These criteria are frozen before translation results are observed.

Passing means only that translation plausibly repairs the documented language-coverage mechanism. It does **not** establish clinical label accuracy and does not authorize MRI model promotion.

## Gold diagnostic

The sample contains reused official gold reports. Translated-B6 definite-call coverage, accuracy and positive/negative call precision are recorded only as descriptive safety diagnostics.

They are not a formal acceptance gate because:

- the 58-study gold surface has already been reused repeatedly;
- only six sampled gold studies are non-Latin;
- the Phase-5 design directly inspected these report texts.

No script-specific clinical performance claim is allowed from this sample.

## Outputs

```text
runs/report_translation_rescue_pilot/
├── pilot_summary.json
├── pilot_cell_audit.csv
├── translation_results.jsonl
└── translation_failures.csv       # only when failures occur
```

`translation_results.jsonl` contains translated competition report content and is a **local-only artifact**. Do not commit it.

## Run

First ensure the same pinned local model is installed and Ollama is available:

```bash
ollama list
```

Then:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main

PYTHONPATH=developments/src \
python -m rsna_knee.report_translation_rescue_pilot \
  --sample-jsonl runs/report_supervision_gap_audit/report_text_sample.jsonl \
  --out-root runs/report_translation_rescue_pilot \
  --model qwen3:14b \
  --num-ctx 8192 \
  --max-new-tokens 4096
```

Package only for local review/upload to ChatGPT; do not commit outputs:

```bash
zip -r report_translation_rescue_phase6.zip \
  runs/report_translation_rescue_pilot
```

## Decision after pilot

If the frozen feasibility gate fails, do not compensate by tuning per-script thresholds, target-specific rescue rules, or selective report inclusion on this sample.

If it passes, the next permissible step is to freeze a full-population translation-rescue generation protocol for the 1,229 original B6-inactive report-only studies, inspect aggregate label balance/domain recovery, and only then decide whether a matched downstream MRI experiment is scientifically justified.

B6 v1.2.1 remains frozen in either case.
