# B55 runbook — the three changes the field measured

**Status: implemented, unrun.** Run it after B53 finishes.

B55 is a competition endpoint, not an experiment. It bundles three changes and
cannot attribute its result to any one of them. That is B52's limitation,
accepted for B52's reason: this project scores `0.716` against public notebooks
at `0.899`, and that is not a gap a single careful ablation closes.

```text
teacher      graded against the competition's severity rubric
crop         130 mm of knee, not 90% of whatever the scanner produced
laterality   one canonical orientation, correctly per plane
resolution   336 reference area, not 448
```

## Why these, and not the other two you asked for

Two of the four items were already implemented, and are not rebuilt:

```text
per-finding spatial pooling   b36 already scores every token per pathology
                              and pools top-k per pathology
per-finding series attention  b12_1 already carries 12 pathology_tokens
                              querying the series
wide slice band               _centers already spans the full stack
```

## Step 1 — regrade the teacher

The competition's labels are expert and image-derived, and borderline findings
are graded **NEGATIVE**. A report teacher tuned for clinical correctness
over-fires: "mild effusion", intrasubstance meniscal degeneration and low-grade
chondromalacia are all real and all competition-negative.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull origin main

PYTHONPATH=developments/src python -m rsna_knee.b55_rubric_teacher \
  --labels-root runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all \
  --reports-csv rsna-knee-abnormality-detection/train.csv \
  --out-root runs/089_Experiment_B55_physical_geometry/teacher_rubric
```

Check the reports CSV path first — the module looks for a `report`,
`report_text` or `text` column and says so if it finds none.

**Read the audit before training on it.** It prints a per-target breakdown and
writes `rubric_changes.csv` with the sentence behind every downgraded cell:

```text
  positive cells before      15,357
  downgraded to negated       ?,???
  fraction of positives       ??.?%

  by target
    Effusion              ?,???
    ...
  never gated: Contusion, Fracture
```

There is no threshold at which this is automatically right or wrong. What to
look for:

* **Effusion and Baker's should dominate.** Their rubric is the most explicitly
  severity-gated ("moderate or large only").
* **Spot-check twenty rows of `rubric_changes.csv`.** Each carries the matched
  term and the surrounding sentence. If the matched word plainly belongs to a
  different finding in the same sentence, the vocabulary is over-firing and the
  run should not proceed.
* **Contusion and Fracture must be zero.** The published rubric gives them no
  severity threshold, so they are never gated.

If the gate looks too aggressive, `--downgrade-to uncertain` makes it discard
the cell rather than flip it — weaker, safer, and a smaller change.

## Step 2 — preflight

```bash
PYTHONPATH=developments/src python -m rsna_knee.b55_physical_geometry_training \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --labels-root runs/089_Experiment_B55_physical_geometry/teacher_rubric \
  --series-policy runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --domain-split runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split \
  --num-workers 6 \
  --preflight-only
```

Expect:

```text
[B55] crop=130 mm  canonical_side=L  reference=336^2 (B42 used 448^2)
[B52] loader workers=6 (command line), sharing=file_system, worker={...}
```

### The guard that may fire

`_report_only_surface` asserts an exact usable-cell count. **Downgrading a
positive to a negative does not change how many cells are usable** — a negative
cell is still supervision — so the count should be unchanged and the guard
should pass.

If it fires, the rubric did something other than downgrade, and the run should
stop until you know what. Only then declare the new number with
`--expected-supervision-cells`. Do not reach for that flag to make the message
go away; it exists to record a count you have understood.

## Step 3 — train

```bash
mkdir -p runs/089_Experiment_B55_physical_geometry

PYTHONPATH=developments/src python -m rsna_knee.b55_physical_geometry_training \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --labels-root runs/089_Experiment_B55_physical_geometry/teacher_rubric \
  --series-policy runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --domain-split runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split \
  --num-workers 6 \
  --epochs 8 \
  --out-root runs/089_Experiment_B55_physical_geometry \
  2>&1 | tee runs/089_Experiment_B55_physical_geometry/b55_train.log
```

Add `--augment` only once B53 has reported positive — see the next section.

It resumes if interrupted — re-run the identical command.

**Expect it to be faster than B53.** 336² is about 56% of 448²'s pixels, so the
encoder does roughly half the work per slice.

## Augmentation

`--augment` applies B53's augmentation to the **training surface only**, using
the same policy read from the same config. It is **off by default**, and that is
a decision rather than an oversight: whether augmentation helps at these
settings is exactly what B53 is measuring right now, and switching it on here
would bundle a fourth unvalidated change into a run that already cannot
attribute.

Turn it on once B53 reports positive. The composition is already built and
tested — `B55AugmentedDataset` inherits B53's `__getitem__` and B55's
`_load_b42`, and a test runs the delegation rather than reading it, because
both classes override `_load_b42` and a broken chain would still produce valid
pixels.

## Reading the result — and the comparison you cannot make

**B55's validation macro AUC is not comparable with B52's `0.834998` or with
B53's.** The teacher changed, so the validation labels changed too. A score
against different labels answers a different question, and the difference
between the two numbers would be mostly the ruler.

This is the real cost of bundling the teacher with the geometry, and it is worth
being blunt about: **the local surface cannot tell you whether B55 is better.**

Three things it *can* do:

* **Support a submission.** The leaderboard scores against labels neither run
  controls, so it is the one comparison the teacher change does not corrupt.
* **Show the training curve.** Whether B55 converges, plateaus or overfits is
  visible regardless of which labels the surface uses.
* **Be made comparable, with a tool that does not yet exist.** Scoring B55's
  checkpoint against the *old* labels on the same 548 studies would restore a
  common ruler and isolate the geometry. That evaluator has not been built. Ask
  for it before drawing a local conclusion from B55, rather than after.

Given the measured exchange rate — local movement reaches Kaggle at about a
quarter — the honest expectation is that the crop and laterality bundle is worth
roughly `+0.030` on the leaderboard *if* it reproduces here, and that the
teacher regrade is the larger unknown in both directions.

## What this does not license

Do not tune the crop millimetres, the reference side, the epoch count or the
rubric vocabulary against B55's score. Three changes at once already means the
result cannot attribute; tuning against it would make the next result
uninterpretable as well.
