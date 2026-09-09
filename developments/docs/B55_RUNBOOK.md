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

The report text comes from the competition's own `Report` column, which is
what `data.load_train_csv` requires. Other spellings (`report_text`, `text`)
are accepted, and the match ignores case; if none is found the error lists the
columns the file actually has. It prints which column it used.

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

**Do not add `--augment`.** B53 has now reported, and it reported negative —
see the next section.

It resumes if interrupted — re-run the identical command.

**Expect it to be faster than B53.** 336² is about 56% of 448²'s pixels, so the
encoder does roughly half the work per slice.

## Augmentation

**Settled: leave it off.** B53 finished at `0.826853` against B52's
`0.834998` — `-0.008145` at eight epochs on the identical split. The rule
written here before the run was that `--augment` goes on only if B53 reported
positive. It did not, so it stays off, and the fourth unvalidated change stays
out of a run that already cannot attribute.

B53 had not converged — its best epoch was its last, and the gap was still
closing — so this is "not worth bundling here", not "augmentation is useless".
The distinction is recorded in `B53_AUGMENTATION_APPLIED.md`; it does not
change what B55 should do.

`--augment` applies B53's augmentation to the **training surface only**, using
the same policy read from the same config. It is **off by default**. The
composition is already built and
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
* **Be made comparable — `common_ruler_eval` now does this.** See step 4.

Given the measured exchange rate — local movement reaches Kaggle at about a
quarter — the honest expectation is that the crop and laterality bundle is worth
roughly `+0.030` on the leaderboard *if* it reproduces here, and that the
teacher regrade is the larger unknown in both directions.

## What this does not license

Do not tune the crop millimetres, the reference side, the epoch count or the
rubric vocabulary against B55's score. Three changes at once already means the
result cannot attribute; tuning against it would make the next result
uninterpretable as well.

## Step 4 — score every checkpoint on one ruler

Because the teacher changed, B55's own validation number is not on the same
scale as B52's or B53's. `common_ruler_eval` fixes that: **one labels root, one
study set, each model's own geometry.** Same answer key, same studies, each
model seeing what it was trained to see — so the difference is the model.

Run it twice, once per ruler, and read the pair:

```bash
cd /media/talafha/Disk_1/CNN_CPC
export COMMON="--data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --series-policy runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --domain-split runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split \
  --num-workers 6 \
  --checkpoint runs/087_Experiment_B52_full_data/b52_best_model.pt \
  --checkpoint runs/088_Experiment_B53_augmentation_applied/b53_best_model.pt \
  --checkpoint runs/089_Experiment_B55_physical_geometry/b52_best_model.pt"

# ruler 1: the old teacher, the scale B52's 0.834998 is already on
PYTHONPATH=developments/src python -m rsna_knee.common_ruler_eval $COMMON \
  --labels-root runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all \
  --out-json runs/089_Experiment_B55_physical_geometry/ruler_old_teacher.json

# ruler 2: the regraded teacher, the target the competition actually scores
PYTHONPATH=developments/src python -m rsna_knee.common_ruler_eval $COMMON \
  --labels-root runs/089_Experiment_B55_physical_geometry/teacher_rubric \
  --expected-supervision-cells <the count step 1 reported> \
  --out-json runs/089_Experiment_B55_physical_geometry/ruler_rubric.json
```

Each is 548 studies with no training — minutes, not hours.

Every row is scored through **the geometry its own checkpoint records**, read
out of the `b55_geometry` field `train_b55` writes. So a second B55 run at a
different `--crop-mm` or `--reference-side` is scored at *its* numbers, not at
this run's, and the JSON says which under `geometry_used`. A B55 checkpoint
carrying no recorded geometry is refused rather than scored at the defaults —
the defaults happening to be right is exactly what would hide the mistake.

### Reading the pair, and what it cannot tell you

**A common ruler compares complete models. It does not attribute.** B55's
weights were trained on regraded labels *and* B55's geometry, so scoring it
against the old teacher does not isolate the geometry: it scores a
differently-trained model on a familiar scale. Nothing here separates the
crop from the laterality from the resolution from the teacher, and no choice
of ruler can, because all four are baked into the weights.

What the two rulers actually give you:

```text
ruler = old teacher       B55 as a whole, on the scale B52's 0.834998 and
                          B53's number already sit on. Comparable, and still
                          a comparison of complete models.

ruler = regraded teacher  the same three models on the answer key that is
                          closer to the competition's severity rule. No
                          historical numbers exist on this scale.
```

Both rulers are report-derived, and **the competition's labels are neither**:
two MSK radiologists reading images, with borderline findings graded negative.
The regraded teacher is a closer proxy than the old one; it is still a proxy.

So: if B55 wins on both rulers, it is a better model and worth a submission. If
it wins on one, that says which answer key favours it, not which change caused
it. If it loses on both, the bundle is not working and the slot is better spent
elsewhere. Attribution would need one change per run, which is the trade this
endpoint deliberately gave up.

### What this is not

It is **not** a selection. Every checkpoint in that table was already chosen by
its own run on its own surface. Promoting the winner of this comparison would be
the post-hoc selection the archive forbids, and the tool prints that under every
table it produces.

### The guard it keeps

`--expected-supervision-cells` is required when the ruler is a regraded teacher
whose usable-cell count differs from the frozen 34,010. Two label sets with
different cell counts are not the same ruler in the sense that matters, so the
count is declared rather than allowed to drift silently.
