# B55 runbook — the three changes the field measured

**Status: run and scored. B55 loses on both rulers.** `-0.015343` against B52
on the old teacher, `-0.001144` on the regraded one — and the second of those
is B55's home ground, where it had the epoch-selection advantage and B52 had
none. Do not submit it. See [The pair, read
together](#the-pair-read-together), and [B56](#b56--the-same-geometry-on-the-old-teacher)
for the one-flag follow-up that makes the geometry attributable.

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

## The run

Eight epochs, 3,801 studies, **21.0 hours** — faster than B53's 24.6 and B52's
26.6, as 336² predicted. About 157 minutes an epoch against B53's 184, roughly
15% off for the same work.

```text
epoch   train      validation   macro AUC    minutes
  1     1.074443   1.028110     0.744899     171.9
  2     0.993281   1.011958     0.779983     174.0
  3     0.951890   1.095444     0.810476     159.2
  4     0.912311   0.992895     0.798892     142.9
  5     0.868069   0.990408     0.807177     148.8
  6     0.830097   0.995655     0.814174     159.6     <- selected
  7     0.798812   1.003626     0.806800     152.1
  8     0.783751   1.008701     0.810947     149.2
```

**It converged.** From epoch 3 the score sits in a band of `0.015` with no
trend, which is B52's shape rather than B53's — B53's best epoch was its last
and it was still climbing. Training loss keeps falling to `0.784` while
validation loss flattens near `1.00` from epoch 4, so the last few epochs are
memorising rather than learning. Eight epochs was enough; a longer schedule
would not have helped this run.

### The supervision guard passed, and that is a result

The run was started **without** `--expected-supervision-cells` and did not
stop. That flag defaulting to `None` means the frozen `34,010` is enforced, so
the regraded teacher produced exactly the same number of usable cells as the
old one.

Which is what was predicted here before the run: **downgrading a positive to a
negative does not change how many cells are usable** — a negative cell is still
supervision. The rubric moved states and values, and moved nothing else.

Step 4's second ruler therefore needs no `--expected-supervision-cells` either.
The two label sets are the same ruler in the sense the guard cares about.

### What `0.814174` does and does not mean

**It is not `-0.020` against B52.** B52's `0.834998` was scored against the old
teacher and this against the regraded one. Two different answer keys.

There is also a reason to expect the regraded ruler to give *lower* numbers for
any model, better or worse: downgrading borderline positives removes the
easiest positives from each target and makes it more imbalanced. A model that
had not changed at all would likely score lower here. That confound is exactly
what step 4 removes, and until step 4 has run, `0.814174` supports **no**
comparison with anything.

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
  --out-json runs/089_Experiment_B55_physical_geometry/ruler_rubric.json
```

No `--expected-supervision-cells` on either: the training run proved the
regraded teacher still yields the frozen `34,010`, so the guard's default is
correct for both rulers.

Each is 548 studies with no training — minutes, not hours.

Every row is scored through **the geometry its own checkpoint records**, read
out of the `b55_geometry` field `train_b55` writes. So a second B55 run at a
different `--crop-mm` or `--reference-side` is scored at *its* numbers, not at
this run's, and the JSON says which under `geometry_used`. A B55 checkpoint
carrying no recorded geometry is refused rather than scored at the defaults —
the defaults happening to be right is exactly what would hide the mistake.

### Ruler 1, the old teacher — run 2026-09-10

```text
    B52_COMPETITION_FULL_FINETUNE  0.834998  geometry=b42
    B53_AUGMENTATION_APPLIED       0.826853  geometry=b42
    B55_PHYSICAL_GEOMETRY          0.819655  geometry=b55

    spread +0.015344
```

**The evaluator verified itself on the way past.** B52 and B53 came back at
`0.834998` and `0.826853` — their own trainers' recorded numbers, to six
decimal places, through a separately written loop with a different grad guard.
That is not a small thing: it means the geometry dispatch, the surface builder
and the scoring path all reproduce the training-time result exactly, so B55's
`0.819655` is a trustworthy number rather than a new tool's first guess.

**B55 is last, by `-0.015343` against B52.**

One asymmetry has to be stated before that is read as final. **B52 and B53 each
chose their best epoch on this exact surface. B55 chose its epoch on the other
one.** Ruler 1 is B52's and B53's home ground, and picking the best of six
plateau epochs is worth something — B55's own plateau spanned `0.015`, so an
epoch chosen on the right surface could plausibly be a few thousandths higher
than one chosen on the wrong one.

That bias runs the other way on ruler 2, which is B55's home ground. **The pair
brackets the answer; neither ruler alone is it.** That is the whole reason for
running both, and it is why no verdict belongs here until ruler 2 has reported.

### Ruler 2, the regraded teacher — run 2026-09-11

```text
    B52_COMPETITION_FULL_FINETUNE  0.815318  geometry=b42
    B55_PHYSICAL_GEOMETRY          0.814174  geometry=b55
    B53_AUGMENTATION_APPLIED       0.803564  geometry=b42

    spread +0.011754
```

**The evaluator verified itself a second time.** B55 came back at `0.814174` —
its own trainer's selected-epoch value, to six decimal places, on the labels it
was selected against. Ruler 1 had already reproduced B52's and B53's. All three
checkpoints now match their training-time numbers on their home surface.

### The pair, read together

```text
                      ruler 1        ruler 2        B55 - B52
                   (old teacher) (regraded)
B52                   0.834998      0.815318
B55                   0.819655      0.814174
B53                   0.826853      0.803564

B55 - B52            -0.015343     -0.001144
B55 - B53            -0.007198     +0.010610
```

**B55 loses on both rulers.** The pre-registered reading of that is written
above: the bundle is not working, and the slot is better spent elsewhere.

The margins differ enormously, and the reason is the selection asymmetry —
which makes the result *stronger*, not weaker:

```text
ruler 1 -> ruler 2, every model drops
  B52   -0.019680
  B53   -0.023289
  B55   -0.005481     <- trained on these labels, so it drops least
```

Every model scores lower on the regraded ruler, exactly as predicted before the
run: downgrading borderline positives removes the easiest positives and leaves
each finding more lopsided. B55 drops least because ruler 2 is its home ground,
and the size of that home advantage is measurable — `+0.014199` relative to
B52.

So ruler 2 is the comparison that favours B55 about as much as any comparison
can: **B55 picked its epoch on these labels and B52 did not.** And B52 still
wins. A model playing away, on an answer key it was never selected against,
beats B55 at home.

That is the sentence this whole exercise was built to be able to say. B55 does
not win anywhere, and it loses by most on the one ruler where the comparison is
fair to B52.

One consolation, and it is small: B55 beats B53 on ruler 2 by `+0.010610`
having lost to it by `-0.007198` on ruler 1. That is two models swapping places
between answer keys, which is a statement about the answer keys.

### What this does not say

**It does not condemn the physical crop.** B55 changed four things, and the
teacher regrade is one of them — the change with a documented precedent of harm
in this archive. Ruler 1 scores a model *trained on regraded labels* against
the old ones, so its `-0.015343` mixes the geometry with a teacher the surface
disagrees with. Nothing here separates them, which was the accepted cost of
bundling.

The field measured the crop and laterality at `+0.030`. This run does not
refute that, because this run never tested it alone. **B56 does** — see below.

### What not to reach for next

The Expert-58 surface is the instrument this archive trusts most for teacher
changes — `THE_CLEANER_TEACHER_MADE_A_WORSE_MODEL` used it to veto a teacher
that looked `+0.0277` better on report labels — and it is the wrong tool here,
for two reasons worth writing down so the idea does not keep resurfacing:

* **It cannot resolve a difference this small.** `B51_EXPERT58_RESOLUTION` is
  `0.03` on 58 studies. The veto it fired before was `-0.0399`, comfortably
  outside that. B55's differences are `0.015` and smaller, which the surface
  would report as inconclusive by construction.
* **It would need the same geometry fix that `common_ruler_eval` just got.**
  `evaluate_b42` builds its datasets with B42's 90% native crop baked in, so
  scoring B55 through it would repeat exactly the bug of scoring a 130 mm/336
  model at 448 — and would report the result as a worse model.

The coverage confound that doc identified, at least, does **not** apply here.
The negated-only teacher cut supervision from 34,010 cells to 25,524, and
coverage predicts report AUC at Pearson `-0.931`. B55's regrade kept all 34,010
— it flipped states without dropping any. Whatever is happening to B55, it is
not that.

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

## B56 — the same geometry on the old teacher

**Status: specified, unrun.** One flag different from B55's training command.

### Why it is worth 21 hours

B55 bundled four changes and cannot attribute. The field measured **one** of
them — a physical-millimetre crop with canonical laterality — at `+0.030`, and
that specific claim has still never been tested here on its own.

B56 tests it, because it changes exactly one thing against B52:

```text
                    B52            B55            B56
teacher             old            regraded       old
crop                90% native     130 mm         130 mm
laterality          as scanned     canonical      canonical
reference           448            336            336
```

B56 against B52 is therefore **the geometry bundle alone**, on the same labels,
the same split, the same seed, the same schedule — and its validation number
lands directly on the scale B52's `0.834998` already sits on. No common ruler
needed; the trainer prints the comparison itself.

### Why this is the right next run rather than B47

B47 is the pre-registered answer to the representational reading and it is
`~38` GPU-hours that have never been spent. B56 is `~21`, reuses code that is
already written and tested, and settles a `+0.030` claim that is currently the
largest unexamined number in this project's field notes. If B56 is flat, the
representational reading is what remains and B47 is the run.

### The command

```bash
mkdir -p runs/090_Experiment_B56_geometry_only

PYTHONPATH=developments/src python -m rsna_knee.b55_physical_geometry_training \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --labels-root runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all \
  --series-policy runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --domain-split runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split \
  --num-workers 6 \
  --epochs 8 \
  --out-root runs/090_Experiment_B56_geometry_only \
  2>&1 | tee runs/090_Experiment_B56_geometry_only/b56_train.log
```

The only change from B55's command is `--labels-root`, pointing back at the old
teacher, and a new `--out-root` so B55's artefacts are not overwritten.

**Its checkpoint will say `B55_PHYSICAL_GEOMETRY`.** That is not a mistake to
fix by editing the payload after the fact — it is the same code — but it is the
same trap B54 fell into, so the run root is what distinguishes them and
`runs/090_...` must not be reused for anything else.

### Declared before the run

```text
noise floor            0.002    what this archive already reads as noise
success                B56 - B52 > +0.010 on the identical validation surface
nothing happened       |B56 - B52| <= 0.002
the crop is harmful    B56 - B52 < -0.010
```

The middle band is the likeliest outcome and the one to prepare for: it would
mean the field's `+0.030` does not reproduce on this pipeline, and that the
`-0.015` B55 showed on ruler 1 was mostly the teacher rather than the geometry.

Do not tune `--crop-mm` or `--reference-side` against B56's score. One change
per run is the entire point of this run.
