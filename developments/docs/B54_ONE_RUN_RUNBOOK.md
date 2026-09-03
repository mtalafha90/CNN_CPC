# B54: one run, four fixes

## Status

**BUILT. NOT RUN.** Every command below is real and tested; none has been
executed against the corpus.

## What is in it

```text
1  spacing conditioning   the model is told how much knee each input holds
2  rebuilt teacher        B52's base, not 091's, plus the rescued cells
3  B6 v1.3                an OA vocabulary, and the list-negation guard
4  B47 native grid        the head stops pooling 196 cells onto 36
```

## The one thing to understand before starting

Four changes in one run means the score cannot be attributed. If it rises you
will not know which change earned it; if it falls you will not know which to
remove.

**One of the four is exempt.** The spacing conditioning is zero-initialised and
switchable, so a single trained checkpoint yields both arms: evaluate once
normally, once with `set_spacing_enabled(model, False)`, and the difference is
the spacing effect exactly. The teacher change and the grid change remain
confounded with each other. That is the accepted cost of one run.

## Why the teacher is rebuilt rather than reused

`runs/092_rescued_negated_fill` is the rescue applied on top of **091**, the
negated-only teacher. 091's teacher is the one already measured: the teacher
swap scored 0.638317 against B52's 0.678247, a loss of 0.0399 — twice the veto
threshold, worse on 9 of 12 targets.

The rescue is good; it took blank studies from 321 to 93. The base underneath
it is not. Steps 2 and 3 below reapply the same rescued cells to B52's teacher
instead.

---

## Step 0 — get the code

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull origin main
PYTHONPATH=developments/src python -m pytest developments/tests -q
```

Expect 1,765 passed, 1 skipped.

## Step 1 — B6 v1.3 report labels

```bash
PYTHONPATH=developments/src python -m rsna_knee.b6_v13_report_labels \
  --train-csv /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection/train.csv \
  --out-root runs/085_B54/b6_v13
```

**Gate.** Read `runs/085_B54/b6_v13/v13_changes.json` before going on:

```text
cells_newly_answered                 expect roughly 220 studies' worth
fallback_cells_now_quoted            expect this to be most of the movement
cells_flipped_by_list_negation_guard expect around 83
cells_silenced                       must be 0
```

If `cells_silenced` is not zero, stop — v1.3 is removing calls, which it is not
allowed to do, and the run should not proceed.

## Step 2 — fill the silent cells, on the v1.3 base

Locate the LLM fill export first; the numbered directory differs per machine:

```bash
find runs -name structured_labels.csv -path '*LLM_FILL*' | head
```

Then:

```bash
PYTHONPATH=developments/src python -m rsna_knee.b23_fill_merge \
  --base runs/085_B54/b6_v13 \
  --filler <the LLM fill export from the find above> \
  --fill-states both \
  --out-root runs/085_B54/teacher_step2
```

`--fill-states both` is B52's rule and is deliberate. `negated` is the rule
that produced the −0.0399 teacher; do not use it.

## Step 3 — add the rescued cells

```bash
find runs -path '*translation_rescue*' -name structured_labels.csv | head
```

```bash
PYTHONPATH=developments/src python -m rsna_knee.b23_fill_merge \
  --base runs/085_B54/teacher_step2 \
  --filler <the Phase-7 rescue export from the find above> \
  --only-silent-studies \
  --out-root runs/085_B54/teacher_final
```

`--only-silent-studies` is the frozen Phase-8 policy: filling a study the base
says nothing whatever about. Filling one the base has already reached is a new
and unmeasured policy.

**Gate.** The final teacher should have *more* answered cells than B52's 34,010
and *fewer* blank studies than its 57. If either moved the wrong way, stop.

```bash
PYTHONPATH=developments/src python -m rsna_knee.teacher_coverage_audit \
  --teacher runs/085_B54/teacher_final \
  --b6-export runs/085_B54/b6_v13
```

This also reports provenance. Expect the quoted share to rise above 33.8%,
because v1.3 converts evidence-free OA calls into quoted ones.

## Step 4 — the geometry table

Already produced. If it is missing:

```bash
PYTHONPATH=developments/src python -m rsna_knee.slice_geometry_scan \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --workers 8 \
  --out-csv runs/slice_geometry_scan/series_geometry.csv \
  --out-json runs/slice_geometry_scan/summary.json
```

## Step 5 — preflight

`b54_spacing_run.preflight` refuses two ways this run could be silently
pointless: the spacing failing to resolve for most series, and the conditioning
never being installed. Call it from the training entry point before the first
epoch, with the assembled series records and the constructed model, and abort
on `passed: False`.

Expect `resolved_fraction` at 1.000 — the scan measured a usable spacing for
all 24,371 series — and `conditioning_sites` at 1 or 2 depending on whether the
head alone or the base as well is conditioned.

## Step 6 — train

Wire, in this order:

1. `attach_spacing(records, series_geometry_csv=..., data_root=...)`
2. `B54SpacingDataset` and `collate_b54` in place of the variable-series pair
3. `install_spacing_conditioning(...)` on each module that sums metadata
4. `spacing_metadata(module, series_meta, series_spacing)` where
   `plane + fluid + fat` is computed today
5. `training_resume.resume(...)` before the loop, `save_checkpoint(...)` after
   every epoch

The resume is not optional here. Runs are already nineteen hours and this one
is longer; a crash without it costs the whole attempt.

## Step 7 — evaluate twice, from one checkpoint

```python
set_spacing_enabled(model, True)    # the arm
set_spacing_enabled(model, False)   # its own control
```

Run the Expert-58 audit on both. The difference is the spacing effect, free of
any second training run.

## Step 8 — read it against the right threshold

`VETO_DELTA = -0.020` on the Expert-58 macro. B52 sits at **0.678247**.

And the standing caution, which has been earned three times this project: at 58
studies the surface cannot resolve small per-target differences. Read the macro,
not the twelve.
