# B54: one run, four fixes

## Status

**COMPONENTS BUILT AND TESTED. ONE WIRING STEP OUTSTANDING. NOT RUN.**

Steps 0 to 5 and 7 to 8 are runnable commands today. **Step 6 is not yet
code.** The spacing has to enter the forward pass, and that pass lives in
`b37_highres_sparse_mil._base_logits_from_global`, which every experiment from
B37 to B52 shares. Threading `series_spacing` through it means a subclass that
overrides `_base_logits_from_global` and `forward` — about forty lines copied
from frozen logic, which deserves its own pass with a model-construction test
rather than being appended to a long session. Everything it needs
(`spacing_metadata`, `install_spacing_conditioning`, `with_spacing`,
`training_resume`) exists and is tested; only that subclass is missing.

## What is in it

```text
1  spacing conditioning   the model is told how much knee each input holds
2  rebuilt teacher        B52's fill rule on a v1.3.1 base, not 091's
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

Step 2 below rebuilds it with B52's own fill rule on a v1.3.1 parser base. The
rescued cells were then measured against that teacher and turned out to be
spent — see step 3.

---

## Step 0 — get the code

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull origin main
PYTHONPATH=developments/src python -m pytest developments/tests -q
```

Expect 1,793 passed, 1 skipped.

## Step 1 — B6 v1.3 report labels

```bash
PYTHONPATH=developments/src python -m rsna_knee.b6_v13_report_labels \
  --train-csv /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection/train.csv \
  --out-root runs/085_B54/b6_v13
```

**Gate.** Read `runs/085_B54/b6_v13/v13_changes.json` before going on:

```text
cells_newly_answered                      1,814 at v1.3.0
fallback_cells_now_quoted                 2,093 at v1.3.0
cells_flipped_by_list_negation_guard         26 at v1.3.0
cells_silenced                            must be 0
cells_weakened_by_new_vocabulary          must be 0
cells_weakened_after_list_negation_guard  expect a handful
```

The first three should barely move at v1.3.1: the fix only suppresses answers
the vocabulary could not commit to, so placements and re-quotings stay.

`cells_silenced` above zero means v1.3 is removing calls, which it is not
allowed to do.

`cells_weakened_by_new_vocabulary` above zero is a hard stop. These are broad
anatomy words, validated for *placing* a cell and never for arbitrating one,
so they answer only with a committed state. A bare anatomy word must not turn
a confident call into `uncertain`, which drops its confidence from 0.90 to
0.25 on exactly the three targets with the worst coverage already.

**This gate has already fired once, on the real corpus, at 360 cells** — 338
of them PF OA. v1.3.0 protected calls the *aliases* made and not calls the
*fallback* made, and the OA targets live almost entirely on the fallback.
v1.3.1 fixes it. The counter stays so the fix is checked rather than trusted.

`cells_weakened_after_list_negation_guard` is a different thing and a small
number is expected. When the guard correctly reads "ACL: intact" as a negation
and the report also describes a tear, the contradiction is real; v1.2.1 only
looked confident there because it misread the list entry. It appears only on
targets that also show guard flips.

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

## Step 3 — add the rescued cells: MEASURED AND DROPPED

Do not run this. It was measured against the step-2 teacher and offers **one
cell in one study**; blank studies would go 48 to 47. Its other 203 cells need
the policy Phase 8 refused — filling studies the filler already reached.

`translation_rescue_supervision_merge` also refuses a v1.3.1 base by design
(`REQUIRED_B6_VERSION = "1.2.1"`, `EXPECTED_ORIGINAL_USABLE = 14123`), and the
measurement shows that refusal cost nothing.

**`runs/085_B54/teacher_step2` is the final teacher.** Promote it so the paths
below hold:

```bash
cp -r runs/085_B54/teacher_step2 runs/085_B54/teacher_final
```

See `THE_B54_TEACHER_AND_A_SPENT_RESCUE.md` for the full numbers.

### What step 2 produced, against B52

```text
                             B52 teacher   B54 teacher    change
cells answered                    34,010        34,842      +832
studies with no answer                57            48        -9
parser, clause recorded           11,491        15,004    +3,513
parser, no clause                  2,632           896    -1,736
no clause at all                   66.2%         56.9%
```

Every gate passed. The evidence-free osteoarthritis calls fell by two thirds,
which is what B6 v1.3 was built for.

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
2. `with_spacing(B42ConstantAreaAspectDataset)` in place of the dataset class
   `b52_competition_training._build_dataset` constructs. **`collate_b42` needs
   no change**: it is `list(items)`, so the spacing travels inside each item
   and nothing has to pad it. `collate_b54` is only for the
   `collate_variable_series` path.
3. **load the pretrained base checkpoint first**
4. *then* `install_spacing_conditioning(...)` on each module that sums metadata
5. `spacing_metadata(module, series_meta, series_spacing)` where
   `plane + fluid + fat` is computed today
6. `training_resume.resume(...)` before the loop, `save_checkpoint(...)` after
   every epoch

**Steps 3 and 4 are in that order for a reason.** Installing the conditioning
adds `spacing_conditioning.projection.weight` to the state dict. A pretrained
checkpoint from before B54 does not have that key, so loading it strictly into
an already-installed model raises. Load first, install second, and the load is
clean and the new module starts at zero. Two tests in
`test_b54_spacing_run.py` pin both halves of this.

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
