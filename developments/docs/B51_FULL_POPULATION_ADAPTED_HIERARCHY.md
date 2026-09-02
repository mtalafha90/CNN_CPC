# B51 — the adapted hierarchy at full scale, for submission

**Date:** 2026-08-29
**Status:** COMPLETED / KAGGLE `0.713` / NOT PROMOTED.

The protocol below was frozen before any B51 training. The completed hidden
result is recorded immediately after it.

## Completed hidden result

**COMPLETED / KAGGLE `0.713` / NOT PROMOTED.**

```text
B42 (frozen hierarchy)      0.714
B51 (adapted hierarchy)     0.713      -0.001
```

B50's controlled comparison measured `+0.011221` on 548 unseen-scanner
studies, all twelve targets improved. On roughly 1,300 hidden studies the
same change measures `-0.001`. The runbook predeclared this outcome as one of
the two it expected -- "may display `0.714` again and be indistinguishable" --
and predeclared that it authorises nothing. It does not.

**Do not read `-0.001` as a decrease.** Kaggle displays three decimals, and one
unit in the last place on this many studies is inside the noise of the
measurement. The honest statement is that the adapted hierarchy is
indistinguishable from the frozen one on hidden data, not that it is worse.

### One confound, narrowed but not closed

This run was the first through the hidden-safe execution contract, which
**drops a series it cannot decode** instead of ending the run. B42's `0.714`
was measured under the strict reader, where an undecodable series was
impossible by construction: the run crashed instead.

All local and visible data is uncompressed (`1.2.840.10008.1.2.1`), and the
competition contract says the hidden set may contain JPEG Lossless and JPEG
2000. A decoder preflight in the B52 notebook raised `no decoder for JPEG
Lossless P14`, which looked like a missing codec. It was not:

```text
gdcm OK 3.0.24
1.2.840.10008.1.2.4.57  available=True
1.2.840.10008.1.2.4.70  available=True
1.2.840.10008.1.2.4.90  available=True
1.2.840.10008.1.2.4.91  available=True
```

The four `missing_dependencies` entries name *pylibjpeg*, an optional second
plugin. GDCM alone covers every syntax the contract mentions.

The earlier failure was an **import-cache artefact**: `pip install` into a
running kernel left a stale path cache, so pydicom's decoder module tried
`import gdcm`, failed, and cached the plugin as unavailable. pydicom resolves
plugin availability when `pydicom.pixels.decoders` is first imported, not per
call, so that verdict then stands for the life of the kernel.

**What is still unknown.** The B51 submission notebook installed the wheel and
then imported pydicom in a later cell without calling
`importlib.invalidate_caches()`. Whether that kernel resolved GDCM depends on
whether its path cache was stale at that moment, and the hidden run's log is
not visible. The visible three studies prove nothing either way -- they are
uncompressed and need no decoder.

So `0.713` is either clean or a floor, and which one cannot be recovered for
this submission. Every future submission should call
`importlib.invalidate_caches()` before the first pydicom import and assert
`is_available` on all four syntaxes, which now passes and therefore means
something when it fails.

### What this does and does not settle

* The hidden-safe execution path works. Three previous attempts on this family
  -- B39, B41 and B51's own first try -- ended in `Notebook Threw Exception`.
* B50's `+0.011` on 548 report-labelled unseen-scanner studies did not
  reproduce on ~1,300 hidden studies with expert labels. Either the effect is
  smaller than the domain-shift split suggested, or the two surfaces measure
  different things. This project has now seen `0.694`, `0.707`, `0.713` and
  `0.714` from four architectures, which is a plateau, not a ranking.
* Nothing here authorises tuning the hierarchy learning rate, the epoch count,
  the seed or the geometry against this number.

## What B51 is

**B42, with one change: the study hierarchy trains.**

That change is the one B50 validated. B42 is the endpoint behind your `0.714`
hidden score, so B51 differs from a submitted model by exactly one thing, which
is the cleanest submission this project has ever been able to make.

```text
                    B42 (0.714 hidden)      B51
population          4,349 report-only       4,349 report-only   unchanged
geometry            constant-area 448       constant-area 448   unchanged
supervision         B6-preserved LLM fill   same                unchanged
epochs              exactly 2, no selection same                unchanged
seed                2026                    same                unchanged
study hierarchy     FROZEN                  trains at 0.05x     <- the change
```

## Why there is no control arm

B50 already ran the controlled comparison, and it was supported: `+0.011219` on
548 unseen-scanner studies, all 12 targets improved, with a discordant ceiling
of `0.030651` giving the measurement 2.7x the headroom it needed.

B51 is not an experiment. It is a production run of a mechanism that has already
been tested, and its control is B42's existing hidden score. Re-running the
frozen arm at full scale would cost another ~8.5 hours to reproduce a number the
leaderboard already holds.

## What is being extrapolated, and the risk in it

B50 trained on **1,447** studies because its fresh gate excluded every row B48
and B49 validation had spent. B51 trains on **4,349**, three times more.

That cuts both ways and neither direction is guaranteed:

- more data usually **helps** an unfrozen 18.95M-parameter hierarchy, since
  overfitting was the main risk and it is the risk more data reduces;
- but the effect size measured at 1,447 need not hold at 4,349, and a mechanism
  that helps a data-starved model can matter less to a better-fed one.

**This extrapolation is the main assumption in B51 and it is not tested by
anything.** It is recorded here rather than discovered afterwards.

## Preconditions

B51 must not start until both hold:

1. **B50's expert-58 audit has been read.** B50's endpoint is report-derived,
   and this project has twice seen a large weak-surface gain fail to reach
   expert truth — B15 gained `+0.167` on teacher agreement and lost `0.008` on
   expert AUC; B25X gained `+0.058` weak and `+0.002` on eleven targets. The 58
   gold studies are the closest proxy available for the hidden test's expert
   reference standard.

   ```text
   expert delta >= +0.010    proceed; the gain reaches expert truth
   -0.010 to +0.010          inconclusive at 58 studies, which resolve to about
                             +/-0.03. Proceed, and record that B51 rests on the
                             report-derived result alone
   expert delta <= -0.020    STOP. This is B15's failure again: the hierarchy
                             learned to predict reports, not to read knees
   ```

   The middle band is the likely one and it is not a failure. It is the honest
   limit of a 58-study surface.

2. **The competition's submission budget allows it.** B51 consumes one hidden
   submission, and hidden submissions are the only trustworthy ruler this
   project has.

## Cost

Scaled from B50's recorded timing, which is the first wall clock this line ever
kept:

```text
B50, 1,447 studies     ~85 min per epoch      ~2.8 h for two epochs
B51, 4,349 studies     ~255 min per epoch     ~8.5 h for two epochs
```

That is an overnight run. It will not fit inside one nine-hour session with any
margin, so the runner must be resumable and the checkpoint written only at the
end of epoch 2.

## Inference, and why it carries no new risk

**A B51 checkpoint loads into a plain B42 model.** Verified by construction:
B50's model subclasses B42's and changes only `requires_grad`, adding and
removing no parameters, so the `base` and `head` state dictionaries are
key-for-key and shape-for-shape identical.

`requires_grad` has no effect on a forward pass, so at inference a B51
checkpoint *is* a B42 checkpoint.

That means the existing hidden-safe dual-T4 streaming path
(`b42_constant_area_aspect_sparse_submission_dualgpu_fast`) runs it unchanged,
with no new inference code. This matters more than it might appear: B41's first
hidden submission failed operationally, not scientifically, and every line of
new inference code written for a submission is a chance to repeat that. B51
writes none.

The only work needed is a small converter that rewrites the B51 checkpoint's
metadata into the B42 format the submission loader expects. It must not touch
the weights, and a test should assert the tensors are bit-identical before and
after.

## Endpoint and governance

```text
epochs                exactly 2
checkpoint selection  none
seed                  2026, frozen
hierarchy lr scale    0.05, frozen, inherited from B50
gold studies          zero, in gradients and everywhere else
```

Post-training, record the expert-58 macro AUC and the focal six as an audit, and
the final gate values for comparison against B50's. **Nothing may be selected
from them.** If B51's hidden score disappoints, that is a result about the
mechanism at full scale, not an invitation to tune the learning rate and try
again — that is precisely how this project spent B43 through B45.

## What would make this protocol wrong

- If B50's `+0.011` came from the smaller training population rather than from
  the adapted hierarchy, B51 will find nothing. The gate evidence argues
  against that reading, but it does not exclude it.
- If the effect is real but smaller than Kaggle's third-decimal rounding, B51
  will display `0.714` again and be indistinguishable from B42. `+0.011` on the
  weak surface is comfortably above that, but transfer is not guaranteed.
- If the hierarchy overfits at full scale in a way it did not at 1,447 studies,
  the hidden score falls. The fixed two-epoch endpoint is the only guard, and it
  is the same guard B42 had.

## Implementation

Reuses B50's model and training modules with the population source changed from
the fresh gate's `train` split to the full report-only surface. Still to write:

```text
config/b51_full_population_adapted_hierarchy.yaml
developments/src/rsna_knee/b51_full_population_training.py
developments/src/rsna_knee/b51_checkpoint_to_b42_format.py
developments/scripts/run_b51.sh
```

Nothing under `b37_*`, `b42_*`, `b46_*`, `b48_*`, `b49_*` or `b50_*` may be
modified: all are completed frozen experiments.
