# The rebuilt teacher, and why the translation rescue is finished

## Status

**COMPLETE. MEASURED ON THE CORPUS. THE TEACHER IS BUILT.**

`runs/085_B54/teacher_step2` — B6 v1.3.1, then the B23 LLM fill on its silent
cells, `--fill-states both`. It is the final B54 teacher: step 3b was measured
and dropped.

## Against B52's teacher

```text
                             B52 teacher   B54 teacher    change
cells answered                    34,010        34,842      +832
coverage                           65.2%         66.8%
studies with no answer                57            48        -9

parser, clause recorded           11,491        15,004    +3,513
parser, no clause                  2,632           896    -1,736
filled, no clause exists          19,887        18,942      -945
------------------------------------------------------------
no clause at all                  22,519        19,838    -2,681
                                   66.2%         56.9%
```

Every gate passed. The one to look at is the third row: **the evidence-free
osteoarthritis calls fell by two thirds**, from 2,632 to 896, which is the
whole reason B6 v1.3 exists.

The quoted share rose from 33.8% to 43.1%, and it rose two ways at once. 945
cells moved from an LLM guess with no clause to a parser call with a
quotation — same cells, better provenance. 832 cells are new supervision that
neither the parser nor the filler had before.

## Where the remaining unquoted calls sit

```text
              unquoted before   after   reduction
Medial OA                 816     140       83%
Lateral OA                783     129       84%
PF OA                   1,033     627       39%
------------------------------------------------
total                   2,632     896       66%
```

The two compartment targets are nearly cleaned out. **PF OA is not**, and it is
now three quarters of what remains. The compartment patterns match the phrase
the report uses; the patellofemoral ones match an anatomy word that has to
survive a soft-tissue guard, so more of them fall through to the fallback. If
a v1.4 is ever wanted, that is where it is.

## The translation rescue is spent, and this retires it

Measured against this teacher rather than assumed:

```text
cells the rescue offers                   3,901
already answered here                     3,697

under the frozen Phase-8 policy
  (studies still wholly silent)               1 cell, 1 study
under a new, unmeasured policy
  (studies the filler already reached)      203 cells, 181 studies
```

**One cell.** Blank studies would go 48 to 47.

The other 203 cells need the policy that was refused before: filling studies
the filler has already reached. That is not the frozen Phase-8 rule and has
never been measured.

So step 3b is dropped. Not skipped for convenience — building a long-format
converter to add one cell would risk getting the policy subtly wrong for no
measurable gain. B6 v1.3.1 and the LLM fill between them have absorbed 3,697
of the rescue's 3,901 cells.

This also closes the standing note to "train `runs/092_rescued_negated_fill`
as a passenger". There is nothing left in it to carry.

## What could not be done, and why that is right

`translation_rescue_supervision_merge` refuses this base outright:

```python
REQUIRED_B6_VERSION = "1.2.1"
EXPECTED_ORIGINAL_USABLE = 14123
```

Phase 8 froze a specific artifact and checks it by version and by exact cell
count. A v1.3.1 base with 15,900 parser cells is not that artifact. The refusal
is the governance working, and the measurement above shows it cost nothing.

## What this does not claim

That the model will score better. The teacher is wider (+832 cells), better
evidenced (+3,513 quoted), and has fewer blank studies (−9). Whether any of
that reaches the Expert-58 macro is unknown, and the one previous teacher
change measured on that surface **lost** 0.0399. Coverage predicting model
performance (r = −0.93) is the reason to expect better; it is not a result.
