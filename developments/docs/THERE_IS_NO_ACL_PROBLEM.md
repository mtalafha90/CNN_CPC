# There is no ACL problem. There is a universal expert gap.

## Status

**COMPLETE. A PLANNED EXPERIMENT IS CANCELLED BY IT.**

The next experiment was going to be geometry for ACL and MCL, on the grounds
that both sit at chance against expert truth. Measured against a surface large
enough to resolve them, neither does.

## What the two surfaces say

Report AUC is the run's own held-out validation, 548 unseen-scanner studies.
Expert AUC is the 58, with the Hanley-McNeil interval its own class counts
support.

```text
target              report  expert     gap   expert 95% interval     teacher err
Lateral OA          0.8940  0.5725  +0.322   [0.378, 0.767]           33.3%
Contusion           0.8588  0.5735  +0.285   [0.413, 0.734]           47.4%
PF OA               0.8798  0.6512  +0.229   [0.500, 0.802]           26.3%
Synovitis           0.9954  0.7778  +0.218   [0.656, 0.900]           28.6%
ACL                 0.7582  0.5478  +0.210   [0.396, 0.700]           12.5%
MCL                 0.7005  0.5011  +0.199   [0.294, 0.708]           16.7%
Fracture            0.7708  0.6139  +0.157   [0.453, 0.775]            6.7%
Medial OA           0.9315  0.7907  +0.141   [0.644, 0.938]           28.6%
Medial Meniscus     0.8087  0.7188  +0.090   [0.584, 0.854]           14.3%
Lateral Meniscus    0.7644  0.6783  +0.086   [0.534, 0.823]           11.5%
Baker's             0.8605  0.8152  +0.045   [0.660, 0.970]           16.7%
Effusion            0.7973  0.8981  -0.101   [0.818, 0.978]           30.6%

mean gap  +0.157      median +0.178      11 of 12 positive
```

## ACL and MCL are learned, not missed

```text
ACL   0.7582 against its teacher      expert interval [0.396, 0.700]
MCL   0.7005 against its teacher      expert interval [0.294, 0.708]
```

Both intervals span from below chance to 0.70. They were never evidence of
blindness; they were evidence that 58 studies cannot resolve one target. On 548
studies both are plainly learned.

They remain the **weakest two on the report surface**, and that ordering is
trustworthy where the expert ordering was not. But 0.70 and 0.76 is a different
problem from 0.50, and it does not support the geometry story that a torn
cruciate is too small for a 6x6 grid to find. A model that finds it at 0.76
against one labeller is finding it.

## What is actually happening

Every target but one scores worse against experts than against its teacher, by
an average of `+0.157`. That single number is the whole of the gap that has been
puzzling this project:

```text
report labels, 548 studies    0.8350
hidden test,  ~1300 studies   0.7160
                              ------
                              0.1190
```

The model is very good at predicting what a radiologist's report would say. The
competition scores whether an expert agrees. Those are different tasks, and the
distance between them is roughly 0.16 per target.

The gap tracks the teacher's own measured error against the experts:

```text
all twelve targets                       Pearson +0.313   Spearman +0.371
excluding Effusion and Synovitis         Pearson +0.659   Spearman +0.588
```

**Take the first line as the result.** Dropping two of twelve points after
seeing them is how a correlation gets manufactured, and the second line is
recorded only because both exclusions have a reason that does not come from the
correlation itself: Effusion is the only negative gap in the table, and
Synovitis is discussed below. Neither reason is strong enough to make `+0.659`
the headline.

## Synovitis at 0.9954 is not believable

A near-perfect AUC on a subtle synovial finding, against a teacher that answers
only 723 positive cells and is 28.6% wrong on the experts, does not describe a
model reading synovium. Something about the teacher's Synovitis labels is
trivially predictable — correlation with effusion, a scanner or protocol
signature, or a reporting habit tied to one site.

It matters beyond curiosity: Synovitis contributes a twelfth of the macro the
runs select their best epoch on. If that twelfth is inflated by something the
hidden test will not reward, epoch selection is partly being driven by it.

This is the one loose thread worth pulling, and it is cheap: correlate the
teacher's Synovitis column against the other eleven and against acquisition
metadata.

## What this changes

```text
cancelled   geometry work aimed at ACL and MCL as chance-level targets
kept        ACL and MCL as the weakest two, now on a surface that can measure them
opened      Synovitis 0.9954, which may be inflating epoch selection
confirmed   the 0.835 -> 0.716 gap is the teacher/expert definition gap, ~0.16
```

Most of the remaining distance to expert agreement is not reachable by
architecture. Point 2 already showed the teacher sits near the ceiling of what
report text supports, and this shows the model has already learned that teacher
well. Between them they say the report-derived route is close to spent.

## The rule this establishes

**The 58-study surface can veto a macro, and nothing else.** Its per-target rows
carry intervals 0.1 to 0.2 wide and cannot rank, cannot detect a change, and
cannot establish that a target is failing. Development measurement belongs on
the 548-study report surface, where a 0.02 move is real; the experts stay what
they have always been, a veto on a macro gain, spent once per experiment.
