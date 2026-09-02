# The teacher is near the ceiling of what report text can give

## Status

**COMPLETE. NO CHANGE MADE. B6 v1.2.1 STAYS FROZEN.**

Point 2 asked how to reduce the teacher's 18.2% error against the 58 experts.
The answer is that most of it is not error.

## Where the errors are

```text
                            cells   wrong     err
every definite call           251      55   21.9%
  positive calls              168      52   31.0%
  negated calls                83       3    3.6%
```

52 of 55 are positive calls, and 55 of the teacher's 57 total errors are the
parser's rather than the filler's. The parser's negatives are 96.4% correct --
the third independent confirmation that the negated-only fill rule was right.

By the rule that fired:

```text
explicit_pathology_mention        76 cells   26 wrong   34.2%  +/-0.054
explicit_structural_abnormality   64 cells   14 wrong   21.9%  +/-0.052
compartment_aware_oa_context      28 cells   12 wrong   42.9%  +/-0.094
```

## What the wrong calls actually are

Nine were read against the report text. **Eight are not parser failures.**

Every Contusion row is the parser reading correctly and the expert disagreeing:

```text
"bone contusion with neglected fracture line at fibular head"
"kemik kontuzyonu mevcut"          Turkish: bone contusion is present
"tiny bone bruise at the posterior aspect of the lateral tibial plateau"
"focal bone marrow edema at the anteromedial tibial plateau"
"edema oseo periferico posterior de platillo tibial lateral"
"discreto edema oseo del aspecto posterior de platillo tibial lateral"
"insufficient fracture ... with adjacent bone marrow oedema"
```

Seven of seven assert bone oedema, bruising or contusion. The parser found each
one. The expert labelled all of them negative.

That is a **definitional disagreement**, not a defect. The reports use "bone
marrow oedema" broadly; the expert appears to reserve Contusion for traumatic
bruising, excluding oedema adjacent to a fracture and excluding anything
qualified as "tiny" or "discreto". Contusion's 47.4% error rate -- the worst of
the twelve -- is a ceiling.

The ninth is the same disagreement in reverse: `"there is no fracture or bone
contusion"`, parser negated, expert positive. `"trace baker's cyst"` is the
milder version: the report says a trace cyst is there, the expert says that does
not count.

**No labeller, no prompt and no stronger model reaches any of these.**

## The one real defect, and its size

```text
Baker's    parser positive    expert negative
           evidence: "baker cyst: none"
```

B6 negates by grammar -- "no", "without", "is not". A structured report lists its
findings, and `finding: none` negates by layout. Nothing in the frozen lexicon
looks at a colon.

Counted across all 4,407 studies, with no labels needed:

```text
positive calls                  7,039
of those, list-negated             83    1.18%
studies affected                   57
among the 58 expert studies         1

  ACL                 592 positive    24    4.1%
  MCL                 279 positive    17    6.1%
  Effusion          1,369 positive    17    1.2%
  Medial Meniscus   1,146 positive    10    0.9%
  Lateral Meniscus    464 positive     7    1.5%
  Baker's             564 positive     4    0.7%
  Contusion           404 positive     2    0.5%
  Fracture            212 positive     2    0.9%
```

It concentrates in the ligaments, and the reason is obvious once seen: ligaments
are the findings reported as a list. `ACL: intact. PCL: normal. MCL: intact.`
Everything else sits under 1.5%.

**It is not fixed.** 83 cells is 1.18% of positive calls and 0.32% of the 26,202
the teacher supervises. Fixing it means unfreezing B6, a new parser version, a
new export, a new merge, and a new teacher whose fingerprint no existing
checkpoint matches. That is not a trade worth making for 83 cells. It is
recorded here so that a B6 v1.3 undertaken for some other reason adds the colon
rule while it is open.

## The finding that matters most

Per target, the teacher's error rate beside the model's AUC, both against the
same 58 experts:

```text
target             label err   model AUC
Fracture                6.7%      0.6139
Lateral Meniscus       11.5%      0.6783
ACL                    12.5%      0.5478
Medial Meniscus        14.3%      0.7188
MCL                    16.7%      0.5011
Baker's                16.7%      0.8152
PF OA                  26.3%      0.6512
Medial OA              28.6%      0.7907
Synovitis              28.6%      0.7778
Effusion               30.6%      0.8981
Lateral OA             33.3%      0.5725
Contusion              47.4%      0.5735

Pearson  r = +0.091
Spearman r = +0.175      n = 12
```

**No relationship**, and what tilt exists runs the wrong way. Effusion has among
the worst labels and the best score. MCL has clean labels and sits exactly at
chance.

Twelve points, both columns from the same 58 studies, per-target AUC carrying
about +/-0.13 -- so this is suggestive, not settled. But if label noise were the
binding constraint, a clear negative slope should be visible, and there is not
even a hint of one.

There is also a mechanism for why it would be weak: the teacher's errors are
largely *systematic* offsets rather than random flips. A consistent definitional
shift moves every affected cell the same way, and AUC is rank-based, so a
consistent offset costs far less than the same number of random errors would.

## What this closes, and what it opens

```text
1  blank studies         FIXED     321 -> 93
2  18.2% error rate      CLOSED    mostly ceiling; the one real bug is 1.18%
3  constant confidence   CLOSED    nothing exists to put in it
```

All three points are answered, and none of them was the bottleneck.

The teacher is close to the limit of what radiology reports can support, because
the reports were written by radiologists who define these findings differently
from whoever produced the expert labels. Further labelling work is spending
effort on the wrong axis.

**ACL 0.5478 and MCL 0.5011, on labels that are 12.5% and 16.7% wrong.** Those
two are not a teaching failure. Lifting them to 0.70 is worth +0.029 macro,
larger than the entire spread between the five architectures submitted so far
(0.694 to 0.716). The geometry predicts the failure: a constant-area 448-pixel
resize and a 6x6 evidence grid give a torn cruciate very few cells to be found
in, while an effusion is large and bright and survives any of it.

That is the next experiment, and it is a model problem, not a label one.
