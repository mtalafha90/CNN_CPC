# B59 — is report silence a negative?

## Status

**PROSPECTIVE / IMPLEMENTED / NOT RUN.**

Every threshold in this document was frozen before the audit was run and before
any number it produces was inspected. No figure below is a result.

```text
developments/src/rsna_knee/b59_silence_audit.py
developments/tests/test_b59_silence_audit.py      35 tests
```

## The question

Two thirds of the teacher's grid is answered and one third is blank. The blank
third is not missing data — it is the two states the loss discards:

```text
positive     -> target 0.85, weight 0.50    used
negated      -> target 0.05, weight 1.00    used
uncertain    -> weight 0.00                 discarded
unmentioned  -> weight 0.00                 discarded
```

```text
34,010 of 52,188 cells answered      65.2%
18,178 cells blank                   34.8%
```

Converting `unmentioned` to a negative takes coverage to essentially 100% for no
compute. Nothing else moves the number materially: reading the reports harder has
about one point left in it (B6 v1.3 bought `+832` cells, and the translation
rescue left 93 unreachable studies with "no built mechanism left to try"), and
the confidence column is a constant `0.90`, so lowering the `0.75` threshold does
nothing at all.

## Why the standing ban is worth re-examining

`B6_STRUCTURED_REPORT_LABELS.md` and `RAISING_AUC.md` both say in bold: do not
assume `unmentioned = negative`. That was written on **2026-08-12** and it is
good medicine — absence of evidence is not evidence of absence.

On **2026-09-08** this project established what the hidden labels actually are:
image-derived by two MSK radiologists with a third adjudicating, under a severity
rubric in which **borderline findings are graded NEGATIVE**.

```text
ACL / MCL   high-grade or full-thickness only; low-grade sprain -> NEGATIVE
Meniscus    definite surface contact on >=2 images; degeneration -> NEGATIVE
Effusion    moderate or large only; trace and mild -> NEGATIVE
Baker's     moderate or large only
```

The target is not "is anything wrong with this structure". It is "would two
specialists call this definite and severe". A finding the reporting radiologist
did not think worth one sentence is unlikely to clear that bar.

**The rule that blocks the conversion was written before the project knew what it
was aiming at.** That does not make the conversion right. It makes it unmeasured.

## Why this is not the bet that failed five times

Every coverage increase this project has tried added **positives**:

```text
the LLM fill          added cells 27.1% wrong against B6's own 21.9%
translation rescue    both piles ~97% positive; the 2,047-cell pile was refused
B26 Synovitis fill    the target it was built to fix fell 0.055
```

The teacher's measured fault is entirely one-directional:

```text
111 wrong cells = 106 false positives + 5 false negatives
sensitivity 0.9768     specificity 0.4988
```

It almost never misses a positive. It calls a negative positive half the time.
Silence-to-negative is the only coverage increase that adds almost entirely
**negatives**, so it is aimed at the defect rather than feeding it.

That is an argument for measuring it. It is not a prediction that it works, and
this document does not contain one.

## The ruler: Expert-58, and nothing else

**Project policy from 2026-09-19: every measurement is read on the 58 expert
studies. The 548-study report-derived surface is retired as a decision surface.**

For this experiment the reason is specific and unusually strong. The report
surface scores against the same label process that produced the teacher, so it
cannot referee a change to that process. Worse:

```text
coverage vs report AUC    Pearson -0.931    Spearman -0.951
```

Raising coverage pushes that number **down** whether or not the model improves,
because it adds back the hard, unremarkable cases the teacher was quietly
excluding from its own exam. The negated-only teacher proved it in reverse:
cutting coverage moved the report surface `+0.0277` while the expert surface fell
`-0.0399`. Reading B59 on the report surface would report a real gain as a
failure.

Expert-58 labels all twelve findings on all 58 studies whatever the report
happened to say, so it carries no mention-selection at all. It is small, its
resolution is about `±0.03`, and it has been looked at many times. It is still
the only instrument here that points at the thing being maximised.

`b59_silence_audit` has no report-validation path, and a test asserts that adding
report-only studies changes the size of the intervention and nothing about the
evidence or the verdict.

## What is measured

For each of the twelve findings and each of the four states, on the 58 expert
studies:

```text
n                    cells in that state
gold_positive        how many the experts called positive
p_gold_positive      the proportion
wilson_low/high      95% binomial interval
prevalence           that finding's expert positive rate across all 58
```

The interval is **Wilson**, not the normal approximation. Several findings will
have a handful of silent gold cells and some will have no positives at all, where
`p ± z·sqrt(p(1-p)/n)` gives a zero-width interval around zero and would wave a
conversion through on no evidence whatsoever.

`prevalence` is the comparator. Converting silence to a negative is only
justified where silence is measurably **less** likely to be positive than a study
drawn at random. A state whose rate merely matches the base rate carries no
information about absence, and asserting a negative there manufactures noise.

## The decision rule, frozen

Applied per finding, per convertible state, in order:

```text
1. n < 15                          -> no_evidence
2. wilson_high >= prevalence       -> refuse_not_informative
3. wilson_high > 0.20              -> refuse_too_many_false_negatives
   otherwise                       -> convert
```

**Clause 1** exists because a rarely-silent finding can reach a
confident-looking proportion on a handful of rows. This project has already made
that mistake, on 45 OA cells, and the correction is recorded in
`TWO_THIRDS_OF_THE_TEACHER_HAS_NO_EVIDENCE`.

**Clause 2** is the one that matters most and the one that is easiest to omit. A
finding positive in 5% of knees will be silent-and-negative about 95% of the time
whatever silence means. Without the prevalence comparison that reads as strong
evidence, and converting would add thousands of cells carrying no signal at all.

Note the asymmetry: an interval upper bound is compared against a **point**
estimate of prevalence, because prevalence is measured on these same 58 studies
and giving it an interval too would compare two overlapping samples. This is a
screen, not a proof.

**Clause 3** is an absolute ceiling on manufactured false negatives. Even a
finding where silence beats its base rate is refused above it, because the cells
being added are asserted negatives in a teacher that currently has five false
negatives in total.

`uncertain` is measured beside `unmentioned` and receives its own verdict. It is
a different claim — the report hedged rather than said nothing — and it never
inherits silence's answer.

## What a converted cell is worth

```text
target   0.05     the same as an explicit negation
weight   0.25     one quarter of an explicit negation
```

One value, declared here, never swept. Four silent cells are needed to carry the
influence of one sentence that actually says "no". `RAISING_AUC` anticipated this
shape — "conservative soft targets/low weights rather than unsupported hard
labels" — and this is that, with the ratio written down in advance.

## Governance

* An **empty policy is a result.** If nothing converts, that says report silence
  carries no usable information on this teacher, and it is recorded rather than
  reopened from memory later.
* Do not use any number this writes to choose a checkpoint, an epoch, a blend or
  a submission. It audits a **labeller**, never a model. Auditing a labeller
  against gold is defensible where selecting a checkpoint against gold is not:
  the question is "does silence indicate absence more often than chance", not "is
  model A better than B by 0.002".
* Do not tune `SILENCE_MIN_GOLD_CELLS`, `SILENCE_MAX_POSITIVE_RATE` or
  `SILENCE_NEGATIVE_WEIGHT` from the table this produces. A rule adjusted after
  seeing its own output is not a rule.
* A positive audit authorises **one** training run against the existing B52
  recipe with the teacher otherwise unchanged, judged on Expert-58 as a macro
  veto. It authorises no hidden submission.

## Running it

The audit is read-only and needs no GPU. It is a join over 696 cells.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull origin main

PYTHONPATH=developments/src python -m rsna_knee.b59_silence_audit \
  --train-csv rsna-knee-abnormality-detection/train.csv \
  --structured-csv runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all/structured_labels.csv \
  --out-root runs/095_Experiment_B59_silence_audit
```

It must be `structured_labels.csv`, which carries all 4,407 studies.
`training_targets.csv` has the gold rows removed by design, and the module says
so rather than failing obscurely.

Outputs:

```text
state_truth_by_target.csv    all four states x twelve findings, with intervals
silence_decision.csv         the verdict and its reason, per finding per state
training_cells.csv           how many cells each conversion would actually add
silence_audit.json           the summary, with the frozen rule beside it
silence_policy.json          the machine-readable outcome
```

The printed table is the thing to read:

```text
[B59] unmentioned
  ACL                n=12  p=0.083 [0.015,0.350] base=0.483  no_evidence
  Effusion           n=31  p=0.065 [0.018,0.208] base=0.420  convert
  ...
[B59] coverage 0.652 -> 0.874 (11,584 cells added)
```

Those figures are **illustrative formatting, not results.** Nothing has been run.

## Applying it, if it passes

`apply_policy` writes the converted cells into an existing B7 target/weight pair,
only where the weight is currently zero. A cell the teacher already answers is
never touched — the same guarantee `base_cells_overridden: 0` gives the LLM
merge, and worth as little. It protects reproducibility, not accuracy, and the
merge's own history is the proof: every B6 call survived and the pooled quality
still fell.
