# Two thirds of the teacher has nothing recorded behind it

## Status

**COMPLETE. MEASURED ACROSS THE CORPUS. NO CHANGE MADE.**

## The count

B52's teacher, 4,349 report-only studies, gold excluded:

```text
cells it answers                34,010    65.2% of the grid

  parser, clause recorded       11,491    33.8%
  parser, no clause              2,632     7.7%
  filled, no clause exists      19,887    58.5%
  ------------------------------------------------
  no clause at all              22,519    66.2%
```

The parser's 11,491 + 2,632 = 14,123 matches B6's own cell count exactly, which
confirms the model of the merge this rests on: every parser call is preserved,
and the filler writes only where the parser was silent.

A filled cell having no clause is not a defect. The filler was asked precisely
because the parser found nothing, so there is nothing to quote. What it does
mean is that **58.5% of the teacher rests entirely on the LLM's reading, with
nothing recorded to check it against.**

## The unexpected part: the OA lexicon never fires

Every one of the 2,632 unquoted parser calls came through one rule,
`compartment_aware_oa_context`, and they fall almost entirely on three targets:

```text
                        quoted   unquoted   % unquoted
the three OA targets        24      2,632        99.1%
the other nine          11,467          0         0.0%
```

```text
PF OA          21 quoted    1,033 unquoted
Medial OA       2 quoted      816 unquoted
Lateral OA      1 quoted      783 unquoted
```

`compartment_aware_oa_context` is the legacy fallback: it runs **only when no
alias matched at all**. So B6's osteoarthritis vocabulary essentially never
matches. Twenty-four quoted calls across three targets and 4,349 studies is not
a lexicon that works; it is one that is bypassed.

Every OA call the frozen parser makes is produced by a fallback that records a
rule name where a quotation would go. And that fallback is already the worst of
the three rules measured against the experts:

```text
explicit_pathology_mention        76 cells   26 wrong   34.2%
explicit_structural_abnormality   64 cells   14 wrong   21.9%
compartment_aware_oa_context      28 cells   12 wrong   42.9%
```

## What this corrects

All session the argument has leaned on parser cells being accountable and filled
cells not. That is right for nine targets and wrong for three. For OA the parser
is not quoting anything either — it is guessing from context through a path with
the worst measured error rate, and 18.6% of all parser cells are that path.

The distinction survives, but narrower than it was being used: **quoted cells are
accountable, and quoted cells are 33.8% of the teacher, not 41.5%.**

## What the negated-only rule also cost

The same audit on `runs/091`:

```text
                        B52 teacher    negated-only
cells answered               34,010          25,524
no clause at all              66.2%           55.0%
studies with no answer           57             321
```

The cleaner-looking 55% is not an improvement in accountability; it is the same
11,491 quoted cells over a smaller total. And the last row is a cost not counted
before: **264 studies went from having some supervision to having none.**

## What is not being proposed

The obvious thought is that OA cells would be better answered by the filler than
by an evidence-free fallback at 42.9%. It is not proposed, for two reasons.

Replacing parser calls with filler calls is B23, measured and refused: it lost
specificity, and B24X put the value of replacement at nothing at all, 95% CI
`[-0.0100, +0.0035]`. Scoping that to OA alone would need evidence, and the
evidence available is 45 OA cells across the 58 experts, which cannot support a
per-target decision — the lesson already learned from the ACL claim this week.

The finding is structural and does not need the experts to state it: **B6 has no
working OA vocabulary.** That is a fact about the parser, observable without a
single label, and it belongs in whatever justifies a B6 v1.3 — alongside the
list-negation gap, which is the other defect found and not fixed.
