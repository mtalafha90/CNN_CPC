# Dataset contract audit Phase 5 result — report-supervision failure modes

> **Completed 2026-08-17.** This is a manual/structural inspection of the deterministic 79-report local sample created by `report_supervision_gap_audit.py`. It does not change frozen B6 v1.2.1, does not define B35, and is not an independent validation experiment. Raw competition report text remains local-only and is not committed.

## Frozen inspection sample

The uploaded Phase-5 artifact matched the predeclared sampling contract:

```text
audit version                     report_supervision_gap_sample_v1
salt                              CNN_CPC|report-gap-sample|2026-08-17
selected studies                  79
gold non-Latin                    6
Latin gold controls              12
Latin B6-inactive                12
Greek B6-inactive                12
Cyrillic B6-inactive             12
Latin B6-active controls         12
Greek B6-active controls         12
Cyrillic B6-active                1
```

The text sample is deliberately non-exhaustive. Manual language observations below describe this deterministic sample only and must not be interpreted as population language prevalence.

## Main finding: zero B6 cells does not mean a clinically silent report

All 36 sampled report-only studies from the three B6-inactive strata contained clear target-relevant diagnostic statements on direct inspection.

The reports contain both asserted abnormalities and explicit normal/negative findings for competition targets such as cruciate/collateral ligaments, menisci, compartmental cartilage/OA, effusion, Baker cyst, bone-marrow injury and fracture. Their zero usable B6-cell status therefore reflects parser vocabulary/language coverage rather than absence of clinically useful supervision.

Manual composition of the 12 Latin-script inactive sample was:

```text
South-Slavic Latin-script reports    8
Turkish reports                       3
Spanish report                        1
```

The 12 Greek-script inactive reports use native Greek radiology wording. The 12 Cyrillic-script inactive reports use Bulgarian radiology wording. These counts are sample descriptions only.

## Why the current B6 rule set misses the non-Latin reports

`normalize_report()` lowercases text and removes combining diacritics, but it does not transliterate Greek or Cyrillic into Latin characters.

The frozen B6 lexicon and its normality/negation/uncertainty regular expressions contain multilingual Latin-script vocabulary, but no native Greek-script or Cyrillic-script target vocabulary. Therefore native Greek/Cyrillic terms cannot match the current aliases or context rules.

B6 should consequently be described precisely as a **conservative multilingual rule parser with substantial Latin-script vocabulary**, not as an English-only parser and not as a parser with effective native Greek/Cyrillic coverage.

## The apparent Greek/Cyrillic B6 activity is incidental embedded English

The active-control inspection makes the mechanism especially clear.

### Greek active controls

All 12 sampled Greek B6-active reports have exactly one usable B6 cell:

```text
Contusion = positive
```

Every one contains the literal embedded English phrase `bone bruise`. The native Greek statements about other targets remain unparsed.

### Cyrillic active population

There is only one B6-active Cyrillic report in the report-only population. It also has exactly one usable cell:

```text
Fracture = positive
```

Its conclusion includes the English parenthetical phrase `subchondral insufficiency fracture`. Native Cyrillic findings in the same report remain unparsed.

Thus the very small non-Latin B6 coverage observed in Phase 1 is not evidence of functional Greek/Cyrillic parsing; it is largely explained by occasional embedded English diagnostic terminology.

## Latin-script failures are also real

The 12 Latin-script B6-inactive examples show that transliteration alone is not the complete problem.

Observed mechanisms include:

- South-Slavic anatomy, pathology, normality and negation vocabulary not covered by the frozen aliases/regexes;
- Turkish target wording and pathology/normality expressions that are only partially covered by B6;
- a Spanish MCL report using the acronym `LCM`, which is not a frozen MCL alias;
- native terminology for chondromalacia/gonarthrosis, rupture, degeneration, effusion and intact/normal structures that does not satisfy the current target-local rules.

Therefore a Greek/Cyrillic-only lexicon patch would not repair the overall 1,229-study B6-inactive population.

## Gold anchoring is small and reused

The sample includes all six non-Latin gold studies plus 12 Latin gold controls.

```text
Greek gold       3
Cyrillic gold    3
Latin controls  12
```

These reports confirm that the official target semantics can be expressed in the same non-Latin reporting styles, but six non-Latin gold cases are far too few for strong script-specific accuracy claims. In addition, the 58-study gold surface has already been reused throughout development and the Phase-5 design has now inspected these reports directly.

Any gold analysis from this stage onward is therefore a **diagnostic safety check only**, never independent acceptance or promotion evidence.

## Relationship to B23/B24X/B25X

This finding must not be interpreted as permission to revive unrestricted B23 replacement.

Historical B23 already showed that a local multilingual LLM can extract much denser labels, but its predeclared reused-gold specificity gate failed (`0.5678` versus frozen B6 `0.6061`).

B24X and B25X subsequently showed a narrower and more useful mechanism: preserving B6 and filling B6-silent cells can add useful supervision, but the observed downstream gains were exploratory and in B25X were dominated by Synovitis class-balance repair.

Phase 5 therefore motivates a **more constrained rescue mechanism**, not direct replacement of B6.

## Phase-5 decision

```text
B6-inactive reports are clinically silent                    REJECTED
native Greek/Cyrillic support in frozen B6 is adequate       REJECTED
Greek/Cyrillic-only repair is sufficient                     REJECTED
modify B6 v1.2.1 in place                                    NO-GO
revive unrestricted B23 replacement                          NO-GO
define B35                                                   NO-GO
test a separately versioned translation -> frozen-B6 rescue  GO
```

## Next hypothesis

The next experiment is a supervision-only feasibility pilot:

```text
original report
   -> deterministic pinned local translation to English
   -> unchanged frozen B6 v1.2.1 parser
   -> use translated-B6 cells ONLY when the original report has zero usable B6 cells
```

Existing original-B6 cells are never overwritten. The purpose is to test whether language normalization alone can recover excluded supervision while preserving the conservative B6 target/state semantics.

This candidate is not B6 v1.2.2 and not B35. It is a separately versioned translation-rescue supervision experiment.
