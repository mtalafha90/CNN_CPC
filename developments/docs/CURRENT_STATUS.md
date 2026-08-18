# Current project status

**Snapshot:** 2026-08-18
**Package:** `0.30.0`
**Primary metric:** macro ROC AUC across 12 targets
**Independent evidence to date:** none — no competition submission has been made

## Where things stand

**Nothing has been promoted.** Every result below comes from surfaces built in
this repository. The frozen governance position remains that B20 is the last
model promoted on evidence, and no later experiment has cleared a promotion
path.

**The top-level interface targets the B34/B31 architecture.** That is an
interface decision recorded in `docs/WORKING_MODEL.md`, made because B34 is the
strongest candidate on both internal surfaces and has the simplest inference
path. It is not a promotion, and it does not change any frozen experiment
record. The interface and the governance record therefore disagree on purpose,
and the disagreement resolves when a hidden-evaluation result exists.

## The two findings that matter

**1. The reports are multilingual, and the parser could not read a quarter of
them.** Phase 5 inspected the zero-cell population and found every sampled
report contained clear target-relevant statements. Their zero-label status was
parser coverage, not clinical silence. The decisive detail: all 12 sampled
Greek B6-active reports had exactly one usable cell, `Contusion = positive`,
each from the incidental English phrase `bone bruise`. Apparent non-Latin
coverage was embedded English, not parsing.

Translating before running the unchanged parser (Phases 6-8) produced:

```text
rescued studies      1053 / 1229 = 85.68%
coverage             71.74% -> 95.95%
usable cells         14123 -> 18024   (+3901, +27.62%)
by script            Cyrillic 99.54%   Latin 83.22%   Greek 81.43%
```

**2. A powered validation surface now exists.** The prospective weak splits
(PV1/PV2) give 499-624 validation studies, against 58 for the expert surface.
PV2's primary test returned a 95% interval entirely below zero
(`[-0.01257, -0.00399]`, P = 0.9998) — something the expert surface has never
produced in 34 experiments.

## Architecture ladder: essentially flat

Reused 58-study expert surface, paired against B20 `0.6674066371`:

| Experiment | Macro AUC | Outcome |
|---|---:|---|
| B26.2 supervision repair | 0.6663 | closed, not promoted |
| B27.1 pathology routing | 0.6599 | closed, not promoted |
| B28 max-evidence residual | 0.6383 | closed, not promoted |
| B29 complementary pool | 0.6769 | frozen candidate, not promoted |
| B30 projected complementary | 0.6547 | formulation closed |
| B31 local context | **0.6823** | highest on this surface |
| B32 dispersion summary | ~tied | formulation closed |
| B33 uniform mean | 0.6764 | simplification of B29 |
| B34 train-only scaffold | — | B31-equivalent, simpler inference |

Eight experiments, roughly +0.015 of point estimate, every interval crossing
zero. The architecture direction is exhausted in its current form.

## Phase 9 v2 — the matched supervision test

Both arms trained on 3,850 studies with the 499-study PV2 partition held out,
and scored on original frozen labels only, so the evaluation is not circular.

```text
BCE        -0.00988   CI [-0.01990, +0.00008]   P = 0.9742
macro AUC  +0.00322   CI [-0.00847, +0.01508]   P = 0.6897
```

Both aggregates favour the merged supervision; both intervals include zero. The
BCE upper bound sits at `+0.000084`, about as close to excluding zero as is
possible without doing so.

Two per-target results reached significance, but **only one survives correction
for testing 12 targets**:

```text
Contusion  +0.0554  CI [+0.0206, +0.0933]  P=0.9990  two-sided p ~ 0.0020
           survives Bonferroni (0.05/12 = 0.00417) and Benjamini-Hochberg

Effusion   -0.0262  CI [-0.0483, -0.0052]  P=0.0082  two-sided p ~ 0.0164
           survives neither
```

Removing Contusion flips the macro sign (`+0.0032 -> -0.0015`). The entire
aggregate rests on one target.

**This is the third time an aggregate has dissolved into a single target under
audit** — B25X was 96.4% Synovitis, B24X-Density showed the gain was coverage
rather than correction, and Phase 9 is Contusion. Leave-one-target-out belongs
in the standing protocol, not in a post-hoc check.

## Open hypothesis on the Contusion result

Contusion is the only target where the non-Latin population had any
pre-existing label coverage — the incidental `bone bruise` positives from
Phase 5. It is also the only target driving the Phase 9 macro result. That
coincidence is untested.

If the control arm learned a site shortcut from a single-target, positive-only
signal in a distinct scanner population, the +0.055 would be partly the
*removal of a control artefact* rather than new signal in the candidate.

The discriminating check is cheap and non-circular: stratify the stored PV2
per-target results by report script (Latin / Greek / Cyrillic). No retraining,
no change to the rescue set. Concentration in the non-Latin strata supports the
shortcut reading; an even spread across Latin studies does not.

## Blocked

The Phase-7 rescue evidence — `translation_cache.jsonl`,
`full_population_rescue_audit.csv`, `recovered_cells.csv` — was not found at
the attempted local path, so the label-generation mechanism audit cannot run.
`phase9_v2_rescue_mechanism_audit.py` exists and is ready; unlike the other
modules in this campaign it has no test file.

The same path ambiguity affects training: the merged-label export is recorded
under two different roots across the archive. Verify with the pinned
fingerprints before launching anything:

```text
training_targets.csv   c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a
recovered_cells.csv    ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0
```

## Next

1. **Submit.** Zero submissions after 34 experiments and 9 audit phases. A
   Kaggle-run notebook costs none of the 9-hour session budget, and the
   leaderboard is the only measurement not built here. Submitting a *matched
   pair* — original versus merged supervision — resolves what Phase 9 could
   not.
2. **Script stratification** of the stored PV2 predictions (CPU, minutes).
3. **Re-score PV1/PV2 on macro AUC with paired bootstrap.** Those surfaces
   currently rank on soft BCE only; macro AUC is recorded without an interval
   and is never used to rank. PV1 itself notes B31 and B33 had near-identical
   macro AUC while the primary metric separated them at P = 0.0050 — so the
   ladder that selected B31 and B34 has never been checked on the metric the
   competition scores.
4. **Locate the Phase-7 artifacts** and run the mechanism audit.

Detailed records for every experiment named here are in this directory and are
frozen; they are not revised when the project's understanding moves on.
