# Raising macro AUC beyond B13

> **Status — 2026-08-11.** Analysis and tooling. No training config is changed by
> this document; B14 is unaffected.

## Where the ceiling actually is

The B6 gold audit measured the supervision the image model is trained against:

```text
sensitivity          0.975
specificity          0.606     <- ~39% of true negatives labelled positive
positive precision   0.690
balanced accuracy    0.790
coverage             0.361     <- 64% of cells carry no label at all
```

The model's optimisation target *is* these labels, on 3,120 studies. Random,
class-conditional label noise largely preserves AUC ranking, which is why weak
supervision works at all. But B6's errors are driven by report phrasing, which
correlates with site, language and how severity is described — that is
instance-dependent noise, and it biases the ranking rather than averaging out.

So the practical ceiling this supervision supports is roughly **0.75-0.80**.
Architecture changes move the model towards that ceiling; they do not raise it.

For context on the higher targets sometimes quoted: MRNet reported meniscal tear
at about 0.85 using 1,370 *fully expert-labelled* studies. A macro 0.94 across
twelve targets, several of them harder than meniscal tear, is above published
knee-MRI results even under full supervision.

## The measurement problem that blocks everything else

The 58 gold studies give a 95% CI of about +/-0.06. Seventeen sequential
decisions have been taken on that surface. Every remaining candidate change is
smaller than its resolution, so structure cannot be chosen empirically there —
the comparison returns noise, and B8/B9/B10/B11.1/B12 are five instances of
exactly that.

`weak_validation.py` addresses this. Holding part of the B6 corpus out of
training gives a surface of ~3,120 studies, and interval width scales as
`1/sqrt(n)`:

```text
58 studies      CI width ~0.115
3,120 studies   CI width ~0.015     (about 7x tighter)
```

The protocol this enables:

```text
weak holdout  ->  rank many candidate structures   (high power, biased)
58 gold       ->  confirm the single winner        (low power, unbiased)
```

The weak surface measures agreement with the teacher, not truth, so its absolute
number is biased and is never a gold or leaderboard estimate. It is for
*ranking*. Spending the gold surface on search is what exhausted it; spending it
only on confirmation preserves it.

## The structural hypothesis worth testing next

B13's per-target results split cleanly by lesion morphology:

| Works (diffuse) | | Fails (focal) | |
|---|---:|---|---:|
| Effusion | 0.768 | ACL | 0.474 |
| Baker's | 0.748 | Contusion | 0.553 |
| Synovitis | 0.711 | MCL | 0.556 |

An ACL tear occupies a handful of contiguous slices and needs fine detail.
Training samples **16 positions per series at 224x224**. Diffuse fluid
collections survive that; focal structural lesions may not.

Nothing in the repository has ever measured how many slices the series actually
contain, so the sampling fraction is unknown. `slice_audit.py` measures it,
using DICOM headers only — no pixels, no labels, no gold studies:

```bash
rsna-knee-slice-audit \
  --config configs/b13_imagenet_init.yaml \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --out runs/slice_audit
```

It reports slices per series, the fraction seen at 16 positions, the sampling
stride, and the millimetre gap between consecutive sampled positions. The report
ends with an explicit interpretation:

- **stride ~1** — sampling already covers everything; slice count is not the
  constraint, and attention should go to in-plane resolution or supervision.
- **stride >= 2** — a lesion spanning fewer slices than the stride can fall
  entirely between sampled positions, which would explain the focal/diffuse
  split and makes the slice budget the strongest next experiment.

Run this before choosing the next structure. It costs minutes, needs no
training, and either supports or kills the hypothesis outright.

## Priority order

1. **Slice-coverage audit.** Label-free, minutes, decides whether the slice
   budget is limiting. Do this first.
2. **Weak validation surface.** Makes structure comparisons resolvable. Without
   it, every subsequent experiment repeats the B8-B12 pattern.
3. **Supervision quality.** Specificity 0.606 and coverage 0.361 are the ceiling.
   Better negation/uncertainty scoping, or a learned multilingual classifier
   audited against the gold studies, lifts every target at once. This is the only
   item on the list that raises the ceiling rather than approaching it.
4. **Resolution and slice budget**, conditional on item 1.
5. **Schedule and capacity.** Four epochs, batch 2 and encoder LR 1e-5 were all
   chosen for the old SSL encoder and are untested since ImageNet initialisation.
6. **Ensembling.** Multi-seed rank averaging is a reliable but modest gain; do it
   last, when the structure is settled.

## What not to do

The existing interpretation policies still hold. Do not build target-wise
hybrids from the per-target table, do not tune thresholds or ensemble weights on
the 58 studies, and do not treat a weak-surface score as a gold estimate.
