# The unread frames are cheap. The triplet's depth is not.

## Status

**COMPLETE. MEASURED ACROSS THE WHOLE CORPUS. NO CHANGE MADE.**

`slice_geometry_scan`, 24,371 series, three DICOM headers each, no pixels
decoded, no labels touched.

## What was suspected, and what is true

The inventory reported that **134,361 frames (16.4%)** are never read, and left
open which frames those were. Two things turned out to be wrong with that
number, and they pull in opposite directions.

### The arithmetic was wrong

The inventory counted the frames read as `min(frames, centres)` — one frame per
centre. The pipeline is 2.5D: every centre pulls `[-gap, centre, +gap]`, and on
a long series those triplets do not overlap. Thirty-two centres on a 320-frame
series touch 96 frames, not 32.

```text
frames present                     819,078
read, one frame per centre         684,717    lost 134,361   16.4%
read, 2.5D triplets                767,064    lost  52,014    6.4%
```

**The real loss is 6.4%, not 16.4%.** The earlier figure overstated it by
82,347 frames.

### And the loss that remains is in the cheapest possible place

Every single unread frame sits in a fine-spaced 3D volume:

```text
acquisition   series     frames   never read   share of the loss
2D            22,329    661,670            0        0.0%
3D               836    129,368       52,014      100.0%
unknown        1,206     28,040            0        0.0%

spacing       series     frames   never read   share of the loss
<1.5 mm        3,324    204,833       52,014      100.0%
1.5-3 mm       4,687    141,638            0        0.0%
3-4.5 mm      13,221    399,679            0        0.0%
>=4.5 mm       3,139     72,928            0        0.0%
```

762 series — 3.1% of the corpus — carry the entire loss, and they are the ones
where losing a frame costs least: at 0.6 mm spacing a slice and its neighbour
are nearly the same picture. 40.2% of the frames in 3D series are never read,
and that is close to free.

**The missing-data worry is closed. There is no hidden pile of unseen anatomy.**

## The assumption that was checked rather than assumed

Long series were assumed to be the thin ones. They are.

```text
length      series    median spacing
<=32        15,188          3.50 mm
33-48        8,121          3.30 mm
49-80          300          1.91 mm
81-160         449          0.60 mm
>160           313          0.60 mm

Spearman, length vs spacing   -0.410
```

## The finding that replaces it

The 2.5D input is `[-gap, centre, +gap]` where `b7_triplet_gap: 1` counts in
**slices**, not millimetres. Its physical depth is therefore `2 x spacing`, and
spacing is not a constant of this dataset:

```text
slice spacing, mm    p05 0.80   p50 3.30   p95 5.00   max 8.33
triplet depth, mm    p05 1.59   p50 6.60   p95 10.00  max 16.66
```

```text
triplet depth      series
<2 mm               1,756
2-4 mm              3,153
4-6 mm              3,102
6-8 mm             10,737
8-12 mm             5,387
>=12 mm               236
```

**The same three channels carry between 1.59 mm and 16.66 mm of knee — a
ten-fold range, and nothing in the pipeline is aware of it.**

For scale, against the structures the twelve targets ask about: articular
cartilage is roughly 2-4 mm thick, a meniscus roughly 3-5 mm, the ACL roughly
7-11 mm across. At the median depth of 6.60 mm the outer two channels are
already a whole meniscus away from the centre one. In the 236 series above
12 mm they are further apart than the ACL is wide. At the other end, in the
1,756 series below 2 mm, the three channels are near-duplicates and the input
is effectively 2D with two wasted channels.

## Per study, which is what a prediction is

`study_geometry_rollup`, 4,407 studies, median 5 series each. A first draft of
this document warned that the corpus might not be the input, because the model
might read a selected handful per series. **That warning was wrong.** Since B12
the policy is `all_repaired_anatomical_series_v1`: every series with a
recognised plane is read, no winner picked, no cap. The corpus *is* the input,
and the legacy six-per-study `dual` policy changes almost nothing anyway
(66.1% against 69.1% on the headline row below).

```text
                               all      expert 58    report only
studies                      4,407             58          4,349
spread within a study, p50    4.97 mm        4.73 mm       4.97 mm
spread within a study, max   14.65 mm       10.20 mm      14.65 mm
thickest >= 2x thinnest      69.1%          67.2%         69.2%
mixes <2 mm with >=8 mm      11.1%           8.6%         11.1%
studies losing any frame     12.9%           6.9%         12.9%
```

**In 3,047 studies — 69.1% — the thickest triplet is at least twice the
thinnest.** The model fuses those views into one prediction. Half of all
studies span 4.97 mm or more inside themselves, and one spans 14.65 mm.

The 58 experts are geometrically typical. Every apparent difference is inside
the noise of 58 draws: 8.6% mixed against 11.1% has a standard error of 3.7
points, 6.9% losing frames against 12.9% has 3.3, and 67.2% against 69.2% has
6.2. Nothing separates them. **The veto surface is not geometrically skewed**,
which is one confound it does not have.

The loss is concentrated rather than smeared: 567 studies lose anything at all,
and those that do lose a median 19.5% and up to 57.4% of their own frames. That
still costs little, because all of it is the redundant interior of fine
volumes.

## The mechanism gap this exposes

The model is told three things about each series, as embeddings in
`b12_1_hierarchical`: `plane_embedding`, `fluid_embedding`, `fat_embedding`.
Plane, fluid sensitivity, fat suppression.

**It is not told the slice spacing.** So it is told what kind of sequence it is
looking at, and not how much of the knee each input holds — the one fact that
changes what the pixels mean.

This is not the experiment that was already refused. `b12_use_physical_scale`
is `false`, but B10 was **in-plane** normalisation: PixelSpacing and field of
view, left to right. Its own first line says so. Through-plane geometry has
never been normalised, and has never been given to the model either.

## What this does not establish

**A physical gap is not free.** Making the gap depend on spacing — `gap =
round(target / 2 / spacing)` — would widen the thin volumes and cannot narrow
the thick ones, since a gap below 1 does not exist. So it changes the bottom of
the range and leaves the top alone.

**A measured fault is not a predicted gain.** The surface that would judge any
fix is the 58 experts, and that surface has already been shown unable to
resolve small per-target differences. This document records a fault in the
input. It does not forecast a score.

**This project has refused physical scaling before.** `b12_use_physical_scale`
is `false`, and a run of mechanism changes has measured at nothing. A measured
fault in the input is not a prediction about the score, and this document does
not make one.

## What it does establish

A fact about the input that holds without a single label: the depth of the 2.5D
window is set by the scanner, not by the design. Any experiment that varies the
triplet should now state which depth it is varying, in millimetres.
