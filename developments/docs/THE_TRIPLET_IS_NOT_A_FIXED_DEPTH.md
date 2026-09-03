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

## What this does not establish

Three limits, stated so they are not quietly stepped over.

**This is the corpus, not the input.** The model reads only the series
`select_series` picks per study and plane, not all 24,371. The distribution
above is therefore not the distribution the model sees. The number that would
matter is the triplet depth of the **selected** series, per study, and it has
not been measured.

**A physical gap is not free.** Making the gap depend on spacing — `gap =
round(target / 2 / spacing)` — would widen the thin volumes and cannot narrow
the thick ones, since a gap below 1 does not exist. So it changes the bottom of
the range and leaves the top alone.

**This project has refused physical scaling before.** `b12_use_physical_scale`
is `false`, and a run of mechanism changes has measured at nothing. A measured
fault in the input is not a prediction about the score, and this document does
not make one.

## What it does establish

A fact about the input that holds without a single label: the depth of the 2.5D
window is set by the scanner, not by the design. Any experiment that varies the
triplet should now state which depth it is varying, in millimetres.
