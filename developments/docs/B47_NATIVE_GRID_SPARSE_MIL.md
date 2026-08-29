# B47 — Score evidence on the grid the encoder actually produced

## Status

**PROSPECTIVE / IMPLEMENTED / NOT RUN.**

Frozen before any B47 training and before any B47 result is inspected. No
number in this document is a result.

The B46 completion guard has now been met, but B47 remains unrun. The historical
sequencing rule existed because each B46 fold was a separate Python process and
an intervening source revision could have let later folds load different code.
B48 and B49 subsequently completed as independent frozen experiments; neither
outcome automatically authorises a B47 run or changes this document's contract.

## The fault

B37 raised the input to 448 pixels so that focal findings would have detail to
be found in. ConvNeXt-tiny has an output stride of 32, so a 448×448 slice leaves
a 14×14 feature map — 196 places the head could look:

```text
input                        448 x 448
encoder feature map           14 x 14   =  196 cells
scored by the sparse head      6 x 6    =   36 cells
discarded                                   5.44x
```

`_encode_chunk` average-pools the map to a fixed 6×6 before any evidence is
scored. At the B42 reference area each surviving cell averages about 74.7 input
pixels in each direction — roughly **23 mm of knee** at the audited median
in-plane sampling of 0.3125 mm/pixel. For a meniscal tear or a subtle cruciate
signal change that is a very coarse notion of where the evidence is.

The sparse residual has never been worth more than about six thousandths of
Expert-58 macro AUC. Discarding five-sixths of the localisation the resolution
was bought for is a plausible reason.

### The second fault, which only B42 has

B42 preserves the native aspect ratio, so a series can reach the head as 10×20
cells rather than 14×14. Pooling a rectangle onto a fixed **square** grid
produces bins that are anisotropic and whose physical extent differs from series
to series. `region_embedding` is a table indexed by grid position, so its row 7
describes a different piece of anatomy depending on how the scan was shaped.

B37's square input did not have this problem. B42 introduced it, and the B42
record frames the geometry change purely as an improvement.

## What B47 changes

Exactly one capability: **where evidence is scored**.

```text
evidence grid     aspect-preserving, sized by a cell BUDGET rather than a fixed
                  side, so cells stay square-ish and a study's series are
                  described on comparable scales
region identity   a continuous function of the cell's normalised centre,
                  replacing the position-indexed lookup table
ragged regions    series with differing cell counts are padded to the study
                  maximum and the padding is masked out before the top-k
evidence dtype    fp32 through scoring and selection
```

The region encoding is machinery **forced by** the grid change, not a separate
bet: keeping the encoder's cells means their number varies, and a 36-row lookup
table cannot describe a varying number of cells. The fp32 change is included
because the top-k selects the best 8 of tens of thousands of tightly clustered
scores, and bfloat16 carries two to three significant digits, so it orders
instances inside that band arbitrarily and differently on different hardware.
Raising the cell count from 36 to 240 makes that worse, so fixing it is part of
the same change rather than a separate one.

The region projection is **zero-initialised**, exactly as `region_embedding` is,
so at step zero the head asks the pretrained local representation the same
question B36's head does.

### The budget is measured, not chosen

```text
square reference                 14 x 14 = 196 cells
measured range over all B42 geometries      196 to 240
frozen native budget                        240
frozen control budget                        36
```

B42 holds the *anatomical* area near 448² and then reflection-pads up to stride
alignment, so a rectangular series carries padding a square one does not.
Sweeping every source shape from 120 to 1300 pixels through
`constant_area_shape` gives between 196 and 240 cells, the maximum arising at
extreme aspect ratios — a 144×1120 source aligns to 192×1280, i.e. 6×40 = 240.

A budget of 196 would therefore pool almost every non-square series, and the
native arm would not be native. This is checked by a test, not remembered.

## The two arms

```text
control   region budget  36    B37/B42's effective resolution, reached through
                               B47's new machinery
native    region budget 240    the encoder's own grid, unpooled
```

The control is the point of the design. It reproduces the old resolution while
using the new continuous region encoding, which separates **"more places to
look"** from **"a different way of saying where"**. Without it, a positive
result could not be attributed and B47 would repeat B37's mistake of changing
several things and winning uninterpretably.

Both arms train from the same base checkpoint, on the same supervision, for the
same fixed two epochs, with no checkpoint selection.

## Frozen parent contract

B47 inherits the complete B42 contract unchanged, verified by
`require_b47_contract` calling `require_b42_contract`:

```text
full-native percentile normalization
90% native center crop
constant-area native-aspect resize, reference area 448^2
reflection pad only to stride 32
ragged per-series encoding
32 deterministic 2.5D centers, gap=1
top-k=8
temperature=1.0
local auxiliary weight=1.0
zero-start target-wise sparse residual gate
final ConvNeXt stage/output norm trainable
B34 non-encoder hierarchy frozen
head LR=1e-4, encoder-tail LR=5e-6
weight decay=1e-4, grad clip=1.0
effective studies/update=2
exactly 2 epochs, no checkpoint selection
TTA [-1,0,+1]
```

An arm may not change its own budget: `require_b47_contract` refuses any
`b47_region_budget` that does not match the declared arm. Otherwise the control
is not a control.

## Cost, and the risk that it does not fit

The native arm multiplies the MIL token count by **6.7×** (36 → 240 cells per
slice). Tokens per study are `cells × 32 slices × series`, so a median 5.53-series
study goes from roughly 6,400 tokens to about 42,000.

**This may not fit in 16 GB.** It is the main practical risk, and it must be
established by a preflight before any training run, not discovered part-way
through an epoch. If the native arm does not fit, the honest response is to
report that and run an intermediate budget as a separate declared arm — **not**
to quietly shrink the native budget, which would make the arm something other
than what this document froze.

Wall-clock cost is unrecorded for every experiment in this line, so B47 should
be timed on its first fold-equivalent and the figure written down.

## Predeclared decision rule

Primary surface: Expert-58 macro AUC, as a **post-training audit only**, against
the frozen B42 combined prediction on the same 58 UIDs.

```text
Delta_native  = B47 native  - B42 combined
Delta_control = B47 control - B42 combined
```

**The grid hypothesis is supported** only if all hold:

```text
Delta_native >= +0.010
paired 95% bootstrap CI lower bound > 0
Delta_native - Delta_control >= +0.005
at least 7 of 12 target AUCs improve versus B42
all 12 leave-one-target-out macro deltas remain > 0
```

The third clause is what makes this an experiment rather than another
uninterpretable win: if the control moves as much as the native arm, the gain
came from the new region encoding and not from the extra resolution, and the
conclusion is about the encoding.

**Not supported** if `Delta_native < +0.005`, or if the CI lower bound is
below zero. Otherwise **inconclusive**.

The focal six (ACL, MCL, medial and lateral meniscus, contusion, fracture) are
recorded separately, because they are the findings the localisation hypothesis
predicts should move. A macro gain carried entirely by non-focal targets does
not support the hypothesis even if it clears the thresholds.

## Governance

Do not use the B47 result to tune the region budget, the basis width, top-k,
temperature, the grid policy, the learning rates, the epoch count, or any target
subset. Expert-58 has been reused extensively and is a descriptive audit surface,
not independent evidence; a B47 result does not authorise a hidden submission.

If B47 is ever run, its outcome must be interpreted under the frozen rule above.
The historical forward path to B48 is no longer an active instruction: B48 and
B49 have already completed independently and both were not promoted. A future
B47 decision requires a separately current protocol decision, not post-hoc
selection from those outcomes.

## Implementation

```text
developments/src/rsna_knee/b47_native_grid_sparse_mil.py
developments/tests/test_b47_native_grid_sparse_mil.py
```

Nothing under `b36_*`, `b37_*`, `b42_*` or `b46_*` is modified. B47 subclasses
`B42ConstantAreaAspectSparseMILResidual` and `B36SparseMILHead`, so every
existing experiment and any B46 run in flight are untouched.

A training entry point and a config are still to be written; they must reuse
`b42_constant_area_aspect_sparse_training` with the model class swapped and the
arm declared, so that nothing but the head geometry differs from B42.
