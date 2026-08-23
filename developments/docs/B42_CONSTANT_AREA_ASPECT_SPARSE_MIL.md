# B42 — constant-area native-aspect rectangular sparse MIL

## Status

**PROSPECTIVE / NOT RUN.**

B42 is frozen before implementation, training, Expert-58 evaluation, or hidden
competition evidence.  It is motivated directly by the completed B37/B41
comparison and is not a post-hoc target-specific adjustment.

## Motivation

B37's direct square resize achieved the current proven hidden Kaggle score of
`0.714`, but rectangular acquisitions are geometrically stretched.  B41 fixed
that geometry by resizing the 90% native crop to fit inside a `448x448` square
and zero-padding the remainder.  On the reused Expert-58 surface:

```text
                         B37 E2          B41 E2          B41 - B37
Global 448 macro         0.6794831901    0.6717205944    -0.0077625956
Combined macro           0.6858177916    0.6778722842    -0.0079455074
Sparse residual gain    +0.0063346016   +0.0061516898     approximately equal
Focal-six combined       0.5841648772    0.5674468541    -0.0167180231
```

The sparse residual contribution remained almost unchanged, while the global
image representation fell.  This supports a specific mechanism: B41 preserved
shape but reduced anatomical occupancy and effective spatial sampling for
rectangular series.

For a `640x1280` source matrix, B41 performs:

```text
640x1280 native
-> 576x1152 central 90% crop
-> 224x448 aspect-preserving resize
-> zero-pad to 448x448
```

Only about half of the square input area contains the resized anatomy.  The
model therefore pays the compute/memory cost of a 448-square image while the
short anatomical dimension has only 224 pixels.

B42 tests the alternative: preserve the same aspect ratio **and** preserve the
approximately `448^2` anatomical pixel budget, using a rectangular tensor with
no large blank bars.

## Frozen B42 geometry

Let the retained 90% native crop have height `H` and width `W`.  Define the
reference pixel area

```text
A0 = 448 * 448 = 200704 pixels.
```

Use one isotropic scale factor

```text
s = sqrt(A0 / (H * W))
```

and resize once to

```text
h = round(H * s)
w = round(W * s).
```

The same `s` is applied to both axes, so the anatomy is never stretched and
`h*w` remains approximately `A0`.

Examples:

```text
576x576   -> about 448x448
576x1152  -> about 317x634
1152x576  -> about 634x317
```

For ConvNeXt stride alignment only, B42 may symmetrically reflection-pad each
resized rectangle to the next multiple of 32 independently in height and width.
This is **not** square padding.  The total added margin is always less than 32
pixels per axis and contains reflected boundary signal rather than a large zero
field.

The `576x1152` example therefore becomes approximately:

```text
317x634 anatomical image
-> 320x640 stride-aligned reflected tensor
-> about 10x20 final ConvNeXt feature map
```

Compared with B41's roughly half-occupied `14x14` square feature surface, B42
keeps about the same total feature-cell budget as B37 while every interior cell
corresponds to anatomy rather than a large blank margin.

## Ragged-series encoding

Different MRI series within one study may have different rectangular shapes.
B42 must **not** pad all series to a shared square before the encoder.

The dataset returns each series as its own tensor:

```text
series_i: [32, 3, H_i, W_i]
```

The encoder processes each series at its own rectangular shape in fixed chunks.
For each series:

1. run the existing ConvNeXt feature extractor;
2. global-average-pool the real rectangular final feature map for the B34 global
   hierarchy;
3. adaptive-average-pool that same real feature map to the unchanged `6x6`
   B37/B41 local grid;
4. pass those 36 pathology-specific local regions into the unchanged top-k=8
   sparse-MIL head.

The `6x6` local grid stays fixed in B42 deliberately.  B42 tests the input
occupancy/representation mechanism only; changing local token count at the same
time would confound the experiment.

## Effective batch-size preservation

B37/B41 use study micro-batch 2.  Rectangular series cannot be stacked into one
ordinary `[B,K,32,3,H,W]` tensor without reintroducing large padding.

B42 therefore keeps an **effective study batch of two**:

- shuffle study UIDs with the frozen B37 loader seed;
- take two studies per optimizer update;
- process their ragged series sequentially;
- divide each study loss by two and accumulate gradients;
- perform one AdamW step after both studies;
- process the final odd study as a singleton exactly as the historical batch-2
  loader does.

ConvNeXt uses LayerNorm rather than BatchNorm, so there is no cross-study batch
statistic that would be changed by sequential study encoding.

## Frozen model and training contract

B42 changes only the geometric tensor construction and the ragged encoder
plumbing required to avoid large padding.

Unchanged from B37:

```text
base checkpoint                 exact full-fill B34 checkpoint
training studies                4349 report-only
training series                 24035
supervision cells               34010
expert labels in gradients      0
2.5D centres                    32 deterministic centres
triplet gap                     1
targets                         all 12
local grid                      6x6
sparse top-k                    8
temperature                     1.0
local auxiliary weight          1.0
trainable encoder depth         final ConvNeXt stage + output norm only
head LR                         1e-4
encoder-tail LR                 5e-6
weight decay                    1e-4
grad clip                       1.0
training duration               exactly 2 epochs
TTA evaluation offsets          [-1, 0, +1]
```

For the cleanest geometry comparison, B42 reuses B37's construction and loader
random streams rather than introducing a new sparse-head initialization solely
because the preprocessing changed.

No stochastic MRI augmentation is introduced.

## Preflight requirements

Before training, B42 must pass all of the following without an optimizer step:

1. square `448x448` input path;
2. representative `2:1` rectangular path (`~320x640` after stride alignment);
3. representative `1:2` path;
4. a high-aspect-ratio synthetic path at approximately constant pixel area;
5. the real study with the largest eligible-series count;
6. nonzero gradients in the trainable encoder tail and sparse evidence head;
7. zero gradients in frozen B34 non-encoder hierarchy;
8. finite global and combined logits;
9. peak host/CUDA memory recorded.

If memory fails, infrastructure changes must be declared before the fixed run;
resolution area, model depth, grid, top-k, supervision, or epoch count must not
be silently changed.

## Evaluation

Expert-58 remains a reused development diagnostic.  Report all three fixed
comparators through their matching preprocessing paths:

```text
historical 224 full-fill base
B37 E2 direct-square 448
B41 E2 aspect-fit + zero-pad 448
B42 E2 constant-area rectangular native-aspect
```

Report separately:

- global macro AUC;
- combined sparse-MIL macro AUC;
- focal-six mean;
- per-target AUC;
- sparse residual increment;
- paired B42-minus-B37 bootstrap;
- paired B42-minus-B41 bootstrap;
- input-content occupancy statistics;
- target rectangular size and final feature-grid distributions.

The key mechanistic question is whether recovering anatomical occupancy restores
B37's global representation while retaining B41's correct geometry.

## Hidden-test governance

B42 is one fixed candidate, not a geometry sweep.  Do not select among target
areas, padding modes, aspect-ratio caps, local-grid sizes, or epoch counts using
Expert-58.

If the fixed endpoint is completed successfully, one unchanged hidden Kaggle
submission is scientifically justified.  Promotion requires independent hidden
competition evidence.  The benchmark to beat remains B37's `0.714` hidden score.

## Interpretation map

```text
B42 > B37 hidden:
    aspect preservation was useful once the lost anatomical pixel/feature
    occupancy was restored.

B42 ~= B37 hidden and > B41:
    geometry can be corrected without paying B41's occupancy penalty, but the
    square-stretch itself was not the dominant remaining bottleneck.

B42 ~= B41 or worse:
    B41's deficit is not explained mainly by blank-canvas occupancy; investigate
    representation pretraining, physical scale, or supervision instead of more
    resize variants.
```
