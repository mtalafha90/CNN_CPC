# The encoder, the resolution, and a mismatch between them

**Written 2026-09-08** from two independent research passes: field intelligence
on the competition itself, and the published literature on DINOv3/DINOv2 versus
ImageNet backbones for MRI. Nothing here is a result from this repository. It is
recorded because it changes what is worth running next.

## The DINOv3 test in this repo was the worst possible configuration

`DINOV3_TINY` and `DINOV3_WIDE` both sit at
`local_artifacts_result_not_in_repo` — no number was ever recorded. What can be
recovered is how it was run:

```text
                    the test that ran          what the evidence says
resolution          224                        the advantage appears above ~256
encoder             freeze_encoder() + tail    freezing collapses the advantage
epochs              PHASE9_FIXED_EPOCHS = 2    a fresh head has not converged
```

**Freezing is not a neutral test, and an earlier note in this session said it
was.** That was wrong. The Medical Slice Transformer paper (arXiv:2411.15802,
Sci Rep 2025) measured the same encoder frozen against fine-tuned, in an
architecture that is almost exactly this one — a per-slice 2D encoder feeding a
Transformer over slices:

```text
breast MRI    fine-tuned 0.94 +/- 0.01     frozen 0.62 +/- 0.03
chest CT      fine-tuned 0.95 +/- 0.01     frozen 0.66 +/- 0.03
```

A study on chest radiographs (arXiv:2510.07191, 816,183 images) found frozen
features from the **7-billion-parameter** DINOv3 teacher losing to fully
fine-tuned 86–89M backbones. Scale does not buy you out of freezing.

So the earlier test carries no information, and its absence from the repo costs
nothing.

## That same paper is the closest external evidence this project has

The Medical Slice Transformer is a per-slice DINOv2 encoder feeding a
Transformer across slices — this repository's topology. On **MRNet knee MRI**:

```text
MST (DINOv2)      0.85 +/- 0.04
3D ResNet         0.69 +/- 0.05        p = 0.001
```

And swapping DINOv2 for an ImageNet ResNet *inside the same architecture*
reduced AUC on all three of its datasets. The gain is attributable to the
encoder rather than to the aggregation.

## The mismatch worth acting on

The DINOv3 paper's Table 15 compares distilled DINOv3 ConvNeXts against
supervised ImageNet-22k ConvNeXts by linear probe at two resolutions:

```text
ConvNeXt-T      supervised @256  87.3     supervised @512  83.0     -4.3
                DINOv3     @256  86.6     DINOv3     @512  87.7     +1.1
```

At 256 the supervised model is slightly ahead. **At 512 the supervised model
degrades badly and DINOv3 improves.** The chest-radiograph study reproduces the
same shape independently: DINOv3's edge appears at 512, not at 224.

**This project runs an ImageNet ConvNeXt-Tiny at 448.** That is the top-left to
bottom-left transition in that table — the specific combination measured to lose
several points.

It also reconciles a finding that looked contradictory. Competitors measured
384 px against 224 px at **-0.004 for 3.5x the cost** and concluded resolution
does not pay. Both can be true at once: resolution does not pay *for a
supervised ImageNet backbone*, because that is exactly where the supervised
backbone falls apart.

So there are two coherent configurations, and this repository is running
neither:

```text
A   keep the ImageNet encoder, drop to ~256      cheap, and the field's
                                                 measured position
B   keep 448, switch to DINOv3 ConvNeXt          matches what the literature
                                                 says 448 is good for
```

A is nearly free to test. B is one argument plus a fine-tuned run.

## Calibrate the prize before spending on it

The best-controlled comparison puts the DINOv3-over-ImageNet gain at **+0.8 to
+1.5 percentage points of AUROC**, at 512, fully fine-tuned. The gap from
`0.716` to `0.952` is **23.6 points**. The encoder is not the bottleneck, and
nothing in this document should be read as suggesting it is.

## A tension in the evidence, and its likely resolution

The competition's own model board has DINOv3-ViT-B/16 at `0.771` while
DINOv2-**small** reaches `0.914`, and one competitor measured v3 below v2 on an
identical pipeline. That looks like a direct contradiction of the literature
above.

The probable explanation is that these are **ViT** numbers. The one study
comparing DINOv3 ConvNeXt against DINOv3 ViT on a medical classification task
with localised findings found **ConvNeXt-B beating ViT-B/16 at every resolution
and every initialisation**. The ViT-favourable results elsewhere are all dense
segmentation over patch tokens, which is a different task shape from MIL
pooling for classification.

Stated plainly: this is inference, not a measurement. What it argues is that
"DINOv3 underperforms here" may be a ViT-versus-ConvNeXt result wearing a
DINOv2-versus-DINOv3 label. It does not license ignoring the competition
evidence.

## Two cheap things from the same literature

**Drop the positional embedding from the study Transformer.** The Medical Slice
Transformer's aggregation ablation on knee MRI:

```text
Transformer, no positional embedding      0.85
Transformer, additive positional embed    0.83
mean pooling                              0.82
linear                                    0.78
```

B35 adds `_position_basis`, a relative slice position. This is one flag to test
and it is the same shape of question the spacing conditioning just answered
negatively.

**RadImageNet is the only checkpoint with direct knee-MRI evidence:** `+4.8%`
AUC on ACL injury and `+4.5%` on meniscal tear against ImageNet. Both shrink on
larger datasets (`+1.7%`, `+0.9%`), and its architectures are dated
(ResNet50, DenseNet121). Worth an ensemble member rather than a replacement.

## Practical notes if B is attempted

All verified against timm 1.0.29 and the DINOv3 repository:

```text
weights          convnext_tiny.dinov3_lvd1689m, timm >= 1.0.20
features         768-d, identical key set and shapes to the current encoder
normalisation    ImageNet mean/std -- unchanged. Only crop_pct differs (1.0)
num_classes      0. There is no classifier; head.norm may be absent, so use
                 forward_features and pool explicitly rather than trusting a
                 silently random LayerNorm
licence          Meta's DINOv3 licence, not Apache. The competition rules
                 explicitly accommodate incompatibly licensed pretrained
                 models, but fine-tuned weights stay under it
offline          the weights are gated on Hugging Face; mirror them into a
                 Kaggle Dataset before scoring, where internet is off
avoid            convnext_*.eupe_lvd1689m -- a stricter non-commercial licence
```

## What this changes about the plan

Nothing about B53, which is running and tests an orthogonal question.

It adds one cheap experiment — **resolution 256 against the current 448 with the
existing encoder** — which the field's own measurement predicts should be at
worst free and at best a real gain, and which this repository has never run.

It does not promote DINOv3 to the front of the queue. The teacher recalibration
against the competition's severity rubric remains the larger lever, because that
is a target mismatch rather than a one-point representation gain.
