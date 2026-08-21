# B35 — target-conditioned dense spatial residual

> **Status — prospective / implementation ready.** This experiment is defined before any B35 expert or hidden result is observed. It is not a promotion and it does not modify the frozen B34 base model.

## Why B35 exists

The current model has now received a much denser supervision surface without a commensurate jump in expert performance:

```text
Phase-8 all-script supervision       18,024 usable cells
B6-preserved + full LLM fill         34,010 usable cells
increase                              +88.7%

one-stage FT expert reference        ~0.663
full LLM-fill expert result          0.6688249
```

The new supervision is therefore useful but does not by itself explain the ~0.69 hidden-test plateau.

The current image path destroys local evidence very early:

```text
ConvNeXt final map       ~7 x 7 x 768
        ↓ global average pool
one 768-vector / slice
        ↓ generic series pool
one 768-vector / MRI series
        ↓ study Transformer
        ↓ pathology queries
```

ACL, MCL, meniscal lesions, contusions and fractures are the weakest current targets and are also the findings for which small, localized image evidence is most plausible. B35 tests that mechanism directly.

## Scientific question

> Does allowing each pathology to attend to coarse local ConvNeXt features *before* global slice pooling and generic series pooling materially improve expert and hidden ranking, especially for focal findings?

## Frozen base

B35 Phase A requires exactly the completed full-fill base:

```text
supervision                    B6 preserved + LLM fills B6-silent cells
excluded targets               none
training studies               4,349 report-only
training MRI series            24,035
training supervision cells     34,010
architecture                   B34
encoder                        B16 report-aligned ConvNeXt-Tiny
encoder trainable stages       1 in the already-completed base
base endpoint                  fixed E2
expert labels in base gradient 0
```

During B35 Phase A the entire completed base, including its already fine-tuned encoder, is frozen.

## B35 sampling

For each real MRI series and each centre-offset view:

```text
historical B34 centres     16
additional dense centres   16
--------------------------------
combined centres           32
```

The first 16 combined centres are **exactly** the ordinary B34 centres. The additional 16 are deterministically selected from a 32-centre grid, excluding historical centres where possible.

This lets one ConvNeXt pass provide both:

```text
first 16 centres -> ordinary globally pooled B34 slice vectors
all 32 centres   -> 3 x 3 local ConvNeXt tokens
```

The training code verifies on the first batch that reconstructing B34 from the shared encoder pass matches the ordinary B34 forward path within a small numerical tolerance.

## Local branch

For a typical five-series study:

```text
5 series x 32 centres x 9 regions = 1,440 local tokens
```

There is **no self-attention over 1,440 tokens**. Instead, each of the 12 pathology queries scores the local tokens directly:

```text
local ConvNeXt token
    + continuous through-plane position
    + coarse region identity
    + plane/fluid/fat metadata
        ↓
12 target-specific attention distributions
        ↓
12 target-specific local summaries
        ↓
12 local residual logits
```

Geometry/acquisition embeddings start at zero so the branch initially reads the pretrained image features directly rather than overwriting them with random positional structure.

## Exact zero-gated safety contract

For target `t`:

```text
B35_logit[t] = frozen_B34_logit[t] + tanh(g[t]) * local_logit[t]
```

with

```text
g[t] = 0 at initialization
```

Therefore:

```text
B35(step 0) == frozen B34 exactly
```

The branch must learn a non-zero gate from report-derived training supervision before it can perturb the deployed base predictor.

## Phase-A optimization

Only the B35 local head is trainable.

```text
base B34 parameters       frozen
ConvNeXt encoder          frozen
B35 local head            trainable
micro-batch               1 study
gradient accumulation     2
effective batch           2 studies
head LR                    1e-4
weight decay               1e-4
epochs                     2 fixed
scheduler                  none / constant LR
expert labels in gradient  0
expert checkpoint select   no
```

No scheduler, encoder-depth, label, crop, resolution, or seed change is mixed into this experiment.

## Evaluation

Expert-58 remains a reused development diagnostic only. B35 evaluation uses the same fixed centre-offset TTA:

```text
[-1, 0, +1]
```

and reports both the frozen B34 base and B35 candidate through the same B35 data path.

Primary diagnostic:

```text
12-target macro AUC delta: B35 - frozen B34
```

Mechanistic diagnostic, predeclared before the result:

```text
focal-six mean AUC
ACL
MCL
Medial Meniscus
Lateral Meniscus
Contusion
Fracture
```

The evaluator also records per-target residual gates and attention concentration.

## Stop/go rule

Because the 58-study surface has resolution of roughly 0.03 macro AUC, small movements should not trigger another architecture campaign.

```text
GO to integrated B35 / hidden submission:
    macro delta >= +0.03
    OR a large focal-six improvement with no global collapse

WEAK / do not over-interpret:
    macro delta between about +0.01 and +0.03

KILL this mechanism as the main explanation:
    macro delta <= +0.01 and focal-six does not improve materially
```

The hidden competition test remains the only trustworthy promotion signal.

## Commands

Train:

```bash
python -m rsna_knee.b35_training \
  --data-root "$DATA_ROOT" \
  --labels runs/b6_plus_llm_fill_all \
  --series-policy "$SERIES_POLICY" \
  --base-checkpoint runs/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --out-root runs/b35_target_spatial_v1
```

Evaluate:

```bash
python -m rsna_knee.b35_eval \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b35_target_spatial_v1/b35_model.pt \
  --out runs/b35_target_spatial_v1/expert58.json
```

## What B35 does *not* test

B35 Phase A does not test:

- higher in-plane resolution;
- crop-before-resize;
- DINOv3;
- more trainable encoder stages;
- a different scheduler;
- pure-LLM replacement of B6;
- target-specific post-hoc model selection;
- gold-supervised fine-tuning.

Those remain separate experiments so a B35 result has one interpretable cause.
