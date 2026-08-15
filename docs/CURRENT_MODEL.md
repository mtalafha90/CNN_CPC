# Current working model

## Status

The active working model is **B20 (`B20_crop_only_joint_focus`)**, with canonical checkpoint:

```text
runs/b20_crop_focus/b20_model.pt
```

Canonical epoch: **2**. Historical development evidence after B20 does not replace this checkpoint.

## Architecture

The current model is a CNN-based 2.5D knee MRI classifier with hierarchical study aggregation:

```text
MRI study
  -> all eligible real MRI series
  -> 16 sampled slice positions per series
  -> 3-channel adjacent-slice triplets
  -> 224 x 224 input
  -> deterministic centered 90% crop and resize
  -> frozen ConvNeXt-Tiny image encoder
  -> learned attention pooling to one token per MRI series
  -> study-level Transformer context
  -> 12 learned pathology queries
  -> 12 logits / sigmoid probabilities
```

The visual encoder was adapted during the historical B15/B16 stages and is frozen in B20. Those stages are preserved under `developments/` for reproducibility but are not exposed as separate top-level workflows anymore.

## Targets

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

## Training supervision

The canonical B20 model was trained from frozen B6 report-derived supervision. Positive and negated cells use asymmetric soft targets/weights; uncertain and unmentioned cells are ignored.

Later B24X/B25X experiments identified a severe class-coverage problem for Synovitis, but these experiments are development evidence only. They are preserved in `developments/` and have not promoted a replacement for B20.

## Evaluation roles

- `training/`: reproduces the recorded B20 training recipe.
- `validation/`: evaluates on the reused 58-study expert development surface. This is **not independent test evidence** because those studies influenced model development/checkpoint selection.
- `testing/`: generates predictions for the released competition test metadata and submission file. Hidden competition labels remain organizer-side.

## Reproducibility archive

All B0--B25X experiment code, documentation, configurations, tests, Kaggle notes, manuscript material and historical workflows are preserved under `developments/`.
