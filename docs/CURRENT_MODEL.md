# Current working model

## Status

The active working model remains **B20 (`B20_crop_only_joint_focus`)**, with canonical checkpoint:

```text
runs/b20_crop_focus/b20_model.pt
```

Canonical epoch: **2**. Later development evidence does not replace this checkpoint without independent hidden or external evidence.

The post-B20 development register is now:

```text
B20  active historical/predictive checkpoint
B31  PV1-selected downstream development architecture
B34  frozen successful training-scaffold simplification/mechanistic architecture
B29  frozen no-scaffold mechanistic comparator
B33  frozen simple complementary-mean comparator
```

B31 and B34 are not promoted to active independent-test status.

## Architecture

The current active model is a CNN-based 2.5D knee MRI classifier with hierarchical study aggregation:

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

The visual encoder was adapted during the historical B15/B16 stages and is frozen in B20 and the later fixed-encoder downstream experiments. Those stages are preserved under `developments/` for reproducibility.

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

## Later development evidence

### PV1

PV1 froze a StudyInstanceUID-only 2,496/624 split of the 3,120 active B6 studies and compared fixed-E2 B20, B31, and B33 under the same frozen B16 encoder. The primary macro weighted soft-label BCE was:

```text
B20  0.6155808446
B31  0.5743065510
B33  0.5849690647
```

The secondary macro AUC was:

```text
B20  0.5727579473
B31  0.7567308761
B33  0.7565223439
```

B31 was selected as the downstream development architecture under PV1, but PV1 is weak-label validation and does not replace B20 as the active historical model.

### B31 context-zero audit

The same trained B31 checkpoint was evaluated after setting only `local_context.weight` to exact zero at inference. The primary loss changed from `0.5743065510` to `0.5743052782`; the paired 95% interval for context-zero minus normal B31 was approximately `[-1.10e-5,+8.61e-6]`. This showed no resolved direct inference benefit from the trained local-context branch and motivated a training-path hypothesis.

### PV2 and B34

PV2 was then frozen inside the old PV1 training partition: 1,997 studies for matched training and 499 for validation, while the original 624 PV1 validation studies remained locked. B29, B31, and B34 were trained to fixed E2.

```text
Model  macro weighted soft BCE   macro AUC
B29          0.5992237910        0.7471635770
B31          0.5909689396        0.7528002817
B34          0.5909695511        0.7527943588
```

The predeclared primary scaffold comparison, B34 minus B29, had median `-0.0083070` with 95% interval `[-0.0125729,-0.0039875]`. The B34-minus-B31 interval was `[-3.13e-6,+4.67e-6]`, entirely inside the predeclared `[-0.001,+0.001]` equivalence margin. Therefore the B34 training-scaffold mechanism passed: local context improves the matched training trajectory, while B34 can bypass all 2,304 local-context parameters at inference without a resolved loss at the chosen metric resolution.

PV2 remains an internal weak-label mechanism test with historical downstream exposure. It is not independent clinical validation and does not authorize B34.1, target-specific switching, blending, kernel retuning, or active-model promotion. See `developments/docs/PV2_B34_RESULT.md`.

## Evaluation roles

- `training/`: reproduces the recorded B20 training recipe.
- `validation/`: evaluates on the reused 58-study expert development surface. This is **not independent test evidence** because those studies influenced model development/checkpoint selection.
- PV1: prospective weak-label downstream architecture-selection evidence under the exact frozen B16 encoder.
- PV2: post-PV1 internal weak-label mechanism evidence for the frozen B34 hypothesis; not historically untouched.
- `testing/`: generates predictions for the released competition test metadata and submission file. Hidden competition labels remain organizer-side.

## Reproducibility archive

The full experiment ladder, prospective-validation code, Kaggle notes, manuscript material, tests, and historical workflows are preserved under `developments/`.
