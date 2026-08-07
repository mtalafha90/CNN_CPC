# Modeling strategy

## Current production model

The repository now has one supported model rather than a runnable experiment matrix:

```text
fold-safe calibrated report teacher
  -> confidence-weighted training labels
  -> dual-sequence 2.5D knee MRI
  -> pretrained ConvNeXt-Tiny slice encoder
  -> per-target slice attention
  -> per-target stream attention
  -> 12 outputs
```

The auxiliary ranking term is deliberately small and confidence-gated. It uses official gold cells and sufficiently confident/extreme calibrated pseudo-labels; low-confidence report silence does not create ranking pairs.

## Validation discipline

The trusted surface is very small, so model development must remain conservative.

- Fold assignments are deterministic for a fixed seed.
- Duplicate normalized gold reports stay together.
- Validation-report duplicates are excluded from training.
- Teacher calibration is fit only on out-of-fold gold studies.
- Validation uses raw official cells with NaNs preserved.
- Per-target AUC is undefined, rather than invented, when a target has only one class in a sample.
- The reported macro AUC is accompanied by a study-level bootstrap interval.
- Future alternatives should be compared on identical OOF studies with the paired bootstrap implementation.

## Why this production architecture

The selected design incorporates the strongest low-complexity ideas from the public-methodology review without preserving a large amount of speculative code:

- **2.5D** captures local through-plane context while retaining mature 2D pretrained encoders.
- **Dual fluid/structural routing** preserves complementary MRI contrast information.
- **ConvNeXt-Tiny** provides a stronger modern 2D encoder than the original ResNet18 baseline while remaining practical.
- **Target-specific hierarchical attention** lets ACL, meniscus, OA, fluid, marrow and fracture targets use different slices and sequences.
- **Fold-safe report supervision** uses the 4,349 non-gold studies without treating them as negatives.
- **Confidence-gated ranking** nudges ordering for an AUC metric without trusting ambiguous pseudo-labels equally.

None of these choices should be described as leaderboard-optimal until measured OOF results exist.

## Memory strategy

The model processes many images per study (`streams × sampled slices`). ConvNeXt encoding is therefore split into bounded encoder micro-batches. During training, gradient checkpointing recomputes encoder activations during backward instead of storing every slice activation simultaneously.

This produces a clean single-GPU reference implementation before distributed execution is introduced.

## Next development step: multi-GPU DDP

The next code change should be proper PyTorch `DistributedDataParallel`, not `nn.DataParallel`.

The DDP implementation should add:

1. one process per GPU and explicit local-rank device binding;
2. `DistributedSampler` for training data;
3. deterministic epoch reseeding with `sampler.set_epoch(epoch)`;
4. synchronized/aggregated validation predictions across ranks;
5. rank-0-only checkpoint/CSV/JSON writes;
6. global effective batch-size handling and learning-rate policy;
7. clean single-GPU fallback through the same entry point;
8. launch examples for `torchrun` and the target cluster/Kaggle environment.

The current repository intentionally removes the previous `nn.DataParallel` path so DDP can be added without supporting two competing parallelization models.

## Before changing the model again

The immediate empirical step remains to train the current three folds and freeze their OOF predictions. Only after that should architecture changes be reintroduced, one at a time, against the same folds.

Avoid:

- treating unlabeled cells as negatives;
- test-time report dependence;
- tuning many hyperparameters against 58 gold studies;
- introducing a new backbone without a controlled OOF comparison;
- claiming CV/leaderboard improvements before running them;
- committing competition data, reports, credentials or model artifacts.
