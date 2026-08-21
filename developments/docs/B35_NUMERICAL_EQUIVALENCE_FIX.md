# B35 numerical-equivalence implementation fix

> Status: implementation correction made before any B35 training result or expert/hidden B35 score exists.

The first B35 launch stopped at its pre-training base-equivalence guard with:

```text
[B35] exact-base reconstruction max|delta|=0.0022049546
```

The prospective tolerance was `0.002`. The tolerance is **not relaxed**.

## Cause

The original B35 shared encoder pass flattened all 32 sampled centres for every active series before splitting the tensor into ConvNeXt encoder chunks. Ordinary B34 flattens only its historical 16 centres before splitting into the same configured encoder batch size.

Under BF16/cuDNN this changes the shapes and boundaries of the ConvNeXt chunks. Identical images and weights can therefore receive slightly different floating-point rounding, producing a small logit discrepancy even though the first 16 B35 centres are the same images as B34.

This is an implementation-level numerical issue, not evidence for or against the B35 scientific hypothesis.

## Correction

`b35_exact_batch.py` encodes the historical 16-centre group and the additional 16-centre group separately. The historical group now has exactly the same:

- active-series order;
- centre order;
- flattened image order;
- encoder chunk size;
- encoder chunk boundaries

as ordinary B34.

The extra 16 centres are then encoded separately and concatenated after the encoder. Each of the 32 sampled images is still encoded only once.

The base-equivalence tolerance remains:

```text
max |B34 ordinary logits - B35 reconstructed B34 logits| <= 0.002
```

No supervision, labels, architecture, optimizer, seed, crop policy, number of centres, spatial grid, training epochs, or evaluation rule changed.

## Segmentation fault seen after the failed guard

The failed equivalence check raises while a multiprocessing DataLoader with persistent spawned workers is alive. A native-process failure during teardown can obscure the intended Python exception on some systems. The numerical mismatch is corrected rather than bypassed. If a segmentation fault recurs after the corrected guard passes, it is a separate runtime issue and should be diagnosed with `PYTHONFAULTHANDLER=1` and a zero-worker reproduction.

## Corrected entrypoints

Training:

```bash
python -m rsna_knee.b35_training_v2 ...
```

Evaluation:

```bash
python -m rsna_knee.b35_eval_v2 ...
```

These entrypoints preserve the original B35 checkpoint schema and all prospective scientific contracts while substituting only the exact-batch encoder implementation.
