# B20 crop-order implementation audit

> **Status — 2026-08-13:** VERIFIED. Historical B20 is unchanged; this audit corrects the description of its executed preprocessing order.

The B20 dataset calls the parent `VariableSeriesKneeDataset.__getitem__()` first and then applies `apply_crop_focus()` to the returned tensor. The parent path has already run `preprocess_triplets()`, which bilinearly resizes the sampled MRI triplets to `224 x 224`.

Therefore historical B20 actually executes:

```text
native DICOM volume
  -> normalize / sample 2.5D triplets
  -> bilinear resize to 224 x 224
  -> centered 90% crop of the 224 image (about 202 x 202)
  -> bilinear resize back to 224 x 224
```

It does **not** execute a native-resolution crop before the first resize.

This finding does not invalidate or modify any stored B20 result. B20 remains the active working model and its checkpoint remains:

```text
runs/b20_crop_focus/b20_model.pt
```

The controlled B21 experiment tests only the corrected ordering of the same 90% field-of-view restriction:

```text
native DICOM volume
  -> centered 90% crop at native resolution
  -> normalize / sample 2.5D triplets
  -> single resize to 224 x 224
```

Because the historical B16 encoder and historical B20 downstream checkpoint both saw weak-v2 holdout studies during earlier training stages, neither is eligible for a clean weak-v2 model-ranking comparison. B21 therefore uses a newly trained weak-v2-safe B16 report encoder and a newly trained matched B20-v2 control before the paired weak-v2 comparison.

See [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md) for the frozen optimization protocol.
