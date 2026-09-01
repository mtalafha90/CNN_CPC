# B53 — the augmentation B52 configured but never applied

**Status:** code ready, unrun. It is the follow-up to B52's full-data run.

## The finding

B52 claimed three changes. Two of them happened.

```text
the encoder trains                    real
cosine completes, best epoch kept     real
augmentation on                       inert
```

`train_b52` calls `make_b7_dataset_config(settings, root, train=True)`, which
faithfully sets `noise_std=0.02`, `rotation_deg=5.0` and the rest on the
`DatasetConfig`. The dataset it then builds never reads those fields.

Two independent causes, either one sufficient:

```text
b37_highres_sparse_mil.py:184   super().__init__(..., train=False)   hard-coded,
                                not read from the config
B42ConstantAreaAspectDataset    writes its own _load_b42, which goes straight
                                from DICOM to cropped triplets; _augment_mri is
                                never on that code path at all
```

## The measurement

Building the B42 dataset twice from the same DICOM series, once with every
augmentation field set and once with none:

```text
two draws with augmentation ON, identical to each other : True
augmentation ON identical to augmentation OFF           : True
maximum absolute difference                             : 0.0
```

Byte for byte identical. This is pinned as
`test_b52s_dataset_does_not_augment`, which will start failing if the upstream
dataset ever begins augmenting on its own — at which point B53 would be applying
augmentation twice.

## What this does and does not change about B52

**It does not undo B52's result.** `+0.0395` on the gate split and `+0.0719` on
the full population are real. They came from two changes rather than three.

**It means one standard remedy has never been tried here.** Augmentation is the
usual answer to a model memorising a few thousand studies, and no run in this
archive has ever had it.

**The Colab notebook and `b52_standalone.py` are unaffected.** Those carry their
own `augment_series`, which is tested and mutation-verified to change the image.
The gap was only ever in the `developments/` trainer.

## What B53 changes

One thing. The pixels the model trains on are distorted, using the values the
config has always carried and the operations `dataset._augment_mri` has always
defined, applied to the tensor `B42ConstantAreaAspectDataset` actually produces.

```text
rotation        +/- 5 degrees          b7_rotation_deg
translate       +/- 3% per axis        b7_translate_frac
scale           +/- 5%                 b7_scale_jitter
gamma           1 +/- 0.12             b7_gamma_jitter
bias field      +/- 0.08, clamped      b7_bias_field_strength
noise           std 0.02               b7_noise_std
slice dropout   8% of slices           b7_slice_dropout
```

Every value is read from `config/b42_constant_area_aspect_sparse.yaml` rather
than written into B53, so none of them is a second change smuggled in.

Everything else is B52's, **imported from `b52_competition_training` rather than
restated**: `b52_parameter_groups`, `select_train_and_validation`,
`evaluate_split`, and every default rate, stage count, split and seed. A test
asserts the identity of those objects, so a second change cannot hide in code
that is meant to be shared.

Three small deliberate differences from the original `_augment_mri`, each
because B53 runs it somewhere the original never did:

* **Each axis shifts by its own side.** B42 series are rectangles of roughly
  constant area, not squares. Scaling both shifts by one `image_size` would move
  a tall series much further sideways than up.
* **Slice dropout never empties a series.** At `p=0.08` over 32 slices this
  essentially never fires; the case it prevents is a blank study still carrying a
  real label, which teaches something false rather than nothing.
* **Draws come from an explicit generator**, seeded by run seed, epoch and study
  index, not from the global random state. A run is reproducible, and DataLoader
  workers cannot repeat one another's numbers.

## What it deliberately leaves alone

Two of the nine configured settings change *which slices are chosen* rather than
what the chosen pixels look like: `center_jitter` and `train_gap_choices`. Slice
choice is the frozen B35 contract — `b35_centers` hard-codes `jitter=0` — and
moving it would be a second change in the same run.

`--slice-jitter N` implements it, is tested, and defaults to `0`.

## The check that would have caught B52

`b53_preflight` draws the same study twice and **refuses to start** unless the
two tensors differ:

```text
[B53 preflight] augmentation reaches the pixels: 2/2 series changed,
                max |diff| 0.318...
```

The result goes into every checkpoint as `augmentation_verified`, beside the
policy itself. B52 wrote `augmentation_enabled: true` and trained on identical
pixels for 27 hours; a boolean nobody measured is what made that possible.

## Running it

From a clone of the repository:

```bash
python -m rsna_knee.b53_augmented_training \
  --config config/b42_constant_area_aspect_sparse.yaml \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --labels-root runs/067_.../b6_plus_llm_fill_all \
  --series-policy runs/020_.../audit/series_policy.json \
  --base-checkpoint runs/067_.../train/llm-filled/model.pt \
  --domain-split runs/083_.../b50_ordered_slice_selection_split \
  --out-root runs/087_Experiment_B53_augmentation_applied \
  --epochs 6 --all-data --preflight-only
```

Drop `--preflight-only` to train. From a standalone bundle, `./run_b53.sh 6
--all-data` does the same thing.

## Reading the result

B53 prints its own comparison at the end, against the matching B52 run on the
identical 548 unseen-scanner studies:

```text
B52, gate train split, 1,447 studies    0.802666
B52, --all-data, 3,801 studies          0.834998
```

Both are **selection statistics** — the best of several epochs on the surface
used to choose the epoch — so they are optimistically biased by construction,
comparable with each other and with B53, and with nothing else.

Three outcomes, and what each would mean:

* **B53 clearly above B52.** Augmentation was worth having, and the obvious next
  questions are a longer schedule and stronger settings, since a model that can
  no longer memorise usually tolerates both.
* **B53 level with B52.** Six epochs on 3,801 studies was not enough training for
  memorisation to be the binding constraint. Augmentation would then be expected
  to pay off only alongside a longer schedule.
* **B53 below B52.** The settings are too strong for this data, most likely the
  geometric ones on a task where small structures matter. Worth re-running with
  rotation and scale halved before concluding anything.

## Governance

Like B52, B53 selects its checkpoint on a held-out split. That is competition
practice and deliberately not the frozen-endpoint policy the scientific line
uses. It sits beside that line and takes nothing from it.
