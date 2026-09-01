# The two B52 notebooks

There are two, because one file cannot do both jobs.

```text
notebook/b52_colab_subset.ipynb   Colab, a Google Drive subset, self-contained
notebook/b52_local_full.ipynb     a local machine, the full dataset, real code
```

Each is generated from a builder and must be rebuilt after any edit to it:

```bash
python notebook/build_b52_colab_subset_notebook.py
python notebook/build_b52_local_notebook.py
python -m pytest notebook/ -q
```

A test in each suite compares every cell of the checked-in notebook against its
builder, so a forgotten rebuild fails rather than shipping quietly.

## Why two, and not one

The real trainer refuses to run on a subset, on purpose:

```python
expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
if not expected_train_sha or sha256_file(root / "train.csv") != expected_train_sha:
    raise ValueError("B52 domain split source train.csv fingerprint mismatch")
```

The B50 scanner gate records the SHA-256 of the `train.csv` it was built from,
and a Drive subset has a different `train.csv` by definition. `_report_only_surface`
likewise expects the full 4,349 report-only studies.

That check is worth keeping. It is what stops a run silently training on a
different population and reporting a number nobody can interpret. So the subset
notebook rebuilds B52's **regime** on its own self-contained model rather than
defeating the check, and the local notebook drives the **real code** on a machine
that has the whole dataset.

## What each one actually does

### `b52_colab_subset.ipynb`

Inherits everything structural from `build_notebook.py` — the Drive archive
contract, DICOM decoding, the 448 geometry, the dataset and the sparse-MIL head
— and adds B52's regime on top.

Of B52's three changes it genuinely adds two and inherits the third:

```text
the encoder learns       inherited. The notebook's model is built from scratch,
                         so there was never a pretrained encoder to freeze. Its
                         version of this lever is the 0.05x hierarchy rate.
augmentation on          added. Seven of the nine, reproduced in the notebook.
a schedule that
finishes, best epoch     added. Cosine with T_max = epochs, and the best
kept                     hold-out epoch restored at the end.
```

Two of the nine augmentations are deliberately absent, and both for reasons:

* **centre jitter** moves the crop window before the crop happens, and by the
  time this code sees a study the crop is done. Applying it afterwards would
  crop twice and change the geometry every experiment holds fixed.
* **no left-right flip at all.** Mirroring a knee swaps medial and lateral, and
  `Medial Meniscus` and `Lateral Meniscus` are two different answers. A test
  reads the augmentation source and fails if a flip primitive ever appears.

Checkpoint selection happens on a **report-labelled** hold-out split, not on the
58 expert studies. Hidden leaderboard scores have run consistently above the
expert-58 surface — `0.694` hidden against roughly `0.66` local, `0.714` against
`0.683` — so a report-labelled hold-out is the better guide. The gold studies are
scored every epoch and nothing is selected from them.

**One honest gap.** The real B52 holds out whole scanner models. This notebook
splits at random, because a small subset may not hold enough distinct scanners to
hold any of them out. A random split is easier, so the number here reads better
than the real one. It is a fair guide to whether the model is learning and not a
fair estimate of anything else.

Nothing from B48, B50 or B51's comparison sections is carried over. A test
asserts they are gone, including the inherited gold-only build cell, which would
otherwise still run and still train on the 58 expert studies without saying so.

### `b52_local_full.ipynb`

A front end for one command, with the checks that stop a 27-hour run failing for
a boring reason. It defines no model — a notebook copy of the trainer would be a
second implementation to keep in step.

```text
2. resolve the layout      works for both the standalone bundle and a repo clone
3. gate fingerprint        hashed here, before the trainer loads anything
   out_root collision      the trainer refuses to overwrite b52_best_model.pt,
                           but only after loading the base checkpoint
4. GPU check               refuses a torch build with no kernels for the card
5. verify.sh, preflight    checksums, then one real forward and backward pass
6. train                   streamed, so the cell shows progress
7. read the result         from history.json and the checkpoint, not from memory
```

The GPU check is the one that earns its place. An RTX 50-series card is compute
capability 12.0 and needs a CUDA 12.8 torch wheel. With an older wheel everything
imports and looks fine, and the failure — `no kernel image is available for
execution on the device` — arrives at the first forward pass. The check compares
the card's capability against `torch.cuda.get_arch_list()` and then runs a real
kernel.

Its command line is checked against the trainer's own argument parser: a test
reads `add_argument` calls out of `b52_competition_training.py` and fails if the
notebook passes a flag the trainer does not accept, or omits one it requires.
Another test pins `CHECKPOINT_NAME` to the trainer's `B52_CHECKPOINT_NAME`.

## What the subset notebook's numbers are worth

Nothing, in absolute terms. It starts from random weights on a subset. What
transfers is the shape: whether the training loss keeps falling, whether the
hold-out score keeps rising, and which epoch it peaks on.

The full-data notebook's numbers are **selection statistics** — each is the best
of several epochs on the surface used to choose the epoch, so each is
optimistically biased by construction. They are not effect sizes and are not
comparable with the `0.714` leaderboard score. The checkpoint says so in its own
`governance` field, and a test requires the notebook to say so too.
