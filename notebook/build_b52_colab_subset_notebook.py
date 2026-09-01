"""Generate the standalone B52 Colab notebook for a Google Drive subset.

B52 is not an architecture change. It is the discovery that every experiment
from B37 to B51 was measured on a model that had barely been trained: the pixel
encoder was frozen, all nine augmentations were switched off, and the whole run
was 3,120 optimiser steps over two fixed epochs with no checkpoint selection.
Against a leaderboard top of `0.952` and a local `0.714`, an architecture
ablation moving `0.003` was never going to be the answer.

This notebook reproduces **B52's training regime** on whatever subset of the
data fits in Drive. Three things define that regime, and all three are here:

```text
1. the encoder learns          it reads the pixels; freezing it froze the model
2. augmentation is on          nine settings existed and were all zeroed
3. a real schedule, and the
   best epoch is kept          not two fixed epochs with whatever came out
```

It cannot run the real `rsna_knee.b52_competition_training` code, and the reason
is worth stating rather than hiding. That trainer checks the SHA-256 of the data
folder's `train.csv` against the fingerprint recorded in the B50 scanner gate,
and refuses to start if they differ. A Drive subset has a different `train.csv`
by definition. The refusal is correct -- it is what stops a run silently
training on the wrong population -- so the subset notebook rebuilds the regime
rather than defeating the check. The companion notebook,
``b52_local_full.ipynb``, drives the real trainer on the full data.

Everything not part of the B52 regime is inherited unchanged from
``build_notebook.py``: the Drive archive contract, DICOM decoding, the 448
geometry, the dataset and the sparse-MIL head. Nothing from B48, B50 or B51's
comparison sections is carried over; the questions those asked are answered and
re-running them here would only add cells that do not train the model.
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path

BASE_BUILDER = Path(__file__).with_name("build_notebook.py")
BASE_NAMESPACE = runpy.run_path(str(BASE_BUILDER))
CELLS: list[tuple[str, str]] = list(BASE_NAMESPACE["CELLS"])


def replace_cell(index: int, kind: str, text: str) -> None:
    """Replace one inherited cell without modifying the base builder."""
    CELLS[index] = (kind, text.strip("\n"))


def find_cell(marker: str) -> int:
    """Locate an inherited cell by its content, so edits survive reordering."""
    matches = [index for index, (_, text) in enumerate(CELLS) if marker in text]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one cell containing {marker!r}, found {len(matches)}"
        )
    return matches[0]


def rewrite_cell(marker: str, kind: str, text: str) -> None:
    """Replace the one inherited cell that contains a marker."""
    CELLS[find_cell(marker)] = (kind, text.strip("\n"))


def retitle_cell(marker: str, old: str, new: str) -> None:
    """Renumber an inherited heading now that B52 sections come before it."""
    index = find_cell(marker)
    kind, text = CELLS[index]
    if old not in text:
        raise RuntimeError(f"heading {old!r} not found in cell {index}")
    CELLS[index] = (kind, text.replace(old, new))


def insert_cells(marker: str, new_cells: list) -> None:
    """Insert B52's definitions immediately before an inherited cell."""
    index = find_cell(marker)
    CELLS[index:index] = [(kind, text.strip("\n")) for kind, text in new_cells]


# --- what this notebook is -------------------------------------------------

replace_cell(
    1,
    "markdown",
    """
# B52 — actually train the model

Eight experiments in a row searched for a missing piece of architecture and none
of them moved the score. B52 asked a different question: was anything missing at
all, or had the model underneath those experiments simply never been trained?

This is what every one of them inherited:

```text
the pixel encoder was frozen            learning rate exactly 0.0
one stage of five thawed                at 0.05x, so 5e-6
all nine augmentations were zeroed      the settings existed, switched off
two epochs, fixed                       3,120 optimiser steps in total
no checkpoint selection                 whatever epoch 2 produced was the answer
```

B52 changes only the training regime and nothing else. On the real data that was
worth **+0.0395** macro AUC against the same frozen baseline on the same 548
studies, and **+0.0719** once it trained on the full population. For comparison,
the best architecture result in the archive before it was `+0.011`.

## The three changes

```text
1. the encoder learns      the part that reads pixels, at a real rate
2. augmentation is on      rotation, shift, scale, gamma, noise, dropout, bias
3. train to a schedule,    a cosine that finishes, and the best epoch is kept
   then keep the best      rather than the last
```

That is the whole experiment. The geometry, the head, the labels and the loss
are all left exactly as they were, so anything that changes is down to training.
""",
)

replace_cell(
    2,
    "markdown",
    """
## What this notebook is, and is not

**It is** B52's training regime, running on whatever subset of the data you have
in Drive: report-derived labels for training, augmentation on, the encoder
learning, a cosine schedule that finishes, and the best epoch kept.

**It is not** the real B52 run. That one starts from the Phase-9 checkpoint
rather than random weights, trains on 3,801 of the 4,349 studies, and validates
on 548 studies from scanners it has never seen. It takes about 26 hours on an
RTX A4500. This notebook starts from scratch on a subset, so its absolute
numbers mean nothing and must never be compared with a leaderboard score.

What transfers is the shape: whether the loss keeps falling, whether the
held-out score keeps rising, and which epoch it peaks on.

### Why the real B52 code cannot run on a subset

The real trainer reads the B50 scanner gate, and that gate records the SHA-256
of the `train.csv` it was built from. Before doing anything else the trainer
compares it with the `train.csv` in front of it:

```python
if sha256_file(root / "train.csv") != expected_train_sha:
    raise ValueError("B52 domain split source train.csv fingerprint mismatch")
```

A subset has a different `train.csv`, so the check fails — correctly. It exists
to stop a run quietly training on the wrong population. The full-data notebook,
`b52_local_full.ipynb`, drives that real trainer on a machine that has the whole
dataset.

### What you need in your Drive folder

Beside `train.csv` and `train_series.csv`, put the label export:

```text
training_targets.csv
```

That is the file `b23_llm_labels.py` writes (`b6_report_labels.py` writes the
same shape). It has one row per report-only study, and for each of the twelve
targets a probability, a `__confidence` and a `__state`. Section 11 checks the
file before anything else happens.
""",
)


DEFINITIONS: list = []


def define(kind: str, text: str) -> None:
    """Collect a B52 definition cell, inserted before the training section."""
    DEFINITIONS.append((kind, text))


# --- the reports as labels -------------------------------------------------

define(
    "markdown",
    """
## 11. Read the report labels

`train.csv` has twelve label columns and a `Report` column. Only the 58
expert-gold studies have the twelve columns filled in; the other 4,349 have
nothing in them but do have a written report. Training on 58 studies is not
training, so the labels come from the reports.

The export gives one state per target, and each state carries a fixed
probability and a fixed confidence. The two exporters differ on one row:

```text
state          probability   confidence (B23)   confidence (B6)
positive           0.97           0.90               0.90
negated            0.03           0.90               0.90
uncertain          0.50           0.00               0.25
conflict           0.50            --                0.20
unmentioned        0.50           0.00               0.00
```

**Confidence `0.00` means "ignore this cell entirely".** The unmentioned row is
the one that matters most. When a report does not mention a finding, that is
**not** evidence the finding is absent — the radiologist simply did not write
about it.

This is easy to get wrong in a way that quietly ruins training. An unmentioned
cell is stored as probability `0.50`, not as a blank. Fed to an ordinary loss it
would look like a real label, and the model would be pushed towards 0.50 on the
majority of cells. The confidence column is what stops that, so this notebook
never reads a probability without its confidence.
""",
)

define(
    "code",
    '''
REPORT_LABELS_FILENAME = "training_targets.csv"


def report_label_columns() -> list[str]:
    """The exact columns b23_llm_labels.py and b6_report_labels.py write."""
    columns = ["StudyInstanceUID"]
    for target in TARGETS:
        columns.extend([target, f"{target}__confidence", f"{target}__state"])
    return columns


def load_report_labels(path: Path) -> pd.DataFrame:
    """Read the label export and refuse anything that is not the agreed shape."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No report labels at {path}.\\n"
            "Export them with b23_llm_labels.py and copy training_targets.csv "
            "into the same Drive folder as train.csv."
        )

    frame = pd.read_csv(path)
    missing = [name for name in report_label_columns() if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing[:6]}")

    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path.name} lists the same study more than once")

    for target in TARGETS:
        confidence = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce")
        if confidence.isna().any() or float(confidence.min()) < 0 or float(confidence.max()) > 1:
            raise ValueError(f"{target}__confidence must be a number between 0 and 1")
    return frame


def weak_targets_and_confidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split the export into a target matrix and a confidence matrix.

    A cell with zero confidence is blanked to NaN as well as being given zero
    weight. Either alone would be enough; doing both means a later change to the
    loss cannot accidentally start training on report silence.
    """
    targets = np.full((len(frame), len(TARGETS)), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frame), len(TARGETS)), dtype=np.float32)

    for index, target in enumerate(TARGETS):
        column = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float32)
        weight = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").to_numpy(np.float32)
        used = weight > 0
        targets[used, index] = column[used]
        confidence[used, index] = weight[used]
    return targets, confidence


def split_gold_and_report_only(train_table: pd.DataFrame) -> tuple:
    """A study is 'gold' exactly when train.csv already carries a label for it."""
    written = train_table[TARGETS].apply(pd.to_numeric, errors="coerce")
    is_gold = written.notna().any(axis=1)
    return (
        train_table.loc[is_gold].reset_index(drop=True),
        train_table.loc[~is_gold].reset_index(drop=True),
    )


def select_report_training_studies(
    train_table: pd.DataFrame, labels: pd.DataFrame, records
) -> tuple:
    """Choose what trains and what is held back, and refuse a gold leak.

    Kept apart from model building so it can be checked on its own. A mistake
    here would not crash: it would quietly put expert-gold studies into training
    and make every score the notebook prints look better than it is.
    """
    gold_frame, report_only = split_gold_and_report_only(train_table)

    leaked = sorted(set(labels["StudyInstanceUID"]) & set(gold_frame["StudyInstanceUID"]))
    if leaked:
        raise ValueError(
            f"the label export contains {len(leaked)} expert-gold studies "
            f"(for example {leaked[0]}); it must hold report-only studies only"
        )

    keep = labels["StudyInstanceUID"].isin(set(report_only["StudyInstanceUID"])) & labels[
        "StudyInstanceUID"
    ].isin(set(records))
    train_frame = labels.loc[keep].reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(
            "No study is in the export, in train.csv as report-only, and in your "
            "extracted DICOM subset at once. Check that the export covers the "
            "studies you actually downloaded."
        )

    gold_usable = gold_frame.loc[
        gold_frame["StudyInstanceUID"].isin(set(records))
    ].reset_index(drop=True)
    return train_frame, gold_usable


def describe_report_labels(confidence: np.ndarray) -> dict:
    """How much supervision the reports actually provide, per target."""
    used = confidence > 0
    return {
        "studies": int(confidence.shape[0]),
        "cells_total": int(used.size),
        "cells_used": int(used.sum()),
        "coverage": float(used.mean()),
        "per_target_cells": {
            target: int(used[:, index].sum()) for index, target in enumerate(TARGETS)
        },
    }
''',
)


# --- the loss --------------------------------------------------------------

define(
    "markdown",
    """
## 12. A loss that respects confidence

Two adjustments to ordinary cross entropy.

**Each cell is weighted by its confidence.** A confident `positive` (0.90) counts
much more than an `uncertain` (0.25), and an unmentioned cell (0.00) counts not
at all.

**Each target is balanced by how much supervision it has.** Some findings are
written about far more often than others. Without this, the targets the reports
happen to discuss most would dominate every gradient, and the rare ones would
barely train. The multiplier is `mean(mass) / mass`, where a target's mass is the
total confidence it received across the training studies.

This is the same rule the real pipeline uses. B52 changed the training regime and
nothing about supervision, so this part is deliberately identical to what came
before it.
""",
)

define(
    "code",
    '''
def target_balance_multipliers(confidence: np.ndarray) -> np.ndarray:
    """Give every target the same total say, whatever the reports talked about."""
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.ndim != 2 or confidence.shape[1] != len(TARGETS):
        raise ValueError(f"confidence must have shape [N,{len(TARGETS)}]")

    mass = confidence.sum(axis=0)
    if not (mass > 0).all():
        empty = [TARGETS[index] for index in np.flatnonzero(mass <= 0)]
        raise ValueError(f"the reports gave no usable supervision for: {empty}")
    return (float(mass.mean()) / mass).astype(np.float32)


def report_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy weighted by per-cell confidence and per-target balance."""
    if logits.shape != target.shape or logits.shape != confidence.shape:
        raise ValueError("logits, target and confidence must have the same shape")

    # A blank target is unusable whatever its confidence claims.
    known = torch.isfinite(target).float()
    effective = confidence.float() * known * multiplier.to(logits.device)[None, :]

    denominator = effective.sum()
    if float(denominator.detach().cpu()) <= 0:
        # No usable cell in this batch. Return a real zero that still has a
        # gradient path, so the training step stays well defined.
        return logits.sum() * 0.0

    safe_target = torch.nan_to_num(target, nan=0.0)
    cell = F.binary_cross_entropy_with_logits(
        logits.float(), safe_target.float(), reduction="none"
    )
    return (cell * effective).sum() / denominator.clamp_min(1e-8)


class ReportSupervision:
    """Per-study confidence, looked up by study UID rather than by row number.

    The dataset filters and reindexes the frame it is given, so a confidence
    array addressed by position would silently drift out of step with the studies
    the loader actually yields. Addressing by UID cannot drift.
    """

    def __init__(self, confidence_by_study: dict, multiplier: np.ndarray) -> None:
        self.confidence_by_study = confidence_by_study
        self.multiplier = torch.tensor(multiplier, dtype=torch.float32, device=DEVICE)

    def batch(self, study_uids: list) -> torch.Tensor:
        """The confidence rows for one batch, in the batch's own order."""
        missing = [uid for uid in study_uids if uid not in self.confidence_by_study]
        if missing:
            raise KeyError(f"no confidence recorded for {missing[:3]}")
        rows = np.stack([self.confidence_by_study[uid] for uid in study_uids])
        return torch.tensor(rows, dtype=torch.float32, device=DEVICE)
''',
)


# --- change 2: augmentation ------------------------------------------------

define(
    "markdown",
    """
## 13. Change 2 — turn augmentation on

The real config lists nine augmentation settings and then builds its dataset with
`train=False`, and that one flag sets every one of them to zero:

```python
noise_std      = ... if train else 0.0
slice_dropout  = ... if train else 0.0
center_jitter  = ... if train else 0
rotation_deg   = ... if train else 0.0
translate_frac = ... if train else 0.0
scale_jitter   = ... if train else 0.0
gamma_jitter   = ... if train else 0.0
bias_field     = ... if train else 0.0
```

So every model since B37 saw each study in exactly the same way, every epoch. On
a few thousand studies that is a direct invitation to memorise them.

Seven of the nine are reproduced below. Rotation, translation and scale are one
affine warp; then gamma, noise, slice dropout and a smooth bias field.

**Centre jitter is not here, and cannot be.** It moves the crop window before the
crop happens, and by the time this code sees a study the crop is already done.
Doing it afterwards would crop a second time and change the geometry, which is
the one thing every experiment holds fixed.

**There is no left-right flip, deliberately.** Mirroring a knee swaps medial and
lateral, and `meniscus_tear_medial` and `meniscus_tear_lateral` are two different
answers. A flip would teach the model that they are interchangeable.

### Randomness that can be reproduced

Each study gets its own random draw from a seed built out of the run seed, the
epoch number and the study's position. That means the augmentation is different
every epoch, identical if you re-run with the same seed, and correct even if you
raise `num_workers` — worker processes would otherwise each inherit a copy of the
same generator and hand out the same "random" numbers.
""",
)

define(
    "code",
    '''
@dataclass(frozen=True)
class AugmentationPolicy:
    """How hard to distort a training study. Zero everywhere means off."""

    # Rotate the image in plane, up to this many degrees either way.
    rotation_deg: float = 8.0
    # Shift the image, as a fraction of its width and height.
    translate_frac: float = 0.05
    # Zoom in or out by up to this fraction.
    scale_jitter: float = 0.10
    # Brighten or darken the mid-tones, as an exponent around 1.0.
    gamma_jitter: float = 0.20
    # Add Gaussian noise of this standard deviation, in normalised units.
    noise_std: float = 0.02
    # Blank this fraction of slices, so no single slice can carry a study.
    slice_dropout: float = 0.10
    # Multiply by a smooth field of this strength, imitating coil shading.
    bias_field_strength: float = 0.10


AUGMENTATION = AugmentationPolicy()
NO_AUGMENTATION = AugmentationPolicy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def augment_series(
    series: torch.Tensor, policy: AugmentationPolicy, generator: torch.Generator
) -> torch.Tensor:
    """Distort one prepared MRI series of shape [slices, 3, height, width].

    Every draw comes from the generator passed in, so nothing here touches the
    global random state and two runs with the same seed agree exactly.
    """
    if series.ndim != 4:
        raise ValueError(f"expected [slices,3,H,W], got {tuple(series.shape)}")

    def uniform(low: float, high: float) -> float:
        if high <= low:
            return low
        drawn = torch.rand((), generator=generator, dtype=torch.float32)
        return float(low + (high - low) * drawn)

    slices, channels, height, width = series.shape
    out = series.float()

    # --- rotation, translation and scale, as one warp ----------------------
    # Doing them together means one interpolation rather than three, so the
    # image is blurred once instead of three times.
    if policy.rotation_deg > 0 or policy.translate_frac > 0 or policy.scale_jitter > 0:
        angle = math.radians(uniform(-policy.rotation_deg, policy.rotation_deg))
        scale = 1.0 + uniform(-policy.scale_jitter, policy.scale_jitter)
        shift_x = uniform(-policy.translate_frac, policy.translate_frac) * 2.0
        shift_y = uniform(-policy.translate_frac, policy.translate_frac) * 2.0

        cosine, sine = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor(
            [[cosine, -sine, shift_x], [sine, cosine, shift_y]], dtype=torch.float32
        ).expand(slices, 2, 3)
        grid = F.affine_grid(theta, list(out.shape), align_corners=False)
        # Zero padding matches the notebook's pad_value, so a rotated corner
        # looks like the padding the geometry policy already produces.
        out = F.grid_sample(
            out, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    # --- gamma -------------------------------------------------------------
    # Applied on the positive part only. These images are percentile-normalised
    # and can hold small negatives, and a fractional power of a negative number
    # is not a real number.
    if policy.gamma_jitter > 0:
        gamma = math.exp(uniform(-policy.gamma_jitter, policy.gamma_jitter))
        positive = out.clamp_min(0.0)
        out = positive.pow(gamma) + (out - positive)

    # --- smooth bias field -------------------------------------------------
    # A coarse 4x4 grid stretched up to full size: slow shading across the
    # image, which is what an imperfect receive coil actually produces.
    if policy.bias_field_strength > 0:
        coarse = torch.rand((1, 1, 4, 4), generator=generator, dtype=torch.float32)
        field = F.interpolate(
            coarse, size=(height, width), mode="bilinear", align_corners=False
        )
        field = 1.0 + policy.bias_field_strength * (2.0 * field - 1.0)
        out = out * field

    # --- noise -------------------------------------------------------------
    if policy.noise_std > 0:
        noise = torch.randn(out.shape, generator=generator, dtype=torch.float32)
        out = out + policy.noise_std * noise

    # --- slice dropout -----------------------------------------------------
    # Never drop every slice: a study with nothing left in it would be a
    # blank input carrying a real label, which teaches the wrong thing.
    if policy.slice_dropout > 0 and slices > 1:
        keep = torch.rand(slices, generator=generator, dtype=torch.float32)
        drop = keep < policy.slice_dropout
        if bool(drop.all()):
            drop[int(torch.argmax(keep))] = False
        out = out * (~drop).float().view(slices, 1, 1, 1)

    return out


class AugmentedKneeMRIDataset(KneeMRIDataset):
    """The inherited dataset, with B52's augmentation on the training split.

    Subclassed rather than edited so the validation and test paths keep using
    the original, untouched decoding. An augmented validation set would make
    every score noisier and none of them comparable.
    """

    def __init__(self, *args, policy: AugmentationPolicy = NO_AUGMENTATION, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Give the next pass a different draw. The training loop calls this."""
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        """A generator fixed by run seed, epoch and study position.

        Not shared state: a DataLoader worker holds a copy of the dataset, so a
        single shared generator would give every worker the same numbers.
        """
        generator = torch.Generator()
        generator.manual_seed(
            (int(self.config.seed) * 1_000_003 + self.epoch * 9_176 + index) % (2**31 - 1)
        )
        return generator

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if self.policy == NO_AUGMENTATION:
            return item

        generator = self._generator(index)
        volumes, present = item["volumes"], item["present"]
        augmented = []
        for position in range(volumes.shape[0]):
            # A masked series is a zero placeholder the model ignores. Warping
            # it would only cost time.
            if float(present[position]) <= 0:
                augmented.append(volumes[position])
            else:
                augmented.append(augment_series(volumes[position], self.policy, generator))
        item["volumes"] = torch.stack(augmented)
        return item


def describe_augmentation(policy: AugmentationPolicy) -> dict:
    """Which augmentations are actually switched on, so it can be checked."""
    active = {
        name: float(getattr(policy, name))
        for name in (
            "rotation_deg",
            "translate_frac",
            "scale_jitter",
            "gamma_jitter",
            "noise_std",
            "slice_dropout",
            "bias_field_strength",
        )
        if float(getattr(policy, name)) > 0
    }
    return {"active": active, "count": len(active)}
''',
)


# --- change 3: a held-out split and a schedule -----------------------------

define(
    "markdown",
    """
## 14. Change 3 — a held-out split, a schedule, and keeping the best epoch

Two fixed epochs is not a training run, and taking whatever epoch 2 produced is
not model selection. B52 replaces both.

**A cosine schedule that finishes.** The learning rate falls smoothly from full
to nearly zero across exactly the epochs you run. Two epochs of a long cosine
spends the whole run at 100% and 90.5% of the rate — the model never gets the
low-rate settling phase where most of the final quality appears.

**A held-out split, and the best epoch is kept.** After every epoch the model is
scored on studies it did not train on, and the best one is saved. On the real
data the peak was epoch 5 of 6 and epoch 6 was worse, so taking the last epoch
would have thrown away the run's best model.

### Which studies are held out, and why not the gold 58

Selection happens on **report-labelled studies**, not on the 58 expert ones.

Hidden leaderboard scores in this project have consistently run above the
expert-58 surface — `0.694` hidden against roughly `0.66` local, `0.714` against
`0.683`. The likeliest reason is that the hidden labels are report-derived too,
which makes a report-labelled hold-out the better guide and the 58 expert studies
the misleading one. The gold studies are still scored every epoch, but only as
something to read; nothing is chosen from them.

**One honest gap.** The real B52 holds out whole scanner models, so the
validation studies come from machines the model has never seen. This notebook
splits at random, because a small Drive subset may not contain enough different
scanners to hold any of them out. A random split is easier than an unseen-scanner
split, so the score here will look better than the real one. It is a fair guide
to whether the model is learning, and not a fair estimate of anything else.
""",
)

define(
    "code",
    '''
def split_report_studies(
    study_uids: list, validation_fraction: float, seed: int
) -> tuple[list, list]:
    """Divide the report-labelled studies into a training and a hold-out part.

    Sorted before shuffling so the split depends on the seed alone, not on the
    order pandas happened to read the file in.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    ordered = sorted(str(uid) for uid in study_uids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("the same study appears twice in the split input")

    shuffled = list(np.random.default_rng(int(seed)).permutation(ordered))
    held = max(1, int(round(len(shuffled) * float(validation_fraction))))
    if held >= len(shuffled):
        raise ValueError(
            f"{len(shuffled)} studies cannot give both a training and a hold-out "
            "part at this fraction; use more studies or a smaller fraction"
        )

    validation = [str(uid) for uid in shuffled[:held]]
    training = [str(uid) for uid in shuffled[held:]]
    overlap = set(training) & set(validation)
    if overlap:
        raise ValueError(f"{len(overlap)} studies ended up on both sides of the split")
    return training, validation


def build_cosine_schedule(optimizer, epochs: int):
    """Fall from the full learning rate to nearly zero across exactly `epochs`.

    `T_max` equal to the epochs actually run is the whole point. A longer T_max
    leaves the run stopping while the rate is still high, which is what two fixed
    epochs of a long schedule did.
    """
    if int(epochs) < 1:
        raise ValueError("a schedule needs at least one epoch")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=0.0
    )


class BestEpoch:
    """Remember the weights of the best epoch, rather than the last one.

    The comparison is deliberately strict: on a tie the earlier epoch is kept,
    because a later epoch that only matched it is not evidence of improvement.
    """

    def __init__(self) -> None:
        self.epoch: int | None = None
        self.score: float | None = None
        self.weights: dict | None = None

    def offer(self, epoch: int, score: float | None, model: nn.Module) -> bool:
        """Keep this epoch if it beats the best so far. Returns whether it did."""
        if score is None or not np.isfinite(score):
            return False
        if self.score is not None and float(score) <= self.score:
            return False
        self.epoch = int(epoch)
        self.score = float(score)
        self.weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> int:
        """Put the best epoch's weights back into the model."""
        if self.weights is None:
            raise RuntimeError("no epoch was ever scored, so there is nothing to restore")
        model.load_state_dict(self.weights)
        return int(self.epoch)
''',
)


# --- change 1: the encoder learns, and the run is built --------------------

define(
    "markdown",
    """
## 15. Change 1 — the encoder learns, and building the run

The model has two paths and a gate that mixes them:

```text
encoder             reads pixels
global_projection   ┐
global_classifier   ┘  the study hierarchy -> base logits
sparse_head            the local branch    -> local logits
fusion_gate            tanh(g), how much the local branch is trusted
```

`logits = base + tanh(g) * local`.

**In the real pipeline this is the single biggest change B52 makes.** There the
encoder is a pretrained ConvNeXt with `encoder_lr: 0.0`, so the part of the
network that reads pixels never moved; B52 thaws all five of its stages at 0.10x
the head's rate. Trainable parameters went from `50,712` to `46,506,660`.

**In this notebook the encoder was already learning**, because the model is built
from scratch and there is no pretrained encoder to freeze. That is worth saying
plainly rather than pretending to add a change that is already there. The
notebook's version of this lever is the reduced rate for the study hierarchy,
`0.05x`, which is the value B50 measured and B52 kept.

So of B52's three changes, this notebook genuinely adds two — augmentation, and
the schedule with best-epoch selection — and inherits the third.

**A frozen parameter must never reach the optimiser.** If it did, weight decay
would still move it, and "frozen" would not mean frozen. Only tensors that
require gradients are put into a group.
""",
)

define(
    "code",
    '''
HIERARCHY_PREFIXES = ("global_projection.", "global_classifier.")
HIERARCHY_LR_SCALE = 0.05  # the value B50 measured and B52 kept unchanged


def hierarchy_parameter_names(model: nn.Module) -> list[str]:
    """Name every parameter that belongs to the study hierarchy."""
    return [
        name
        for name, _ in model.named_parameters()
        if name.startswith(HIERARCHY_PREFIXES)
    ]


def build_parameter_groups(model: nn.Module, head_lr: float) -> list[dict]:
    """Encoder and head at full rate, the study hierarchy at a reduced one."""
    hierarchy_names = set(hierarchy_parameter_names(model))
    head, hierarchy = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (hierarchy if name in hierarchy_names else head).append(parameter)

    if not head:
        raise RuntimeError("nothing outside the hierarchy is trainable; B52 trains the encoder")

    groups = [{"params": head, "lr": float(head_lr), "name": "encoder_and_head"}]
    if hierarchy:
        groups.append(
            {
                "params": hierarchy,
                "lr": float(head_lr) * HIERARCHY_LR_SCALE,
                "name": "study_hierarchy",
            }
        )
    return groups


def describe_trainable(model: nn.Module) -> dict:
    """What is actually learning, so a setting can be checked rather than assumed."""
    hierarchy_names = set(hierarchy_parameter_names(model))
    counts = {"encoder": 0, "hierarchy": 0, "head_and_rest": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            counts["encoder"] += parameter.numel()
        elif name in hierarchy_names:
            counts["hierarchy"] += parameter.numel()
        else:
            counts["head_and_rest"] += parameter.numel()
    return counts


def read_fusion_gate(model: nn.Module) -> np.ndarray:
    """tanh(g): how much of the local branch reaches the score, per target."""
    return torch.tanh(model.fusion_gate.detach()).cpu().numpy()
''',
)

define(
    "markdown",
    """
## 16. Build the B52 run

Who gets what:

```text
report-labelled studies, most of them   ->  training, with augmentation
report-labelled studies, a fifth        ->  hold-out, used to pick the epoch
the 58 expert-gold studies              ->  scored each epoch, never selected on
```

Gold studies never enter training, which is what makes the number at the end mean
anything. The builder refuses an export containing gold studies outright, because
a leak there would be invisible in the result and would inflate every number the
notebook prints.

Set `CONFIG.epochs` before running this. Two is the inherited default and is the
thing B52 exists to replace — on the real data the best epoch was the fifth of
six. Six is a sensible starting point here; more than that is worth it only if
the hold-out score is still climbing at the end.
""",
)

define(
    "code",
    '''
@dataclass
class B52Run:
    """Everything one B52 run needs, kept in one place."""

    experiment: Experiment
    supervision: ReportSupervision
    scheduler: object
    gold_loader: DataLoader | None
    train_dataset: AugmentedKneeMRIDataset
    best: BestEpoch


def build_b52_run(
    paths: DrivePaths,
    config: RunConfig = CONFIG,
    *,
    policy: AugmentationPolicy = AUGMENTATION,
    labels_path: Path | None = None,
) -> B52Run:
    """Assemble B52's regime: augmentation on, a real schedule, a hold-out split."""
    set_seed(config.seed)
    validate_dataset(paths)

    train_table = pd.read_csv(paths.train_csv)
    train_table["StudyInstanceUID"] = train_table["StudyInstanceUID"].astype(str)
    series_table = pd.read_csv(paths.series_csv)
    records = build_series_records(series_table, config)

    labels = load_report_labels(labels_path or paths.data_root / REPORT_LABELS_FILENAME)
    labelled, gold_usable = select_report_training_studies(train_table, labels, records)

    targets, confidence = weak_targets_and_confidence(labelled)

    # A report that mentions none of the twelve findings supervises nothing. Such
    # a study would cost a full DICOM decode per epoch and teach nothing, and the
    # inherited preflight refuses a batch with no usable label at all.
    usable = confidence.sum(axis=1) > 0
    if not usable.all():
        print(f"skipping {int((~usable).sum())} studies whose report mentions no finding")
        labelled = labelled.loc[usable].reset_index(drop=True)
        targets, confidence = targets[usable], confidence[usable]
    if labelled.empty:
        raise ValueError("no report in the export mentions any of the twelve findings")

    for index, target in enumerate(TARGETS):
        labelled[target] = targets[:, index]

    confidence_by_study = {
        uid: confidence[row] for row, uid in enumerate(labelled["StudyInstanceUID"])
    }
    # Gold labels are real, so every known gold cell carries full confidence.
    for uid in gold_usable["StudyInstanceUID"]:
        confidence_by_study[uid] = np.ones(len(TARGETS), dtype=np.float32)

    train_uids, holdout_uids = split_report_studies(
        list(labelled["StudyInstanceUID"]), config.validation_fraction, config.seed
    )
    train_frame = labelled.loc[
        labelled["StudyInstanceUID"].isin(set(train_uids))
    ].reset_index(drop=True)
    holdout_frame = labelled.loc[
        labelled["StudyInstanceUID"].isin(set(holdout_uids))
    ].reset_index(drop=True)

    # Augmentation on the training split only. The hold-out and the gold studies
    # are decoded exactly as the inherited notebook decodes them, so their scores
    # stay comparable from epoch to epoch.
    train_dataset = AugmentedKneeMRIDataset(
        train_frame, records, paths, config,
        split="train", include_targets=True, policy=policy,
    )
    holdout_dataset = KneeMRIDataset(
        holdout_frame, records, paths, config, split="train", include_targets=True
    )
    gold_dataset = (
        KneeMRIDataset(gold_usable, records, paths, config, split="train", include_targets=True)
        if not gold_usable.empty else None
    )

    # Balance is measured over the studies the training loader will really yield.
    used_confidence = np.stack([confidence_by_study[uid] for uid in train_dataset.study_uids])
    supervision = ReportSupervision(
        confidence_by_study, target_balance_multipliers(used_confidence)
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
        "collate_fn": collate_studies,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_kwargs)
    gold_loader = (
        DataLoader(gold_dataset, shuffle=False, **loader_kwargs)
        if gold_dataset is not None else None
    )

    model = HighResolutionSparseMIL(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        build_parameter_groups(model, config.learning_rate),
        weight_decay=config.weight_decay,
    )

    experiment = Experiment(
        paths=paths,
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda"),
        train_loader=train_loader,
        validation_loader=holdout_loader,
        # Balance is handled per target by the multiplier, so this stays neutral.
        positive_weight=torch.ones(len(TARGETS), dtype=torch.float32, device=DEVICE),
    )

    summary = describe_report_labels(used_confidence)
    augmentation = describe_augmentation(policy)
    print(f"training studies (reports) : {len(train_dataset)}")
    print(f"hold-out studies (reports) : {len(holdout_dataset)}   <- the epoch is chosen on these")
    print(f"gold studies (read only)   : {0 if gold_dataset is None else len(gold_dataset)}")
    print(f"report cells used          : {summary['cells_used']:,} of {summary['cells_total']:,} "
          f"({summary['coverage']:.1%})")
    print(f"augmentations on           : {augmentation['count']} -> {augmentation['active']}")
    print(f"epochs / cosine T_max      : {config.epochs}")
    print(f"trainable                  : {describe_trainable(model)}")
    print(f"optimiser groups           : {[group['name'] for group in optimizer.param_groups]}")
    if gold_dataset is None:
        print("note: no expert-gold study is in your subset, so the gold column stays blank.")

    return B52Run(
        experiment=experiment,
        supervision=supervision,
        scheduler=build_cosine_schedule(optimizer, config.epochs),
        gold_loader=gold_loader,
        train_dataset=train_dataset,
        best=BestEpoch(),
    )
''',
)


# --- training --------------------------------------------------------------

define(
    "markdown",
    """
## 17. Train, and keep the best epoch

Each epoch: one augmented pass over the training studies, then a score on the
hold-out studies, then a score on the gold studies if your subset has any. The
learning rate steps down its cosine, and the best hold-out epoch is kept.

Three numbers to watch, and what they mean:

```text
train loss     should keep falling. Flat means the run has stopped learning.
holdout AUC    the one the epoch is chosen on. This is what B52 improved.
|gate|         how much the model leans on the local branch. On the real data
               this moved before any score did.
```

At the end the best epoch's weights are put back into the model, so whatever you
save or predict with afterwards is the best model, not the last one.
""",
)

define(
    "code",
    '''
def run_b52_epoch(
    experiment: Experiment,
    loader: DataLoader,
    supervision: ReportSupervision,
    training: bool,
) -> dict:
    """One pass, using each cell's confidence instead of treating all cells alike."""
    experiment.model.train(training)
    losses: list[float] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for batch in loader:
        # Read the UIDs before move_batch drops them.
        confidence = supervision.batch(list(batch["study_uid"]))
        volumes, present, metadata, position, target = move_batch(batch)
        del batch

        if training:
            experiment.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), autocast_context():
            output = experiment.model(volumes, present, metadata, position)
            combined_loss = report_weighted_bce(
                output.logits, target, confidence, supervision.multiplier
            )
            local_loss = report_weighted_bce(
                output.local_logits, target, confidence, supervision.multiplier
            )
            loss = combined_loss + experiment.config.local_loss_weight * local_loss

        if training:
            experiment.scaler.scale(loss).backward()
            experiment.scaler.unscale_(experiment.optimizer)
            torch.nn.utils.clip_grad_norm_(
                experiment.model.parameters(), experiment.config.grad_clip_norm
            )
            experiment.scaler.step(experiment.optimizer)
            experiment.scaler.update()

        losses.append(float(loss.detach().cpu()))
        targets.append(target.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(output.logits).detach().cpu().numpy())
        del volumes, present, metadata, position, target, output
        del loss, combined_loss, local_loss

    return {
        "loss": float(np.mean(losses)),
        "target": np.concatenate(targets, axis=0),
        "probability": np.concatenate(probabilities, axis=0),
    }


def train_b52(run: B52Run) -> list[dict]:
    """Train under B52's regime and leave the best epoch's weights in the model."""
    experiment = run.experiment
    if experiment.validation_loader is None:
        raise RuntimeError("B52 chooses an epoch on a hold-out split, so one is required")

    for epoch in range(1, experiment.config.epochs + 1):
        started = time.time()
        # A different augmentation draw each epoch, reproducible from the seed.
        run.train_dataset.set_epoch(epoch)
        rate = experiment.optimizer.param_groups[0]["lr"]

        train_result = run_b52_epoch(
            experiment, experiment.train_loader, run.supervision, training=True
        )
        holdout = run_b52_epoch(
            experiment, experiment.validation_loader, run.supervision, training=False
        )
        holdout_scores = evaluate_predictions(holdout["target"], holdout["probability"])

        row = {
            "epoch": epoch,
            "learning_rate": float(rate),
            "train_loss": train_result["loss"],
            "validation_loss": holdout["loss"],
            "holdout_macro_auc": holdout_scores["mean_auc"],
            "per_target_auc": holdout_scores["per_target_auc"],
            "gate": float(np.abs(read_fusion_gate(experiment.model)).mean()),
            "seconds": round(time.time() - started, 1),
        }

        if run.gold_loader is not None:
            gold = run_b52_epoch(experiment, run.gold_loader, run.supervision, training=False)
            # Read only. Choosing on the 58 expert studies is exactly what
            # section 14 explains this notebook does not do.
            row["gold_macro_auc"] = evaluate_predictions(
                gold["target"], gold["probability"]
            )["mean_auc"]

        kept = run.best.offer(epoch, row["holdout_macro_auc"], experiment.model)
        row["kept"] = kept
        experiment.history.append(row)
        run.scheduler.step()

        def shown(value) -> str:
            return f"{value:.5f}" if value is not None else "  n/a  "

        print(
            f"epoch {epoch:>2} | lr {rate:.2e} | train {row['train_loss']:.5f} | "
            f"holdout {shown(row['holdout_macro_auc'])} | "
            f"gold {shown(row.get('gold_macro_auc'))} | "
            f"|gate| {row['gate']:.5f} | {row['seconds']}s"
            f"{'  <- best so far' if kept else ''}"
        )

    best_epoch = run.best.restore(experiment.model)
    print()
    print(f"restored epoch {best_epoch}, hold-out macro AUC {run.best.score:.6f}")
    print("the model now holds the best epoch's weights, not the last epoch's")
    return experiment.history
''',
)


# --- put B52 in the inherited flow -----------------------------------------
#
# B52's definitions go in front of the inherited training section, and the cells
# that build and train are rewritten to use them. Leaving the inherited
# gold-only path in place would leave a trap: it still runs, it trains on the 58
# expert studies with no augmentation, and nothing about it announces that it is
# not B52.

TRAINING_SECTION = "Train on the extracted training subset and predict"

insert_cells(TRAINING_SECTION, DEFINITIONS)
retitle_cell(TRAINING_SECTION, "## 11. Train on the", "## 18. Train on the")
retitle_cell("Mandatory no-update memory", "### 11a.", "### 18a.")
retitle_cell("Train, plot losses, review cases", "### 11b.", "### 18b.")

rewrite_cell(
    "EXPERIMENT = build_experiment(PATHS, CONFIG)",
    "code",
    '''
# RunConfig is frozen, so a changed setting is a new object rather than an edit.
from dataclasses import replace

# B52 replaces the inherited two fixed epochs with a schedule that finishes, so
# set the epoch count before building the run: the cosine's T_max comes from it.
CONFIG = replace(CONFIG, epochs=6)

# Build the B52 run: augmentation on, the encoder learning, a report-labelled
# hold-out split for choosing the epoch, and the gold studies kept read-only.
B52_RUN = build_b52_run(PATHS, CONFIG, policy=AUGMENTATION)
EXPERIMENT, SUPERVISION = B52_RUN.experiment, B52_RUN.supervision
''',
)

rewrite_cell(
    "RUN_TRAINING = False",
    "code",
    '''
# Keep training off until the preflight cell prints PASS.
RUN_TRAINING = False

if RUN_TRAINING:
    # Train under B52's regime; the best hold-out epoch is restored at the end.
    HISTORY = train_b52(B52_RUN)
    # Plot the training and hold-out loss curves.
    plot_loss_history(EXPERIMENT)
    # Display the numeric epoch history table.
    RESULTS = show_results(EXPERIMENT)

    print()
    print(f"best epoch {B52_RUN.best.epoch} of {CONFIG.epochs}")
    if B52_RUN.best.epoch == CONFIG.epochs:
        print("the last epoch was the best, so the hold-out score was still")
        print("climbing when the run stopped. More epochs are worth trying.")
    else:
        print("the run peaked before the end, so more epochs would not have helped.")

    print()
    print("per-target hold-out AUC at the best epoch")
    BEST_ROW = HISTORY[B52_RUN.best.epoch - 1]
    for target in TARGETS:
        value = BEST_ROW.get("per_target_auc", {}).get(target)
        print(f"  {target:<28} {('%.5f' % value) if value is not None else 'n/a'}")

    # Plot up to twelve held-out MRI examples and their classifications.
    VALIDATION_CASE_TABLE = show_case_examples(
        EXPERIMENT, max_cases=12, title_prefix="Held-out"
    )
    # Build a local-DICOM loader for the separately extracted test subset.
    TEST_LOADER = build_test_loader(TEST_PATHS, CONFIG)
    # Generate one probability row and thresholded classification per test study.
    TEST_PREDICTIONS = predict_test_set(EXPERIMENT, TEST_LOADER)
    # Plot up to twelve unlabelled test MRI examples with their classifications.
    TEST_CASE_TABLE = show_case_examples(
        EXPERIMENT, loader=TEST_LOADER, max_cases=12, title_prefix="Test"
    )
    # Save the best epoch's model, history, configuration, and test predictions.
    RUN_DIRECTORY = save_results(EXPERIMENT, test_predictions=TEST_PREDICTIONS)
''',
)


def build(path: Path) -> Path:
    """Write the notebook."""
    cells = []
    for kind, text in CELLS:
        source = [line + "\n" for line in text.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    written = build(Path(__file__).with_name("b52_colab_subset.ipynb"))
    print(f"{written}  ({len(CELLS)} cells)")
