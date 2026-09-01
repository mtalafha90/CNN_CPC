"""Generate the local B52 notebook that drives the real trainer on full data.

This notebook is deliberately thin. It does not define a model, a dataset or a
loss, because the real ones already exist in ``rsna_knee.b52_competition_training``
and a notebook copy of them would be a second implementation to keep in step.
Everything here is checking, launching and reading a result back.

The companion notebook, ``b52_colab_subset.ipynb``, rebuilds B52's regime from
scratch for a Google Drive subset. It has to, because the real trainer compares
the SHA-256 of the data folder's ``train.csv`` against the fingerprint recorded
in the B50 scanner gate and refuses to start when they differ -- which a subset's
``train.csv`` always will. On a machine holding the full dataset that check
passes, so this notebook can run the real code.

What it adds over typing the command by hand:

* a GPU check that catches the Blackwell wheel problem before a night is spent
  on a run that will fail at the first forward pass,
* the bundle's own checksum and import verification,
* a preflight that proves gradient actually reaches the encoder,
* streamed training output, so a Jupyter cell shows progress rather than
  appearing to hang for a day,
* the epoch history read back out of the saved checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# B52 on the full dataset — local run

This notebook runs the **real** B52 trainer on the whole dataset, on a machine
that has it. It is a front end for one command, with the checks that stop a long
run failing for a boring reason.

## What B52 is

Every experiment from B37 to B51 was measured on a model that had barely been
trained:

```text
the pixel encoder was frozen            learning rate exactly 0.0
one stage of five thawed                at 0.05x, so 5e-6
all nine augmentations were zeroed      the settings existed, switched off
two epochs, fixed                       3,120 optimiser steps in total
no checkpoint selection                 whatever epoch 2 produced was the answer
```

B52 changes the training regime and nothing else. Geometry, head, labels and
loss are all held exactly as they were, so any difference is down to training.

```text
changed   five encoder stages instead of one, at 0.10x rather than 0.05x
          augmentation on
          a cosine whose T_max equals the epochs actually run
          the epoch chosen by hold-out score, not fixed at two

fixed     the B42 geometry contract
          the sparse-MIL head, 6x6 grid, top-k 8
          the merged B6+LLM label export and its supervision policy
          the Phase-9 llm_fill base checkpoint
          the B50 scanner-grouped gate
```

Trainable parameters go from `50,712` to `46,506,660`.

## What it has produced so far

```text
run                                  studies   validation   macro AUC
B50 frozen control                     1,447          548     0.763117
B50 adapted hierarchy                  1,447          548     0.774336
B52, gate train split, 6 epochs        1,447          548     0.802666   (epoch 5)
B52, --all-data, 6 epochs              3,801          548     0.834998   (epoch 5)
```

All four rows are validated on the same 548 unseen-scanner studies, so they are
directly comparable. Both B52 runs are complete.

**These are selection statistics.** Each is the best of several epochs on the
surface used to pick the epoch, so each is optimistically biased by construction.
They must not be quoted as effect sizes, and they are not comparable with the
`0.714` leaderboard score.
""",
)

markdown(
    """
## 1. What this notebook needs

Two things, and nothing else:

```text
the B52 bundle     built by developments/scripts/package_b52_standalone.sh,
                   or the repository itself
the data folder    holding train.csv, train_series.csv and the DICOM directories
```

The bundle carries the code, the frozen B42 config, the Phase-9 base checkpoint,
the merged label export, the frozen series policy and the B50 gate. The data
folder is not in it: it is large, it is unchanged, and the run reads it directly.

Set both paths in the next cell.
""",
)

code(
    r'''
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

from IPython.display import display

# The unpacked b52_standalone bundle. If you are running from a clone of the
# repository instead, point this at the repository root: the next cell works out
# which layout it is looking at.
BUNDLE = Path("/path/to/b52_standalone")

# The competition data folder, holding train.csv and train_series.csv.
DATA_ROOT = Path("/path/to/rsna-knee-abnormality-detection")

# How many epochs. Six is what both completed runs used, and the full-data run
# peaked at epoch 5 of 6. Twelve is the next thing worth measuring; see section 6.
EPOCHS = 6

# Train on every split except the unseen-scanner validation surface: 3,801
# studies rather than the gate's 1,447, scored on the same 548 either way.
ALL_DATA = True

print("bundle   :", BUNDLE)
print("data root:", DATA_ROOT)
print("epochs   :", EPOCHS, "| all data:", ALL_DATA)
''',
)

markdown(
    """
## 2. Work out the layout and find every artefact

The bundle and the repository put the same files in different places. Rather
than assume, this resolves each path and prints it, so a wrong `BUNDLE` fails
here in a second instead of failing inside the trainer twenty minutes later.
""",
)

code(
    r'''
BUNDLE_LAYOUT = {
    "source": "src",
    "config": "config/b42_constant_area_aspect_sparse.yaml",
    "labels": "labels",
    "series_policy": "policy/series_policy.json",
    "base_checkpoint": "models/phase9_llm_fill_base.pt",
    "gate": "gate",
    "out_root": "runs/b52",
}

REPO_LAYOUT = {
    "source": "developments/src",
    "config": "config/b42_constant_area_aspect_sparse.yaml",
    "labels": (
        "runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/"
        "b6_plus_llm_fill_all"
    ),
    "series_policy": (
        "runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json"
    ),
    "base_checkpoint": (
        "runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/"
        "b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt"
    ),
    "gate": "runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split",
    "out_root": "runs/086_Experiment_B52_competition_full_finetune",
}


def resolve_layout(root: Path) -> dict:
    """Decide whether this is a bundle or a repository, and resolve every path."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory; set BUNDLE")

    layout = BUNDLE_LAYOUT if (root / "src" / "rsna_knee").is_dir() else REPO_LAYOUT
    kind = "bundle" if layout is BUNDLE_LAYOUT else "repository"
    resolved = {"root": root, "kind": kind}

    for name, relative in layout.items():
        path = root / relative
        # The output folder is created by the run; everything else must exist.
        if name != "out_root" and not path.exists():
            raise FileNotFoundError(
                f"this looks like a {kind}, but {name} is missing:\n  {path}"
            )
        resolved[name] = path
    return resolved


def check_data_root(root: Path) -> Path:
    """Confirm the data folder is the thing the trainer will ask it to be."""
    root = Path(root).expanduser().resolve()
    for name in ("train.csv", "train_series.csv"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"no {name} under {root}; set DATA_ROOT")
    return root


CHECKPOINT_NAME = "b52_best_model.pt"


def check_out_root_is_free(out_root: Path) -> None:
    """The trainer refuses to overwrite a checkpoint, so say so now.

    `FileExistsError` arrives only after the base checkpoint, the label export
    and the whole dataset index have been loaded. Reaching it half an hour into
    what looked like a started run is a poor way to find out.
    """
    existing = Path(out_root) / CHECKPOINT_NAME
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists and the trainer will not overwrite it.\n"
            "Move the previous run aside, or point --out-root somewhere new by "
            "changing LAYOUT['out_root'] below."
        )


LAYOUT = resolve_layout(BUNDLE)
DATA_ROOT = check_data_root(DATA_ROOT)
check_out_root_is_free(LAYOUT["out_root"])

print(f"layout: {LAYOUT['kind']}  ({LAYOUT['root']})")
for name in ("source", "config", "labels", "series_policy", "base_checkpoint", "gate"):
    print(f"  {name:<16} {LAYOUT[name]}")
print(f"  {'out_root':<16} {LAYOUT['out_root']}   (free)")
print(f"  {'data_root':<16} {DATA_ROOT}")
''',
)

markdown(
    """
## 3. Check the data folder is the one the gate was built on

The B50 gate records the SHA-256 of the `train.csv` it was built from, and the
trainer refuses to start if the data folder in front of it does not match. That
check is what stops a run training on the wrong population and reporting a
number nobody can interpret.

It is checked here rather than left to the trainer for one reason: hashing a
`train.csv` takes a moment, and the trainer only reaches this check after loading
the base checkpoint and the label export. Failing here is faster and says more.

If it fails, the data folder is a different copy of the dataset — perhaps
re-downloaded, perhaps a subset. It is not something to work around.
""",
)

code(
    r'''
def sha256_of(path: Path) -> str:
    """Hash a file in blocks, so a large CSV never lands in memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_gate_matches_data(gate_dir: Path, data_root: Path) -> dict:
    """Compare the gate's recorded train.csv fingerprint with the real one."""
    payload = json.loads((Path(gate_dir) / "b50_selection_split.json").read_text())
    expected = str(payload.get("source_train_csv_sha256", ""))
    if not expected:
        raise ValueError("this gate records no source_train_csv_sha256")

    actual = sha256_of(Path(data_root) / "train.csv")
    if actual != expected:
        raise ValueError(
            "the data folder is not the one this gate was built on.\n"
            f"  gate expects : {expected}\n"
            f"  data root has: {actual}\n"
            "This is not something to override. Use the dataset the gate was "
            "built from, or rebuild the gate on this dataset."
        )
    return {"train_csv_sha256": actual, "studies": payload.get("studies")}


GATE_CHECK = check_gate_matches_data(LAYOUT["gate"], DATA_ROOT)
print("train.csv fingerprint matches the gate")
print("  sha256 :", GATE_CHECK["train_csv_sha256"])
print("  studies:", GATE_CHECK["studies"])
''',
)

markdown(
    """
## 4. Check the GPU, and the torch build that will drive it

One failure is worth catching before anything else, because it does not appear
at import: **an RTX 50-series card needs a CUDA 12.8 torch wheel.** Blackwell is
compute capability 12.0, and a wheel built for an older CUDA carries no kernels
for it. Everything imports, everything looks fine, and then the first forward
pass raises `no kernel image is available for execution on the device`.

If the check below reports a mismatch, install the right wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
""",
)

code(
    r'''
def check_gpu() -> dict:
    """Report the card, and refuse a torch build that has no kernels for it."""
    import torch

    report = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device. B52 is a multi-hour run on the full dataset and is "
            "not practical on CPU."
        )

    report["device"] = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    report["capability"] = f"{major}.{minor}"
    report["total_memory_gib"] = round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
    )
    report["compiled_for"] = torch.cuda.get_arch_list()

    # The real test: does this build carry kernels for this card at all? A
    # Blackwell card with a pre-12.8 wheel fails exactly here.
    if f"sm_{major}{minor}" not in report["compiled_for"]:
        raise RuntimeError(
            f"this torch build has no kernels for sm_{major}{minor} "
            f"({report['device']}).\n"
            f"  built for: {report['compiled_for']}\n"
            "Install the CUDA 12.8 wheel:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128"
        )

    # A real allocation and a real kernel, not just a capability lookup.
    probe = torch.ones(64, 64, device="cuda")
    report["kernel_check"] = float((probe @ probe).sum().cpu())
    del probe
    torch.cuda.empty_cache()
    return report


GPU = check_gpu()
for name, value in GPU.items():
    print(f"{name:<18} {value}")
''',
)

markdown(
    """
## 5. Verify the bundle, then preflight

Two checks, in order.

**`verify.sh`** re-hashes every artefact against `MANIFEST.sha256`, then actually
loads the base checkpoint and imports the package. A file truncated by a partial
copy fails here in seconds rather than as a strange training result hours later.
It only exists in a bundle; a repository skips it.

**The preflight** is one forward and one backward pass through the real trainer.
It fails loudly if no gradient reaches the encoder — which is the entire point of
B52, and which a silent failure would turn back into the frozen baseline. It also
prints peak GPU memory, so you know the run fits before starting it.
""",
)

code(
    r'''
def stream(command: list, cwd: Path, extra_env: dict | None = None) -> int:
    """Run a command and print its output as it arrives.

    Without this a training cell shows nothing for many hours and looks hung.
    """
    environment = dict(os.environ)
    environment.update(extra_env or {})
    print("$", " ".join(str(part) for part in command), flush=True)

    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def trainer_command(*, preflight: bool) -> list:
    """The exact command line, built once so the two runs cannot drift apart."""
    command = [
        sys.executable, "-m", "rsna_knee.b52_competition_training",
        "--config", LAYOUT["config"],
        "--data-root", DATA_ROOT,
        "--labels-root", LAYOUT["labels"],
        "--series-policy", LAYOUT["series_policy"],
        "--base-checkpoint", LAYOUT["base_checkpoint"],
        "--domain-split", LAYOUT["gate"],
        "--out-root", LAYOUT["out_root"],
        "--epochs", str(EPOCHS),
    ]
    if ALL_DATA:
        command.append("--all-data")
    if preflight:
        command.append("--preflight-only")
    return command


TRAINER_ENV = {"PYTHONPATH": str(LAYOUT["source"]), "PYTHONUNBUFFERED": "1"}

verify = LAYOUT["root"] / "verify.sh"
if verify.is_file():
    if stream(["bash", verify], cwd=LAYOUT["root"]) != 0:
        raise RuntimeError("verify.sh failed; the bundle did not copy cleanly")
else:
    print("no verify.sh here (repository layout), skipping the checksum pass")

print()
print("=" * 68)
print("preflight: one forward and backward pass")
print("=" * 68)
if stream(trainer_command(preflight=True), cwd=LAYOUT["root"], extra_env=TRAINER_ENV) != 0:
    raise RuntimeError("preflight failed; do not start the run until it passes")
''',
)

markdown(
    """
## 6. Train

This is the long one. On an RTX A4500 the full-data epochs took 263 to 275
minutes each, so six epochs is roughly 27 hours. A 5090 should be meaningfully
faster, though how much depends on whether the GPU is being fed.

### Feeding a fast card

`num_workers: 0` in the config means DICOM decoding and all seven active
augmentations happen on the main thread, between GPU steps. That was chosen for
safety in Kaggle submission, where a worker crash loses the run. On a fast card
the GPU then spends its time waiting for the CPU.

If an epoch is slower than the card suggests it should be, raise it in
`config/b42_constant_area_aspect_sparse.yaml`:

```yaml
num_workers: 6
prefetch_factor: 2
```

This changes no maths — the loader is seeded through `worker_init_fn`, and the
B42 geometry contract does not cover worker count. Preflight again afterwards:
each worker is a separate process under `spawn` and costs host RAM.

### What not to change

**Do not add `--no-gradient-checkpointing` without preflighting it.** It is
identical maths and roughly 30% faster, but it keeps every encoder activation
instead of recomputing it, and needs about 15 GiB at this geometry against 1.39
GiB with checkpointing. It runs out of memory on a 16 GiB card. On a 32 GiB card
it should fit — preflight it and read the reported peak before committing.

### How many epochs

Six is what both completed runs used, and on this data it is enough. The
1,447-study run peaked at epoch 5 and fell `0.008` at six. The full-data run
peaked at 5 and came back only `0.0015` at six, with its last four epochs
spanning `0.0065` -- a plateau. Train loss kept falling while validation loss
and AUC went flat, which is memorisation, not undertraining. More epochs are
not the lever; augmentation is. Twelve was the next thing worth measuring on
more training data supports a longer schedule — but it is a measurement, not a
known result. Nothing in the archive establishes it.

### Stopping early is safe

A checkpoint is written whenever an epoch beats the best hold-out score so far,
and each one carries the whole history inside it. Interrupting the cell keeps
everything up to the last improvement.
""",
)

code(
    r'''
# Set this to True once the preflight above has passed.
RUN_TRAINING = False

if RUN_TRAINING:
    status = stream(
        trainer_command(preflight=False), cwd=LAYOUT["root"], extra_env=TRAINER_ENV
    )
    if status != 0:
        raise RuntimeError(f"training exited with status {status}")
    print()
    print("training finished")
else:
    print("RUN_TRAINING is False. Set it to True to start the run.")
''',
)

markdown(
    """
## 7. Read the result back

The run writes two things into `out_root`: `b52_best_model.pt`, which carries the
whole history inside it, and `history.json` beside it. The table below is read
back from those files rather than from this notebook's memory, so it still works
in a fresh kernel and reports what was actually saved.

`history.json` is read first because it is a few kilobytes; the checkpoint is
several hundred megabytes and is only opened for the fields the JSON does not
carry.
""",
)

code(
    r'''
def read_epoch_history(out_root: Path):
    """The per-epoch table, from the small file if it is there."""
    import pandas as pd

    out_root = Path(out_root)
    history_json = out_root / "history.json"
    if history_json.is_file():
        return pd.DataFrame(json.loads(history_json.read_text()))

    checkpoint = out_root / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"neither history.json nor {CHECKPOINT_NAME} under {out_root}; "
            "has an epoch finished yet?"
        )
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return pd.DataFrame(payload["history"])


def describe_selection(out_root: Path) -> dict:
    """What the saved checkpoint says about itself, including its governance note.

    `history.json` is written only when the whole run finishes, so a checkpoint
    from an interrupted run is still readable here.
    """
    import torch

    checkpoint = Path(out_root) / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no {CHECKPOINT_NAME} under {out_root}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in (
        "experiment",
        "selected_epoch",
        "selection_metric",
        "selection_value",
        "epochs_planned",
        "training_studies",
        "validation_studies",
        "encoder_trainable_stages",
        "augmentation_enabled",
        "gold_labels_used",
    ):
        if key in payload:
            print(f"{key:<26} {payload[key]}")

    # The encoder hash before and after is the direct evidence that the encoder
    # actually moved -- which is the whole of B52.
    before = payload.get("encoder_sha256_initial")
    after = payload.get("encoder_sha256_final")
    if before and after:
        print()
        print("encoder changed during training:", before != after)

    if "governance" in payload:
        print()
        print("governance:", payload["governance"])
    return payload


try:
    HISTORY = read_epoch_history(LAYOUT["out_root"])
    display(HISTORY[[c for c in HISTORY.columns if c != "validation_per_target_auc"]])
    print()
    SELECTION = describe_selection(LAYOUT["out_root"])
except FileNotFoundError as error:
    print(error)
''',
)

markdown(
    """
## What this run does and does not settle

**It settles** whether the training regime was the constraint. Eight experiments
searched for missing architecture on a model with a frozen encoder, no
augmentation and 3,120 optimiser steps. Anything measured that way was measured
through a floor.

**It does not settle** anything about the hidden leaderboard score. This is a
report-derived label surface, and the archive records cases where a gain here did
not survive the move to expert truth: B50 gained `+0.011` on this surface and lost
`0.012` against the 58 expert studies. Two things are different this time — the
size of the gain, and that the mechanism is the model learning to read images
rather than a small architectural edit — but neither is proof.

**It is not yet a submission.** The B42 submission loader requires
`training_studies == 4349`, and `--all-data` trains on 3,801, holding 548 back so
there is something honest to select the epoch on. A submission run trains on all
4,349 with no hold-out, using the epoch count this run establishes.
""",
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
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    written = build(Path(__file__).with_name("b52_local_full.ipynb"))
    print(f"{written}  ({len(CELLS)} cells)")
