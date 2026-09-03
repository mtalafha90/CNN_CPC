"""Survive a crash thirty hours into a run.

`_save_recovery` already writes `recovery_latest.pt` after every epoch, and it
writes the model weights, the epoch number and the history. Three things are
missing, and each on its own is enough to lose the run:

```text
nothing reads it     there is no restore path anywhere in the codebase, so
                     the file has never been used for what it is named after
half the state       optimiser moments, the learning-rate schedule, the AMP
                     loss scale and every random generator are absent. Loading
                     the weights alone restarts Adam from zero and the schedule
                     from step zero, which is not the run continuing
one copy             the save is not atomic. A crash *during* the write leaves
                     a truncated file where the only recovery point was
```

Runs here are already nineteen hours. The single run that carries the spacing
conditioning, the native grid and the rebuilt teacher is longer than that. A
resume that does not restore the optimiser is not a resume; it is a warm start
with a different trajectory, which quietly makes the run unreproducible.

## What is guaranteed

Restoring is exact for everything torch can serialise: parameters, optimiser
state, scheduler step, gradient scaler, and the Python, NumPy and torch random
generators including CUDA. Data-loader worker order is *not* restored, because
`DataLoader` holds no resumable position; an epoch boundary is the only safe
place to stop, which is where the save happens.

## What is refused

A checkpoint whose `version` does not match the running experiment. Loading
B47 weights into a B49 model would either raise deep in `load_state_dict` or,
worse, succeed on a partial match. The version is checked first and the error
names both sides.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

RESUME_VERSION = "training_resume_v1"
RECOVERY_NAME = "recovery_latest.pt"


def rng_state() -> dict:
    """Every generator that can change what the next epoch does."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict | None) -> dict:
    """Restore what is present and report what was skipped, rather than raising.

    A run that moves between machines will not have the same device count, and
    refusing to resume over that would cost more than the exactness is worth.
    The caller is told, so a partial restore is never silent.
    """
    skipped: list[str] = []
    if not state:
        return {"restored": [], "skipped": ["all"]}

    restored: list[str] = []
    if state.get("python") is not None:
        random.setstate(state["python"])
        restored.append("python")
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
        restored.append("numpy")
    if state.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
        restored.append("torch")

    cuda = state.get("cuda")
    if cuda is None:
        skipped.append("cuda")
    elif not torch.cuda.is_available():
        skipped.append("cuda: no device")
    elif len(cuda) != torch.cuda.device_count():
        skipped.append(
            f"cuda: saved on {len(cuda)} device(s), running on "
            f"{torch.cuda.device_count()}"
        )
    else:
        torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8) for x in cuda])
        restored.append("cuda")
    return {"restored": restored, "skipped": skipped}


def save_checkpoint(
    out: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    version: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    history: list[dict] | None = None,
    extra: dict | None = None,
    name: str = RECOVERY_NAME,
) -> Path:
    """Write a resumable point, atomically.

    The write goes to a temporary file in the same directory and is then moved
    into place with `os.replace`, which is atomic on every platform this runs
    on. A crash mid-write therefore destroys the temporary file and leaves the
    previous recovery point intact, instead of truncating the only copy.
    """
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "resume_version": RESUME_VERSION,
        "version": str(version),
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state(),
        "history": list(history or []),
        "extra": dict(extra or {}),
    }
    final = directory / name
    temporary = directory / f"{name}.writing"
    torch.save(payload, temporary)
    os.replace(temporary, final)
    return final


def load_checkpoint(
    out: str | Path, *, name: str = RECOVERY_NAME, map_location: Any = "cpu"
) -> dict | None:
    """The saved point, or None when there is nothing to resume from."""
    path = Path(out) / name
    if not path.is_file():
        return None
    # These files are written by this module, so the richer loader is correct
    # here; `weights_only=True` cannot carry the history or the RNG state.
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass
class ResumeState:
    """What a caller needs to continue the loop."""

    restored: bool
    start_epoch: int
    history: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    rng: dict = field(default_factory=dict)
    parts: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.restored:
            return "no recovery point; starting from epoch 1"
        return (
            f"resumed after epoch {self.start_epoch - 1}, restoring "
            f"{', '.join(self.parts) or 'weights only'}"
        )


def resume(
    out: str | Path,
    *,
    model: torch.nn.Module,
    version: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    name: str = RECOVERY_NAME,
    map_location: Any = "cpu",
    strict: bool = True,
) -> ResumeState:
    """Restore a run in place, or report that there is nothing to restore.

    `start_epoch` is the next epoch to run, so a caller writes
    `for epoch in range(state.start_epoch, epochs + 1)` and needs no other
    change to its loop.
    """
    payload = load_checkpoint(out, name=name, map_location=map_location)
    if payload is None:
        return ResumeState(restored=False, start_epoch=1)

    saved_version = str(payload.get("version", ""))
    if saved_version != str(version):
        raise ValueError(
            f"refusing to resume: {Path(out) / name} was written by "
            f"{saved_version!r} and this run is {version!r}. Move or delete it "
            "if the restart is deliberate."
        )

    parts: list[str] = []
    model.load_state_dict(payload["model_state"], strict=strict)
    parts.append("model")
    for component, key, label in (
        (optimizer, "optimizer_state", "optimizer"),
        (scheduler, "scheduler_state", "scheduler"),
        (scaler, "scaler_state", "scaler"),
    ):
        if component is not None and payload.get(key) is not None:
            component.load_state_dict(payload[key])
            parts.append(label)

    rng = set_rng_state(payload.get("rng_state"))
    if rng["restored"]:
        parts.append("rng")

    return ResumeState(
        restored=True,
        start_epoch=int(payload["epoch"]) + 1,
        history=list(payload.get("history", [])),
        extra=dict(payload.get("extra", {})),
        rng=rng,
        parts=parts,
    )
