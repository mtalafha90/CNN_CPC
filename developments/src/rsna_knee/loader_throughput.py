"""Let the training runs use DataLoader workers, which cost a third of the time.

## The measurement this exists for

B53's partial run measured **3.0 to 3.3 hours per epoch** where B52 took **4.5**
for the same work on the same machine — roughly a third off — the only
difference being `num_workers=6` with the file-system sharing strategy.

That fix lives in `notebook/b52_standalone.py`, which takes `--num-workers` on
its command line. The `developments/` trainers do not: they read
`num_workers: 0` out of `config/b42_constant_area_aspect_sparse.yaml` and offer
no way to say otherwise, so every run through them still pays the old cost.
B54 v2's six epochs took 11.9 hours at `workers=0`.

This is not a lever on the score. It is a multiplier on every lever that is: a
third more experiments per week, permanently.

## Why it was zero in the first place

`num_workers: 0`, `pin_memory: false`, `persistent_workers: false` and
`prefetch_factor: 1` are B42's deliberate settings, and B39's submission
contract names `workers = 0` explicitly under "operational safety". The reason
is real — worker processes exhaust file descriptors on this dataset, because
each one holds open DICOM handles and passes tensors back through shared
memory, and the default sharing strategy allocates a descriptor per tensor.

The answer is not to avoid workers. It is `file_system` sharing, which passes a
name rather than a descriptor. B53 ran three epochs and a validation pass at
`workers=6` under it with no crash.

## What this deliberately does not change

Only `num_workers`, and the sharing strategy that makes it safe. `pin_memory`,
`persistent_workers` and `prefetch_factor` stay exactly as the config sets
them, because changing four things and measuring one is how B52 ended up unable
to say which of its changes mattered.

## Whether it can change a result

For B42-descended datasets, no. `B42ConstantAreaAspectDataset` is deterministic
— `B53_AUGMENTATION_APPLIED` measured two draws of the same study as identical
to `0.0` — so which process reads a study cannot change what the model sees.

B53's augmentation is worker-safe by construction: its draws come from a
generator seeded by run seed, epoch and study index rather than from the global
random state, precisely so that workers cannot repeat one another's numbers.

`seed_worker` in `runtime.loader_kwargs` already reseeds each worker, and is
unchanged here.

## The parent process is not the one that matters

Setting the strategy in the trainer looked sufficient and was not. The config
uses `multiprocessing_context: spawn`, and a spawned worker is a **fresh Python
process**: it imports the modules it needs and inherits none of the parent's
global state, the sharing strategy included.

The sender chooses how a tensor crosses the queue, and the sender is the worker.
So a run could print `sharing=file_system`, have every worker use
`file_descriptor` anyway, and die with

```text
RuntimeError: received 0 items of ancdata
  ... in rebuild_storage_fd
```

which is the descriptor table running out — the exact failure the strategy
exists to prevent, under a log line saying it had been prevented.

The fix is `worker_init`, which runs *inside* each worker. It is a module-level
function so `spawn` can pickle it by reference and the child re-imports this
module to call it.

`verify_worker_sharing` then asks a real worker what strategy it ended up with,
rather than asking the parent about itself. That is the same shape as B53's
augmentation preflight, and for the same reason: a claim nobody measured is how
B52 trained on identical pixels for 27 hours.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .runtime import seed_worker

#: What B53 measured, per epoch on 3,801 studies, on an RTX A4500.
MEASURED_HOURS_WITHOUT_WORKERS = 4.5
MEASURED_HOURS_WITH_WORKERS = 3.15

#: The strategy that makes workers survive this dataset. `file_descriptor` is
#: PyTorch's default and is what exhausts the descriptor table here.
SHARING_STRATEGY = "file_system"


def measured_speedup() -> float:
    """The fraction of epoch time B53's workers removed. Reported, not assumed."""
    return 1.0 - MEASURED_HOURS_WITH_WORKERS / MEASURED_HOURS_WITHOUT_WORKERS


def use_file_system_sharing() -> str:
    """Pass tensor names between processes rather than file descriptors.

    Must be called before any worker starts. Returns the strategy actually in
    force, so a caller can record it rather than trust that this worked: on a
    platform without `file_system` the request is ignored rather than raising,
    and a run that silently kept the default would be the one that dies eight
    hours in.
    """
    available = torch.multiprocessing.get_all_sharing_strategies()
    if SHARING_STRATEGY in available:
        torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
    return torch.multiprocessing.get_sharing_strategy()


def worker_init(worker_id: int) -> None:
    """Runs inside each DataLoader worker, before it produces anything.

    A module-level function so `spawn` can pickle it by reference: the child
    re-imports this module and calls it, which is the only way a setting can
    reach a process that inherits nothing.

    It sets the sharing strategy first, then delegates to `runtime.seed_worker`
    so the seeding contract every other experiment relies on is unchanged.
    """
    use_file_system_sharing()
    seed_worker(worker_id)


def _first(batch):
    """Module level, not a lambda: `spawn` has to pickle the collate function."""
    return batch[0]


class _StrategyProbe(Dataset):
    """Reports the sharing strategy of whichever process reads it.

    Module level, not a closure, because a spawned worker has to import it.
    """

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> str:
        return torch.multiprocessing.get_sharing_strategy()


def verify_worker_sharing(num_workers: int, context: str | None) -> dict:
    """Ask a real worker what strategy it has, rather than asking the parent.

    Costs a second or two and starts a genuine worker process under the same
    context the run will use. Without it a trainer can print
    `sharing=file_system`, be describing only itself, and die hours later when
    the descriptor table runs out.
    """
    if int(num_workers) < 1:
        return {"checked": False, "reason": "no workers to check"}

    loader = DataLoader(
        _StrategyProbe(),
        batch_size=1,
        num_workers=1,
        worker_init_fn=worker_init,
        multiprocessing_context=context or None,
        collate_fn=_first,
    )
    try:
        observed = next(iter(loader))
    finally:
        del loader

    if observed != SHARING_STRATEGY:
        raise RuntimeError(
            f"a DataLoader worker reports sharing strategy {observed!r}, not "
            f"{SHARING_STRATEGY!r}. Under the {context!r} start method a worker "
            "inherits nothing from this process, so setting the strategy here "
            "does not reach it, and the run would fail with 'received 0 items "
            "of ancdata' once the descriptor table filled. Re-run with "
            "--num-workers 0."
        )
    return {"checked": True, "worker_strategy": observed, "context": context}


def apply_worker_override(settings: dict, num_workers: int | None) -> dict:
    """Put `num_workers` into the settings `resolve_runtime` will read.

    `None` leaves the config alone, so every existing run is byte-identical to
    what it was. Any explicit value wins, including `0`, which is how a run
    reproduces the old behaviour on purpose rather than by default.

    Returns what was decided, for the checkpoint's audit.
    """
    if num_workers is None:
        return {
            "num_workers": int(settings.get("num_workers", 0) or 0),
            "source": "config",
            "sharing_strategy": torch.multiprocessing.get_sharing_strategy(),
        }

    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")

    settings["num_workers"] = workers
    strategy = use_file_system_sharing() if workers > 0 else (
        torch.multiprocessing.get_sharing_strategy()
    )
    if workers > 0 and strategy != SHARING_STRATEGY:
        raise RuntimeError(
            f"{workers} DataLoader workers were requested but the sharing "
            f"strategy is {strategy!r}, not {SHARING_STRATEGY!r}. On this "
            "dataset that exhausts the file-descriptor table part-way through "
            "a run, which is the failure the strategy exists to prevent. "
            "Re-run with --num-workers 0 rather than risking it."
        )
    # The parent's strategy is not the one that decides. Ask a worker.
    verified = verify_worker_sharing(
        workers, settings.get("multiprocessing_context", "spawn")
    )
    return {
        "num_workers": workers,
        "source": "command line",
        "sharing_strategy": strategy,
        "worker_verified": verified,
    }


def loader_kwargs_with_sharing(runtime, *, seed: int | None = None) -> dict:
    """`runtime.loader_kwargs`, with the worker init that sets the strategy.

    `loader_kwargs` installs `seed_worker`, which seeds but does not share.
    This replaces it with `worker_init`, which does both. Everything else the
    runtime decided is passed through untouched.
    """
    kwargs = runtime.loader_kwargs(seed=seed)
    if int(kwargs.get("num_workers", 0)) > 0:
        kwargs["worker_init_fn"] = worker_init
    return kwargs


def add_worker_argument(parser) -> None:
    """The flag, worded the same wherever it appears."""
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        # No percent sign anywhere in this string: argparse runs help text
        # through `%` formatting, so a literal `%` raises when --help renders.
        help=(
            "DataLoader worker processes. Omit to use the config's value, which "
            "is 0. B53 measured about a third off each epoch at 6 workers with "
            "file-system sharing. The dataset is deterministic, so this cannot "
            "change what the model sees."
        ),
    )


__all__ = [
    "MEASURED_HOURS_WITHOUT_WORKERS",
    "MEASURED_HOURS_WITH_WORKERS",
    "SHARING_STRATEGY",
    "add_worker_argument",
    "apply_worker_override",
    "measured_speedup",
    "use_file_system_sharing",
]
