"""Balanced study sampling for one-GPU weakly supervised MRI training."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

DEFAULT_MAX_BATCHES_PER_EPOCH = 300
ENV_MAX_BATCHES = "RSNA_MAX_TRAIN_BATCHES"
ENV_RUN_DEADLINE = "RSNA_RUN_DEADLINE_MONOTONIC"


class TwoPoolBatchSampler(Sampler[list[int]]):
    """Deterministically mix trusted/general studies with hard work/time caps.

    For even batch sizes the trusted quota is emitted in pairs. This preserves
    the requested trusted-row fraction to within at most one pair over an epoch,
    while allowing confidence-gated pairwise ranking to see positive/negative
    trusted examples in the same minibatch. With the production batch size of
    two, batches are therefore either a trusted pair or a general pair rather
    than almost always containing exactly one trusted study.

    ``deadline_monotonic`` is the preferred production contract. The environment
    variable remains as a CLI/backward-compatible fallback, but direct calls to
    ``train_fold`` can now enforce the same deadline without relying on CLI state.
    """

    def __init__(
        self,
        trusted_mask: np.ndarray,
        batch_size: int,
        *,
        trusted_fraction: float = 0.30,
        seed: int = 2026,
        drop_last: bool = True,
        max_batches: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> None:
        mask = np.asarray(trusted_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("trusted_mask must be one-dimensional")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= trusted_fraction <= 1.0:
            raise ValueError("trusted_fraction must be in [0,1]")

        if max_batches is None:
            env_value = os.environ.get(ENV_MAX_BATCHES)
            max_batches = int(env_value) if env_value else DEFAULT_MAX_BATCHES_PER_EPOCH
        if int(max_batches) < 1:
            raise ValueError("max_batches must be >=1")

        if deadline_monotonic is None:
            deadline_text = os.environ.get(ENV_RUN_DEADLINE)
            deadline_monotonic = float(deadline_text) if deadline_text else None
        self.deadline = None if deadline_monotonic is None else float(deadline_monotonic)

        self.trusted = np.flatnonzero(mask)
        self.general = np.flatnonzero(~mask)
        if trusted_fraction > 0 and len(self.trusted) == 0:
            raise ValueError("trusted pool is empty")
        if trusted_fraction < 1 and len(self.general) == 0:
            raise ValueError("general pool is empty")
        self.n = len(mask)
        self.batch_size = int(batch_size)
        self.fraction = float(trusted_fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        full_batches = self.n // self.batch_size if self.drop_last else math.ceil(self.n / self.batch_size)
        self.n_batches = min(full_batches, int(max_batches))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def _time_available(self) -> bool:
        return self.deadline is None or time.monotonic() < self.deadline

    @staticmethod
    def _draw(pool: np.ndarray, count: int, rng: np.random.Generator, state: dict) -> list[int]:
        result = []
        while len(result) < count:
            order = state.get("order")
            pos = int(state.get("pos", 0))
            if order is None or pos >= len(order):
                order = rng.permutation(pool)
                pos = 0
            take = min(count - len(result), len(order) - pos)
            result.extend(int(x) for x in order[pos:pos + take])
            state["order"], state["pos"] = order, pos + take
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 10007 * self.epoch)
        trusted_state, general_state = {}, {}

        # Pair-friendly quota for even batch sizes. Grouping is only needed when
        # both pools are active; all-trusted/all-general cases keep exact legacy
        # behavior. For B=2 and f=0.30 this emits two trusted rows in 30% of the
        # batches (on average) and zero trusted rows in the remainder.
        pair_group = 2 if self.batch_size % 2 == 0 and 0.0 < self.fraction < 1.0 else 1
        previous_group_quota = 0

        for batch_index in range(self.n_batches):
            if not self._time_available():
                break

            cumulative_groups = int(
                math.floor(
                    ((batch_index + 1) * self.batch_size * self.fraction) / pair_group
                    + 1e-9
                )
            )
            n_trusted = (cumulative_groups - previous_group_quota) * pair_group
            previous_group_quota = cumulative_groups
            n_trusted = min(self.batch_size, max(0, int(n_trusted)))
            n_general = self.batch_size - n_trusted

            batch = []
            if n_trusted:
                batch.extend(self._draw(self.trusted, n_trusted, rng, trusted_state))
            if n_general:
                batch.extend(self._draw(self.general, n_general, rng, general_state))
            rng.shuffle(batch)
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch


def trusted_study_mask(gold_rows: np.ndarray, weights: np.ndarray, pseudo_threshold: float = 0.60) -> np.ndarray:
    gold = np.asarray(gold_rows, dtype=bool)
    weights = np.asarray(weights, float)
    if weights.ndim != 2 or weights.shape[0] != gold.shape[0]:
        raise ValueError("weights/gold shape mismatch")
    return gold | (np.nanmax(weights, axis=1) >= float(pseudo_threshold))
