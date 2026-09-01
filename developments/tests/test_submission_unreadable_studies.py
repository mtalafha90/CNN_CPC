"""One unreadable study must not destroy a 1,300-study hidden run.

The B51 hidden submission threw an exception Kaggle would not show. The code was
identical to the commit run that had just passed on the three visible example
studies, so the failure was in the data, and the frozen path has four places
where one bad study ends everything:

```text
a study whose series all have an unrecognised plane   ValueError before inference
a series directory that cannot be found               FileNotFoundError, strict_dicom
a series that cannot be decoded                       the decoder's exception, strict_dicom
a study whose probabilities come out non-finite       RuntimeError
```

None can fire on three clean studies. Across roughly 7,000 hidden series, at
least one is close to certain. These tests run the real shard loop against a
dataset that fails on chosen rows, and check both halves: `raise` still ends the
run exactly as the frozen path did, and `fallback` finishes with a row for every
study and an honest record of which ones it had to guess.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.b42_constant_area_aspect_sparse_submission_dualgpu_fast import (
    DEFAULT_FALLBACK_PROBABILITY,
    ON_UNREADABLE_FALLBACK,
    ON_UNREADABLE_RAISE,
    _infer_shard,
)
from rsna_knee.constants import TARGETS

CPU = torch.device("cpu")
OFFSETS = 3
SERIES = 2
SLICES = 32


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _Model:
    """Returns fixed logits. The shard's job is control flow, not arithmetic.

    The default is deliberately not 0: sigmoid(0) is 0.5, which is the fallback
    constant, and a stub whose real output collides with the fallback makes every
    "this row was really predicted" assertion vacuous.
    """

    def __init__(self, value: float = 2.0):
        self.value = value

    def __call__(self, volumes, present, series_meta, position):
        return _Output(torch.full((1, len(TARGETS)), float(self.value)))


class _Dataset:
    """A test dataset that fails on chosen rows, the way hidden data does."""

    def __init__(self, count: int, *, failing: dict[int, Exception] | None = None):
        self.count = count
        self.failing = failing or {}

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        if index in self.failing:
            raise self.failing[index]
        return {
            "study_uid": f"study-{index}",
            "volumes": [torch.zeros(OFFSETS, SLICES, 3, 64, 48) for _ in range(SERIES)],
            "slice_position": torch.zeros(SERIES, OFFSETS, SLICES),
            "present": torch.ones(SERIES),
            "series_meta": torch.zeros(SERIES, 3, dtype=torch.long),
        }


def _run(dataset, indices, *, mode, uids=None, model=None):
    return _infer_shard(
        rank=0,
        indices=indices,
        model=model or _Model(),
        dataset=dataset,
        global_started=torch.zeros(1).item(),
        max_hours=8.25,
        reserve_minutes=30.0,
        uids=uids or [f"study-{i}" for i in range(len(dataset))],
        on_unreadable=mode,
        device=CPU,
    )


# --- the frozen behaviour is still available and still strict ---------------


def test_raise_mode_still_ends_the_run_on_one_bad_study():
    """B42's 0.714 hidden run was made under this. It must stay reachable."""
    dataset = _Dataset(4, failing={2: FileNotFoundError("missing series s/42")})
    with pytest.raises(FileNotFoundError, match="missing series"):
        _run(dataset, [0, 1, 2, 3], mode=ON_UNREADABLE_RAISE)


def test_raise_is_the_default():
    """A caller that passes nothing gets exactly the frozen behaviour."""
    import inspect  # noqa: PLC0415

    default = inspect.signature(_infer_shard).parameters["on_unreadable"].default
    assert default == ON_UNREADABLE_RAISE


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="on_unreadable must be one of"):
        _run(_Dataset(1), [0], mode="carry_on_regardless")


# --- fallback finishes the run ---------------------------------------------


def test_fallback_predicts_the_rest_of_the_studies():
    """The property the whole change exists for."""
    dataset = _Dataset(4, failing={2: FileNotFoundError("missing series s/42")})
    rows, failures = _run(dataset, [0, 1, 2, 3], mode=ON_UNREADABLE_FALLBACK)

    assert [row[0] for row in rows] == [0, 1, 2, 3], "every study needs a row"
    assert len(failures) == 1
    assert failures[0]["index"] == 2
    assert failures[0]["study_uid"] == "study-2"
    assert "FileNotFoundError" in failures[0]["error"]


def test_the_fallback_row_is_the_constant_and_the_others_are_not():
    dataset = _Dataset(3, failing={1: RuntimeError("row 1 produced invalid probabilities")})
    rows, _ = _run(dataset, [0, 1, 2], mode=ON_UNREADABLE_FALLBACK, model=_Model(2.0))

    by_index = {row[0]: row[2] for row in rows}
    assert np.allclose(by_index[1], DEFAULT_FALLBACK_PROBABILITY)
    for good in (0, 2):
        assert not np.allclose(by_index[good], DEFAULT_FALLBACK_PROBABILITY)
        assert by_index[good].shape == (len(TARGETS),)


def test_a_fallback_row_has_the_right_width():
    """A short row would break the submission frame, not just one study."""
    dataset = _Dataset(1, failing={0: ValueError("nothing readable")})
    rows, _ = _run(dataset, [0], mode=ON_UNREADABLE_FALLBACK)
    assert rows[0][2].shape == (len(TARGETS),)
    assert rows[0][3] == [], "a study that was never read reports no series shapes"


def test_every_failure_mode_the_hidden_set_can_produce_is_survived():
    """The four raise sites, exercised together rather than argued about."""
    dataset = _Dataset(6, failing={
        0: FileNotFoundError("missing series"),
        1: RuntimeError("B42 row 1 has no ragged series tensors"),
        2: ValueError("cannot decode JPEG2000"),
        3: RuntimeError("B42 row 3 produced invalid probabilities"),
    })
    rows, failures = _run(dataset, list(range(6)), mode=ON_UNREADABLE_FALLBACK)

    assert len(rows) == 6
    assert sorted(record["index"] for record in failures) == [0, 1, 2, 3]


def test_the_uid_is_recovered_even_when_the_study_never_loaded():
    """`item["study_uid"]` is unreachable when `dataset[index]` is what threw."""
    dataset = _Dataset(2, failing={0: FileNotFoundError("gone")})
    rows, failures = _run(
        dataset, [0, 1], mode=ON_UNREADABLE_FALLBACK, uids=["real-uid-a", "real-uid-b"]
    )
    assert failures[0]["study_uid"] == "real-uid-a"
    assert rows[0][1] == "real-uid-a", "the submission row must carry the real UID"


# --- memory, the other thing that only appears at scale --------------------


class _OnceOutOfMemory(_Dataset):
    """Fails with an OOM the first time, succeeds the second."""

    def __init__(self, count: int, index: int):
        super().__init__(count)
        self.index = index
        self.hits = 0

    def __getitem__(self, index: int) -> dict:
        if index == self.index:
            self.hits += 1
            if self.hits == 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
        return super().__getitem__(index)


def test_an_out_of_memory_study_is_retried_once_before_being_given_up():
    """The allocator cache is kept warm between studies, so a large study can
    fail where it would have fitted from clean. One clean retry costs a study's
    time and can save the row."""
    dataset = _OnceOutOfMemory(3, index=1)
    rows, failures = _run(dataset, [0, 1, 2], mode=ON_UNREADABLE_FALLBACK)

    assert dataset.hits == 2, "the study was not retried"
    assert failures == [], "the retry succeeded, so nothing should have fallen back"
    assert not np.allclose(rows[1][2], DEFAULT_FALLBACK_PROBABILITY)


def test_a_study_that_is_out_of_memory_twice_falls_back():
    dataset = _Dataset(2, failing={0: torch.OutOfMemoryError("CUDA out of memory")})
    rows, failures = _run(dataset, [0, 1], mode=ON_UNREADABLE_FALLBACK)
    assert len(failures) == 1
    assert "OutOfMemoryError" in failures[0]["error"]
    assert np.allclose(rows[0][2], DEFAULT_FALLBACK_PROBABILITY)
