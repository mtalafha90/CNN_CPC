"""The inventory has to be right about which file is which.

Its whole purpose is to be trusted when a filename cannot be, so the two cases
that matter are a checkpoint whose name lies about it and a checkpoint that is
too damaged to read. Getting either wrong would be worse than not having the
tool: it would put a confident sentence behind a false answer.
"""

from __future__ import annotations

import torch

from tools.checkpoint_inventory import describe_checkpoint, duplicates, inventory


def _write(path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {}, **payload}, path)
    return path


def test_a_fine_tuned_checkpoint_is_recognised_by_its_fingerprints(tmp_path):
    path = _write(
        tmp_path / "model.pt",
        encoder_sha256_initial="aaa",
        encoder_sha256_final="bbb",
        encoder_trainable_stages=1,
    )
    record = describe_checkpoint(path)
    assert record["fine_tuned"] is True
    assert record["stages_free"] == 1


def test_a_frozen_checkpoint_is_recognised(tmp_path):
    path = _write(
        tmp_path / "model.pt",
        encoder_sha256_initial="aaa",
        encoder_sha256_final="aaa",
    )
    assert describe_checkpoint(path)["fine_tuned"] is False


def test_the_stale_frozen_flag_does_not_win(tmp_path):
    """Old checkpoints claim `encoder_frozen: True` whatever really happened."""
    path = _write(
        tmp_path / "model_finetuned.pt",
        encoder_sha256_initial="aaa",
        encoder_sha256_final="bbb",
        encoder_frozen=True,
    )
    assert describe_checkpoint(path)["fine_tuned"] is True


def test_a_misleading_filename_is_ignored(tmp_path):
    """The name says fine-tuned; the record says otherwise, and wins."""
    path = _write(
        tmp_path / "model_finetuned.pt",
        encoder_sha256_initial="aaa",
        encoder_sha256_final="aaa",
    )
    assert describe_checkpoint(path)["fine_tuned"] is False


def test_a_damaged_file_is_reported_rather_than_crashing(tmp_path):
    """A half-copied checkpoint is precisely what a merge produces."""
    path = tmp_path / "truncated.pt"
    path.write_bytes(b"not a checkpoint at all")
    record = describe_checkpoint(path)
    assert record["readable"] is False
    assert record["problem"]


def test_identical_copies_are_grouped(tmp_path):
    _write(tmp_path / "a" / "model.pt", encoder_sha256_final="x")
    _write(tmp_path / "b" / "model.pt", encoder_sha256_final="x")
    _write(tmp_path / "c" / "model.pt", encoder_sha256_final="y", completed_epochs=9)

    records = inventory([tmp_path])
    assert len(records) == 3
    groups = duplicates(records)
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 2


def test_the_same_file_reached_twice_is_counted_once(tmp_path):
    """Passing two folders that overlap must not double-count."""
    _write(tmp_path / "runs" / "model.pt", encoder_sha256_final="x")
    records = inventory([tmp_path, tmp_path / "runs"])
    assert len(records) == 1
