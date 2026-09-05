"""The five edits that wire B54 into B52's trainer.

B52 produced 0.716 and its path must not move. Every edit is gated on
`spacing_geometry_csv`, which defaults to None, so the whole of B54 is
unreachable unless the flag is given. These tests hold that gate shut, and
hold the wiring correct on the other side of it.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from rsna_knee import b52_competition_training as trainer
from rsna_knee.b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
)
from rsna_knee.b42_constant_area_aspect_sparse_training import _losses, _move_study
from rsna_knee.b50_adapted_hierarchy_mil import B50AdaptedHierarchySparseMILResidual
from rsna_knee.b54_spacing_conditioned_mil import (
    B54SpacingConditionedMIL,
    losses_with_spacing,
    move_study_with_spacing,
)


# --- the gate is shut by default ----------------------------------------------


def test_the_spacing_is_off_unless_asked_for():
    default = inspect.signature(trainer.train_b52).parameters[
        "spacing_geometry_csv"
    ].default
    assert default is None


def test_the_flag_defaults_to_none_on_the_command_line():
    source = inspect.getsource(trainer.main)
    assert "--spacing-geometry-csv" in source
    block = source.split("--spacing-geometry-csv", 1)[1].split("parser.add_argument", 1)[0]
    assert "default=None" in block


def test_every_b54_branch_is_behind_use_spacing():
    """No B54 call may run on B52's own path."""
    source = inspect.getsource(trainer.train_b52)
    for call in (
        "attach_spacing(",
        "install_spacing_conditioning(",
        "assert_conditioning_will_train(",
        "B54SpacingConditionedMIL",
    ):
        assert call in source, call
    assert "use_spacing = spacing_geometry_csv is not None" in source


def test_evaluate_split_defaults_to_the_frozen_pair():
    parameters = inspect.signature(trainer.evaluate_split).parameters
    assert parameters["move"].default is _move_study
    assert parameters["losses"].default is _losses


def test_the_dataset_class_is_unchanged_without_the_flag():
    source = inspect.getsource(trainer._build_dataset)
    assert inspect.signature(trainer._build_dataset).parameters["spacing"].default is False
    assert "B42ConstantAreaAspectDataset" in source


def test_the_dataset_builder_swaps_only_when_asked():
    from rsna_knee.b54_spacing_run import with_spacing

    wrapped = with_spacing(B42ConstantAreaAspectDataset)
    assert issubclass(wrapped, B42ConstantAreaAspectDataset)
    assert "with_spacing" in inspect.getsource(trainer._build_dataset)


# --- the model choice ---------------------------------------------------------


def test_the_residual_class_is_chosen_by_the_flag():
    source = inspect.getsource(trainer.train_b52)
    assert "B54SpacingConditionedMIL if use_spacing else" in source
    assert "B50AdaptedHierarchySparseMILResidual" in source


def test_b54_is_a_drop_in_for_b50():
    """Same constructor, so the call site needs no other change."""
    assert issubclass(B54SpacingConditionedMIL, B50AdaptedHierarchySparseMILResidual)
    b50 = inspect.signature(B50AdaptedHierarchySparseMILResidual.__init__)
    assert "adapt_hierarchy" in b50.parameters


def test_the_conditioning_is_installed_after_the_checkpoint_is_loaded():
    """Installing first adds a key the pretrained checkpoint does not have."""
    source = inspect.getsource(trainer.train_b52)
    load_at = source.index("load_phase9_checkpoint")
    install_at = source.index("install_spacing_conditioning")
    assert load_at < install_at


def test_the_preflight_runs_before_the_first_epoch():
    source = inspect.getsource(trainer.train_b52)
    assert source.index("preflight(train_index") < source.index("for epoch in range(")
    assert "B54 preflight failed" in source


# --- the optimiser, which is where this would have failed silently ------------


def test_the_optimiser_is_checked_after_it_is_built():
    source = inspect.getsource(trainer.train_b52)
    assert source.index("torch.optim.AdamW") < source.index(
        "assert_conditioning_will_train"
    )


def test_the_run_refuses_to_finish_if_the_conditioning_never_moved():
    source = inspect.getsource(trainer.train_b52)
    assert "conditioning_has_moved" in source
    assert "still exactly zero after training" in source


def test_the_backstop_runs_before_the_payload_is_written():
    source = inspect.getsource(trainer.train_b52)
    assert source.index("_check_spacing_learned()") < source.index("payload = {")


# --- the loss path ------------------------------------------------------------


def test_the_b54_loss_matches_the_frozen_one_line_for_line():
    """Only the forward call may differ; the loss arithmetic must not."""

    def meaningful(function):
        return [
            line.strip()
            for line in inspect.getsource(function).splitlines()
            if line.strip().startswith(("combined =", "local =", "total ="))
        ]

    assert meaningful(losses_with_spacing) == meaningful(_losses)


def test_only_the_forward_call_gained_an_argument():
    frozen = inspect.getsource(_losses)
    b54 = inspect.getsource(losses_with_spacing)
    assert "model(volumes, present, meta, position)" in frozen
    assert "model(volumes, present, meta, position, spacing)" in b54


def test_the_move_helper_extends_the_frozen_tuple_rather_than_replacing_it():
    source = inspect.getsource(move_study_with_spacing)
    assert "_move_study(item, device)" in source
    assert "*_move_study" in source


def test_a_study_with_no_spacing_moves_to_none():
    item = {
        "volumes": [torch.zeros(1, 3, 4, 4)],
        "slice_position": torch.zeros(1, 1),
        "present": torch.ones(1),
        "series_meta": torch.zeros(1, 3, dtype=torch.long),
        "target": torch.zeros(12),
        "weight": torch.ones(12),
    }
    assert move_study_with_spacing(item, torch.device("cpu"))[-1] is None


def test_a_study_with_spacing_moves_it_with_a_batch_dimension():
    item = {
        "volumes": [torch.zeros(1, 3, 4, 4)],
        "slice_position": torch.zeros(1, 1),
        "present": torch.ones(1),
        "series_meta": torch.zeros(1, 3, dtype=torch.long),
        "target": torch.zeros(12),
        "weight": torch.ones(12),
        "series_spacing": torch.tensor([3.3]),
    }
    spacing = move_study_with_spacing(item, torch.device("cpu"))[-1]

    assert spacing.shape == (1, 1)
    assert spacing[0, 0].item() == pytest.approx(3.3)


def test_the_tuple_is_one_longer_than_the_frozen_one():
    item = {
        "volumes": [torch.zeros(1, 3, 4, 4)],
        "slice_position": torch.zeros(1, 1),
        "present": torch.ones(1),
        "series_meta": torch.zeros(1, 3, dtype=torch.long),
        "target": torch.zeros(12),
        "weight": torch.ones(12),
    }
    device = torch.device("cpu")
    assert len(move_study_with_spacing(item, device)) == len(_move_study(item, device)) + 1


# --- resume -------------------------------------------------------------------


def test_the_loop_starts_where_the_resume_says():
    source = inspect.getsource(trainer.train_b52)
    assert "resume(" in source
    assert "for epoch in range(resumed.start_epoch, int(epochs) + 1)" in source


def test_a_checkpoint_is_written_every_epoch():
    source = inspect.getsource(trainer.train_b52)
    assert "save_checkpoint(" in source
    block = source.split("save_checkpoint(", 1)[1].split(")", 1)[0]
    for part in ("optimizer=optimizer", "scheduler=scheduler", "scaler=scaler"):
        assert part in block, part


def test_the_restored_history_is_not_thrown_away():
    source = inspect.getsource(trainer.train_b52)
    assert "history = list(resumed.history)" in source


def test_the_checkpoint_is_versioned_so_a_wrong_one_is_refused():
    source = inspect.getsource(trainer.train_b52)
    assert "version=B52_VERSION" in source


# --- what lands in the audit --------------------------------------------------


def test_the_payload_records_the_spacing_state():
    source = inspect.getsource(trainer.train_b52)
    assert '"spacing": spacing_state' in source


def test_the_state_is_empty_when_the_flag_is_absent():
    source = inspect.getsource(trainer.train_b52)
    assert 'spacing_state: dict = {"enabled": False}' in source


class _Base(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)


def test_parameter_groups_would_carry_the_conditioning():
    """The end-to-end version of the silent failure, on a stand-in model."""
    from rsna_knee.b54_spacing_conditioned_mil import (
        assert_conditioning_will_train,
        conditioning_parameters,
    )
    from rsna_knee.b54_spacing_run import install_spacing_conditioning

    base = _Base()
    install_spacing_conditioning(base)
    optimizer = torch.optim.AdamW(conditioning_parameters(base), lr=1e-3)

    assert assert_conditioning_will_train(base, optimizer)["all_reach_the_optimiser"]


# --- resume and the overwrite guard must not fight ----------------------------


def test_the_overwrite_guard_allows_a_resume():
    """The guard stops a fresh run clobbering a finished one. A resume is the
    one case where the best checkpoint legitimately already exists."""
    source = inspect.getsource(trainer.train_b52)
    assert "checkpoint_path.exists() and load_checkpoint(out) is None" in source


def test_the_guard_still_refuses_a_rerun_into_a_finished_directory(tmp_path):
    """No recovery point beside it means the run finished; refuse."""
    from rsna_knee.training_resume import load_checkpoint

    (tmp_path / trainer.B52_CHECKPOINT_NAME).write_bytes(b"finished")
    assert load_checkpoint(tmp_path) is None


def test_the_guard_stands_down_when_a_recovery_point_is_present(tmp_path):
    from rsna_knee.training_resume import load_checkpoint, save_checkpoint

    (tmp_path / trainer.B52_CHECKPOINT_NAME).write_bytes(b"best so far")
    save_checkpoint(tmp_path, epoch=2, model=_Base(), version=trainer.B52_VERSION)
    assert load_checkpoint(tmp_path) is not None
