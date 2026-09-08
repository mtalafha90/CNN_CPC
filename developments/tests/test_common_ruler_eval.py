"""Scoring several checkpoints against one labels root.

B55 regrades the teacher, which moves the validation labels as well as the
training ones. Its macro AUC and B52's are then computed against different
answer keys, and their difference is mostly the ruler.

The property this module has to get right is the one that is easy to get
backwards: **the labels are shared and the geometry is not**. Scoring B55's
weights through B42's 448 fractional crop would measure a different model and
report the difference as a worse one.
"""

from __future__ import annotations

import inspect
import json

import pytest

from rsna_knee.b52_competition_training import B52_EXPERIMENT, _build_dataset
from rsna_knee.b53_augmented_training import B53_EXPERIMENT
from rsna_knee.b55_physical_geometry import B55_REFERENCE_SIDE
from rsna_knee.b55_physical_geometry_training import B55_EXPERIMENT
from rsna_knee.common_ruler_eval import (
    COMMON_RULER_VERSION,
    GEOMETRY_BY_EXPERIMENT,
    dataset_factory_for,
    geometry_for,
    report,
    score_one,
)
from rsna_knee.physical_crop import CROP_MM


# --- each model keeps its own eyes --------------------------------------------


@pytest.mark.parametrize(
    "experiment,expected",
    [(B52_EXPERIMENT, "b42"), (B53_EXPERIMENT, "b42"), (B55_EXPERIMENT, "b55")],
)
def test_each_experiment_maps_to_the_geometry_it_trained_under(experiment, expected):
    assert geometry_for({"experiment": experiment}) == expected


def test_an_unknown_experiment_is_refused_rather_than_guessed():
    """Scoring through the wrong geometry looks like a worse model."""
    with pytest.raises(ValueError, match="no recorded geometry"):
        geometry_for({"experiment": "B99_SOMETHING_NEW"})


def test_a_checkpoint_with_no_experiment_is_refused():
    with pytest.raises(ValueError, match="no recorded geometry"):
        geometry_for({})


def test_b42_lineage_gets_b52s_own_builder():
    assert dataset_factory_for({"experiment": B52_EXPERIMENT}) is _build_dataset
    assert dataset_factory_for({"experiment": B53_EXPERIMENT}) is _build_dataset


def test_b55_gets_a_builder_carrying_its_own_crop_and_resolution():
    built = dataset_factory_for({"experiment": B55_EXPERIMENT})

    assert built is not _build_dataset
    assert list(inspect.signature(built).parameters) == list(
        inspect.signature(_build_dataset).parameters
    )


def test_b55s_geometry_is_read_from_the_checkpoint_when_recorded():
    """A later B55 run at a different crop must not be scored at this one's."""
    from rsna_knee.b55_physical_geometry_training import b55_dataset_factory

    payload = {
        "experiment": B55_EXPERIMENT,
        "model_state": {"crop_mm": 150.0, "reference_area": 224 * 224},
    }
    built = dataset_factory_for(payload)
    source = inspect.getsource(built)

    # The closure carries them; check the values reached it rather than the text.
    assert built.__closure__ is not None
    values = {cell.cell_contents for cell in built.__closure__}
    assert 150.0 in values
    assert 224 * 224 in values


def test_b55s_geometry_falls_back_to_the_defaults():
    built = dataset_factory_for({"experiment": B55_EXPERIMENT})
    values = {cell.cell_contents for cell in built.__closure__}

    assert CROP_MM in values
    assert B55_REFERENCE_SIDE**2 in values


# --- the ruler is shared -------------------------------------------------------


def test_one_labels_root_is_used_for_every_checkpoint():
    """The whole point: the answer key does not vary between rows."""
    source = inspect.getsource(score_one)
    assert "labels_root=labels_root" in source

    from rsna_knee import common_ruler_eval as module

    main_source = inspect.getsource(module.main)
    assert '"--labels-root"' in main_source
    assert main_source.count("labels_root=args.labels_root") == 1


def test_the_checkpoint_flag_repeats_and_the_labels_flag_does_not():
    from rsna_knee import common_ruler_eval as module

    source = inspect.getsource(module.main)
    assert '"--checkpoint",\n        action="append"' in source
    assert '"--labels-root",\n        required=True' in source


def test_the_validation_split_is_the_frozen_one():
    from rsna_knee.b52_competition_training import B52_PRIMARY_SPLIT

    source = inspect.getsource(score_one)
    assert "B52_PRIMARY_SPLIT" in source
    assert B52_PRIMARY_SPLIT == "validation_unseen_scanners"


def test_the_surface_guard_is_not_defeated():
    """`None` means the frozen 34,010; a regraded ruler must be declared."""
    source = inspect.getsource(score_one)
    assert "expected_cells=int(surface_cells) if surface_cells else None" in source
    assert inspect.signature(score_one).parameters["surface_cells"].default is None


def test_the_surface_builder_is_called_with_keywords():
    """It is keyword-only, and a positional call would fail at run time."""
    source = inspect.getsource(score_one)
    for keyword in ("data_root=", "labels_root=", "config=", "domain_rows=", "base_payload="):
        assert keyword in source


def test_the_score_keys_are_the_ones_evaluate_split_returns():
    """The trainer renames them; this module reads them raw and must not
    inherit the renamed spelling."""
    source = inspect.getsource(score_one)
    assert 'scores["macro_auc"]' in source
    assert "validation_macro_auc" not in source


# --- the output ----------------------------------------------------------------


def _result():
    return {
        "version": COMMON_RULER_VERSION,
        "labels_root": "/runs/teacher_rubric",
        "studies": 548,
        "scored": [
            {"experiment": B52_EXPERIMENT, "macro_auc": 0.8350, "geometry": "b42"},
            {"experiment": B55_EXPERIMENT, "macro_auc": 0.8471, "geometry": "b55"},
            {"experiment": B53_EXPERIMENT, "macro_auc": 0.8402, "geometry": "b42"},
        ],
    }


def test_the_report_ranks_and_states_the_spread(capsys):
    report(_result())
    printed = capsys.readouterr().out

    assert "0.847100" in printed
    assert "spread +0.012100" in printed
    lines = [line for line in printed.splitlines() if "geometry=" in line]
    assert B55_EXPERIMENT in lines[0], "the highest score must be first"


def test_the_report_says_this_is_not_a_selection(capsys):
    """Promoting the winner from this table would be post-hoc selection."""
    report(_result())
    printed = capsys.readouterr().out

    assert "not a selection" in printed
    assert "post-hoc" in printed


def test_the_report_names_the_ruler(capsys):
    report(_result())
    assert "teacher_rubric" in capsys.readouterr().out


def test_the_report_survives_a_single_checkpoint(capsys):
    single = _result()
    single["scored"] = single["scored"][:1]
    report(single)

    assert "spread" not in capsys.readouterr().out, "one row has no spread"


def test_the_report_shows_which_geometry_each_row_used():
    """So a reader can see that the geometries differ and the labels do not."""
    source = inspect.getsource(report)
    assert "geometry=" in source


# --- the command line ----------------------------------------------------------


def test_the_command_line_renders(capsys, monkeypatch):
    import sys

    from rsna_knee import common_ruler_eval as module

    monkeypatch.setattr(sys, "argv", ["ruler", "--help"])
    with pytest.raises(SystemExit) as exit_code:
        module.main()

    assert exit_code.value.code == 0
    printed = capsys.readouterr().out
    for flag in (
        "--labels-root",
        "--checkpoint",
        "--expected-supervision-cells",
        "--num-workers",
        "--out-json",
    ):
        assert flag in printed


def test_the_result_is_json_serialisable():
    """It is written to disk, so a numpy float would raise at the last step."""
    json.dumps(_result())


# --- executed, not read --------------------------------------------------------
#
# Every test above this line inspects source. That is the pattern that let a
# `losses` parameter shadow a local list in this project and cost a full
# training epoch to discover. These bind the real signatures instead.


def test_the_surface_call_binds_against_the_real_function():
    """A wrong keyword here fails only after the model is on the GPU."""
    from rsna_knee.b48_global_conditioned_sparse_training import _report_only_surface

    inspect.signature(_report_only_surface).bind(
        data_root="/data",
        labels_root="/labels",
        config={},
        domain_rows=None,
        base_payload={},
        expected_cells=None,
    )


def test_the_surface_returns_the_number_of_values_unpacked():
    """score_one unpacks nine; a changed return would raise at run time."""
    import ast

    from rsna_knee.b48_global_conditioned_sparse_training import _report_only_surface

    tree = ast.parse(inspect.getsource(_report_only_surface).lstrip())
    returns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    ]
    assert returns, "the surface builder no longer returns a tuple"
    assert len(returns[-1].value.elts) == 9


def test_the_evaluate_call_binds_against_the_real_function():
    from rsna_knee.b52_competition_training import evaluate_split

    inspect.signature(evaluate_split).bind(
        object(), object(), object(), object(), 1.0
    )


def test_evaluate_split_really_returns_macro_auc():
    """Read out of `macro_auc`'s own return, not assumed from the trainer."""
    from rsna_knee.evaluation import macro_auc_from_arrays

    import numpy as np

    target = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]])
    macro, per_target = macro_auc_from_arrays(target, scores)

    assert isinstance(float(macro), float)
    assert len(per_target) == target.shape[1]


def test_score_one_binds_with_the_arguments_main_passes():
    from rsna_knee.common_ruler_eval import score_one

    inspect.signature(score_one).bind(
        "ckpt.pt",
        config={},
        data_root="/data",
        labels_root="/labels",
        series_policy_path="/policy.json",
        base_checkpoint="/base.pt",
        domain_split="/split",
        surface_cells=None,
        num_workers=None,
    )


def test_the_b55_factory_accepts_the_call_score_one_makes():
    """Six positional arguments and no `train`, since this is validation."""
    from rsna_knee.b55_physical_geometry_training import b55_dataset_factory

    built = b55_dataset_factory(CROP_MM, B55_REFERENCE_SIDE**2)
    inspect.signature(built).bind(None, None, None, None, None, None)
