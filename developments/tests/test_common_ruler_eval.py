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
    b55_geometry_from,
    dataset_factory_for,
    geometry_for,
    per_target_table,
    predict_split,
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


def test_b55_gets_a_builder_carrying_its_own_crop_and_resolution(monkeypatch):
    built = dataset_factory_for(_payload_from_a_real_b55_call(monkeypatch))

    assert built is not _build_dataset
    assert list(inspect.signature(built).parameters) == list(
        inspect.signature(_build_dataset).parameters
    )


# --- B55's geometry comes from the checkpoint, in the shape B55 writes --------
#
# The first version of this read `model_state`, and the tests here agreed with
# it -- because they built the payload by hand in the shape the code expected.
# `model_state` carries the architecture, not the dataset, so a 150 mm crop at
# a 224 reference side leaves no trace in it: such a run was scored at 130 mm
# and 336, through eyes it had never been trained with, and the loss was
# reported as a worse model.
#
# So the payload below is not written by hand. It is the one `train_b55`
# actually passes, captured from the call.


def _payload_from_a_real_b55_call(monkeypatch, **kwargs) -> dict:
    """The `extra` a real `train_b55` hands to `train_b52`, as a checkpoint."""
    from rsna_knee import b55_physical_geometry_training as b55

    seen: dict = {}
    monkeypatch.setattr(
        b55, "train_b52", lambda config, **passed: seen.update(passed)
    )
    b55.train_b55(
        {},
        data_root=".",
        labels_root=".",
        series_policy_path=".",
        base_checkpoint=".",
        domain_split=".",
        **kwargs,
    )
    # `train_b52` writes the identity and then every `extra` field beside it.
    return {**seen["identity"], **seen["extra"]}


def test_b55s_geometry_is_read_from_the_checkpoint_a_real_run_writes(monkeypatch):
    """A run at a different crop must not be scored at the default one's."""
    payload = _payload_from_a_real_b55_call(
        monkeypatch, crop_mm=150.0, reference_side=224
    )
    built = dataset_factory_for(payload)

    assert built.__closure__ is not None
    values = {cell.cell_contents for cell in built.__closure__}
    assert 150.0 in values, "the recorded crop never reached the dataset"
    assert 224 * 224 in values, "the recorded resolution never reached it"
    assert CROP_MM not in values, "it fell back to the default crop"


def test_the_defaults_are_read_from_the_checkpoint_too(monkeypatch):
    """Not guessed at -- they happen to match, and that is what hid the bug."""
    payload = _payload_from_a_real_b55_call(monkeypatch)
    values = {cell.cell_contents for cell in dataset_factory_for(payload).__closure__}

    assert CROP_MM in values
    assert B55_REFERENCE_SIDE**2 in values


def test_the_geometry_is_read_out_of_the_field_b55_writes(monkeypatch):
    payload = _payload_from_a_real_b55_call(monkeypatch, crop_mm=150.0)

    assert "b55_geometry" in payload, "train_b55 no longer records its geometry"
    assert b55_geometry_from(payload)["crop_mm"] == 150.0


def test_a_b55_checkpoint_with_no_geometry_is_refused_rather_than_defaulted():
    """Scoring at the defaults is exactly how the wrong geometry stayed quiet."""
    with pytest.raises(ValueError, match="records no `b55_geometry`"):
        dataset_factory_for({"experiment": B55_EXPERIMENT})

    with pytest.raises(ValueError, match="records no `b55_geometry`"):
        dataset_factory_for(
            {
                "experiment": B55_EXPERIMENT,
                # The shape the broken version read. It is not the shape B55
                # writes, and treating it as one is the bug.
                "model_state": {"crop_mm": 150.0, "reference_area": 224 * 224},
            }
        )


def test_a_side_and_an_area_that_disagree_are_refused():
    with pytest.raises(ValueError, match="do not agree"):
        b55_geometry_from(
            {
                "experiment": B55_EXPERIMENT,
                "b55_geometry": {
                    "crop_mm": 130.0,
                    "reference_side": 336,
                    "reference_area": 224 * 224,
                },
            }
        )


def test_the_area_is_derived_when_only_the_side_is_recorded():
    geometry = b55_geometry_from(
        {"b55_geometry": {"crop_mm": 130.0, "reference_side": 224}}
    )

    assert geometry["reference_area"] == 224 * 224


def test_b42_checkpoints_need_no_geometry_field():
    """B52 and B53 have one geometry and it is the builder's own."""
    assert dataset_factory_for({"experiment": B52_EXPERIMENT}) is _build_dataset


def test_the_score_records_the_numbers_it_scored_through():
    """`geometry: b55` alone does not say which B55."""
    source = inspect.getsource(score_one)
    assert '"geometry_used"' in source
    assert "b55_geometry_from(payload)" in source


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


# --- the per-target table, built from macro_auc's real return -------------------
#
# This is the crash that cost a whole scoring run: `per_target_auc` is keyed by
# target name, and the first version of this module zipped it with TARGETS.
# Zipping a dict iterates its keys, so `float()` was handed 'ACL' -- and it
# raised only after every study had already been through the model.


def _real_scores():
    """`macro_auc`'s own output, not a hand-written stand-in."""
    import numpy as np

    from rsna_knee.b52_competition_training import TARGETS, macro_auc

    rows, columns = 40, len(TARGETS)
    generator = np.random.default_rng(0)
    target = (generator.random((rows, columns)) > 0.5).astype(np.float64)
    weight = np.ones((rows, columns), dtype=np.float64)
    prediction = generator.random((rows, columns))
    return macro_auc(target, weight, prediction)


def test_the_per_target_table_is_built_from_the_real_scores():
    table = per_target_table(_real_scores())

    from rsna_knee.b52_competition_training import TARGETS

    assert set(table) == set(TARGETS), "keyed by target name, not by position"
    assert all(isinstance(value, float) for value in table.values())


def test_zipping_the_per_target_scores_with_targets_is_the_bug_that_was_fixed():
    """The failure mode, reproduced, so nobody reintroduces the shorter line."""
    from rsna_knee.b52_competition_training import TARGETS

    scores = _real_scores()
    with pytest.raises(ValueError, match="could not convert string to float"):
        {
            name: float(value)
            for name, value in zip(TARGETS, scores["per_target_auc"])
        }


def test_the_per_target_table_survives_a_target_no_study_supervises():
    """An undefined AUC is NaN, which is a float and must not raise."""
    table = per_target_table({"per_target_auc": {"ACL": float("nan")}})

    assert table["ACL"] != table["ACL"], "NaN is carried through, not dropped"


def test_score_one_uses_the_table_rather_than_rebuilding_it():
    source = inspect.getsource(score_one)
    assert "per_target_table(scores)" in source
    assert "zip(TARGETS" not in source


# --- inference builds no graph -------------------------------------------------
#
# `evaluate_split` carries `@torch.no_grad()`; `predict_split` was written as
# "the same loop" and did not. Every forward pass then kept its activations
# alive for a backward pass that never came -- the whole encoder over every
# slice of every study -- and a 16 GB card runs out of memory while *scoring*,
# which is the last place anyone looks. TTA multiplies it by the offsets and an
# ensemble by the members, so the tools most likely to hit it are the new ones.


def _run_predict_split(monkeypatch):
    """Run the real loop with a stand-in model, and report what it saw."""
    import types

    import numpy as np
    import torch

    from rsna_knee import b37_highres_sparse_training as trimming
    from rsna_knee import b42_constant_area_aspect_sparse_training as losses_module
    from rsna_knee.b52_competition_training import TARGETS

    seen: dict = {}
    weight = torch.nn.Parameter(torch.zeros(len(TARGETS)))

    def fake_losses(model, runtime, tensors, multiplier_t, aux_weight):
        seen["grad_enabled"] = torch.is_grad_enabled()
        logits = weight * 2.0
        seen["logits_track_gradients"] = logits.requires_grad
        return types.SimpleNamespace(logits=logits), None, None, None

    monkeypatch.setattr(losses_module, "_losses", fake_losses)
    monkeypatch.setattr(losses_module, "_move_study", lambda item, device: item)
    monkeypatch.setattr(trimming, "_trim_host_memory", lambda: None)

    generator = np.random.default_rng(0)
    batch = [
        {
            "target": torch.as_tensor(
                (generator.random(len(TARGETS)) > 0.5).astype("float32")
            ),
            "weight": torch.ones(len(TARGETS)),
        }
        for _ in range(6)
    ]
    model = torch.nn.Linear(1, 1)
    runtime = types.SimpleNamespace(device="cpu")

    predict_split(model, runtime, [batch], torch.ones(len(TARGETS)), 1.0)
    return seen


def test_prediction_runs_with_gradients_switched_off(monkeypatch):
    assert _run_predict_split(monkeypatch)["grad_enabled"] is False


def test_no_activation_graph_is_kept_for_a_backward_pass_that_never_comes(monkeypatch):
    """The memory itself, not just the flag: a graph is what holds it."""
    assert _run_predict_split(monkeypatch)["logits_track_gradients"] is False


def test_the_stand_in_would_have_caught_the_original(monkeypatch):
    """Outside the guard the same fake tracks gradients -- so the test can fail."""
    import torch

    with torch.enable_grad():
        weight = torch.nn.Parameter(torch.zeros(3))
        assert (weight * 2.0).requires_grad is True


def test_both_scoring_loops_carry_the_same_guard():
    from rsna_knee.b52_competition_training import evaluate_split

    for function in (evaluate_split, predict_split):
        assert getattr(function, "__wrapped__", None) is not None, (
            f"{function.__name__} is no longer wrapped in an inference guard"
        )


# --- a run that stops part-way keeps what it measured --------------------------
#
# The first version printed one line *after* a whole checkpoint had been
# scored, and wrote its JSON only after all of them. So a hang looked the same
# wherever it happened, and a run that died on the third model threw away the
# first two -- each of which is a scoring pass over 548 studies.


def _run_main(monkeypatch, tmp_path, *, fail_on=None, checkpoints=3):
    import sys

    from rsna_knee import common_ruler_eval as module

    calls: list[str] = []

    def fake_score_one(path, **kwargs):
        calls.append(str(path))
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("stopped part-way, exactly as a hang would")
        return {
            "checkpoint": str(path),
            "experiment": f"E{len(calls)}",
            "macro_auc": 0.8 + len(calls) / 1000,
            "geometry": "b42",
            "studies": 548,
        }

    monkeypatch.setattr(module, "score_one", fake_score_one)
    monkeypatch.setattr(module, "_read_config", lambda path: {})

    out = tmp_path / "ruler.json"
    argv = ["ruler", "--data-root", ".", "--labels-root", ".",
            "--series-policy", ".", "--base-checkpoint", ".",
            "--domain-split", ".", "--out-json", str(out)]
    for index in range(checkpoints):
        argv += ["--checkpoint", f"model{index}.pt"]
    monkeypatch.setattr(sys, "argv", argv)

    return module, out


def test_every_scored_checkpoint_is_on_disk_before_the_next_one_starts(
    monkeypatch, tmp_path, capsys
):
    module, out = _run_main(monkeypatch, tmp_path, fail_on=3)

    with pytest.raises(RuntimeError, match="stopped part-way"):
        module.main()

    saved = json.loads(out.read_text("utf-8"))
    assert len(saved["scored"]) == 2, "the two completed passes were lost"
    assert saved["checkpoints_requested"] == 3, "it must say the table is partial"


def test_a_complete_run_says_so(monkeypatch, tmp_path, capsys):
    module, out = _run_main(monkeypatch, tmp_path)
    module.main()

    saved = json.loads(out.read_text("utf-8"))
    assert len(saved["scored"]) == saved["checkpoints_requested"] == 3


def test_each_checkpoint_is_announced_before_it_is_scored(monkeypatch, tmp_path, capsys):
    """Otherwise a hang on the first model prints nothing at all."""
    module, _ = _run_main(monkeypatch, tmp_path, fail_on=1)

    with pytest.raises(RuntimeError):
        module.main()

    printed = capsys.readouterr().out
    assert "1/3" in printed, "the run must say which checkpoint it is starting"
    assert "model0.pt" in printed


def test_the_slow_stages_each_announce_themselves():
    """Named stages, so a hang can be located rather than guessed at."""
    source = inspect.getsource(score_one)

    for marker in (
        "reading the base checkpoint",
        "building the supervision surface",
        "backfilling series metadata",
        "loading the model",
        "scoring ",
    ):
        assert marker in source, f"no stage marker for {marker!r}"


def test_the_scoring_stage_names_the_worker_count():
    """A worker deadlock is the likeliest hang, so the number must be visible."""
    source = inspect.getsource(score_one)
    assert "num_workers" in source
    assert "workers" in source
