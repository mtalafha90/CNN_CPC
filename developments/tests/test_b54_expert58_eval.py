"""Scoring B54 on the 58 experts, twice, from one checkpoint.

The contract worth pinning hardest is the loading order. At training the
conditioning is installed *after* the checkpoint load; here it must be *before*.
Both are correct, they are opposite, and getting either wrong raises a strict
load error that is easy to misread as a corrupted checkpoint.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from rsna_knee.b50_adapted_hierarchy_eval import _score_split
from rsna_knee.b54_expert58_eval import (
    B52_EXPERT58_MACRO,
    EXPERT58_RESOLUTION,
    VETO_DELTA,
    _report,
    _scores,
    load_b54_checkpoint,
    set_enabled,
)
from rsna_knee.b54_spacing_run import install_spacing_conditioning
from rsna_knee.constants import TARGETS


class _Base(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.plane_embedding = nn.Embedding(4, d, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, d, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, d, padding_idx=0)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _Base()


# --- the declared thresholds --------------------------------------------------


def test_the_reference_is_b52s_measured_macro():
    assert B52_EXPERT58_MACRO == 0.678247


def test_the_veto_threshold_is_the_projects_own():
    assert VETO_DELTA == -0.020


def test_the_resolution_is_stated():
    """58 studies resolve to about this; a smaller delta is not a finding."""
    assert EXPERT58_RESOLUTION == 0.03


# --- switching the arms -------------------------------------------------------


def test_switching_reports_how_many_sites_moved():
    model = _Model()
    install_spacing_conditioning(model.base)
    assert set_enabled(model, False) == 1


def test_a_model_with_no_conditioning_is_refused():
    """Silently scoring the same thing twice would look like a null result."""
    with pytest.raises(RuntimeError, match="no SpacingConditioning"):
        set_enabled(_Model(), True)


def test_both_arms_can_be_selected_in_turn():
    from rsna_knee.spacing_conditioning import SpacingConditioning

    model = _Model()
    install_spacing_conditioning(model.base)
    set_enabled(model, True)
    assert all(
        m.enabled for m in model.modules() if isinstance(m, SpacingConditioning)
    )
    set_enabled(model, False)
    assert not any(
        m.enabled for m in model.modules() if isinstance(m, SpacingConditioning)
    )


# --- the loading order, which inverts -----------------------------------------


def _body(function) -> str:
    """The source with the docstring removed.

    Searching the whole source matches prose: this module's docstrings discuss
    `load_state_dict` above the line that calls it.
    """
    import ast

    source = inspect.getsource(function).lstrip()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    statements = node.body[1:] if ast.get_docstring(node) else node.body
    lines = source.splitlines()
    return "\n".join(lines[statements[0].lineno - 1 :])


def test_the_conditioning_is_installed_before_the_load():
    """The opposite of training, because now the checkpoint carries the key."""
    body = _body(load_b54_checkpoint)
    assert body.index("install_spacing_conditioning") < body.index("load_state_dict")


def test_the_load_stays_strict():
    """Strictness is what proves the checkpoint was trained with conditioning."""
    assert _body(load_b54_checkpoint).count("strict=True") == 2


def test_installing_first_makes_the_key_loadable():
    """The concrete reason for the order, on a stand-in module."""
    trained = _Base()
    install_spacing_conditioning(trained)
    with torch.no_grad():
        trained.spacing_conditioning.projection.weight.normal_()
    state = trained.state_dict()

    without = _Base()
    with pytest.raises(RuntimeError, match="Unexpected"):
        without.load_state_dict(state, strict=True)

    with_it = _Base()
    install_spacing_conditioning(with_it)
    with_it.load_state_dict(state, strict=True)
    assert torch.allclose(
        with_it.spacing_conditioning.projection.weight,
        trained.spacing_conditioning.projection.weight,
    )


def test_a_checkpoint_without_the_spacing_is_refused():
    source = inspect.getsource(load_b54_checkpoint)
    assert "there is no" in source
    assert 'spacing_state.get("enabled")' in source


def test_an_untrained_conditioning_is_refused():
    """Still exactly zero means the ablation would compare a thing to itself."""
    source = inspect.getsource(load_b54_checkpoint)
    assert "conditioning_has_moved" in source
    assert "still exactly zero" in source


# --- the scorer B50 lends us --------------------------------------------------


def test_the_spacing_flag_defaults_off_so_b50_is_untouched():
    assert inspect.signature(_score_split).parameters["spacing"].default is False


def test_the_flag_swaps_the_dataset_and_the_forward():
    source = inspect.getsource(_score_split)
    assert "with_spacing(B42ConstantAreaAspectDataset)" in source
    assert 'item["series_spacing"]' in source


def test_no_parameter_shadows_a_local_in_the_scorer():
    """The bug that cost an epoch in the trainer, checked here before it can."""
    import ast

    tree = ast.parse(inspect.getsource(_score_split).lstrip())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    parameters = {a.arg for a in function.args.args} | {
        a.arg for a in function.args.kwonlyargs
    }
    assigned = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not (parameters & assigned), sorted(parameters & assigned)


# --- scoring ------------------------------------------------------------------


def _surface(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 58
    target = rng.integers(0, 2, (n, len(TARGETS))).astype(float)
    # Make every column measurable, so no AUC is undefined.
    target[0, :] = 1.0
    target[1, :] = 0.0
    return {
        "target": target,
        "combined": rng.random((n, len(TARGETS))),
        "weight": np.ones((n, len(TARGETS))),
    }


def test_scores_returns_a_macro_and_every_target():
    scores = _scores(_surface())

    assert 0.0 <= scores["macro_auc"] <= 1.0
    assert set(scores["per_target_auc"]) == set(TARGETS)


def test_the_report_runs_on_a_complete_result(capsys):
    """It is the only thing a person actually reads, so it must not raise."""
    on, off = _scores(_surface(1)), _scores(_surface(2))
    delta = on["macro_auc"] - off["macro_auc"]
    _report(
        {
            "expert_studies": 58,
            "arms": {"spacing_on": on, "spacing_off": off},
            "spacing_delta": delta,
            "spacing_delta_is_resolvable": abs(delta) >= EXPERT58_RESOLUTION,
            "resolution": EXPERT58_RESOLUTION,
            "b52_expert58_macro": B52_EXPERT58_MACRO,
            "delta_against_b52": on["macro_auc"] - B52_EXPERT58_MACRO,
            "veto_delta": VETO_DELTA,
            "vetoed_against_b52": False,
            "spacing_delta_per_target": {
                target: on["per_target_auc"][target] - off["per_target_auc"][target]
                for target in TARGETS
            },
        }
    )
    printed = capsys.readouterr().out
    assert "spacing on" in printed
    assert "the spacing effect" in printed
    assert "Read the macro" in printed


def test_a_small_delta_is_reported_as_unresolvable(capsys):
    scores = _scores(_surface(3))
    _report(
        {
            "expert_studies": 58,
            "arms": {"spacing_on": scores, "spacing_off": scores},
            "spacing_delta": 0.0,
            "spacing_delta_is_resolvable": False,
            "resolution": EXPERT58_RESOLUTION,
            "b52_expert58_macro": B52_EXPERT58_MACRO,
            "delta_against_b52": 0.0,
            "veto_delta": VETO_DELTA,
            "vetoed_against_b52": False,
            "spacing_delta_per_target": {t: 0.0 for t in TARGETS},
        }
    )
    assert "below the 0.03" in capsys.readouterr().out


def test_the_veto_is_announced_when_it_fires(capsys):
    scores = _scores(_surface(4))
    _report(
        {
            "expert_studies": 58,
            "arms": {"spacing_on": scores, "spacing_off": scores},
            "spacing_delta": 0.0,
            "spacing_delta_is_resolvable": False,
            "resolution": EXPERT58_RESOLUTION,
            "b52_expert58_macro": B52_EXPERT58_MACRO,
            "delta_against_b52": -0.05,
            "veto_delta": VETO_DELTA,
            "vetoed_against_b52": True,
            "spacing_delta_per_target": {t: 0.0 for t in TARGETS},
        }
    )
    assert "VETOED" in capsys.readouterr().out
