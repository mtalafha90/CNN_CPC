"""The B51 notebook's comparison works, executed rather than read.

The point of this notebook is that two arms differ in exactly one thing. Two
mistakes would break that silently, and both are tested here: freezing by
switching to eval mode instead of clearing `requires_grad`, which would also
change dropout, and handing a frozen tensor to the optimiser, which would let
weight decay keep moving it.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

BUILDER = Path(__file__).with_name("build_b51_colab_notebook.py")
NOTEBOOK = Path(__file__).with_name("b51_adapted_hierarchy_colab.ipynb")


@pytest.fixture(scope="module")
def cells() -> list[tuple[str, str]]:
    return list(runpy.run_path(str(BUILDER))["CELLS"])


@pytest.fixture(scope="module")
def namespace(cells) -> dict:
    """Execute only the cells the B51 mechanism needs, in order."""
    scope: dict = {"__name__": "__main__", "torch": torch, "nn": nn}
    import numpy as np

    scope["np"] = np
    for _kind, text in cells:
        if "def hierarchy_parameter_names" in text or "def discordant_pair_fraction" in text:
            exec(compile(text, "<b51-cell>", "exec"), scope)
    return scope


class _Model(nn.Module):
    """The shape the notebook's model has: encoder, hierarchy, head, gate."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.global_projection = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
        self.global_classifier = nn.Linear(4, 12)
        self.sparse_head = nn.Linear(4, 12)
        self.fusion_gate = nn.Parameter(torch.zeros(12))


# --- the notebook builds ---------------------------------------------------


def test_the_notebook_is_current(cells):
    """A stale checked-in notebook would teach something the builder no longer says.

    Cell counts alone would pass while an edited cell went stale, so every cell
    is compared by its content.
    """
    written = json.loads(NOTEBOOK.read_text())["cells"]
    assert len(written) == len(cells), "rebuild the notebook from its builder"

    for index, ((kind, text), cell) in enumerate(zip(cells, written)):
        assert cell["cell_type"] == kind, f"cell {index} changed type"
        assert "".join(cell["source"]) == text, f"cell {index} is stale; rebuild the notebook"


def test_every_code_cell_parses(cells):
    import ast

    for index, (kind, text) in enumerate(cells):
        if kind == "code":
            ast.parse(text)  # raises with the offending cell's content


def test_the_notebook_says_what_it_is_not(cells):
    """A reader must not carry this subset run's absolute numbers anywhere.

    Emphasis markers are stripped before matching, so re-wrapping a line or
    bolding a phrase cannot fail this test for the wrong reason.
    """
    opening = "\n".join(text for kind, text in cells[:4] if kind == "markdown")
    scope = opening.lower().replace("*", "").replace("\n", " ")

    assert "is not the b51 protocol" in scope, "the notebook must disclaim being the real protocol"
    assert "absolute numbers mean nothing" in scope, "subset numbers must be disclaimed"
    assert "4,349" in scope, "it must say which population the real protocol uses"


# --- freezing --------------------------------------------------------------


def test_the_stub_matches_the_model_the_notebook_actually_builds(namespace, cells):
    """Every test below uses a stub, so the stub must not drift from the real model.

    If the notebook renamed `global_projection`, the stub would keep passing
    while `HIERARCHY_PREFIXES` silently selected nothing on the real model.
    """
    import ast  # noqa: PLC0415

    # The model class lives inside a notebook cell, not in the builder's own code.
    real = None
    for _kind, text in cells:
        for node in ast.walk(ast.parse(text)) if _kind == "code" else ():
            if isinstance(node, ast.ClassDef) and node.name == "HighResolutionSparseMIL":
                real = node
    assert real is not None, "the notebook no longer defines HighResolutionSparseMIL"
    assigned = {
        target.attr
        for node in ast.walk(real)
        for target in getattr(node, "targets", [])
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }

    for prefix in namespace["HIERARCHY_PREFIXES"]:
        attribute = prefix.rstrip(".")
        assert attribute in assigned, f"the real model has no {attribute!r}"

    stub = {name for name, _ in _Model().named_children()} | {"fusion_gate"}
    assert stub <= assigned, f"the stub invented {stub - assigned}"


def test_the_hierarchy_is_exactly_the_two_study_blocks(namespace):
    names = namespace["hierarchy_parameter_names"](_Model())
    assert names, "the hierarchy must be findable"
    assert all(n.startswith(("global_projection.", "global_classifier.")) for n in names)
    assert not any(n.startswith(("encoder.", "sparse_head.")) for n in names)
    assert "fusion_gate" not in names, "the gate belongs to the head, not the hierarchy"


def test_freezing_clears_gradients_without_touching_training_mode(namespace):
    """Freezing via eval() would also change dropout, so the arms would differ twice."""
    model = _Model()
    model.train()
    moved = namespace["set_hierarchy_trainable"](model, False)

    assert moved > 0
    assert model.training, "train/eval mode must be untouched"
    assert model.global_projection.training
    assert not model.global_classifier.weight.requires_grad
    assert model.encoder.weight.requires_grad, "only the hierarchy freezes"
    assert model.sparse_head.weight.requires_grad


def test_unfreezing_restores_the_same_parameters(namespace):
    model = _Model()
    frozen = namespace["set_hierarchy_trainable"](model, False)
    thawed = namespace["set_hierarchy_trainable"](model, True)
    assert frozen == thawed
    assert model.global_classifier.weight.requires_grad


def test_the_counts_describe_what_is_actually_learning(namespace):
    model = _Model()
    namespace["set_hierarchy_trainable"](model, False)
    counts = namespace["describe_trainable"](model)
    assert counts["hierarchy"] == 0
    assert counts["everything_else"] > 0

    namespace["set_hierarchy_trainable"](model, True)
    counts = namespace["describe_trainable"](model)
    assert counts["hierarchy"] > 0


# --- the optimiser ---------------------------------------------------------


def test_a_frozen_parameter_never_reaches_the_optimiser(namespace):
    """Weight decay would keep moving it, and the frozen arm would not be frozen."""
    model = _Model()
    namespace["set_hierarchy_trainable"](model, False)
    groups = namespace["build_parameter_groups"](model, 1e-4)

    assert [g["name"] for g in groups] == ["encoder_and_head"]
    handed = {id(p) for g in groups for p in g["params"]}
    for name in namespace["hierarchy_parameter_names"](model):
        parameter = dict(model.named_parameters())[name]
        assert id(parameter) not in handed, f"{name} was handed to the optimiser"


def test_the_adapted_arm_gets_a_reduced_rate(namespace):
    model = _Model()
    namespace["set_hierarchy_trainable"](model, True)
    groups = {g["name"]: g for g in namespace["build_parameter_groups"](model, 1e-4)}

    assert set(groups) == {"encoder_and_head", "study_hierarchy"}
    assert groups["encoder_and_head"]["lr"] == pytest.approx(1e-4)
    assert groups["study_hierarchy"]["lr"] == pytest.approx(
        1e-4 * namespace["HIERARCHY_LR_SCALE"]
    )
    assert namespace["HIERARCHY_LR_SCALE"] == 0.05, "the value B50 froze"


def test_no_parameter_is_updated_twice(namespace):
    model = _Model()
    namespace["set_hierarchy_trainable"](model, True)
    seen = set()
    for group in namespace["build_parameter_groups"](model, 1e-4):
        for parameter in group["params"]:
            assert id(parameter) not in seen
            seen.add(id(parameter))


# --- the ceiling -----------------------------------------------------------


def test_identical_arms_could_not_move_any_auc(namespace):
    import numpy as np

    values = np.random.default_rng(0).random((15, 12))
    assert namespace["discordant_pair_fraction"](values, values.copy()) == 0.0


def test_a_reversed_ranking_moves_every_pair(namespace):
    import numpy as np

    values = np.random.default_rng(1).random((15, 12))
    assert namespace["discordant_pair_fraction"](values, -values) == pytest.approx(1.0)


def test_rescaling_is_not_reordering(namespace):
    import numpy as np

    values = np.random.default_rng(2).random((15, 12))
    assert namespace["discordant_pair_fraction"](values, values * 4.0 + 2.0) == pytest.approx(0.0)


def test_the_gate_is_read_through_tanh(namespace):
    model = _Model()
    with torch.no_grad():
        model.fusion_gate.copy_(torch.full((12,), 0.5))
    gate = namespace["read_fusion_gate"](model)
    assert gate.shape == (12,)
    assert gate[0] == pytest.approx(float(torch.tanh(torch.tensor(0.5))))
