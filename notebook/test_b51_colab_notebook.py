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


WANTED = (
    "def hierarchy_parameter_names",
    "def discordant_pair_fraction",
    "def load_report_labels",
    "def target_balance_multipliers",
)

# Stand-ins for the notebook's twelve findings. Every function under test is
# generic over this list, and a separate test pins the real length at twelve.
FAKE_TARGETS = [f"finding_{index}" for index in range(12)]


@pytest.fixture(scope="module")
def namespace(cells) -> dict:
    """Execute only the cells the B51 mechanism needs, in order."""
    import numpy as np
    import pandas as pd

    scope: dict = {
        "__name__": "__main__",
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "np": np,
        "pd": pd,
        "Path": Path,
        "TARGETS": list(FAKE_TARGETS),
        "DEVICE": torch.device("cpu"),
    }
    for _kind, text in cells:
        if any(marker in text for marker in WANTED):
            exec(compile(text, "<b51-cell>", "exec"), scope)
    return scope


def _export_frame(states: dict | None = None):
    """A label export shaped exactly like the one b23_llm_labels.py writes."""
    import pandas as pd  # noqa: PLC0415

    fixed = {
        "positive": (0.97, 0.90),
        "negated": (0.03, 0.90),
        "uncertain": (0.50, 0.25),
        "conflict": (0.50, 0.20),
        "unmentioned": (0.50, 0.00),
    }
    states = states or {}
    rows = []
    for study in ("weak-a", "weak-b"):
        row = {"StudyInstanceUID": study}
        for target in FAKE_TARGETS:
            state = states.get((study, target), "positive")
            probability, confidence = fixed[state]
            row[target] = probability
            row[f"{target}__confidence"] = confidence
            row[f"{target}__state"] = state
        rows.append(row)
    return pd.DataFrame(rows)


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


def _bound_names(tree) -> set[str]:
    """Every name this tree binds, at any nesting level."""
    import ast  # noqa: PLC0415

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            arguments = getattr(node, "args", None)
            if arguments is not None:
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                ):
                    names.add(argument.arg)
                for extra in (arguments.vararg, arguments.kwarg):
                    if extra is not None:
                        names.add(extra.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def test_no_cell_uses_a_name_the_notebook_never_defines(cells):
    """The failure mode that a parse check cannot see.

    Every code cell parses happily while referring to a name that does not
    exist; the NameError only appears when that cell is executed, which for a
    training cell may be an hour into a Colab session.
    """
    import ast  # noqa: PLC0415
    import builtins  # noqa: PLC0415

    available = set(dir(builtins)) | {"__name__", "__file__", "get_ipython"}
    used: dict[str, int] = {}

    for index, (kind, text) in enumerate(cells):
        if kind != "code":
            continue
        tree = ast.parse(text)
        available |= _bound_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.setdefault(node.id, index)

    unknown = sorted(name for name in used if name not in available)
    assert not unknown, "names used but never defined: " + ", ".join(
        f"{name} (cell {used[name]})" for name in unknown
    )


def _top_level_loads(tree) -> set[str]:
    """Names read by code that runs when the cell runs, ignoring function bodies.

    A function may safely refer to something defined in a later cell, because
    the name is resolved when it is called. Code at the top level of a cell may
    not.
    """
    import ast  # noqa: PLC0415

    names: set[str] = set()
    stack = list(getattr(tree, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # runs later, not now
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return names


def test_definitions_come_before_the_cells_that_run_them(cells):
    """Cells run top to bottom, so a definition placed after its use is a crash.

    B51 inserts its definitions ahead of the inherited training section rather
    than appending them, and this is what holds that ordering in place.
    """
    import ast  # noqa: PLC0415
    import builtins  # noqa: PLC0415

    available = set(dir(builtins)) | {"__name__", "__file__", "get_ipython"}
    for index, (kind, text) in enumerate(cells):
        if kind != "code":
            continue
        tree = ast.parse(text)
        # A cell's own imports and definitions count as available to it; the
        # question here is only whether an earlier cell needs a later one.
        available |= _bound_names(tree)
        too_early = sorted(_top_level_loads(tree) - available)
        assert not too_early, (
            f"cell {index} runs before {', '.join(too_early)} is defined"
        )


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


# --- the report labels -----------------------------------------------------


def test_the_notebook_has_twelve_findings(cells):
    """The stub target list is only honest if the real one is the same length."""
    import ast  # noqa: PLC0415

    for kind, text in cells:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "TARGETS" for t in node.targets
                )
                and isinstance(node.value, ast.List)
            ):
                assert len(node.value.elts) == len(FAKE_TARGETS) == 12
                return
    pytest.fail("the notebook no longer defines TARGETS as a list")


def test_a_well_formed_export_loads(namespace, tmp_path):
    path = tmp_path / "training_targets.csv"
    _export_frame().to_csv(path, index=False)
    frame = namespace["load_report_labels"](path)
    assert list(frame["StudyInstanceUID"]) == ["weak-a", "weak-b"]


def test_a_missing_export_says_how_to_make_one(namespace, tmp_path):
    with pytest.raises(FileNotFoundError, match="b23_llm_labels"):
        namespace["load_report_labels"](tmp_path / "training_targets.csv")


def test_an_export_missing_confidence_is_refused(namespace, tmp_path):
    """Probabilities without confidence are exactly the dangerous case."""
    frame = _export_frame().drop(columns=[f"{FAKE_TARGETS[0]}__confidence"])
    path = tmp_path / "training_targets.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        namespace["load_report_labels"](path)


def test_a_repeated_study_is_refused(namespace, tmp_path):
    import pandas as pd  # noqa: PLC0415

    frame = _export_frame()
    path = tmp_path / "training_targets.csv"
    pd.concat([frame, frame.head(1)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than once"):
        namespace["load_report_labels"](path)


def test_an_impossible_confidence_is_refused(namespace, tmp_path):
    frame = _export_frame()
    frame.loc[0, f"{FAKE_TARGETS[0]}__confidence"] = 4.2
    path = tmp_path / "training_targets.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        namespace["load_report_labels"](path)


def test_an_unmentioned_finding_becomes_a_blank(namespace):
    """Report silence is not a negative -- the single most important rule here.

    The export stores an unmentioned cell as probability 0.50, which is a real
    number an ordinary loss would happily train on.
    """
    import numpy as np  # noqa: PLC0415

    quiet = {("weak-a", FAKE_TARGETS[0]): "unmentioned"}
    targets, confidence = namespace["weak_targets_and_confidence"](_export_frame(quiet))

    assert np.isnan(targets[0, 0]), "an unmentioned cell must not carry a 0.50 target"
    assert confidence[0, 0] == 0.0
    assert targets[0, 1] == pytest.approx(0.97), "mentioned cells are untouched"
    assert confidence[0, 1] == pytest.approx(0.90)


def test_the_file_decides_the_values_not_the_notebook(namespace):
    """B6 and B23 disagree about `uncertain`, so no policy may be hard-coded.

    B6 gives an uncertain cell confidence 0.25; B23 pins it to 0.00 so a hedged
    reading can never become a training label. Both must load correctly.
    """
    states = {
        ("weak-a", FAKE_TARGETS[0]): "positive",
        ("weak-a", FAKE_TARGETS[1]): "negated",
        ("weak-a", FAKE_TARGETS[2]): "uncertain",
        ("weak-a", FAKE_TARGETS[3]): "conflict",
    }
    targets, confidence = namespace["weak_targets_and_confidence"](_export_frame(states))
    assert (targets[0, 0], confidence[0, 0]) == pytest.approx((0.97, 0.90))
    assert (targets[0, 1], confidence[0, 1]) == pytest.approx((0.03, 0.90))
    assert (targets[0, 2], confidence[0, 2]) == pytest.approx((0.50, 0.25))
    assert (targets[0, 3], confidence[0, 3]) == pytest.approx((0.50, 0.20))


def test_a_b23_export_ignores_its_uncertain_cells(namespace):
    """B23 pins uncertain to confidence 0.00, which must blank the cell."""
    import numpy as np  # noqa: PLC0415

    frame = _export_frame({("weak-a", FAKE_TARGETS[2]): "uncertain"})
    frame.loc[0, f"{FAKE_TARGETS[2]}__confidence"] = 0.00  # the B23 policy
    targets, confidence = namespace["weak_targets_and_confidence"](frame)

    assert np.isnan(targets[0, 2]), "a B23 uncertain cell must not train the model"
    assert confidence[0, 2] == 0.0


# --- the assembly the tests cannot execute ---------------------------------
#
# build_report_supervised_experiment wires the real dataset, model and Experiment
# together. It cannot be executed here -- the notebook's first cell runs pip and
# mounts Drive -- so its constructor calls are checked against the real
# definitions instead. A wrong keyword would otherwise surface only in Colab.


def _find_class(cells, name):
    import ast  # noqa: PLC0415

    for kind, text in cells:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
    return None


def _calls_to(cells, name):
    import ast  # noqa: PLC0415

    found = []
    for kind, text in cells:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
            ):
                found.append(node)
    return found


def test_the_experiment_is_built_with_the_fields_it_declares(cells):
    import ast  # noqa: PLC0415

    declared = _find_class(cells, "Experiment")
    assert declared is not None, "the notebook no longer defines Experiment"

    required, optional = [], []
    for node in declared.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            (optional if node.value is not None else required).append(node.target.id)

    calls = _calls_to(cells, "Experiment")
    assert calls, "nothing builds an Experiment any more"
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords}
        assert not set(passed) - set(required + optional), (
            f"Experiment(...) passes unknown fields: {passed - set(required + optional)}"
        )
        assert not set(required) - passed, (
            f"Experiment(...) is missing required fields: {set(required) - passed}"
        )


def test_the_dataset_is_built_with_the_arguments_it_takes(cells):
    import ast  # noqa: PLC0415

    declared = _find_class(cells, "KneeMRIDataset")
    assert declared is not None, "the notebook no longer defines KneeMRIDataset"

    init = next(
        node
        for node in declared.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameters = [argument.arg for argument in init.args.args][1:]  # drop self
    keyword_only = [argument.arg for argument in init.args.kwonlyargs]
    defaulted = len(init.args.defaults)
    mandatory = parameters if not defaulted else parameters[:-defaulted]

    calls = _calls_to(cells, "KneeMRIDataset")
    assert calls, "nothing builds a KneeMRIDataset any more"
    for call in calls:
        supplied = list(call.args)
        named = {keyword.arg for keyword in call.keywords}
        assert not named - set(parameters + keyword_only), (
            f"KneeMRIDataset(...) passes unknown arguments: {named - set(parameters)}"
        )
        covered = set(parameters[: len(supplied)]) | named
        assert not set(mandatory) - covered, (
            f"KneeMRIDataset(...) is missing arguments: {set(mandatory) - covered}"
        )


# --- who trains on what ----------------------------------------------------


def _train_csv(report_only=("weak-a", "weak-b"), gold=("gold-a",)):
    """train.csv as the competition ships it: gold labelled, the rest blank."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    rows = []
    for study in report_only:
        row = {"StudyInstanceUID": study, "Report": "knee mri"}
        row.update({target: np.nan for target in FAKE_TARGETS})
        rows.append(row)
    for study in gold:
        row = {"StudyInstanceUID": study, "Report": "knee mri"}
        row.update({target: 1.0 for target in FAKE_TARGETS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_gold_is_told_apart_by_having_a_label(namespace):
    gold, report_only = namespace["split_gold_and_report_only"](_train_csv())
    assert list(gold["StudyInstanceUID"]) == ["gold-a"]
    assert list(report_only["StudyInstanceUID"]) == ["weak-a", "weak-b"]


def test_a_study_with_even_one_label_counts_as_gold(namespace):
    """`notna().any()` is the rule; a partly labelled study must not train."""
    import numpy as np  # noqa: PLC0415

    frame = _train_csv(report_only=("weak-a",), gold=())
    frame.loc[0, FAKE_TARGETS[3]] = 0.0
    gold, report_only = namespace["split_gold_and_report_only"](frame)
    assert list(gold["StudyInstanceUID"]) == ["weak-a"]
    assert report_only.empty


def test_only_report_only_studies_train(namespace):
    records = {"weak-a": 1, "weak-b": 1, "gold-a": 1}
    train_frame, gold_usable = namespace["select_report_training_studies"](
        _train_csv(), _export_frame(), records
    )
    assert list(train_frame["StudyInstanceUID"]) == ["weak-a", "weak-b"]
    assert list(gold_usable["StudyInstanceUID"]) == ["gold-a"]


def test_an_export_containing_gold_is_refused(namespace):
    """A leak here would not crash -- it would just make every score look better."""
    leaky = _export_frame()
    leaky.loc[0, "StudyInstanceUID"] = "gold-a"
    with pytest.raises(ValueError, match="expert-gold studies"):
        namespace["select_report_training_studies"](
            _train_csv(), leaky, {"gold-a": 1, "weak-b": 1}
        )


def test_studies_without_downloaded_images_are_dropped(namespace):
    """The Drive subset is usually smaller than the export."""
    train_frame, _ = namespace["select_report_training_studies"](
        _train_csv(), _export_frame(), {"weak-b": 1, "gold-a": 1}
    )
    assert list(train_frame["StudyInstanceUID"]) == ["weak-b"]


def test_no_overlap_at_all_is_an_explaining_error(namespace):
    with pytest.raises(ValueError, match="Check that the export covers"):
        namespace["select_report_training_studies"](
            _train_csv(), _export_frame(), {"someone-else": 1}
        )


def test_a_subset_with_no_gold_still_trains(namespace):
    """Then nothing can be scored, but training must not be blocked."""
    train_frame, gold_usable = namespace["select_report_training_studies"](
        _train_csv(gold=()), _export_frame(), {"weak-a": 1, "weak-b": 1}
    )
    assert len(train_frame) == 2
    assert gold_usable.empty


# --- the weighted loss -----------------------------------------------------


def test_a_silent_cell_cannot_influence_training(namespace):
    """The property the whole confidence mechanism exists to provide."""
    import numpy as np  # noqa: PLC0415

    multiplier = torch.ones(12)
    logits = torch.zeros(2, 12)
    confidence = torch.ones(2, 12)
    confidence[0, 3] = 0.0  # this cell is report silence

    target = torch.full((2, 12), 0.5)
    first = namespace["report_weighted_bce"](logits, target, confidence, multiplier)

    moved = target.clone()
    moved[0, 3] = 1.0  # change the silent cell to anything at all
    second = namespace["report_weighted_bce"](logits, moved, confidence, multiplier)

    assert float(first) == pytest.approx(float(second)), "a zero-confidence cell moved the loss"


def test_an_unsure_disagreement_pulls_less_than_a_sure_one(namespace):
    """Weighting only shows up when the cells carry different losses."""
    multiplier = torch.ones(12)

    # Cell 0: the model is confidently wrong. Cell 1: mildly wrong, always sure.
    logits = torch.zeros(1, 12)
    logits[0, 0] = 5.0
    target = torch.zeros(1, 12)

    sure = torch.zeros(1, 12)
    sure[0, 0], sure[0, 1] = 0.90, 0.90
    unsure = torch.zeros(1, 12)
    unsure[0, 0], unsure[0, 1] = 0.25, 0.90

    confident = float(namespace["report_weighted_bce"](logits, target, sure, multiplier))
    hesitant = float(namespace["report_weighted_bce"](logits, target, unsure, multiplier))
    assert confident > hesitant, "an uncertain disagreement should pull less than a sure one"


def test_a_blank_target_is_ignored_even_when_confidence_claims_otherwise(namespace):
    multiplier = torch.ones(12)
    logits = torch.zeros(1, 12)
    target = torch.zeros(1, 12)
    target[0, 0] = float("nan")
    confidence = torch.ones(1, 12)

    value = namespace["report_weighted_bce"](logits, target, confidence, multiplier)
    assert torch.isfinite(value), "a NaN target must not poison the loss"


def test_a_batch_with_nothing_usable_still_has_a_gradient_path(namespace):
    multiplier = torch.ones(12)
    logits = torch.zeros(1, 12, requires_grad=True)
    target = torch.zeros(1, 12)
    value = namespace["report_weighted_bce"](logits, target, torch.zeros(1, 12), multiplier)

    assert float(value.detach()) == 0.0
    value.backward()  # must not raise
    assert logits.grad is not None


def test_shape_disagreement_is_refused(namespace):
    with pytest.raises(ValueError, match="same shape"):
        namespace["report_weighted_bce"](
            torch.zeros(1, 12), torch.zeros(1, 12), torch.zeros(2, 12), torch.ones(12)
        )


# --- target balance --------------------------------------------------------


def test_equal_supervision_needs_no_correction(namespace):
    import numpy as np  # noqa: PLC0415

    multiplier = namespace["target_balance_multipliers"](np.ones((5, 12), dtype=np.float32))
    assert multiplier == pytest.approx(np.ones(12))


def test_a_rarely_written_finding_is_lifted(namespace):
    """Otherwise the findings radiologists write about most win every gradient."""
    import numpy as np  # noqa: PLC0415

    confidence = np.ones((10, 12), dtype=np.float32)
    confidence[:, 0] = 0.1  # this finding is rarely mentioned
    multiplier = namespace["target_balance_multipliers"](confidence)

    assert multiplier[0] > multiplier[1], "the rare finding must be weighted up"
    mass = confidence.sum(axis=0) * multiplier
    assert mass == pytest.approx(np.full(12, mass[0]), rel=1e-5), "every finding ends level"


def test_a_finding_the_reports_never_mention_is_reported(namespace):
    import numpy as np  # noqa: PLC0415

    confidence = np.ones((4, 12), dtype=np.float32)
    confidence[:, 7] = 0.0
    with pytest.raises(ValueError, match="no usable supervision"):
        namespace["target_balance_multipliers"](confidence)


# --- confidence reaches the right study ------------------------------------


def test_confidence_follows_the_study_not_the_row_number(namespace):
    """The dataset filters and reindexes, so positional lookup would drift."""
    import numpy as np  # noqa: PLC0415

    lookup = {
        "study-a": np.full(12, 0.1, dtype=np.float32),
        "study-b": np.full(12, 0.2, dtype=np.float32),
        "study-c": np.full(12, 0.3, dtype=np.float32),
    }
    supervision = namespace["ReportSupervision"](lookup, np.ones(12, dtype=np.float32))

    # A batch in a different order from the dictionary must still be correct.
    batch = supervision.batch(["study-c", "study-a"])
    assert batch.shape == (2, 12)
    assert float(batch[0, 0]) == pytest.approx(0.3)
    assert float(batch[1, 0]) == pytest.approx(0.1)


def test_a_study_with_no_confidence_is_a_loud_failure(namespace):
    import numpy as np  # noqa: PLC0415

    supervision = namespace["ReportSupervision"]({}, np.ones(12, dtype=np.float32))
    with pytest.raises(KeyError, match="no confidence recorded"):
        supervision.batch(["study-a"])


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
