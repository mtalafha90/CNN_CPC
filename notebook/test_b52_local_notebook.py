"""The local B52 notebook's checks, executed rather than read.

This notebook defines no model. Everything in it is a guard, so what is worth
testing is that each guard actually fires:

* a wrong `BUNDLE` must fail in section 2, not inside the trainer twenty minutes
  later;
* a data folder that is not the one the B50 gate was built on must be refused,
  because the whole point of the gate is that it names a population;
* an `out_root` that already holds a checkpoint must be reported now, since the
  trainer's own `FileExistsError` arrives only after loading everything;
* the command line it builds must be the one the trainer accepts, argument for
  argument.
"""

from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path

import pytest

BUILDER = Path(__file__).with_name("build_b52_local_notebook.py")
NOTEBOOK = Path(__file__).with_name("b52_local_full.ipynb")
TRAINER = (
    Path(__file__).resolve().parents[1]
    / "developments"
    / "src"
    / "rsna_knee"
    / "b52_competition_training.py"
)

# The checks under test, keyed by the definition that identifies their cell.
WANTED = (
    "def resolve_layout",
    "def sha256_of",
    "def stream",
)


@pytest.fixture(scope="module")
def cells() -> list[tuple[str, str]]:
    return list(runpy.run_path(str(BUILDER))["CELLS"])


def _is_literal(node) -> bool:
    """Whether an assigned value is a plain constant, so it is safe to evaluate."""
    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False
    return True


@pytest.fixture(scope="module")
def namespace(cells) -> dict:
    """Execute only the cells that hold the checks, in order.

    The first cell, which sets `BUNDLE` and `DATA_ROOT` to placeholder paths, is
    deliberately not run: it would fail on paths that do not exist, which is the
    behaviour a user wants and not something to reproduce in a test.
    """
    import os
    import subprocess
    import sys

    scope: dict = {
        "__name__": "__main__",
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "display": lambda *args, **kwargs: None,
    }
    for _kind, text in cells:
        if any(marker in text for marker in WANTED):
            # Only the definitions and plain constants. A line such as
            # `LAYOUT = resolve_layout(BUNDLE)` is the notebook doing its job on
            # the user's real paths, and has nothing to run against here.
            tree = ast.parse(text)
            tree.body = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
                or (isinstance(node, ast.Assign) and _is_literal(node.value))
            ]
            exec(compile(tree, "<b52-local-cell>", "exec"), scope)
    return scope


def _bundle(tmp_path: Path, train_csv: str = "study,label\na,1\n") -> tuple[Path, Path]:
    """A bundle and a data folder shaped exactly as the notebook expects."""
    bundle = tmp_path / "b52_standalone"
    for relative in ("src/rsna_knee", "labels", "policy", "models", "gate", "config"):
        (bundle / relative).mkdir(parents=True, exist_ok=True)
    (bundle / "config" / "b42_constant_area_aspect_sparse.yaml").write_text("b7_n_slices: 16\n")
    (bundle / "models" / "phase9_llm_fill_base.pt").write_bytes(b"not really a checkpoint")
    (bundle / "policy" / "series_policy.json").write_text("{}")

    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text(train_csv)
    (data / "train_series.csv").write_text("StudyInstanceUID,SeriesInstanceUID\na,s\n")

    digest = hashlib.sha256(train_csv.encode()).hexdigest()
    (bundle / "gate" / "b50_selection_split.json").write_text(
        json.dumps({"source_train_csv_sha256": digest, "studies": 4349})
    )
    return bundle, data


# --- the notebook builds ---------------------------------------------------


def test_the_notebook_is_current(cells):
    """A stale checked-in notebook would run something the builder no longer says."""
    written = json.loads(NOTEBOOK.read_text())["cells"]
    assert len(written) == len(cells), "rebuild the notebook from its builder"

    for index, ((kind, text), cell) in enumerate(zip(cells, written)):
        assert cell["cell_type"] == kind, f"cell {index} changed type"
        assert "".join(cell["source"]) == text, f"cell {index} is stale; rebuild the notebook"


def test_every_code_cell_parses(cells):
    for kind, text in cells:
        if kind == "code":
            ast.parse(text)


def _bound_names(tree) -> set[str]:
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
    """A NameError in the training cell would land after the preflight passed."""
    import builtins

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
    names: set[str] = set()
    stack = list(getattr(tree, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return names


def test_definitions_come_before_the_cells_that_run_them(cells):
    import builtins

    available = set(dir(builtins)) | {"__name__", "__file__", "get_ipython"}
    for index, (kind, text) in enumerate(cells):
        if kind != "code":
            continue
        tree = ast.parse(text)
        available |= _bound_names(tree)
        too_early = sorted(_top_level_loads(tree) - available)
        assert not too_early, f"cell {index} runs before {', '.join(too_early)} is defined"


def test_training_is_off_until_it_is_turned_on(cells):
    """A 27-hour run must never start from 'Run all'."""
    everything = "\n".join(text for _kind, text in cells)
    assert "RUN_TRAINING = False" in everything
    assert "RUN_TRAINING = True" not in everything


def test_the_notebook_carries_no_old_experiment(cells):
    """This is a B52 notebook. Nothing else should be runnable from it."""
    everything = "\n".join(text for _kind, text in cells)
    for stale in (
        "b51_full_population_training",
        "b50_ordered_slice_selection_training",
        "b48_global_conditioned",
        "RUN_B51_COMPARISON",
        "build_experiment(",
    ):
        assert stale not in everything, f"{stale} is a leftover from an older notebook"


def test_the_selection_numbers_are_labelled_as_selection_statistics(cells):
    """These are maxima over epochs on the surface used to choose the epoch.

    Quoting one as an effect size, or beside the 0.714 leaderboard score, is the
    specific mistake the checkpoint's own governance field exists to prevent.
    """
    opening = "\n".join(text for kind, text in cells[:3] if kind == "markdown").lower()
    assert "selection statistic" in opening
    assert "optimistically biased" in opening
    assert "0.714" in opening


# --- the layout check ------------------------------------------------------


def test_a_bundle_is_recognised_and_fully_resolved(namespace, tmp_path):
    bundle, _data = _bundle(tmp_path)
    layout = namespace["resolve_layout"](bundle)

    assert layout["kind"] == "bundle"
    assert layout["source"] == bundle / "src"
    assert layout["base_checkpoint"].is_file()
    assert layout["gate"].is_dir()


def test_a_bundle_missing_a_file_fails_here_not_in_the_trainer(namespace, tmp_path):
    """Twenty minutes into a run is a poor time to learn the checkpoint is absent."""
    bundle, _data = _bundle(tmp_path)
    (bundle / "models" / "phase9_llm_fill_base.pt").unlink()

    with pytest.raises(FileNotFoundError, match="base_checkpoint is missing"):
        namespace["resolve_layout"](bundle)


def test_a_path_that_is_not_a_bundle_at_all_says_so(namespace, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        namespace["resolve_layout"](empty)


def test_a_missing_bundle_directory_says_to_set_bundle(namespace, tmp_path):
    with pytest.raises(FileNotFoundError, match="set BUNDLE"):
        namespace["resolve_layout"](tmp_path / "does-not-exist")


def test_a_data_folder_without_train_csv_is_refused(namespace, tmp_path):
    empty = tmp_path / "data"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="set DATA_ROOT"):
        namespace["check_data_root"](empty)


# --- the gate fingerprint --------------------------------------------------


def test_the_matching_data_folder_passes(namespace, tmp_path):
    bundle, data = _bundle(tmp_path)
    result = namespace["check_gate_matches_data"](bundle / "gate", data)
    assert result["studies"] == 4349
    assert len(result["train_csv_sha256"]) == 64


def test_a_different_dataset_is_refused_and_not_offered_a_workaround(namespace, tmp_path):
    """This is the check that stops a run training on the wrong population.

    It is also the reason the subset notebook exists: a Drive subset can never
    pass it, so it rebuilds the regime instead of defeating the check.
    """
    bundle, data = _bundle(tmp_path)
    (data / "train.csv").write_text("study,label\nsomething,else\n")

    with pytest.raises(ValueError) as caught:
        namespace["check_gate_matches_data"](bundle / "gate", data)

    message = str(caught.value)
    assert "not something to override" in message
    assert "gate expects" in message and "data root has" in message


def test_a_gate_with_no_recorded_fingerprint_is_refused(namespace, tmp_path):
    """An unfingerprinted gate cannot say which population it describes."""
    bundle, data = _bundle(tmp_path)
    (bundle / "gate" / "b50_selection_split.json").write_text(json.dumps({"studies": 4349}))

    with pytest.raises(ValueError, match="records no source_train_csv_sha256"):
        namespace["check_gate_matches_data"](bundle / "gate", data)


def test_the_hash_matches_hashlib(namespace, tmp_path):
    """The blockwise read must give the same answer as reading it whole."""
    path = tmp_path / "some.csv"
    path.write_bytes(b"x" * (3 * 1024 * 1024 + 17))  # spans several blocks
    assert namespace["sha256_of"](path) == hashlib.sha256(path.read_bytes()).hexdigest()


# --- the output folder -----------------------------------------------------


def test_an_existing_checkpoint_is_reported_before_the_run_starts(namespace, tmp_path):
    """The trainer refuses to overwrite, but only after loading everything."""
    out = tmp_path / "runs" / "b52"
    out.mkdir(parents=True)
    (out / namespace["CHECKPOINT_NAME"]).write_bytes(b"a previous run")

    with pytest.raises(FileExistsError, match="will not overwrite"):
        namespace["check_out_root_is_free"](out)


def test_a_free_output_folder_passes(namespace, tmp_path):
    namespace["check_out_root_is_free"](tmp_path / "runs" / "b52")  # need not exist yet


def test_the_checkpoint_name_matches_the_trainer(namespace):
    """A renamed checkpoint would make both the collision check and the result
    reader look at a file that is never written."""
    source = TRAINER.read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "B52_CHECKPOINT_NAME"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ):
            assert namespace["CHECKPOINT_NAME"] == node.value.value
            return
    pytest.fail("the trainer no longer defines B52_CHECKPOINT_NAME")


# --- the command line ------------------------------------------------------


def _trainer_flags() -> set[str]:
    """Every flag the trainer's argument parser actually accepts."""
    flags: set[str] = set()
    for node in ast.walk(ast.parse(TRAINER.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for argument in node.args:
                if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                    flags.add(argument.value)
    return flags


def test_every_flag_the_notebook_passes_exists_in_the_trainer(cells):
    """A renamed flag would fail after the preflight cell had already passed."""
    accepted = _trainer_flags()
    assert accepted, "could not read the trainer's arguments"

    command_cell = next(
        text for kind, text in cells if kind == "code" and "def trainer_command" in text
    )
    used = {
        node.value
        for node in ast.walk(ast.parse(command_cell))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("--")
    }
    assert used <= accepted, f"the trainer does not accept: {sorted(used - accepted)}"


def test_the_command_carries_every_required_argument(namespace, cells):
    """The trainer marks six arguments required; missing one fails at launch."""
    required = {
        argument.value
        for node in ast.walk(ast.parse(TRAINER.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(keyword.value, ast.Constant)
            and keyword.arg == "required"
            and keyword.value.value is True
            for keyword in node.keywords
        )
        for argument in node.args
        if isinstance(argument, ast.Constant)
    }
    assert required, "the trainer no longer marks any argument required"

    command_cell = next(
        text for kind, text in cells if kind == "code" and "def trainer_command" in text
    )
    for flag in required:
        assert f'"{flag}"' in command_cell, f"the notebook never passes {flag}"


def _command_builder(namespace, cells, *, all_data: bool):
    """Load the command cell against stand-in paths and return its builder."""
    scope = dict(namespace)
    scope.update(
        {
            "LAYOUT": {
                "root": Path("/bundle"),
                "source": Path("/bundle/src"),
                "config": Path("/c.yaml"),
                "labels": Path("/labels"),
                "series_policy": Path("/policy.json"),
                "base_checkpoint": Path("/base.pt"),
                "gate": Path("/gate"),
                "out_root": Path("/out"),
            },
            "DATA_ROOT": Path("/data"),
            "EPOCHS": 6,
            "ALL_DATA": all_data,
        }
    )
    command_cell = next(
        text for kind, text in cells if kind == "code" and "def trainer_command" in text
    )
    tree = ast.parse(command_cell)
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    exec(compile(tree, "<command-cell>", "exec"), scope)
    return scope["trainer_command"]


def test_the_two_commands_differ_only_by_the_preflight_flag(namespace, cells):
    """Preflighting a different command than the one that runs proves nothing."""
    build = _command_builder(namespace, cells, all_data=True)
    preflight = build(preflight=True)
    real = build(preflight=False)

    assert preflight[:-1] == real, "the preflight runs a different command than the run"
    assert preflight[-1] == "--preflight-only"
    assert "--all-data" in real


def test_all_data_is_what_switches_the_training_population(namespace, cells):
    """3,801 studies against the gate's 1,447 is entirely this one flag."""
    build = _command_builder(namespace, cells, all_data=False)
    assert "--all-data" not in build(preflight=False)


# --- streaming -------------------------------------------------------------


def test_output_is_streamed_rather_than_collected(namespace, capsys):
    """A 27-hour cell that prints nothing until it ends looks like a hang."""
    import sys

    status = namespace["stream"](
        [sys.executable, "-c", "print('first'); print('second')"], cwd=Path.cwd()
    )
    assert status == 0
    captured = capsys.readouterr().out
    assert "first" in captured and "second" in captured


def test_a_failing_command_returns_a_non_zero_status(namespace):
    """Every caller checks this; a swallowed failure would start a doomed run."""
    import sys

    assert namespace["stream"]([sys.executable, "-c", "raise SystemExit(3)"], cwd=Path.cwd()) == 3


def test_stderr_is_shown_too(namespace, capsys):
    """A traceback from the trainer arrives on stderr and must not be dropped."""
    import sys

    namespace["stream"](
        [sys.executable, "-c", "import sys; print('boom', file=sys.stderr)"], cwd=Path.cwd()
    )
    assert "boom" in capsys.readouterr().out
