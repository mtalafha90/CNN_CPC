"""The B52 subset notebook's mechanism, executed rather than read.

B52 is three changes to how the model is trained. Each one has a way of being
present in the prose and absent in the code, and each of those ways is tested
here rather than trusted:

* augmentation that returns the image unchanged, or that flips left to right and
  quietly teaches the model that medial and lateral menisci are the same thing;
* a cosine whose `T_max` is not the number of epochs run, which is exactly the
  bug the frozen contract had;
* "best epoch" selection that keeps the last epoch, or that selects on the 58
  expert studies the notebook says it does not select on.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

BUILDER = Path(__file__).with_name("build_b52_colab_subset_notebook.py")
NOTEBOOK = Path(__file__).with_name("b52_colab_subset.ipynb")


@pytest.fixture(scope="module")
def cells() -> list[tuple[str, str]]:
    return list(runpy.run_path(str(BUILDER))["CELLS"])


WANTED = (
    "def load_report_labels",
    "def target_balance_multipliers",
    "class AugmentationPolicy",
    "def split_report_studies",
    "def hierarchy_parameter_names",
)

# Stand-ins for the notebook's twelve findings. Every function under test is
# generic over this list, and a separate test pins the real length at twelve.
FAKE_TARGETS = [f"finding_{index}" for index in range(12)]


@pytest.fixture(scope="module")
def namespace(cells) -> dict:
    """Execute only the cells the B52 mechanism needs, in order."""
    import math
    from dataclasses import dataclass, field

    import numpy as np
    import pandas as pd

    class _StubDataset:
        """Stands in for the inherited KneeMRIDataset, which needs real DICOMs."""

        def __init__(self, *args, **kwargs) -> None:
            self.config = kwargs.get("config")
            self.study_uids: list[str] = []

        def __getitem__(self, index: int) -> dict:
            return dict(self._items[index])

    scope: dict = {
        "__name__": "__main__",
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "np": np,
        "pd": pd,
        "math": math,
        "Path": Path,
        "dataclass": dataclass,
        "field": field,
        "TARGETS": list(FAKE_TARGETS),
        "DEVICE": torch.device("cpu"),
        "KneeMRIDataset": _StubDataset,
    }
    for _kind, text in cells:
        if any(marker in text for marker in WANTED):
            exec(compile(text, "<b52-cell>", "exec"), scope)
    return scope


def _export_frame(states: dict | None = None):
    """A label export shaped exactly like the one b23_llm_labels.py writes."""
    import pandas as pd  # noqa: PLC0415

    fixed = {
        "positive": (0.97, 0.90),
        "negated": (0.03, 0.90),
        "uncertain": (0.50, 0.25),
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
    import ast  # noqa: PLC0415

    for _index, (kind, text) in enumerate(cells):
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
    """Names read by code that runs when the cell runs, ignoring function bodies."""
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
    """Cells run top to bottom, so a definition placed after its use is a crash."""
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
        assert not too_early, f"cell {index} runs before {', '.join(too_early)} is defined"


def test_the_notebook_says_what_it_is_not(cells):
    """A reader must not carry this subset run's absolute numbers anywhere.

    Emphasis markers are stripped before matching, so re-wrapping a line or
    bolding a phrase cannot fail this test for the wrong reason.
    """
    opening = "\n".join(text for kind, text in cells[:4] if kind == "markdown")
    scope = opening.lower().replace("*", "").replace("\n", " ")

    assert "is not the real b52 run" in scope, "it must disclaim being the real run"
    assert "absolute numbers mean nothing" in scope, "subset numbers must be disclaimed"
    assert "fingerprint mismatch" in scope, (
        "it must explain why the real B52 code cannot run on a subset"
    )


def test_the_old_experiments_are_gone(cells):
    """The user asked for a B52 notebook, not a pile of settled comparisons.

    A leftover arm-comparison section is worse than clutter: it still runs, it
    trains a second model for hours, and it answers a question B50 already
    answered.
    """
    everything = "\n".join(text for _kind, text in cells)

    for stale in (
        "RUN_B51_COMPARISON",
        "def compare_arms",
        "def run_one_arm",
        "def discordant_pair_fraction",
        "frozen_hierarchy_control",
    ):
        assert stale not in everything, f"{stale} is a leftover from an older notebook"

    # The inherited gold-only training path is the subtler leftover: it still
    # runs and it trains on the 58 expert studies without saying so.
    assert "build_experiment(PATHS, CONFIG)" not in everything, (
        "the inherited gold-only build cell must be replaced, not left in place"
    )


def test_the_notebook_has_twelve_findings(cells):
    """The stub target list is only honest if the real one is the same length."""
    import ast  # noqa: PLC0415

    for kind, text in cells:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TARGETS" for t in node.targets)
                and isinstance(node.value, ast.List)
            ):
                assert len(node.value.elts) == len(FAKE_TARGETS) == 12
                return
    pytest.fail("the notebook no longer defines TARGETS as a list")


# --- the report labels -----------------------------------------------------


def test_a_well_formed_export_loads(namespace, tmp_path):
    path = tmp_path / "training_targets.csv"
    _export_frame().to_csv(path, index=False)
    frame = namespace["load_report_labels"](path)
    assert list(frame["StudyInstanceUID"]) == ["weak-a", "weak-b"]


def test_an_export_missing_confidence_is_refused(namespace, tmp_path):
    """Probabilities without confidence are exactly the dangerous case."""
    frame = _export_frame().drop(columns=[f"{FAKE_TARGETS[0]}__confidence"])
    path = tmp_path / "training_targets.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        namespace["load_report_labels"](path)


def test_an_unmentioned_finding_becomes_a_blank(namespace):
    """Report silence is not a negative -- the single most important rule here."""
    import numpy as np  # noqa: PLC0415

    quiet = {("weak-a", FAKE_TARGETS[0]): "unmentioned"}
    targets, confidence = namespace["weak_targets_and_confidence"](_export_frame(quiet))

    assert np.isnan(targets[0, 0]), "an unmentioned cell must not carry a 0.50 target"
    assert confidence[0, 0] == 0.0
    assert targets[0, 1] == pytest.approx(0.97), "mentioned cells are untouched"


# --- change 2: augmentation ------------------------------------------------


def _series(slices: int = 4, size: int = 16) -> torch.Tensor:
    """One prepared series: [slices, 3, H, W], with structure worth preserving."""
    ramp = torch.linspace(0.1, 0.9, size)
    plane = ramp[None, :].expand(size, size)
    return plane[None, None].expand(slices, 3, size, size).clone()


def test_augmentation_actually_changes_the_image(namespace):
    """The failure that would leave the notebook truthful in prose and false in fact.

    An augmentation that silently returns its input would make this notebook
    exactly the frozen baseline it exists to replace.
    """
    generator = torch.Generator().manual_seed(7)
    before = _series()
    after = namespace["augment_series"](before, namespace["AUGMENTATION"], generator)

    assert after.shape == before.shape
    assert not torch.allclose(after, before), "the default policy changed nothing"


def test_the_off_policy_leaves_the_image_alone(namespace):
    """Validation and test studies must be decoded exactly as before."""
    generator = torch.Generator().manual_seed(7)
    before = _series()
    after = namespace["augment_series"](before, namespace["NO_AUGMENTATION"], generator)
    assert torch.allclose(after, before)


def test_the_same_seed_gives_the_same_augmentation(namespace):
    """Reproducibility: two runs of the same seed must agree exactly."""
    policy = namespace["AUGMENTATION"]
    first = namespace["augment_series"](_series(), policy, torch.Generator().manual_seed(3))
    second = namespace["augment_series"](_series(), policy, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_different_seeds_give_different_augmentations(namespace):
    """If every draw were the same, the model would still see one fixed dataset."""
    policy = namespace["AUGMENTATION"]
    first = namespace["augment_series"](_series(), policy, torch.Generator().manual_seed(3))
    second = namespace["augment_series"](_series(), policy, torch.Generator().manual_seed(4))
    assert not torch.allclose(first, second)


def test_the_augmentation_code_contains_no_flip(cells):
    """Mirroring a knee swaps medial and lateral, which are separate targets.

    A horizontal flip is the most tempting augmentation in imaging and the one
    that would silently teach this model that two of its twelve answers are
    interchangeable. This reads the source rather than the behaviour, because a
    flip added behind a policy flag that happens to be off would pass any
    behavioural check while still being one edit away from live.
    """
    source = "\n".join(
        text for kind, text in cells if kind == "code" and "def augment_series" in text
    )
    assert source, "the augmentation cell has moved or been renamed"

    for primitive in ("torch.flip", "fliplr", "hflip", "[::-1]", "flip("):
        assert primitive not in source, f"{primitive} mirrors the image; see the docstring"


def test_augmentation_keeps_the_image_on_the_side_it_started(namespace):
    """The behavioural half: a left-heavy study must stay left-heavy.

    Rotation of 8 degrees and a 5% shift move the centre of mass a little. A
    mirror would move it to the other side of the image entirely.
    """
    size = 16
    bright = torch.zeros(4, 3, size, size)
    bright[:, :, :, : size // 4] = 1.0  # all the signal on the left

    columns = torch.arange(size, dtype=torch.float32)
    before_centre = float((bright.mean((0, 1, 2)) * columns).sum() / bright.mean((0, 1, 2)).sum())

    for seed in range(20):
        after = namespace["augment_series"](
            bright, namespace["AUGMENTATION"], torch.Generator().manual_seed(seed)
        )
        profile = after.mean((0, 1, 2)).clamp_min(0.0)
        after_centre = float((profile * columns).sum() / profile.sum().clamp_min(1e-6))
        assert after_centre < size / 2, (
            f"seed {seed} moved the signal to the right half; the image was mirrored "
            f"(centre {before_centre:.1f} -> {after_centre:.1f})"
        )


def test_slice_dropout_never_empties_a_study(namespace):
    """A study with every slice blanked still carries a real label.

    It would be a blank input paired with a positive finding, which teaches the
    model something false rather than nothing.
    """
    policy = namespace["AugmentationPolicy"](
        rotation_deg=0.0,
        translate_frac=0.0,
        scale_jitter=0.0,
        gamma_jitter=0.0,
        noise_std=0.0,
        slice_dropout=1.0,  # try to drop everything
        bias_field_strength=0.0,
    )
    for seed in range(10):
        after = namespace["augment_series"](
            _series(), policy, torch.Generator().manual_seed(seed)
        )
        alive = [index for index in range(after.shape[0]) if float(after[index].abs().sum()) > 0]
        assert alive, f"seed {seed} blanked every slice"


def test_gamma_does_not_produce_nan_on_negative_values(namespace):
    """Percentile-normalised images hold small negatives.

    A fractional power of a negative number is not a real number, so a naive
    `image ** gamma` fills the batch with NaN and the loss becomes NaN an epoch
    later, far from the cause.
    """
    policy = namespace["AugmentationPolicy"](
        rotation_deg=0.0,
        translate_frac=0.0,
        scale_jitter=0.0,
        gamma_jitter=0.5,
        noise_std=0.0,
        slice_dropout=0.0,
        bias_field_strength=0.0,
    )
    series = _series() - 0.5  # now genuinely negative in places
    assert float(series.min()) < 0

    for seed in range(10):
        after = namespace["augment_series"](
            series, policy, torch.Generator().manual_seed(seed)
        )
        assert torch.isfinite(after).all(), f"seed {seed} produced NaN or infinity"


def test_the_seed_recipe_separates_epochs_and_studies(namespace):
    """Two studies in one epoch, and one study across epochs, must all differ.

    A generator shared across the dataset would also fail this under
    `num_workers > 0`, where each worker holds its own copy.
    """
    dataset = namespace["AugmentedKneeMRIDataset"].__new__(
        namespace["AugmentedKneeMRIDataset"]
    )

    class _Config:
        seed = 2026

    dataset.config = _Config()
    dataset.epoch = 1

    first_study = namespace["augment_series"](
        _series(), namespace["AUGMENTATION"], dataset._generator(0)
    )
    second_study = namespace["augment_series"](
        _series(), namespace["AUGMENTATION"], dataset._generator(1)
    )
    assert not torch.allclose(first_study, second_study), "two studies got the same draw"

    dataset.epoch = 2
    next_epoch = namespace["augment_series"](
        _series(), namespace["AUGMENTATION"], dataset._generator(0)
    )
    assert not torch.allclose(first_study, next_epoch), "the second epoch repeated the first"


def test_the_augmented_dataset_leaves_masked_series_alone(namespace):
    """A masked series is a zero placeholder the model ignores.

    Warping it would cost time and, with noise on, would turn a placeholder into
    something that is no longer zero.
    """
    dataset = namespace["AugmentedKneeMRIDataset"].__new__(
        namespace["AugmentedKneeMRIDataset"]
    )

    class _Config:
        seed = 2026

    dataset.config = _Config()
    dataset.epoch = 1
    dataset.policy = namespace["AUGMENTATION"]

    real, blank = _series(), torch.zeros_like(_series())
    dataset._items = [
        {
            "study_uid": "s",
            "volumes": torch.stack([real, blank]),
            "present": torch.tensor([1.0, 0.0]),
        }
    ]

    out = dataset[0]
    assert torch.equal(out["volumes"][1], blank), "a masked placeholder was augmented"
    assert not torch.allclose(out["volumes"][0], real), "the real series was not augmented"


def test_describe_augmentation_counts_what_is_on(namespace):
    """The printed summary is how a reader checks the notebook is doing this."""
    assert namespace["describe_augmentation"](namespace["NO_AUGMENTATION"])["count"] == 0
    assert namespace["describe_augmentation"](namespace["AUGMENTATION"])["count"] == 7


# --- change 3: the split, the schedule, and the best epoch -----------------


def test_the_split_covers_everything_exactly_once(namespace):
    studies = [f"study-{index}" for index in range(50)]
    train, validation = namespace["split_report_studies"](studies, 0.2, 2026)

    assert len(validation) == 10
    assert len(train) == 40
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(studies)


def test_the_split_depends_on_the_seed_not_the_row_order(namespace):
    """Two reads of the same file in a different order must split identically."""
    studies = [f"study-{index}" for index in range(50)]
    first = namespace["split_report_studies"](studies, 0.2, 2026)
    second = namespace["split_report_studies"](list(reversed(studies)), 0.2, 2026)
    assert sorted(first[1]) == sorted(second[1])


def test_a_different_seed_gives_a_different_split(namespace):
    studies = [f"study-{index}" for index in range(50)]
    first = namespace["split_report_studies"](studies, 0.2, 2026)
    second = namespace["split_report_studies"](studies, 0.2, 7)
    assert sorted(first[1]) != sorted(second[1])


def test_a_split_that_would_leave_no_training_data_is_refused(namespace):
    with pytest.raises(ValueError, match="cannot give both"):
        namespace["split_report_studies"](["a", "b"], 0.9, 2026)


def test_a_duplicated_study_is_refused(namespace):
    """A study on both sides of the split leaks training data into the score."""
    with pytest.raises(ValueError, match="twice"):
        namespace["split_report_studies"](["a", "b", "a"], 0.34, 2026)


def test_the_cosine_finishes_within_the_epochs_run(namespace):
    """The exact bug the frozen contract had: T_max longer than the run.

    Two epochs of a five-epoch cosine spends the whole run between 100% and
    90.5% of the peak rate, so the model never trains at a reduced rate at all.
    """
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = namespace["build_cosine_schedule"](optimizer, 6)

    rates = []
    for _epoch in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()  # the real order: optimiser first, then the schedule
        scheduler.step()

    assert rates[0] == pytest.approx(1e-3), "the run must start at the full rate"
    assert rates == sorted(rates, reverse=True), "the rate must fall monotonically"
    assert rates[-1] < 0.1 * rates[0], (
        "the schedule did not finish; T_max is longer than the run"
    )


def test_a_schedule_of_zero_epochs_is_refused(namespace):
    with pytest.raises(ValueError, match="at least one epoch"):
        namespace["build_cosine_schedule"](torch.optim.AdamW(_Model().parameters()), 0)


def test_the_best_epoch_is_kept_not_the_last(namespace):
    """The result B52 actually produced: epoch 5 beat epoch 6.

    Taking the last epoch would have thrown away the best model of the run.
    """
    best = namespace["BestEpoch"]()
    model = _Model()

    for epoch, score in ((1, 0.70), (2, 0.75), (3, 0.80), (4, 0.78)):
        with torch.no_grad():
            model.fusion_gate.fill_(float(epoch))
        best.offer(epoch, score, model)

    assert best.epoch == 3
    assert best.score == pytest.approx(0.80)

    with torch.no_grad():
        model.fusion_gate.fill_(99.0)
    assert best.restore(model) == 3
    assert float(model.fusion_gate.detach()[0]) == pytest.approx(3.0), (
        "the restored weights are not the best epoch's"
    )


def test_the_kept_weights_are_a_copy_not_a_reference(namespace):
    """Keeping a reference would leave 'best' tracking the live model.

    The saved weights would then change under later training and 'restore' would
    put back whatever the model already had.
    """
    best = namespace["BestEpoch"]()
    model = _Model()
    with torch.no_grad():
        model.fusion_gate.fill_(1.0)
    best.offer(1, 0.70, model)

    with torch.no_grad():
        model.fusion_gate.fill_(2.0)
    best.restore(model)
    assert float(model.fusion_gate.detach()[0]) == pytest.approx(1.0)


def test_a_tie_keeps_the_earlier_epoch(namespace):
    """A later epoch that only matched is not evidence of improvement."""
    best = namespace["BestEpoch"]()
    model = _Model()
    best.offer(1, 0.80, model)
    assert best.offer(2, 0.80, model) is False
    assert best.epoch == 1


def test_an_undefined_score_is_never_selected(namespace):
    """A hold-out split with one class in every target gives no AUC at all."""
    import numpy as np  # noqa: PLC0415

    best = namespace["BestEpoch"]()
    model = _Model()
    assert best.offer(1, None, model) is False
    assert best.offer(2, float(np.nan), model) is False
    with pytest.raises(RuntimeError, match="nothing to restore"):
        best.restore(model)


# --- change 1: what learns -------------------------------------------------


def test_the_hierarchy_gets_a_reduced_rate_and_the_rest_full(namespace):
    model = _Model()
    groups = namespace["build_parameter_groups"](model, 1e-4)

    by_name = {group["name"]: group for group in groups}
    assert by_name["encoder_and_head"]["lr"] == pytest.approx(1e-4)
    assert by_name["study_hierarchy"]["lr"] == pytest.approx(5e-6)


def test_a_frozen_parameter_never_reaches_the_optimiser(namespace):
    """Weight decay moves any parameter the optimiser holds, gradient or not."""
    model = _Model()
    model.global_classifier.weight.requires_grad_(False)

    held = {
        id(parameter)
        for group in namespace["build_parameter_groups"](model, 1e-4)
        for parameter in group["params"]
    }
    assert id(model.global_classifier.weight) not in held


def test_the_encoder_is_counted_as_trainable(namespace):
    """B52's whole claim is that the encoder learns. It must be visible."""
    counts = namespace["describe_trainable"](_Model())
    assert counts["encoder"] > 0, "the encoder is not learning; this is not B52"
    assert counts["hierarchy"] > 0
    assert counts["head_and_rest"] > 0


def test_a_model_with_nothing_but_a_frozen_hierarchy_is_refused(namespace):
    """The frozen baseline, arrived at by accident, must not run silently."""
    model = _Model()
    for name, parameter in model.named_parameters():
        if not name.startswith(("global_projection.", "global_classifier.")):
            parameter.requires_grad_(False)

    with pytest.raises(RuntimeError, match="B52 trains the encoder"):
        namespace["build_parameter_groups"](model, 1e-4)


# --- the loss --------------------------------------------------------------


def test_a_zero_confidence_cell_contributes_nothing(namespace):
    """Report silence must not reach a gradient, whatever the stored probability."""
    logits = torch.zeros(2, 12, requires_grad=True)
    target = torch.full((2, 12), 0.5)
    confidence = torch.zeros(2, 12)
    multiplier = torch.ones(12)

    loss = namespace["report_weighted_bce"](logits, target, confidence, multiplier)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0)
    assert float(logits.grad.abs().sum()) == pytest.approx(0.0)


def test_a_rarely_mentioned_target_gets_a_larger_multiplier(namespace):
    """Otherwise the findings reports talk about most dominate every gradient."""
    import numpy as np  # noqa: PLC0415

    confidence = np.ones((100, 12), dtype=np.float32)
    confidence[:, 0] = 0.1  # this target is rarely written about
    multiplier = namespace["target_balance_multipliers"](confidence)
    assert multiplier[0] > multiplier[1]
