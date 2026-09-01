"""The generated B52 script, run end to end on synthetic MRI studies.

The claim this file has to defend is "run it once and it produces every B52
result". Reading the source cannot establish that, so the main test builds a
handful of real DICOM series on disk, runs `main()` exactly as a person would
from a shell, and checks that every promised file arrives and holds sensible
content.

The rest of the tests guard the transform that generates the script from the
notebook: that Colab-only code and the three inherited traps are gone, that no
statement survived which would fire on import, and that the script on disk still
matches its builder.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pydicom = pytest.importorskip("pydicom")
numpy = pytest.importorskip("numpy")

BUILDER = Path(__file__).with_name("build_b52_script.py")
SCRIPT = Path(__file__).with_name("b52_standalone.py")

# The twelve findings, in the order train.csv carries them.
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


@pytest.fixture(scope="module")
def generated() -> str:
    """The script the builder produces right now, as text."""
    namespace = runpy.run_path(str(BUILDER))
    body, _dropped = namespace["build"](
        Path(namespace["__file__"]).with_name("_b52_generated_check.py")
    ), None
    path = body[0]
    text = path.read_text()
    path.unlink()
    return text


@pytest.fixture(scope="module")
def module():
    """Import the checked-in script, so tests exercise what a user runs."""
    spec = importlib.util.spec_from_file_location("b52_standalone_under_test", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


# --- the script is current and importable ----------------------------------


def test_the_script_on_disk_matches_its_builder(generated):
    """A hand-edited script would silently diverge from the notebook."""
    assert SCRIPT.read_text() == generated, (
        "b52_standalone.py is stale or was edited by hand; "
        "run python notebook/build_b52_script.py"
    )


def test_importing_the_script_runs_nothing(module, capsys):
    """A notebook cell mixes definitions with the lines that run them.

    Carried into a script those lines fire on import -- mounting Drive, reading
    CSVs, building a model. Importing must define things and do nothing.
    """
    captured = capsys.readouterr().out
    assert "Preflight" not in captured
    assert "training studies" not in captured


def test_no_top_level_statement_can_run_on_import(generated):
    """The structural version of the test above, so it cannot pass by luck."""
    allowed_names = {"__doc__", "matplotlib"}
    offenders = []

    for node in ast.parse(generated).body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
        ):
            continue
        if isinstance(node, ast.Assign):
            continue  # module constants; a separate test pins which ones
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the module docstring
        if isinstance(node, ast.If):
            # Only the __main__ guard is allowed to do anything.
            test = ast.dump(node.test)
            assert "__main__" in test, f"a conditional runs on import: {test[:80]}"
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            called = getattr(node.value.func, "attr", getattr(node.value.func, "id", ""))
            owner = getattr(getattr(node.value.func, "value", None), "id", "")
            if owner in allowed_names:
                continue  # matplotlib.use("Agg"), which must run before pyplot
            offenders.append(called)
        else:
            offenders.append(type(node).__name__)

    assert not offenders, f"these would run on import: {offenders}"


def test_the_matplotlib_backend_is_fixed_before_pyplot(generated):
    """A machine with no display raises when pyplot picks an interactive backend."""
    backend = generated.index('matplotlib.use("Agg")')
    pyplot = generated.index("import matplotlib.pyplot")
    assert backend < pyplot, "the backend must be chosen before pyplot is imported"


# --- what the transform removed --------------------------------------------


def test_no_trace_of_b37_survives(generated):
    """B37's 0.714 is a leaderboard score from another model on the full data.

    The base notebook saved it beside every run. In a subset script it invites
    exactly the comparison that means nothing.
    """
    for term in ("B37", "b37"):
        assert term not in generated, f"{term} still appears in the generated script"


def test_the_colab_only_code_is_gone(generated):
    """A script is handed paths; it has no Drive to mount and no zip to unpack."""
    for name in (
        "def mount_drive",
        "def copy_and_extract_archives",
        "def find_extracted_root",
        "google.colab",
        "class ArchivePaths",
    ):
        assert name not in generated, f"{name} is Colab-only and should not be here"


def test_the_three_traps_are_gone(generated):
    """Each of these runs, looks right, and is not B52.

    In a notebook they at least sit under headings that say what they are. In a
    flat script they would be one call away from anything.
    """
    traps = {
        "def build_experiment": "trains on the 58 expert-gold studies",
        "def run_epoch": "ignores confidence, so it trains on report silence",
        "def masked_bce_with_logits": "the unweighted loss, not B52's",
        "def run_preflight": "checks gradient flow with the wrong loss",
    }
    for name, why in traps.items():
        assert name not in generated, f"{name} survived: {why}"


def test_b52_still_has_everything_it_needs(generated):
    """The other half of the removal tests: nothing load-bearing was cut."""
    for name in (
        "def augment_series",
        "class AugmentedKneeMRIDataset",
        "def split_report_studies",
        "def build_cosine_schedule",
        "class BestEpoch",
        "def report_weighted_bce",
        "def target_balance_multipliers",
        "def build_parameter_groups",
        "def build_b52_run",
        "def train_b52",
        "def b52_preflight",
        "def prepare_series_tensor_from_dicom",
        "class HighResolutionSparseMIL",
    ):
        assert name in generated, f"{name} is missing; the script cannot run B52"


def test_the_inherited_comments_survived(generated):
    """The transform slices source text rather than regenerating from a tree.

    Regenerating would be simpler and would strip every comment out of the
    inherited code, which is most of what makes it readable.
    """
    assert "# Return a Path object so later cells use" not in generated  # that one went with Drive
    assert generated.count("#") > 400, "the comments were stripped"
    assert "# Reject an unexpected split name before constructing a filesystem path." in generated


def test_every_dropped_definition_has_a_stated_reason():
    """A removal without a reason is indistinguishable from an accident."""
    namespace = runpy.run_path(str(BUILDER))
    for name, reason in namespace["DROP_DEFINITIONS"].items():
        assert reason and len(reason) > 10, f"{name} is dropped without an explanation"


# --- the command line -------------------------------------------------------


def test_the_help_text_works(module, capsys):
    with pytest.raises(SystemExit) as exit_code:
        module.main(["--help"])
    assert exit_code.value.code == 0
    assert "--data-root" in capsys.readouterr().out


def test_the_required_arguments_are_required(module):
    with pytest.raises(SystemExit):
        module.main([])


def test_a_missing_data_root_says_which_file_is_absent(module, tmp_path):
    arguments = module.build_argument_parser().parse_args(
        ["--data-root", str(tmp_path), "--labels", str(tmp_path / "l.csv"), "--out", str(tmp_path / "o")]
    )
    with pytest.raises(FileNotFoundError, match="train.csv"):
        module.run_b52(arguments)


def test_a_missing_label_export_says_how_to_make_one(module, tmp_path, tiny_dataset):
    data_root, _labels = tiny_dataset
    arguments = module.build_argument_parser().parse_args(
        ["--data-root", str(data_root), "--labels", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o")]
    )
    with pytest.raises(FileNotFoundError, match="b23_llm_labels"):
        module.run_b52(arguments)


# --- a real dataset, and a real run ----------------------------------------


def _write_series(directory: Path, frames: int, size: int, seed: int) -> None:
    """Write one small but genuine DICOM series.

    Real files rather than a mocked reader: decoding is where a subset run
    actually fails, and a mock would pass while the real path was broken.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    generator = numpy.random.default_rng(seed)

    for index in range(frames):
        pixels = generator.integers(0, 2048, size=(size, size), dtype=numpy.uint16)
        # A bright block whose position depends on the seed, so the twelve
        # targets have something a model could in principle key on.
        offset = seed % max(size // 2, 1)
        pixels[offset : offset + size // 4, offset : offset + size // 4] += 6000

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        frame = Dataset()
        frame.file_meta = meta
        frame.SOPClassUID = meta.MediaStorageSOPClassUID
        frame.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        frame.Modality = "MR"
        frame.Rows, frame.Columns = size, size
        frame.BitsAllocated = 16
        frame.BitsStored = 16
        frame.HighBit = 15
        frame.PixelRepresentation = 0
        frame.SamplesPerPixel = 1
        frame.PhotometricInterpretation = "MONOCHROME2"
        frame.InstanceNumber = index + 1
        frame.ImagePositionPatient = [0.0, 0.0, float(index) * 3.0]
        frame.PixelSpacing = [0.5, 0.5]
        frame.PixelData = pixels.tobytes()
        frame.save_as(directory / f"{index:04d}.dcm", enforce_file_format=True)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> tuple:
    """Fourteen synthetic studies: twelve report-labelled and two expert-gold.

    Twelve so that a quarter held out is three studies rather than one. A
    single-study hold-out leaves every target with one class, so every AUC is
    undefined and there is nothing for the best-epoch rule to choose between.
    """
    import pandas as pd

    root = tmp_path_factory.mktemp("b52_data")
    data_root = root / "dataset"
    (data_root).mkdir()

    report_only = [f"report-{index}" for index in range(12)]
    gold = [f"gold-{index}" for index in range(2)]

    study_rows, series_rows = [], []
    for position, study in enumerate(report_only + gold):
        series_uid = f"series-{position}"
        _write_series(data_root / "train_series" / study / series_uid, frames=4, size=24, seed=position)

        row = {"StudyInstanceUID": study}
        for target_index, target in enumerate(TARGETS):
            # Gold studies carry real labels; report-only studies are blank in
            # train.csv, which is exactly how the competition data looks.
            row[target] = float((position + target_index) % 2) if study in gold else ""
        study_rows.append(row)

        series_rows.append(
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": "true" if position % 2 else "false",
                "Fat_Suppression": "false",
                "Anatomical_Plane": ("Sagittal", "Coronal", "Axial")[position % 3],
            }
        )

    pd.DataFrame(study_rows).to_csv(data_root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(data_root / "train_series.csv", index=False)

    # The label export, in the exact shape b23_llm_labels.py writes. Alternating
    # states so no target is one-class, which would leave its AUC undefined.
    fixed = {"positive": (0.97, 0.90), "negated": (0.03, 0.90), "unmentioned": (0.50, 0.00)}
    label_rows = []
    for position, study in enumerate(report_only):
        row = {"StudyInstanceUID": study}
        for target_index, target in enumerate(TARGETS):
            # Decorrelated across targets on purpose. A strict parity rule
            # makes all twelve targets agree, so a hold-out that happens to
            # draw one class leaves every AUC undefined at once.
            state = ("positive", "negated")[(position * 7 + target_index * 3) % 5 < 3]
            if target_index == len(TARGETS) - 1 and position == 0:
                state = "unmentioned"  # one silent cell, so the masking is exercised
            probability, confidence = fixed[state]
            row[target] = probability
            row[f"{target}__confidence"] = confidence
            row[f"{target}__state"] = state
        label_rows.append(row)

    labels = root / "training_targets.csv"
    pd.DataFrame(label_rows).to_csv(labels, index=False)
    return data_root, labels


def _arguments(module, data_root, labels, out, *extra):
    return module.build_argument_parser().parse_args(
        [
            "--data-root", str(data_root),
            "--labels", str(labels),
            "--out", str(out),
            "--seed", "2026",
            # A trial geometry: the real 448px x 32 slices would make every test
            # here a multi-minute run, and none of them is about the geometry.
            "--image-size", "64",
            "--slices-per-series", "4",
            "--validation-fraction", "0.25",
            *extra,
        ]
    )


def test_the_preflight_proves_a_gradient_reaches_the_encoder(module, tiny_dataset, tmp_path):
    """B52's whole claim. A silent failure here is the frozen baseline renamed."""
    data_root, labels = tiny_dataset
    out = tmp_path / "preflight"
    arguments = _arguments(module, data_root, labels, out, "--epochs", "1", "--preflight-only")

    module.run_b52(arguments)
    settings = json.loads((out / "config.json").read_text())
    assert settings["trainable_parameters"]["encoder"] > 0
    assert settings["augmentations_on"]["count"] == 7
    assert not (out / "best_model.pt").exists(), "--preflight-only must not train"


def test_one_run_produces_every_promised_result(module, tiny_dataset, tmp_path):
    """The claim in the module docstring, checked by running it.

    Two epochs rather than one: a single epoch cannot show a schedule stepping
    down or an epoch being chosen over another, which are two of B52's three
    changes.
    """
    import pandas as pd

    data_root, labels = tiny_dataset
    out = tmp_path / "run"
    module.run_b52(_arguments(module, data_root, labels, out, "--epochs", "2"))

    expected = [
        "config.json",
        "labels_summary.json",
        "history.json",
        "history.csv",
        "per_target_auc.csv",
        "holdout_predictions.csv",
        "gold_predictions.csv",
        "loss_curve.png",
        "auc_curve.png",
        "best_model.pt",
        "summary.txt",
    ]
    missing = [name for name in expected if not (out / name).is_file()]
    assert not missing, f"the run did not write: {missing}"

    history = json.loads((out / "history.json").read_text())
    assert len(history) == 2, "one row per epoch"
    assert history[0]["holdout_macro_auc"] is not None, (
        "the hold-out gave no usable AUC, so no epoch could be chosen"
    )
    assert history[0]["learning_rate"] > history[1]["learning_rate"], (
        "the cosine did not step down between epochs"
    )
    assert history[0]["kept"] is True, "the first scored epoch is always the best so far"

    predictions = pd.read_csv(out / "holdout_predictions.csv")
    assert "StudyInstanceUID" in predictions.columns
    assert set(TARGETS) <= set(predictions.columns)
    assert ((predictions[TARGETS] >= 0) & (predictions[TARGETS] <= 1)).all().all(), (
        "predictions must be probabilities"
    )

    gold = pd.read_csv(out / "gold_predictions.csv")
    assert len(gold) == 2, "both expert-gold studies must be scored"

    checkpoint = torch.load(out / "best_model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["gold_labels_used"] is False, "gold must never train"
    assert checkpoint["selected_epoch"] in (1, 2)
    assert "optimistically biased" in checkpoint["governance"]
    assert checkpoint["targets"] == TARGETS

    summary = (out / "summary.txt").read_text()
    assert "best epoch" in summary
    assert "not comparable with any leaderboard score" in summary
    assert "B37" not in summary


def test_the_same_seed_reproduces_the_same_run(module, tiny_dataset, tmp_path):
    """Augmentation is random by design; the run must still be repeatable."""
    data_root, labels = tiny_dataset
    first, second = tmp_path / "a", tmp_path / "b"

    module.run_b52(_arguments(module, data_root, labels, first, "--epochs", "1"))
    module.run_b52(_arguments(module, data_root, labels, second, "--epochs", "1"))

    left = json.loads((first / "history.json").read_text())
    right = json.loads((second / "history.json").read_text())
    assert left[0]["train_loss"] == pytest.approx(right[0]["train_loss"], rel=1e-6)


def test_gold_studies_never_enter_training(module, tiny_dataset, tmp_path):
    """The mistake that would inflate every number without crashing anything."""
    data_root, labels = tiny_dataset
    out = tmp_path / "leak"
    module.run_b52(_arguments(module, data_root, labels, out, "--epochs", "1"))

    import pandas as pd

    gold = set(pd.read_csv(out / "gold_predictions.csv")["StudyInstanceUID"])
    holdout = set(pd.read_csv(out / "holdout_predictions.csv")["StudyInstanceUID"])
    assert not (gold & holdout)
    assert all(uid.startswith("gold-") for uid in gold)
    assert all(uid.startswith("report-") for uid in holdout)


def test_a_label_export_containing_gold_is_refused(module, tiny_dataset, tmp_path):
    """A leak here is invisible in the result and inflates every score."""
    import pandas as pd

    data_root, labels = tiny_dataset
    leaked = pd.read_csv(labels)
    leaked.loc[0, "StudyInstanceUID"] = "gold-0"
    path = tmp_path / "leaked_targets.csv"
    leaked.to_csv(path, index=False)

    with pytest.raises(ValueError, match="expert-gold studies"):
        module.run_b52(_arguments(module, data_root, path, tmp_path / "out", "--epochs", "1"))


def test_max_studies_copies_rather_than_edits_the_export(module, tiny_dataset, tmp_path):
    """The export is the record of what the reports said."""
    _data_root, labels = tiny_dataset
    before = labels.read_bytes()

    trimmed = module.limit_labels(labels, 2, tmp_path / "scratch")
    assert trimmed != labels
    assert labels.read_bytes() == before, "the original export was modified"

    import pandas as pd

    assert len(pd.read_csv(trimmed)) == 2


# --- the augmentation presets ----------------------------------------------


def test_the_b53_preset_matches_the_frozen_config(module):
    """A subset run is only a rehearsal if it uses the same strengths.

    The values are read from config/b42_constant_area_aspect_sparse.yaml, which
    is what B53 reads, so the two cannot drift apart silently.
    """
    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "b42_constant_area_aspect_sparse.yaml")
        .read_text(encoding="utf-8")
    )
    preset = module.AUGMENT_PRESETS["b53"]

    for field, key in (
        ("rotation_deg", "b7_rotation_deg"),
        ("translate_frac", "b7_translate_frac"),
        ("scale_jitter", "b7_scale_jitter"),
        ("gamma_jitter", "b7_gamma_jitter"),
        ("noise_std", "b7_noise_std"),
        ("slice_dropout", "b7_slice_dropout"),
        ("bias_field_strength", "b7_bias_field_strength"),
    ):
        assert getattr(preset, field) == pytest.approx(float(config[key])), (
            f"{field} does not match {key} in the frozen config"
        )


def test_the_notebook_preset_is_the_default(module):
    """Adding a preset must not silently change what an existing command does."""
    arguments = module.build_argument_parser().parse_args(
        ["--data-root", "d", "--labels", "l", "--out", "o"]
    )
    assert arguments.augment_preset == "notebook"
    assert module.AUGMENT_PRESETS["notebook"] is module.AUGMENTATION


def test_the_presets_really_differ(module):
    """If they were the same, --augment-preset would be a decoration."""
    notebook = module.asdict(module.AUGMENT_PRESETS["notebook"])
    b53 = module.asdict(module.AUGMENT_PRESETS["b53"])
    differing = [key for key in notebook if notebook[key] != b53[key]]
    assert len(differing) >= 5, f"only {differing} differ"
    assert all(b53[key] <= notebook[key] for key in b53), (
        "B53's settings are the milder ones on every axis"
    )


def test_an_unknown_preset_is_refused(module):
    with pytest.raises(SystemExit):
        module.build_argument_parser().parse_args(
            ["--data-root", "d", "--labels", "l", "--out", "o", "--augment-preset", "wild"]
        )


def test_the_results_record_which_preset_ran(module, tiny_dataset, tmp_path):
    """A folder of results that cannot say how it was augmented is hard to read."""
    data_root, labels = tiny_dataset
    out = tmp_path / "preset"
    module.run_b52(
        _arguments(module, data_root, labels, out, "--epochs", "1", "--augment-preset", "b53")
    )
    settings = json.loads((out / "config.json").read_text())
    assert settings["augmentation_preset"] == "b53"
    assert settings["augmentation"]["rotation_deg"] == pytest.approx(5.0)
