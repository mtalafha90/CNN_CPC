"""Static checks for the standalone Google Colab notebook."""
from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("knee_mri_model.ipynb")
BUILDER = Path(__file__).with_name("build_notebook.py")


def _notebook() -> dict:
    """Read the generated notebook JSON."""
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _text() -> str:
    """Join notebook sources for simple static contract checks."""
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def test_notebook_is_valid_json_with_gpu_metadata():
    """The generated notebook remains a GPU-ready nbformat v4 file."""
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) >= 14


def test_builder_regenerates_the_tracked_notebook(tmp_path):
    """The builder is the canonical source of the checked-in notebook JSON."""
    namespace = runpy.run_path(str(BUILDER))
    regenerated = tmp_path / "knee_mri_model.ipynb"
    namespace["build"](regenerated)
    assert json.loads(regenerated.read_text()) == _notebook()


def test_every_code_cell_parses_as_python():
    """Comments and teaching text did not introduce invalid notebook Python."""
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_is_standalone_with_clear_functions_and_classes():
    """The notebook exposes its data/model/train workflow without project imports."""
    text = _text()
    for name in (
        "mount_drive",
        "copy_and_extract_archives",
        "safe_extract_zip",
        "find_extracted_root",
        "validate_dataset",
        "validate_test_dataset",
        "read_dicom_volume",
        "prepare_series_tensor",
        "build_experiment",
        "build_test_loader",
        "predict_test_set",
        "run_preflight",
        "train_model",
        "plot_loss_history",
        "show_case_examples",
        "save_results",
        "show_results",
    ):
        assert f"def {name}" in text
    for name in (
        "DrivePaths",
        "ArchivePaths",
        "TestPaths",
        "RunConfig",
        "KneeMRIDataset",
        "SliceEncoder",
        "SparseEvidenceHead",
        "HighResolutionSparseMIL",
    ):
        assert f"class {name}" in text
    assert "from rsna_knee" not in text
    assert "git clone" not in text
    assert "load_state_dict" not in text
    assert "448×448" in text


def test_teaching_outputs_are_enabled_but_training_starts_safe():
    """Loss plotting and the twelve-case review are available behind an opt-in switch."""
    text = _text()
    assert "forward/backward only; no optimizer step" in text
    assert "RUN_TRAINING = False" in text
    assert "plot_loss_history(EXPERIMENT)" in text
    assert "max_cases=12" in text
    assert "colab_subset.zip" in text
    assert "test.zip" in text
    assert "test_predictions.csv" in text
