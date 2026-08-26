"""Static contracts for the separate B48-shaped Colab subset notebook."""
from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("b48_global_conditioned_sparse_mil_colab.ipynb")
BUILDER = Path(__file__).with_name("build_b48_colab_notebook.py")
OLD_NOTEBOOK = Path(__file__).with_name("knee_mri_model.ipynb")


def _notebook() -> dict:
    """Read the generated B48 notebook JSON."""
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _text() -> str:
    """Join sources for concise static contract checks."""
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def test_b48_notebook_is_a_separate_valid_gpu_notebook():
    """The new B48 artifact exists beside, rather than in place of, the old notebook."""
    notebook = _notebook()
    assert NOTEBOOK.is_file()
    assert OLD_NOTEBOOK.is_file()
    assert NOTEBOOK != OLD_NOTEBOOK
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) >= 20


def test_b48_builder_regenerates_the_tracked_new_notebook(tmp_path):
    """The separate builder is the canonical source of the separate artifact."""
    namespace = runpy.run_path(str(BUILDER))
    regenerated = tmp_path / NOTEBOOK.name
    namespace["build"](regenerated)
    assert json.loads(regenerated.read_text(encoding="utf-8")) == _notebook()


def test_every_b48_code_cell_parses_as_python():
    """The teaching-oriented cells remain syntactically runnable in Colab."""
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_b48_uses_the_same_drive_subset_but_isolates_its_outputs():
    """The B48 sandbox reads the prior archives without overwriting old results."""
    text = _text()
    assert '"colab_subset.zip"' in text
    assert '"test.zip"' in text
    assert 'Path("/content/knee_mri_b48_subset")' in text
    assert '"knee_mri_b48_subset_outputs"' in text
    assert "earlier notebook; untouched" in text
    assert "RUN_B48_TRAINING = False" in text


def test_b48_model_exposes_both_matched_query_source_arms():
    """The compact model retains the B48 representation-level comparison."""
    text = _text()
    for name in (
        "CompactGlobalPathologyBranch",
        "B48SubsetSparseEvidenceHead",
        "B48SubsetModel",
        "B48MatchedPair",
        "build_b48_matched_pair",
        "check_zero_start_pair_equivalence",
        "run_b48_pair_preflight",
        "train_b48_matched_pair",
        "evaluate_b48_matched_pair",
        "save_b48_pair_results",
    ):
        assert name in text
    assert '"static_prior_control"' in text
    assert '"post_cross_attention_candidate"' in text
    assert "pathology_prior_before_series_cross_attention" in text
    assert "post_series_cross_attention_query" in text
    assert "context_dim: int = 96" in text
    assert "cosine_low_rank_query_token_compatibility" in text
    assert "global_query.detach()" in text
    assert "context_query = (" in text
    assert ").detach()" in text
    assert "self.context_gate = nn.Parameter(torch.zeros(N_TARGETS))" in text
    assert "self.fusion_gate = nn.Parameter(torch.zeros(N_TARGETS))" in text


def test_b48_pairing_and_preflight_guards_are_explicit():
    """No one-arm or silently mismatched subset comparison can be mistaken for B48."""
    text = _text()
    assert "torch.random.fork_rng" in text
    assert "Matched B48 arms did not start from identical parameter tensors" in text
    assert "zero_start_pair_max_abs_difference" in text
    assert "B48 arms differ before a zero-start gate can open" in text
    assert "common_train_uid_sha256" in text
    assert "common_validation_uid_sha256" in text
    assert "forward/backward only; no optimizer step" in text
    assert "context_gate_gradient_at_zero" in text
    assert "context_projections_zero_at_zero_gate" in text
    assert "context_projections_active_after_opening" in text
    assert "global_branch_isolated_from_local_loss" in text
    assert "b48_subset_comparison.json" in text
    assert "topk_change_fraction_by_target" in text


def test_b48_notebook_states_its_scientific_scope_and_stays_standalone():
    """Subset labels cannot be represented as the full scanner-domain B48 endpoint."""
    text = _text()
    assert "compact subset sandbox" in text
    assert "not the official B48 result" in text
    assert "scanner-domain split" in text
    assert "report-only fill artifact" in text
    assert "from rsna_knee" not in text
    assert "git clone" not in text
    assert "load_state_dict" not in text
    assert "B46" not in text


def test_b48_keeps_the_earlier_notebook_memory_safety_primitives():
    """The separate architecture does not discard the tested Drive/DICOM safeguards."""
    text = _text()
    assert "safe_extract_zip" in text
    assert "percentile_sample_cap: int = 262_144" in text
    assert "gradient_checkpointing: bool = True" in text
    assert "encoder_chunk_size: int = 1" in text
    assert "max_series_per_study: int = 4" in text
    assert "from torch.utils.checkpoint import checkpoint" in text
    assert "host_peak_rss_gib" in text
