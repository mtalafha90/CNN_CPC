"""Safety and provenance tests for the numbered run-archive view."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from tools.organize_runs import (
    DEFAULT_REGISTRY,
    OrganizationConflict,
    RegistryError,
    apply_plan,
    build_plan,
    canonical_directory,
    classify,
    load_registry,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry(DEFAULT_REGISTRY)


def test_history_registry_is_permanent_contiguous_and_unique(registry):
    experiments = registry["experiments"]
    assert len(experiments) == 83
    assert [entry["number"] for entry in experiments] == list(range(1, 84))
    assert len({entry["code"] for entry in experiments}) == 83
    assert registry["history_basis"]["commit_count_reviewed"] == 1143


@pytest.mark.parametrize(
    ("run_name", "expected_code", "expected_number"),
    [
        ("stage1_random", "B0", 1),
        ("b4_3_crossval_ssl", "B4.3", 9),
        ("b6_report_labels_v121", "B6", 11),
        ("b6_b15_gold_diagnostic", "B15", 24),
        ("b23_smoke_400", "B23", 33),
        ("phase9_matched_supervision_v2", "PHASE9_V2", 59),
        ("working_model", "WORKING_CONTROL", 60),
        ("finetune_seed7", "SEED_ENSEMBLE", 65),
        ("b6_plus_llm_fill_all_ft1", "LLM_FILL_ALL", 67),
        (
            "b6_plus_llm_fill_no_synovitis_ft1",
            "LLM_FILL_NO_SYNOVITIS",
            68,
        ),
        ("b35_target_spatial_v2_entry", "B35", 69),
        ("b36_sparse_mil_v1", "B36", 70),
        ("b37_highres_sparse_mil", "B37", 71),
        ("b38_highres_global_tail", "B38", 73),
        ("b41_highres_aspect_sparse_mil", "B41", 76),
        ("b42_constant_area_aspect_sparse_mil", "B42", 77),
        ("b45_plane_calibrated_sparse_mil", "B45", 78),
        ("b46_gold_anchored_crossfit", "B46", 79),
        ("b47_native_grid_sparse_mil", "B47", 80),
        ("b48_global_conditioned_spatial_mil", "B48", 81),
        ("b49_native_tiled_multiscale_mil", "B49", 82),
        ("b50_ordered_slice_selection_split", "B50", 83),
    ],
)
def test_names_visible_in_the_local_archive_map_to_their_lineage(
    registry, run_name, expected_code, expected_number
):
    result = classify(run_name, registry)
    assert result.kind == "experiment"
    assert result.code == expected_code
    assert result.number == expected_number


def test_comparison_and_preflight_files_are_shared_not_fake_experiments(registry):
    comparison = classify("b4_vs_b5.json", registry)
    preflight = classify("preflight_train.json", registry)
    assert comparison.kind == "shared"
    assert comparison.directory == "_Shared/Comparisons"
    assert preflight.kind == "shared"
    assert preflight.directory == "_Shared/Preflight_and_checks"


def test_an_unknown_run_is_retained_for_review(registry):
    result = classify("a_future_experiment", registry)
    assert result.kind == "unclassified"
    assert result.directory == "_Unclassified"


def test_ambiguous_patterns_are_refused_instead_of_guessing(registry):
    ambiguous = copy.deepcopy(registry)
    ambiguous["experiments"][0]["run_patterns"].append("working_model")
    with pytest.raises(RegistryError, match="matches multiple experiments"):
        classify("working_model", ambiguous)


def test_apply_creates_relative_links_and_preserves_originals(tmp_path, registry):
    runs = tmp_path / "runs"
    source = runs / "b35_target_spatial_v2_entry"
    source.mkdir(parents=True)
    (source / "result.json").write_text('{"score": 0.7}', encoding="utf-8")
    (runs / "b4_vs_b5.json").write_text("{}", encoding="utf-8")
    (runs / "unknown_old_run").mkdir()
    output = runs / "by_experiment"

    plan = build_plan(runs, output, registry)
    assert {item.action for item in plan} == {"create_link"}
    apply_plan(plan, output, registry)

    entry = next(item for item in plan if item.source_name == source.name)
    link = Path(entry.destination_path)
    assert link.is_symlink()
    assert not os.path.isabs(os.readlink(link))
    assert link.resolve() == source.resolve()
    assert (source / "result.json").read_text(encoding="utf-8") == '{"score": 0.7}'
    assert (output / "_Shared" / "Comparisons" / "b4_vs_b5.json").is_symlink()
    assert (output / "_Unclassified" / "unknown_old_run").is_symlink()

    index = json.loads((output / "INDEX.json").read_text(encoding="utf-8"))
    assert len(index["runs"]) == 3
    assert (output / "INDEX.csv").is_file()
    assert (output / "README.md").is_file()


def test_every_numbered_experiment_directory_exists_even_without_a_run(
    tmp_path, registry
):
    runs = tmp_path / "runs"
    runs.mkdir()
    output = runs / "by_experiment"
    apply_plan(build_plan(runs, output, registry), output, registry)

    experiment_directories = [
        path
        for path in output.iterdir()
        if path.is_dir() and path.name[0].isdigit()
    ]
    assert len(experiment_directories) == 83
    first = canonical_directory(
        registry["experiments"][0], registry["directory_template"]
    )
    last = canonical_directory(
        registry["experiments"][-1], registry["directory_template"]
    )
    assert (output / first).is_dir()
    assert (output / last).is_dir()


def test_running_twice_is_idempotent_and_does_not_index_the_index(
    tmp_path, registry
):
    runs = tmp_path / "runs"
    (runs / "control").mkdir(parents=True)
    output = runs / "by_experiment"

    first = build_plan(runs, output, registry)
    apply_plan(first, output, registry)
    second = build_plan(runs, output, registry)

    assert len(second) == 1
    assert second[0].source_name == "control"
    assert second[0].action == "already_linked"
    apply_plan(second, output, registry)


def test_a_destination_collision_stops_before_any_overwrite(tmp_path, registry):
    runs = tmp_path / "runs"
    source = runs / "control"
    source.mkdir(parents=True)
    output = runs / "by_experiment"
    classification = classify("control", registry)
    collision = output / classification.directory / "control"
    collision.parent.mkdir(parents=True)
    collision.write_text("keep me", encoding="utf-8")

    plan = build_plan(runs, output, registry)
    assert plan[0].action == "conflict"
    with pytest.raises(OrganizationConflict, match="refusing to overwrite"):
        apply_plan(plan, output, registry)

    assert collision.read_text(encoding="utf-8") == "keep me"
    assert source.is_dir()


def test_an_output_directory_symlink_is_refused(tmp_path, registry):
    runs = tmp_path / "runs"
    (runs / "control").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    output = runs / "by_experiment"
    output.symlink_to(elsewhere, target_is_directory=True)

    plan = build_plan(runs, output, registry)
    with pytest.raises(OrganizationConflict, match="must be a real directory"):
        apply_plan(plan, output, registry)

    assert list(elsewhere.iterdir()) == []


def test_an_index_metadata_symlink_is_refused(tmp_path, registry):
    runs = tmp_path / "runs"
    runs.mkdir()
    output = runs / "by_experiment"
    output.mkdir()
    external = tmp_path / "external.json"
    external.write_text("do not overwrite", encoding="utf-8")
    (output / "INDEX.json").symlink_to(external)

    with pytest.raises(OrganizationConflict, match="metadata path"):
        apply_plan(build_plan(runs, output, registry), output, registry)

    assert external.read_text(encoding="utf-8") == "do not overwrite"
