"""Physical run migration must remain reversible and path-compatible."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.migrate_runs import (
    apply_migration,
    build_migration_plan,
    rollback_migration,
)
from tools.organize_runs import (
    DEFAULT_REGISTRY,
    OrganizationConflict,
    build_plan,
    classify,
    load_registry,
)


@pytest.fixture(scope="module")
def registry():
    return load_registry(DEFAULT_REGISTRY)


def test_plan_selects_real_experiment_directories_only(tmp_path, registry):
    runs = tmp_path / "runs"
    (runs / "b5_report_ssl").mkdir(parents=True)
    (runs / "audit").mkdir()
    (runs / "b35_target_spatial_v1.log").write_text("log", encoding="utf-8")
    (runs / "by_experiment").mkdir()
    (runs / "001_Experiment_B0_random_init_stage1").mkdir()

    plan = build_migration_plan(runs, registry)

    assert [item.source_name for item in plan] == ["b5_report_ssl"]
    assert plan[0].experiment_code == "B5"
    assert plan[0].action == "move_and_link"


def test_apply_moves_directory_and_leaves_relative_compatibility_link(
    tmp_path, registry
):
    runs = tmp_path / "runs"
    source = runs / "b35_target_spatial_v2_entry"
    source.mkdir(parents=True)
    (source / "model.pt").write_bytes(b"checkpoint")
    result = classify(source.name, registry)

    manifest_path = apply_migration(
        build_migration_plan(runs, registry), runs, registry
    )

    container = runs / result.directory
    destination = container / source.name
    assert container.is_dir()
    assert destination.is_dir() and not destination.is_symlink()
    assert (destination / "model.pt").read_bytes() == b"checkpoint"
    assert source.is_symlink()
    assert not os.path.isabs(os.readlink(source))
    assert source.resolve() == destination.resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["compatibility_links"] is True
    assert manifest["items"][0]["state"] == "moved"
    assert manifest["items"][0]["alias_created"] is True


def test_multiple_runs_share_one_numbered_container(tmp_path, registry):
    runs = tmp_path / "runs"
    (runs / "b5_report_ssl").mkdir(parents=True)
    (runs / "b5_frozen_probe").mkdir()

    plan = build_migration_plan(runs, registry)
    assert {item.experiment_code for item in plan} == {"B5"}
    assert len({item.container_name for item in plan}) == 1
    apply_migration(plan, runs, registry)

    container = runs / plan[0].container_name
    assert (container / "b5_report_ssl").is_dir()
    assert (container / "b5_frozen_probe").is_dir()


def test_every_numbered_container_is_created(tmp_path, registry):
    runs = tmp_path / "runs"
    runs.mkdir()
    apply_migration(build_migration_plan(runs, registry), runs, registry)

    containers = [
        path
        for path in runs.iterdir()
        if path.is_dir() and path.name[:3].isdigit() and "_Experiment_" in path.name
    ]
    assert len(containers) == 72


def test_migration_is_idempotent(tmp_path, registry):
    runs = tmp_path / "runs"
    (runs / "control").mkdir(parents=True)
    first = build_migration_plan(runs, registry)
    apply_migration(first, runs, registry)

    second = build_migration_plan(runs, registry)
    assert len(second) == 1
    assert second[0].action == "already_migrated"
    apply_migration(second, runs, registry)


def test_organizer_ignores_physical_numbered_containers(tmp_path, registry):
    runs = tmp_path / "runs"
    (runs / "control").mkdir(parents=True)
    apply_migration(build_migration_plan(runs, registry), runs, registry)
    output = runs / "by_experiment"

    plan = build_plan(runs, output, registry)

    assert [item.source_name for item in plan] == ["control"]
    assert plan[0].action == "create_link"


def test_destination_collision_stops_before_any_move(tmp_path, registry):
    runs = tmp_path / "runs"
    source = runs / "control"
    source.mkdir(parents=True)
    result = classify("control", registry)
    destination = runs / result.directory / "control"
    destination.mkdir(parents=True)
    (destination / "existing.txt").write_text("keep", encoding="utf-8")

    plan = build_migration_plan(runs, registry)
    assert plan[0].action == "conflict"
    with pytest.raises(OrganizationConflict, match="conflicting"):
        apply_migration(plan, runs, registry)

    assert source.is_dir() and not source.is_symlink()
    assert (destination / "existing.txt").read_text(encoding="utf-8") == "keep"
    assert not (runs / "_migration").exists()


def test_rollback_restores_original_locations_and_contents(tmp_path, registry):
    runs = tmp_path / "runs"
    first = runs / "b5_report_ssl"
    second = runs / "b5_frozen_probe"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "history.json").write_text('{"epochs": 8}', encoding="utf-8")

    plan = build_migration_plan(runs, registry)
    manifest = apply_migration(plan, runs, registry)
    rollback_migration(manifest, runs)

    assert first.is_dir() and not first.is_symlink()
    assert second.is_dir() and not second.is_symlink()
    assert (first / "history.json").read_text(encoding="utf-8") == '{"epochs": 8}'
    assert not (runs / plan[0].container_name).exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "rolled_back"
    assert {item["state"] for item in payload["items"]} == {"rolled_back"}


def test_rollback_refuses_a_changed_alias_before_moving_anything(
    tmp_path, registry
):
    runs = tmp_path / "runs"
    first = runs / "b5_report_ssl"
    second = runs / "b5_frozen_probe"
    first.mkdir(parents=True)
    second.mkdir()
    manifest = apply_migration(build_migration_plan(runs, registry), runs, registry)

    first.unlink()
    first.mkdir()
    with pytest.raises(OrganizationConflict, match="old path is not the expected alias"):
        rollback_migration(manifest, runs)

    assert first.is_dir() and not first.is_symlink()
    assert second.is_symlink()
    assert (runs / classify(second.name, registry).directory / second.name).is_dir()
