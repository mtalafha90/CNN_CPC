"""Physically move run directories into numbered experiment containers.

Unlike :mod:`tools.organize_runs`, which creates a read-only-style symlink
index, this tool changes the physical location of run directories.  It keeps
the original leaf directory names inside each numbered container because one
experiment can own several runs.  By default it replaces each old root path
with a relative compatibility symlink so recorded checkpoint and result paths
continue to work.

The default is a dry-run.  Every applied migration writes a durable JSON
manifest that can be passed back with ``--rollback``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # ``python tools/migrate_runs.py`` puts tools/, not the repository root,
    # on sys.path.  Add the root so this documented invocation and module-mode
    # imports use the same implementation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.organize_runs import (
    DEFAULT_REGISTRY,
    OrganizationConflict,
    canonical_directory,
    classify,
    load_registry,
)


NUMBERED_DIRECTORY = re.compile(r"^\d{3}_Experiment_")
RESERVED_NAMES = {"by_experiment", "_migration"}


@dataclass(frozen=True)
class MigrationItem:
    source_name: str
    source_path: str
    experiment_number: int
    experiment_code: str
    experiment_title: str
    experiment_status: str
    container_name: str
    container_path: str
    destination_path: str
    action: str


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except FileNotFoundError:
        return False


def _migration_action(source: Path, destination: Path) -> str:
    if source.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            if _same_path(source, destination):
                return "already_migrated"
        return "conflict"
    if not source.is_dir():
        return "not_a_directory"
    if os.path.lexists(destination):
        return "conflict"
    return "move_and_link"


def build_migration_plan(
    runs_root: Path, registry: dict[str, Any]
) -> list[MigrationItem]:
    """Plan moves for classified top-level experiment directories only."""
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise NotADirectoryError(f"runs root must be a real directory: {runs_root}")

    plan: list[MigrationItem] = []
    for source in sorted(runs_root.iterdir(), key=lambda item: item.name.casefold()):
        if source.name in RESERVED_NAMES or NUMBERED_DIRECTORY.match(source.name):
            continue
        result = classify(source.name, registry)
        if result.kind != "experiment":
            continue
        if result.number is None or result.code is None:
            raise ValueError(f"incomplete experiment classification for {source.name}")

        entry = registry["experiments"][result.number - 1]
        container_name = canonical_directory(entry, registry["directory_template"])
        container = runs_root / container_name
        destination = container / source.name
        action = _migration_action(source, destination)
        if action == "not_a_directory":
            continue
        plan.append(
            MigrationItem(
                source_name=source.name,
                source_path=str(source.absolute()),
                experiment_number=result.number,
                experiment_code=result.code,
                experiment_title=result.title or entry["title"],
                experiment_status=result.status or entry["status"],
                container_name=container_name,
                container_path=str(container.absolute()),
                destination_path=str(destination.absolute()),
                action=action,
            )
        )
    return plan


def _all_containers(runs_root: Path, registry: dict[str, Any]) -> list[Path]:
    return [
        runs_root
        / canonical_directory(entry, registry["directory_template"])
        for entry in registry["experiments"]
    ]


def _preflight_apply(
    plan: list[MigrationItem], runs_root: Path, registry: dict[str, Any]
) -> None:
    conflicts = [item for item in plan if item.action == "conflict"]
    if conflicts:
        details = "\n".join(
            f"  - {item.source_path} -> {item.destination_path}"
            for item in conflicts
        )
        raise OrganizationConflict(
            "migration has conflicting source or destination paths:\n" + details
        )

    for container in _all_containers(runs_root, registry):
        if os.path.lexists(container) and (
            container.is_symlink() or not container.is_dir()
        ):
            raise OrganizationConflict(
                f"numbered container must be a real directory: {container}"
            )

    for item in plan:
        if item.action != "move_and_link":
            continue
        source = Path(item.source_path)
        destination = Path(item.destination_path)
        if source.is_symlink() or not source.is_dir():
            raise OrganizationConflict(f"source changed since dry-run: {source}")
        if os.path.lexists(destination):
            raise OrganizationConflict(
                f"destination appeared since dry-run: {destination}"
            )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _new_manifest(
    plan: list[MigrationItem],
    runs_root: Path,
    registry: dict[str, Any],
    compatibility_links: bool,
) -> tuple[Path, dict[str, Any]]:
    stamp = _utc_stamp()
    path = runs_root / "_migration" / "manifests" / f"migration_{stamp}.json"
    counter = 1
    while os.path.lexists(path):
        path = (
            runs_root
            / "_migration"
            / "manifests"
            / f"migration_{stamp}_{counter}.json"
        )
        counter += 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "applying",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root.resolve()),
        "registry_history": registry.get("history_basis", {}),
        "compatibility_links": compatibility_links,
        "created_containers": [],
        "items": [
            {
                **asdict(item),
                "state": (
                    "already_migrated"
                    if item.action == "already_migrated"
                    else "planned"
                ),
                "alias_created": False,
            }
            for item in plan
        ],
    }
    _write_json_atomic(path, payload)
    return path, payload


def apply_migration(
    plan: list[MigrationItem],
    runs_root: Path,
    registry: dict[str, Any],
    *,
    compatibility_links: bool = True,
) -> Path:
    """Apply a fully preflighted migration and return its rollback manifest."""
    _preflight_apply(plan, runs_root, registry)
    manifest_path, manifest = _new_manifest(
        plan, runs_root, registry, compatibility_links
    )

    try:
        for container in _all_containers(runs_root, registry):
            if not container.exists():
                container.mkdir()
                manifest["created_containers"].append(str(container.absolute()))
                _write_json_atomic(manifest_path, manifest)

        for item, journal in zip(plan, manifest["items"]):
            if item.action == "already_migrated":
                continue

            source = Path(item.source_path)
            destination = Path(item.destination_path)
            source.rename(destination)
            journal["state"] = "moved"
            _write_json_atomic(manifest_path, manifest)

            if compatibility_links:
                relative_target = os.path.relpath(destination, start=source.parent)
                source.symlink_to(relative_target, target_is_directory=True)
                journal["alias_created"] = True
                _write_json_atomic(manifest_path, manifest)

        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json_atomic(manifest_path, manifest)
        raise OrganizationConflict(
            "migration stopped; use the manifest to roll back completed moves: "
            f"{manifest_path}"
        ) from error

    return manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"migration manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid migration manifest: {path}: {error}") from error
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("items"), list
    ):
        raise ValueError(f"unsupported migration manifest: {path}")
    return payload


def _preflight_rollback(manifest: dict[str, Any], runs_root: Path) -> None:
    recorded_root = Path(str(manifest.get("runs_root", "")))
    if recorded_root != runs_root.resolve():
        raise OrganizationConflict(
            f"manifest belongs to {recorded_root}, not {runs_root.resolve()}"
        )
    if manifest.get("status") == "rolled_back":
        raise OrganizationConflict("this migration is already rolled back")

    problems: list[str] = []
    for item in manifest["items"]:
        if item.get("state") != "moved":
            continue
        source = Path(item["source_path"])
        destination = Path(item["destination_path"])
        if destination.is_symlink() or not destination.is_dir():
            problems.append(f"moved directory is missing or changed: {destination}")
        if os.path.lexists(source):
            if not source.is_symlink() or not _same_path(source, destination):
                problems.append(f"old path is not the expected alias: {source}")
    if problems:
        raise OrganizationConflict("rollback preflight failed:\n  - " + "\n  - ".join(problems))


def rollback_migration(manifest_path: Path, runs_root: Path | None = None) -> None:
    """Restore every completed move recorded by a migration manifest."""
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    root = (
        runs_root.expanduser().resolve()
        if runs_root is not None
        else Path(manifest["runs_root"]).resolve()
    )
    _preflight_rollback(manifest, root)

    for item in reversed(manifest["items"]):
        if item.get("state") != "moved":
            continue
        source = Path(item["source_path"])
        destination = Path(item["destination_path"])
        if source.is_symlink():
            source.unlink()
        destination.rename(source)
        item["state"] = "rolled_back"
        item["alias_created"] = False
        _write_json_atomic(manifest_path, manifest)

    for raw_container in reversed(manifest.get("created_containers", [])):
        container = Path(raw_container)
        try:
            container.rmdir()
        except OSError:
            # A container may now hold a user-created file.  Never remove it.
            pass

    manifest["status"] = "rolled_back"
    manifest["rolled_back_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)


def print_plan(plan: list[MigrationItem], applying: bool) -> None:
    moving = sum(item.action == "move_and_link" for item in plan)
    existing = sum(item.action == "already_migrated" for item in plan)
    conflicts = sum(item.action == "conflict" for item in plan)
    print(("APPLY" if applying else "DRY RUN") + " physical run migration")
    print(
        f"directories: {len(plan)} classified; {moving} move; "
        f"{existing} already migrated; {conflicts} conflict"
    )
    print()
    for item in plan:
        print(
            f"[{item.action}] {item.source_name} -> "
            f"{item.container_name}/{item.source_name}"
        )
    if not applying:
        print("\nNo files changed. Re-run with --apply after reviewing this plan.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move run directories into numbered containers with rollback support"
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="existing run archive (default: ./runs)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"experiment registry (default: {DEFAULT_REGISTRY})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="perform the planned moves; default is dry-run",
    )
    mode.add_argument(
        "--rollback",
        type=Path,
        metavar="MANIFEST",
        help="restore a migration from its JSON manifest",
    )
    parser.add_argument(
        "--no-compatibility-links",
        action="store_true",
        help="do not leave old root paths as symlinks (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root.expanduser().resolve()
    if args.rollback:
        rollback_migration(args.rollback, runs_root)
        print(f"Rolled back migration: {args.rollback.resolve()}")
        return 0

    registry = load_registry(args.registry.expanduser().resolve())
    plan = build_migration_plan(runs_root, registry)
    print_plan(plan, applying=args.apply)
    if args.apply:
        manifest = apply_migration(
            plan,
            runs_root,
            registry,
            compatibility_links=not args.no_compatibility_links,
        )
        print(f"\nMigration complete. Rollback manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
