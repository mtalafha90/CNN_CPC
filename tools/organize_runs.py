"""Build a numbered, non-destructive view of the project run archive.

The historical ``runs/`` directory is part experiment archive and part working
area.  Renaming its children in place would break checkpoint paths embedded in
scripts, logs, notebooks, and reports.  This tool instead creates a stable
``runs/by_experiment`` index made from relative symbolic links.  The originals
never move and are never renamed.

Dry-run first::

    python tools/organize_runs.py --runs-root /media/talafha/Disk_1/CNN_CPC/runs

Create the index after reviewing the plan::

    python tools/organize_runs.py --runs-root /media/talafha/Disk_1/CNN_CPC/runs --apply
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "config" / "experiment_registry.json"
DEFAULT_OUTPUT_NAME = "by_experiment"
NUMBERED_DIRECTORY = re.compile(r"^\d{3}_Experiment_")
RESERVED_RUN_DIRECTORIES = {"_migration"}


class RegistryError(ValueError):
    """The experiment registry is incomplete, ambiguous, or malformed."""


class OrganizationConflict(RuntimeError):
    """An existing file prevents the index from being built safely."""


@dataclass(frozen=True)
class Classification:
    kind: str
    directory: str
    code: str | None = None
    number: int | None = None
    title: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class PlanItem:
    source_name: str
    source_path: str
    kind: str
    experiment_number: int | None
    experiment_code: str | None
    experiment_title: str | None
    experiment_status: str | None
    group_directory: str
    destination_path: str
    action: str


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load and validate the permanent experiment-number registry."""
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryError(f"registry does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RegistryError(f"registry is not valid JSON: {path}: {error}") from error
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise RegistryError("registry.experiments must be a non-empty list")

    numbers = [entry.get("number") for entry in experiments]
    expected = list(range(1, len(experiments) + 1))
    if numbers != expected:
        raise RegistryError(
            "experiment numbers must be contiguous and stored in chronological "
            f"order; expected {expected}, got {numbers}"
        )

    codes = [entry.get("code") for entry in experiments]
    if any(not isinstance(code, str) or not code for code in codes):
        raise RegistryError("every experiment needs a non-empty string code")
    if len(set(codes)) != len(codes):
        raise RegistryError("experiment codes must be unique")

    template = registry.get("directory_template")
    if not isinstance(template, str) or not template:
        raise RegistryError("directory_template must be a non-empty string")

    directories: set[str] = set()
    for entry in experiments:
        for field in ("slug", "title", "status", "introduced_commit"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise RegistryError(f"experiment {entry['code']} needs {field}")
        patterns = entry.get("run_patterns")
        if not isinstance(patterns, list) or any(
            not isinstance(pattern, str) or not pattern for pattern in patterns
        ):
            raise RegistryError(
                f"experiment {entry['code']} run_patterns must be a string list"
            )
        directory = canonical_directory(entry, template)
        if Path(directory).name != directory or directory in {".", ".."}:
            raise RegistryError(
                f"experiment {entry['code']} produces an unsafe directory: {directory}"
            )
        if directory in directories:
            raise RegistryError(f"duplicate canonical directory: {directory}")
        directories.add(directory)

    shared = registry.get("shared_groups", [])
    if not isinstance(shared, list):
        raise RegistryError("shared_groups must be a list")
    shared_directories: set[str] = set()
    for group in shared:
        directory = group.get("directory")
        patterns = group.get("patterns")
        if not isinstance(directory, str) or Path(directory).name != directory:
            raise RegistryError(f"unsafe shared directory: {directory!r}")
        if directory in shared_directories:
            raise RegistryError(f"duplicate shared directory: {directory}")
        shared_directories.add(directory)
        if not isinstance(patterns, list) or any(
            not isinstance(pattern, str) or not pattern for pattern in patterns
        ):
            raise RegistryError(
                f"shared group {directory} patterns must be a string list"
            )


def canonical_directory(entry: dict[str, Any], template: str) -> str:
    """Render one permanent ``001_Experiment_...`` directory name."""
    try:
        return template.format(**entry)
    except (KeyError, ValueError) as error:
        raise RegistryError(
            f"cannot render directory for experiment {entry.get('code')}: {error}"
        ) from error


def _pattern_matches(name: str, patterns: Iterable[str]) -> bool:
    folded = name.casefold()
    return any(fnmatch.fnmatchcase(folded, pattern.casefold()) for pattern in patterns)


def classify(name: str, registry: dict[str, Any]) -> Classification:
    """Classify one immediate child of ``runs/`` by its existing name."""
    experiment_matches = [
        entry
        for entry in registry["experiments"]
        if _pattern_matches(name, entry["run_patterns"])
    ]
    if len(experiment_matches) > 1:
        codes = ", ".join(entry["code"] for entry in experiment_matches)
        raise RegistryError(f"{name!r} matches multiple experiments: {codes}")
    if experiment_matches:
        entry = experiment_matches[0]
        directory = canonical_directory(entry, registry["directory_template"])
        return Classification(
            kind="experiment",
            directory=directory,
            code=entry["code"],
            number=entry["number"],
            title=entry["title"],
            status=entry["status"],
        )

    shared_matches = [
        group
        for group in registry.get("shared_groups", [])
        if _pattern_matches(name, group["patterns"])
    ]
    if len(shared_matches) > 1:
        groups = ", ".join(group["directory"] for group in shared_matches)
        raise RegistryError(f"{name!r} matches multiple shared groups: {groups}")
    if shared_matches:
        return Classification(
            kind="shared",
            directory=str(Path("_Shared") / shared_matches[0]["directory"]),
        )

    return Classification(kind="unclassified", directory="_Unclassified")


def _link_action(source: Path, destination: Path) -> str:
    """Return the safe action for a planned link without changing the disk."""
    if not os.path.lexists(destination):
        return "create_link"
    if destination.is_symlink():
        try:
            if destination.resolve(strict=True) == source.resolve(strict=True):
                return "already_linked"
        except FileNotFoundError:
            pass
    return "conflict"


def build_plan(
    runs_root: Path,
    output_root: Path,
    registry: dict[str, Any],
) -> list[PlanItem]:
    """Describe every link needed for the current top-level run archive."""
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")
    if not runs_root.is_dir():
        raise NotADirectoryError(f"runs root is not a directory: {runs_root}")
    if runs_root.resolve() == output_root.resolve():
        raise ValueError("output root must not be the runs root itself")

    output_resolved = output_root.resolve()
    plan: list[PlanItem] = []
    for source in sorted(runs_root.iterdir(), key=lambda item: item.name.casefold()):
        if source.resolve() == output_resolved:
            continue
        # Physical migrations place canonical containers at the run root.
        # They are destinations, not unclassified runs to index again.
        if source.name in RESERVED_RUN_DIRECTORIES or NUMBERED_DIRECTORY.match(
            source.name
        ):
            continue
        classification = classify(source.name, registry)
        destination = output_root / classification.directory / source.name
        plan.append(
            PlanItem(
                source_name=source.name,
                # Point to the archive entry itself, not through it.  If an old
                # run entry is already a symlink, the new view must preserve
                # that indirection rather than silently changing provenance.
                source_path=str(source.absolute()),
                kind=classification.kind,
                experiment_number=classification.number,
                experiment_code=classification.code,
                experiment_title=classification.title,
                experiment_status=classification.status,
                group_directory=classification.directory,
                destination_path=str(destination.absolute()),
                action=_link_action(source, destination),
            )
        )
    return plan


def _all_index_directories(
    output_root: Path, registry: dict[str, Any]
) -> list[Path]:
    directories = [
        output_root
        / canonical_directory(entry, registry["directory_template"])
        for entry in registry["experiments"]
    ]
    directories.extend(
        output_root / "_Shared" / group["directory"]
        for group in registry.get("shared_groups", [])
    )
    directories.append(output_root / "_Unclassified")
    return directories


def _check_output_directories(output_root: Path, registry: dict[str, Any]) -> None:
    if os.path.lexists(output_root) and (
        output_root.is_symlink() or not output_root.is_dir()
    ):
        raise OrganizationConflict(
            f"output root must be a real directory, not a file or symlink: {output_root}"
        )
    for directory in _all_index_directories(output_root, registry):
        if os.path.lexists(directory) and (
            directory.is_symlink() or not directory.is_dir()
        ):
            raise OrganizationConflict(
                "required index directory is occupied by a file or symlink: "
                f"{directory}"
            )
    for filename in ("INDEX.csv", "INDEX.json", "README.md"):
        path = output_root / filename
        if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
            raise OrganizationConflict(
                f"index metadata path must be a real file, not a symlink: {path}"
            )


def apply_plan(
    plan: list[PlanItem],
    output_root: Path,
    registry: dict[str, Any],
) -> None:
    """Create the relative-link index after a full conflict preflight."""
    conflicts = [item for item in plan if item.action == "conflict"]
    if conflicts:
        paths = "\n".join(f"  - {item.destination_path}" for item in conflicts)
        raise OrganizationConflict(
            "refusing to overwrite existing index entries:\n" + paths
        )
    _check_output_directories(output_root, registry)

    for directory in _all_index_directories(output_root, registry):
        directory.mkdir(parents=True, exist_ok=True)

    for item in plan:
        if item.action == "already_linked":
            continue
        source = Path(item.source_path)
        destination = Path(item.destination_path)
        relative_target = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_target, target_is_directory=source.is_dir())

    write_indexes(plan, output_root, registry)


def _index_rows(plan: list[PlanItem]) -> list[dict[str, Any]]:
    return [asdict(item) for item in plan]


def write_indexes(
    plan: list[PlanItem], output_root: Path, registry: dict[str, Any]
) -> None:
    """Write machine-readable indexes and a human-readable experiment ledger."""
    rows = _index_rows(plan)
    json_payload = {
        "schema_version": 1,
        "history_basis": registry.get("history_basis", {}),
        "runs": rows,
    }
    (output_root / "INDEX.json").write_text(
        json.dumps(json_payload, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = list(PlanItem.__dataclass_fields__)
    with (output_root / "INDEX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "README.md").write_text(
        render_readme(plan, registry), encoding="utf-8"
    )


def render_readme(plan: list[PlanItem], registry: dict[str, Any]) -> str:
    matched = Counter(
        item.experiment_number
        for item in plan
        if item.kind == "experiment" and item.experiment_number is not None
    )
    shared_count = sum(item.kind == "shared" for item in plan)
    unclassified_count = sum(item.kind == "unclassified" for item in plan)
    lines = [
        "# Numbered experiment run index",
        "",
        "This directory is a non-destructive view of the original `runs/` archive.",
        "Every run entry here is a relative symbolic link; original folders and files",
        "remain at their old paths.",
        "",
        f"Indexed items: **{len(plan)}**; shared artifacts: **{shared_count}**; "
        f"unclassified: **{unclassified_count}**.",
        "",
        "| Number | Experiment | Status | Matched items | Title |",
        "|---:|---|---|---:|---|",
    ]
    for entry in registry["experiments"]:
        lines.append(
            f"| {entry['number']:03d} | `{entry['code']}` | `{entry['status']}` | "
            f"{matched[entry['number']]} | {entry['title']} |"
        )
    lines.extend(
        [
            "",
            "`INDEX.csv` and `INDEX.json` contain the source-to-index mapping.",
            "Items in `_Unclassified` are retained deliberately for manual review.",
            "",
        ]
    )
    return "\n".join(lines)


def print_plan(plan: list[PlanItem], output_root: Path, applying: bool) -> None:
    prefix = "APPLY" if applying else "DRY RUN"
    counts = Counter(item.kind for item in plan)
    actions = Counter(item.action for item in plan)
    print(f"{prefix}: {len(plan)} top-level run item(s)")
    print(f"output: {output_root.absolute()}")
    print(
        "classification: "
        f"{counts['experiment']} experiment, {counts['shared']} shared, "
        f"{counts['unclassified']} unclassified"
    )
    print(
        "actions: "
        f"{actions['create_link']} create, {actions['already_linked']} existing, "
        f"{actions['conflict']} conflict"
    )
    print()
    for item in plan:
        label = item.experiment_code or item.group_directory
        print(f"[{item.action}] {item.source_name} -> {label}")
    if not applying:
        print("\nNo files changed. Re-run with --apply after reviewing this plan.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a numbered symlink index without moving historical runs"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="existing run archive (default: ./runs)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="index destination (default: RUNS_ROOT/by_experiment)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"experiment registry (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create directories, relative links, and indexes; default is dry-run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else runs_root / DEFAULT_OUTPUT_NAME
    )
    registry = load_registry(args.registry.expanduser().resolve())
    plan = build_plan(runs_root, output_root, registry)
    print_plan(plan, output_root, applying=args.apply)
    if args.apply:
        apply_plan(plan, output_root, registry)
        print(f"\nCreated/updated index: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
