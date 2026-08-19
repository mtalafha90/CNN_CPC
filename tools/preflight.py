"""Check a training run's inputs before spending the GPU on them.

Training takes about ninety minutes and validates its inputs as it reaches
them, so a wrong path can fail at minute one or at minute sixty-eight. That has
already happened here, and the cost is not the error -- it is the hour.

Every check below calls the same loader training calls, so it cannot drift into
approving something training would reject. Nothing is loaded onto the GPU and
no images are read, so this finishes in seconds.

It also reports any input living outside the working directory. That is not an
error -- a path can legitimately point anywhere -- but a run that depends on a
folder you are about to tidy away is worth knowing about while it is still
cheap to fix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model._implementation import ensure_developments_source

ensure_developments_source()


def _check(name: str, function) -> dict:
    """Run one check and turn any failure into a report line rather than a crash."""
    try:
        detail = function()
    except Exception as error:
        return {"name": name, "ok": False, "detail": f"{type(error).__name__}: {error}"}
    return {"name": name, "ok": True, "detail": detail}


def check_data_root(path: Path) -> str:
    from rsna_knee.data import load_train_csv

    train = load_train_csv(path / "train.csv")
    if len(train) != 4407:
        raise ValueError(
            f"training release should hold 4,407 studies, this holds {len(train)}"
        )
    for name in ("train_series.csv", "test.csv"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"missing {name}")
    return f"{len(train)} studies, train_series.csv and test.csv present"


def check_latin_script_labels(path: Path) -> str:
    from rsna_knee.b7_weak_supervision import load_frozen_b6_export

    frame, policy, _ = load_frozen_b6_export(path)
    return f"{len(frame)} rows, parser v{policy.get('version')}"


def check_all_script_labels(path: Path) -> str:
    from rsna_knee.phase9_supervision import load_frozen_phase8_export

    frame, policy, _ = load_frozen_phase8_export(path)
    return f"{len(frame)} rows, merge v{policy.get('version')}, fingerprint matches"


def check_series_policy(path: Path) -> str:
    from rsna_knee.b13_training import B13_SERIES_SIGNATURE

    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = policy.get("series_summary", {}).get("series_signature_sha256")
    if signature != B13_SERIES_SIGNATURE:
        raise ValueError(
            "this is not the frozen all-series policy; its signature is "
            f"{str(signature)[:12]}..., training requires {B13_SERIES_SIGNATURE[:12]}..."
        )
    return "frozen all-series policy"


def check_encoder(path: Path) -> str:
    from rsna_knee.b16_report_ssl import load_b16_report_encoder

    load_b16_report_encoder(path)
    return f"{Path(path).stat().st_size / 1e6:.0f} MB, loads"


CHECKS = {
    "data root": check_data_root,
    "latin-script labels": check_latin_script_labels,
    "all-script labels": check_all_script_labels,
    "series policy": check_series_policy,
    "encoder": check_encoder,
}


def outside_working_directory(path: Path, working: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(working).resolve())
    except ValueError:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a training run's inputs before starting it"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--latin-script-labels", required=True, type=Path)
    parser.add_argument("--all-script-labels", required=True, type=Path)
    parser.add_argument("--series-policy", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--working-directory",
        type=Path,
        default=Path.cwd(),
        help="inputs outside this folder are pointed out; defaults to the current one",
    )
    args = parser.parse_args()

    given = {
        "data root": args.data_root,
        "latin-script labels": args.latin_script_labels,
        "all-script labels": args.all_script_labels,
        "series policy": args.series_policy,
        "encoder": args.encoder_checkpoint,
    }

    results = []
    for name, function in CHECKS.items():
        path = given[name]
        if not path.exists():
            results.append({"name": name, "ok": False, "detail": f"no such path: {path}"})
            continue
        results.append(_check(name, lambda f=function, p=path: f(p)))

    width = max(len(name) for name in given)
    for result in results:
        mark = "ok  " if result["ok"] else "FAIL"
        print(f"{mark}  {result['name']:<{width}}  {result['detail']}")
        print(f"      {'':<{width}}  {given[result['name']]}")

    strays = [
        name
        for name, path in given.items()
        if path.exists() and outside_working_directory(path, args.working_directory)
    ]
    if strays:
        print(
            f"\nOutside {args.working_directory}: {', '.join(strays)}.\n"
            "The run will work, but it depends on a folder you may be about to "
            "move or delete. Copy it in and repoint the variable."
        )

    failed = [r["name"] for r in results if not r["ok"]]
    if failed:
        raise SystemExit(
            f"\n{len(failed)} input(s) not usable: {', '.join(failed)}. "
            "Fix these before starting the run."
        )
    print("\nAll inputs check out. The run can start.")


if __name__ == "__main__":
    main()
