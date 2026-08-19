"""List every checkpoint under a folder and say what each one actually is.

Checkpoints accumulate. They end up in several working copies, under run
directories named after whatever the experiment was called that afternoon, and
a file called `model_finetuned.pt` is only fine-tuned because somebody typed
that when they copied it. A Kaggle submission has already been held up by
exactly this: the filename said one thing and nothing on hand could confirm it.

Every checkpoint carries the record its training run wrote -- which encoder,
how many epochs, which label surface, and the encoder's fingerprint before and
after training. That last pair settles the question a filename cannot: the two
are taken at the start and the end, so they disagree only if the encoder
really moved.

This reads that record and prints one line per file. It opens checkpoints on
the CPU and never builds a model, so it is quick and needs no GPU.

Two files with the same content hash are the same checkpoint copied twice, and
are reported as such -- which is what makes this safe to run before merging
two folders: identical copies can be dropped, and everything else has to be
looked at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

# Reading a whole multi-hundred-megabyte file to identify it is wasteful when a
# prefix already separates checkpoints that differ. Collisions within one
# folder of models are not a realistic concern here.
HASH_BYTES = 8 * 1024 * 1024


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(HASH_BYTES))
    return digest.hexdigest()[:12]


# Which label surface each experiment arm trained on. The arm names are the
# older vocabulary; the surface names say what the arm actually read.
ARM_SURFACES = {"control": "latin-script", "candidate": "all-script"}


def supervision_label(payload: dict) -> str:
    """Name the label surface, whatever shape the checkpoint recorded it in.

    Later runs store a plain surface name. Earlier ones store the full
    supervision summary -- a nested dict of per-target cell counts -- and put
    the arm in a separate field. Printing that dict raw buries the one word the
    reader is looking for in several hundred characters of counts.
    """
    supervision = payload.get("supervision")
    if isinstance(supervision, str) and supervision:
        return supervision

    arm = str(payload.get("arm") or payload.get("supervision_source", {}).get("arm", ""))
    if arm in ARM_SURFACES:
        return f"{ARM_SURFACES[arm]} ({arm} arm)"
    return arm or "unrecorded"


def training_population(payload: dict) -> str | None:
    """How many studies trained this model, when it held some out.

    A model trained on a subset is not comparable with one trained on the whole
    population, and that difference is invisible in a filename.
    """
    supervision = payload.get("supervision")
    if not isinstance(supervision, dict):
        return None
    trained = supervision.get("training_studies")
    held_out = supervision.get("held_out_pv2_studies")
    if trained is None:
        return None
    if held_out:
        return f"{trained:,} studies ({held_out} held out)"
    return f"{trained:,} studies"


def describe_checkpoint(path: Path) -> dict:
    """Read one checkpoint's own record of the run that produced it."""
    record: dict = {
        "path": str(path),
        "name": path.name,
        "megabytes": round(path.stat().st_size / 1e6, 1),
        "saved": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "content": _content_hash(path),
    }
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # a truncated or half-copied file
        record["readable"] = False
        record["problem"] = f"{type(error).__name__}: {error}"
        return record

    if not isinstance(payload, dict):
        record["readable"] = False
        record["problem"] = f"payload is {type(payload).__name__}, not a checkpoint"
        return record

    before = payload.get("encoder_sha256_initial")
    after = payload.get("encoder_sha256_final")
    record.update(
        {
            "readable": True,
            "encoder": str(payload.get("encoder_source", "report-aligned")),
            "epochs": payload.get("completed_epochs"),
            "supervision": supervision_label(payload),
            "training_population": training_population(payload),
            "seed": (payload.get("config") or {}).get("seed"),
            "stages_free": payload.get("encoder_trainable_stages", 0),
            # The fingerprints outrank any stored flag: `encoder_frozen` was
            # written as a constant True for every run before fine-tuning
            # existed, so it still claims a frozen encoder for runs that moved.
            "fine_tuned": bool(before and after and before != after),
            "encoder_sha": (after or "")[:12] or None,
        }
    )
    return record


def checkpoints_under(root: Path) -> list[Path]:
    """The checkpoints a given path refers to.

    A single checkpoint is as reasonable a thing to ask about as a folder full
    of them -- "is this the file I think it is" is the question this tool
    exists for. `rglob` on a file quietly yields nothing, so naming one used to
    report "no .pt files found", which reads as an answer and is not one.
    """
    if not root.exists():
        raise FileNotFoundError(f"no such file or folder: {root}")
    if root.is_file():
        if root.suffix != ".pt":
            raise ValueError(f"not a checkpoint: {root} (expected a .pt file)")
        return [root]
    return sorted(root.rglob("*.pt"))


def inventory(roots: list[Path]) -> list[dict]:
    found: list[dict] = []
    seen: set[Path] = set()
    for root in roots:
        for path in checkpoints_under(root):
            resolved = path.resolve()
            if resolved in seen:  # the same file reached by two paths
                continue
            seen.add(resolved)
            found.append(describe_checkpoint(path))
    return found


def duplicates(records: list[dict]) -> dict[str, list[str]]:
    """Group checkpoints whose contents match, so copies can be spotted."""
    by_content: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_content[record["content"]].append(record["path"])
    return {k: v for k, v in by_content.items() if len(v) > 1}


def _report(records: list[dict], roots: list[Path] | None = None) -> None:
    if not records:
        # Say where the search looked. "Nothing found" without that is a
        # statement the reader cannot act on or check.
        print("no .pt files found under:")
        for root in roots or []:
            print(f"    {Path(root).resolve()}")
        return

    print(f"{len(records)} checkpoint(s)\n")
    for record in sorted(records, key=lambda r: r["saved"]):
        print(record["path"])
        if not record["readable"]:
            print(f"    UNREADABLE -- {record['problem']}")
            print("    this file is damaged; do not submit or copy it over a good one\n")
            continue
        tuned = "fine-tuned" if record["fine_tuned"] else "frozen encoder"
        print(
            f"    {record['saved']}  {record['megabytes']} MB  {record['content']}"
        )
        print(
            f"    {record['encoder']} | {tuned}"
            f" | stages free {record['stages_free']}"
            f" | epochs {record['epochs']}"
        )
        line = f"    supervision {record['supervision']} | seed {record['seed']}"
        if record.get("training_population"):
            line += f" | trained on {record['training_population']}"
        print(line)
        print()

    copies = duplicates(records)
    if copies:
        print("identical copies (same contents, so keeping one is enough):")
        for content, paths in copies.items():
            print(f"  {content}")
            for path in paths:
                print(f"      {path}")
        print()

    seeds = {r["seed"] for r in records if r["readable"] and r["seed"] is not None}
    if len(records) > 1 and len(seeds) == 1:
        print(
            "every checkpoint shares one seed, so these models saw the same "
            "initialisation and the same data order. They make much the same "
            "mistakes, and averaging them gains little -- vary --seed to build "
            "an ensemble worth having."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Say what each checkpoint on disk actually is"
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="folders to search; pass two to compare working copies before merging",
    )
    parser.add_argument("--json", type=Path, help="also write the findings here")
    args = parser.parse_args()

    records = inventory(args.roots)
    _report(records, args.roots)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(args.json)


if __name__ == "__main__":
    main()
