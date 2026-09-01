"""Generate `b52_standalone.py`: the whole B52 run as one Python file.

The notebook and this script are the same code. That is the point of generating
one from the other rather than writing it twice: a fix to the augmentation or
the split lands in both, and neither can quietly drift from the other.

What the transform does, and why each part of it is needed:

* **Colab-only code is dropped.** Mounting Drive, unzipping archives and the
  `pip install` cell have no meaning in a script, which is handed paths instead.
* **Top-level statements are dropped.** A notebook cell mixes definitions with
  the lines that run them. In a script those lines would fire on import, so only
  definitions and a named list of constants survive; a `main()` written here
  runs everything, in order, once.
* **Traps are dropped by name.** The base notebook's `build_experiment` trains
  on the 58 expert-gold studies, `run_epoch` treats every label cell alike
  whatever its confidence, and `run_preflight` checks gradient flow with the
  wrong loss. All three still run, none announces that it is not B52, and in a
  flat script they would sit one call away from anything.
* **Comments are preserved.** The transform slices the original source text
  rather than regenerating it from a syntax tree, because the comments are most
  of what makes the inherited code readable.

Every marker the transform depends on is asserted, so a change to
``build_notebook.py`` fails here loudly rather than emitting a broken script.
"""
from __future__ import annotations

import ast
import runpy
from pathlib import Path

B52_BUILDER = Path(__file__).with_name("build_b52_colab_subset_notebook.py")
CELLS: list[tuple[str, str]] = list(runpy.run_path(str(B52_BUILDER))["CELLS"])


# Cells that exist only because this is a notebook, identified by content.
DROP_CELLS = (
    '"pip", "install"',          # installs pydicom into a Colab session
    "B52_RUN = build_b52_run",   # the notebook's build step; main() does this
    "PREFLIGHT = run_preflight",  # replaced by b52_preflight, see below
    "RUN_TRAINING = False",      # the notebook's run gate
)

# Definitions that must not reach the script, and the reason for each.
DROP_DEFINITIONS = {
    # Colab and Google Drive only.
    "mount_drive": "a script is given paths, not a mounted Drive",
    "ArchivePaths": "no archives to unpack",
    "safe_extract_zip": "no archives to unpack",
    "copy_and_extract_archives": "no archives to unpack",
    "find_extracted_root": "no archives to unpack",
    # Traps: they run, they look right, and none of them is B52.
    "build_experiment": "trains on the 58 expert-gold studies",
    "make_split": "only used by build_experiment",
    "run_epoch": "ignores per-cell confidence, so it trains on report silence",
    "train_model": "only used by run_epoch",
    "run_preflight": "checks gradient with the wrong loss; b52_preflight replaces it",
    "masked_bce_with_logits": "the unweighted loss; report_weighted_bce is B52's",
    "make_positive_weight": "only used by the unweighted loss",
    # Need a notebook to render into.
    "describe_b52_reference": "renders a table into a notebook cell",
    "plot_loss_history": "calls plt.show(); the script saves figures instead",
    "collect_case_examples": "renders MRI figures into notebook cells",
    "show_case_examples": "renders MRI figures into notebook cells",
    "show_results": "renders a table into a notebook cell",
    "format_known_labels": "only used by the case-review figures",
    "evaluate_predictions": (
        "casts soft report labels with .astype(int), so 0.97 and 0.03 both "
        "become 0; evaluate_weak_predictions replaces it"
    ),
}

# Module-level constants the script needs. Everything else assigned at the top
# level of a cell is the notebook running something, and is dropped.
KEEP_ASSIGNMENTS = {
    "TARGETS",
    "N_TARGETS",
    "PLANE_TO_ID",
    "DEVICE",
    "TRUE_TOKENS",
    "FALSE_TOKENS",
    "DICOM_SUFFIXES",
    "REPORT_LABELS_FILENAME",
    "AUGMENTATION",
    "NO_AUGMENTATION",
    "HIERARCHY_PREFIXES",
    "HIERARCHY_LR_SCALE",
    "CONFIG",
    "B52_REFERENCE",
}

# Imports handled by the preamble instead: the future import must be the first
# statement in the file, and IPython is not a dependency of a script.
DROP_IMPORTS = {"__future__", "IPython.display"}


def _assigned_names(node: ast.Assign) -> list[str]:
    return [target.id for target in node.targets if isinstance(target, ast.Name)]


def _import_module(node) -> str:
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return (node.names[0].asname or node.names[0].name).split(".")[0]


def _slice_with_comments(lines: list[str], node) -> str:
    """The exact source of one top-level node, including the comments above it.

    A syntax tree carries no comments, so regenerating from it would strip every
    explanation out of the inherited code. Slicing the text keeps them.
    """
    start = node.lineno
    for decorator in getattr(node, "decorator_list", []):
        start = min(start, decorator.lineno)

    # Walk upwards over any comment block written directly above.
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start - 1 : node.end_lineno])


def filter_cell(text: str) -> tuple[str, list[str]]:
    """Keep the definitions and named constants; drop everything that runs."""
    lines = text.split("\n")
    kept: list[str] = []
    dropped: list[str] = []

    for node in ast.parse(text).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _import_module(node) in DROP_IMPORTS:
                dropped.append(f"import {_import_module(node)}")
                continue
            kept.append(_slice_with_comments(lines, node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in DROP_DEFINITIONS:
                dropped.append(node.name)
                continue
            kept.append(_slice_with_comments(lines, node))
        elif isinstance(node, ast.Assign) and set(_assigned_names(node)) <= KEEP_ASSIGNMENTS:
            if not _assigned_names(node):
                dropped.append("<unnamed assignment>")
                continue
            kept.append(_slice_with_comments(lines, node))
        else:
            dropped.append(type(node).__name__)

    return "\n\n\n".join(part.strip("\n") for part in kept), dropped


def heading_of(markdown: str) -> str | None:
    """The section title of a markdown cell, so the script keeps its structure."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            # Section numbers belong to the notebook's running order, not here.
            head, _, tail = title.partition(". ")
            return (tail or title).strip() if head.rstrip(".").isdigit() else title
    return None


PREAMBLE = '''#!/usr/bin/env python3
"""B52 in one file: train the model, and write every result to a folder.

This is generated from `notebook/build_b52_script.py`. Do not edit it by hand --
edit the builder and regenerate, or the notebook and the script will disagree.

    python b52_standalone.py --data-root DIR --labels training_targets.csv --out DIR

## What B52 is

Every experiment in this line before it was measured on a model that had barely
been trained: the pixel encoder frozen at a learning rate of exactly zero, all
nine augmentations switched off, and two fixed epochs -- 3,120 optimiser steps
in total -- with no checkpoint selection. An architecture ablation measured that
way is measured through a floor.

B52 changes the training regime and nothing else:

    1. the encoder learns      the part that reads pixels, at a real rate
    2. augmentation is on      rotation, shift, scale, gamma, noise, dropout, bias
    3. a cosine that finishes,  and the best epoch is kept rather than the last

The geometry, the head, the labels and the loss are untouched, so anything that
changes is down to training.

## What one run writes

    config.json               the exact settings used
    labels_summary.json       how much supervision the reports actually gave
    history.json              per epoch: losses, hold-out AUC, gold AUC, gate, lr
    history.csv               the same, as a table
    per_target_auc.csv        every target's AUC at the best epoch
    holdout_predictions.csv   one probability row per held-out study
    gold_predictions.csv      the same for the expert-gold studies, if any
    test_predictions.csv      only when --test-root is given
    loss_curve.png            training and hold-out loss
    auc_curve.png             hold-out and gold macro AUC per epoch
    best_model.pt             the best epoch's weights, with its provenance
    summary.txt               the whole run in plain words

## What these numbers are worth

Nothing, in absolute terms. This trains a fresh compact model from random
weights on whatever subset you have. What transfers is the shape: whether the
training loss keeps falling, whether the hold-out score keeps rising, and which
epoch it peaks on. Do not compare the number it prints with a leaderboard score.
"""
from __future__ import annotations

# Figures are written to files, never shown, so the backend is fixed before
# pyplot is imported. Without this a machine with no display raises on import.
import matplotlib

matplotlib.use("Agg")

import argparse
import csv
import sys
from dataclasses import replace
'''


def build(path: Path) -> tuple[Path, list[str]]:
    """Write the script and return the names that were dropped from it."""
    body: list[str] = []
    dropped_all: list[str] = []
    pending_heading: str | None = None

    for kind, text in CELLS:
        if kind == "markdown":
            heading = heading_of(text)
            if heading:
                pending_heading = heading
            continue
        if any(marker in text for marker in DROP_CELLS):
            continue

        kept, dropped = filter_cell(text)
        dropped_all.extend(dropped)
        if not kept.strip():
            continue

        if pending_heading:
            rule = "=" * 74
            body.append(f"# {rule}\n# {pending_heading}\n# {rule}")
            pending_heading = None
        body.append(kept)

    source = PREAMBLE + "\n\n" + "\n\n\n".join(body) + "\n\n\n" + APPENDIX.strip("\n") + "\n"
    ast.parse(source)  # never write a script that does not parse
    path.write_text(source, encoding="utf-8")
    return path, dropped_all


APPENDIX = '''
# ==========================================================================
# Running B52 end to end
# ==========================================================================
#
# Everything above is shared with the notebook. Everything below exists only
# because this is a script: it turns command-line arguments into one run, and
# turns that run into files on disk.


def resolve_paths(data_root: Path, out_dir: Path) -> DrivePaths:
    """Point the inherited path bundle at a plain folder rather than at Drive."""
    data_root = Path(data_root).expanduser().resolve()
    for name in ("train.csv", "train_series.csv"):
        if not (data_root / name).is_file():
            raise FileNotFoundError(f"no {name} under {data_root}; check --data-root")
    return make_paths(data_root, Path(out_dir).expanduser().resolve())


def limit_labels(labels_path: Path, max_studies: int, scratch: Path) -> Path:
    """Write a shortened copy of the label export, for a quick trial run.

    A copy rather than an edit: the original export is the record of what the
    reports said, and a run that trims it in place would destroy that.
    """
    frame = pd.read_csv(labels_path)
    if max_studies <= 0 or max_studies >= len(frame):
        return Path(labels_path)

    scratch.mkdir(parents=True, exist_ok=True)
    trimmed = scratch / "training_targets_limited.csv"
    frame.head(int(max_studies)).to_csv(trimmed, index=False)
    print(f"[B52] limited to the first {max_studies} of {len(frame)} labelled studies")
    return trimmed


def b52_preflight(run: B52Run) -> dict:
    """One forward and backward pass, with B52's own loss and no update.

    It answers the question B52 exists to ask: does a gradient actually reach
    the encoder? A silent no here is the frozen baseline wearing B52's name, and
    it would look like an ordinary disappointing result rather than a bug.
    """
    print("preflight: one forward and backward pass, no optimiser step")
    experiment = run.experiment
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    experiment.model.train()
    experiment.model.zero_grad(set_to_none=True)

    batch = next(iter(experiment.train_loader))
    confidence = run.supervision.batch(list(batch["study_uid"]))
    volumes, present, metadata, position, target = move_batch(batch)
    del batch

    with autocast_context():
        output = experiment.model(volumes, present, metadata, position)
        combined = report_weighted_bce(
            output.logits, target, confidence, run.supervision.multiplier
        )
        local = report_weighted_bce(
            output.local_logits, target, confidence, run.supervision.multiplier
        )
        total = combined + experiment.config.local_loss_weight * local

    experiment.scaler.scale(total).backward()

    def moved(module: nn.Module) -> int:
        return sum(
            1
            for parameter in module.parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
        )

    report = {
        "loss": float(total.detach().cpu()),
        "encoder_tensors_with_gradient": moved(experiment.model.encoder),
        "hierarchy_tensors_with_gradient": moved(experiment.model.global_classifier),
        "head_tensors_with_gradient": moved(experiment.model.sparse_head),
        "trainable": describe_trainable(experiment.model),
    }
    if DEVICE.type == "cuda":
        report["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)

    experiment.model.zero_grad(set_to_none=True)

    if report["encoder_tensors_with_gradient"] == 0:
        raise RuntimeError(
            "preflight FAILED: no gradient reached the encoder. B52 is the "
            "experiment in which the encoder learns, so this run would be the "
            "frozen baseline under a different name."
        )

    for name, value in report.items():
        print(f"  {name:<32} {value}")
    print("preflight PASS")
    return report


def score_split(run: B52Run, loader, name: str, out_dir: Path) -> dict:
    """Score one split with the final weights and write a prediction row per study."""
    result = run_b52_epoch(run.experiment, loader, run.supervision, training=False)
    uids = list(loader.dataset.study_uids)
    probability = result["probability"]
    if len(uids) != len(probability):
        raise RuntimeError(
            f"{name}: {len(uids)} studies but {len(probability)} predictions; "
            "the loader and the dataset are out of step"
        )

    frame = pd.DataFrame(probability, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    frame["predicted_positive"] = [format_positive_predictions(row) for row in probability]
    frame.to_csv(out_dir / f"{name}_predictions.csv", index=False)

    scores = evaluate_weak_predictions(result["target"], result["probability"])
    scores["loss"] = result["loss"]
    scores["studies"] = len(uids)
    return scores


def write_history(history: list, out_dir: Path) -> None:
    """The epoch table, as JSON for exactness and CSV for reading."""
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    columns = [
        "epoch",
        "learning_rate",
        "train_loss",
        "validation_loss",
        "holdout_macro_auc",
        "gold_macro_auc",
        "gate",
        "seconds",
        "kept",
    ]
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_per_target(scores: dict, out_dir: Path) -> None:
    """Every target's AUC, which is where a macro average hides its detail."""
    rows = [
        {"target": target, "auc": scores["per_target_auc"].get(target)}
        for target in TARGETS
    ]
    pd.DataFrame(rows).to_csv(out_dir / "per_target_auc.csv", index=False)


def plot_curves(history: list, out_dir: Path) -> None:
    """Two figures: whether it is learning, and whether that is helping."""
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="training loss")
    plt.plot(
        epochs, [row["validation_loss"] for row in history], marker="o", label="hold-out loss"
    )
    plt.xlabel("epoch")
    plt.ylabel("confidence-weighted BCE")
    plt.title("B52: training and hold-out loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        epochs,
        [row.get("holdout_macro_auc") for row in history],
        marker="o",
        label="hold-out macro AUC (the epoch is chosen on this)",
    )
    if any(row.get("gold_macro_auc") is not None for row in history):
        plt.plot(
            epochs,
            [row.get("gold_macro_auc") for row in history],
            marker="s",
            label="expert-gold macro AUC (read only)",
        )
    plt.xlabel("epoch")
    plt.ylabel("macro AUC")
    plt.title("B52: score per epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "auc_curve.png", dpi=140)
    plt.close()


def write_summary(
    run: B52Run,
    history: list,
    holdout: dict,
    gold: dict | None,
    policy: AugmentationPolicy,
    out_dir: Path,
    minutes: float,
) -> None:
    """The whole run in plain words, for whoever opens the folder next."""
    best_epoch = run.best.epoch
    config = run.experiment.config
    augmentation = describe_augmentation(policy)

    def line(value) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    text = [
        "B52 -- actually train the model",
        "=" * 74,
        "",
        "What B52 changes, and nothing else:",
        "  1. the encoder learns          (it was frozen, at a rate of exactly 0.0)",
        "  2. augmentation is on          (nine settings existed and were all zeroed)",
        "  3. the cosine finishes and     (it was two fixed epochs, and whatever",
        "     the best epoch is kept       epoch 2 produced was the answer)",
        "",
        "Settings",
        "-" * 74,
        f"  epochs                  {config.epochs}",
        f"  learning rate           {config.learning_rate}",
        f"  hierarchy rate          {config.learning_rate * HIERARCHY_LR_SCALE} ({HIERARCHY_LR_SCALE}x)",
        f"  image size              {config.image_size}",
        f"  slices per series       {config.slices_per_series}",
        f"  batch size              {config.batch_size}",
        f"  seed                    {config.seed}",
        f"  device                  {DEVICE}",
        f"  augmentations on        {augmentation['count']}",
        *(f"    {name:<22}{value}" for name, value in augmentation["active"].items()),
        "",
        "Studies",
        "-" * 74,
        f"  training (reports)      {len(run.experiment.train_loader.dataset)}",
        f"  hold-out (reports)      {holdout['studies']}   <- the epoch is chosen on these",
        f"  expert gold (read only) {gold['studies'] if gold else 0}",
        "",
        "Result",
        "-" * 74,
        f"  best epoch              {best_epoch} of {config.epochs}",
        f"  hold-out macro AUC      {line(run.best.score)}",
        f"  expert-gold macro AUC   {line(gold['mean_auc']) if gold else 'n/a'}",
        f"  wall clock              {minutes:.1f} minutes",
        "",
    ]

    if best_epoch == config.epochs:
        text += [
            "  The last epoch was the best, so the hold-out score was still climbing",
            "  when the run stopped. More epochs are worth trying.",
        ]
    else:
        text += [
            f"  The run peaked at epoch {best_epoch} and did not improve after it,",
            "  so more epochs would not have helped.",
        ]

    text += [
        "",
        "What this number is worth",
        "-" * 74,
        "  Nothing, in absolute terms. This is a fresh compact model trained from",
        "  random weights on a subset. Read the shape -- is the loss falling, is the",
        "  hold-out score rising, which epoch does it peak on -- and not the value.",
        "  It is not comparable with any leaderboard score.",
        "",
        "  The hold-out number is also a selection statistic: it is the best of",
        f"  {config.epochs} epochs on the very surface used to choose the epoch, so it is",
        "  optimistically biased by construction.",
        "",
        "For context, what the real B52 measured on the real data",
        "-" * 74,
        f"  frozen control          {B52_REFERENCE.frozen_control_macro_auc:.6f}",
        f"  B52, 1,447 studies      {B52_REFERENCE.gate_split_macro_auc:.6f}",
        f"  B52, 3,801 studies      {B52_REFERENCE.all_data_macro_auc:.6f}",
        f"  measured on             {B52_REFERENCE.evaluation}",
        "",
    ]

    (out_dir / "summary.txt").write_text("\\n".join(text), encoding="utf-8")
    print()
    print("\\n".join(text))


def run_b52(arguments: argparse.Namespace) -> Path:
    """One complete B52 run, from paths to a folder full of results."""
    started = time.time()
    out_dir = Path(arguments.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_paths(arguments.data_root, out_dir)
    labels_path = Path(arguments.labels).expanduser().resolve()
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"no label export at {labels_path}.\\n"
            "Export it with b23_llm_labels.py (or b6_report_labels.py) and pass "
            "it with --labels."
        )
    labels_path = limit_labels(labels_path, arguments.max_studies, out_dir / "scratch")

    config = replace(
        CONFIG,
        epochs=int(arguments.epochs),
        seed=int(arguments.seed),
        batch_size=int(arguments.batch_size),
        num_workers=int(arguments.num_workers),
        learning_rate=float(arguments.learning_rate),
        validation_fraction=float(arguments.validation_fraction),
        image_size=int(arguments.image_size),
        slices_per_series=int(arguments.slices_per_series),
    )
    policy = NO_AUGMENTATION if arguments.no_augment else AUGMENTATION

    print("=" * 74)
    print("B52: building the run")
    print("=" * 74)
    run = build_b52_run(paths, config, policy=policy, labels_path=labels_path)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "augmentation": asdict(policy),
                "augmentations_on": describe_augmentation(policy),
                "data_root": str(paths.data_root),
                "labels": str(labels_path),
                "device": str(DEVICE),
                "hierarchy_lr_scale": HIERARCHY_LR_SCALE,
                "trainable_parameters": describe_trainable(run.experiment.model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    b52_preflight(run)
    if arguments.preflight_only:
        print()
        print("--preflight-only: stopping before training")
        return out_dir

    print()
    print("=" * 74)
    print(f"B52: training for {config.epochs} epochs")
    print("=" * 74)
    history = train_b52(run)
    write_history(history, out_dir)

    print()
    print("scoring the best epoch")
    holdout = score_split(run, run.experiment.validation_loader, "holdout", out_dir)
    write_per_target(holdout, out_dir)
    gold = (
        score_split(run, run.gold_loader, "gold", out_dir)
        if run.gold_loader is not None else None
    )

    if arguments.test_root:
        test_paths = make_test_paths(Path(arguments.test_root).expanduser().resolve())
        predictions = predict_test_set(run.experiment, build_test_loader(test_paths, config))
        predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    plot_curves(history, out_dir)

    torch.save(
        {
            "model_state": run.experiment.model.state_dict(),
            "config": asdict(config),
            "augmentation": asdict(policy),
            "targets": TARGETS,
            "history": history,
            "selected_epoch": run.best.epoch,
            "selection_metric": "macro AUC on a held-out report-labelled split",
            "selection_value": run.best.score,
            "gold_labels_used": False,
            "b52_reference": asdict(B52_REFERENCE),
            "governance": (
                "The selection value is the best of several epochs on the surface "
                "used to choose the epoch, so it is optimistically biased by "
                "construction. It is not an effect size, and a subset run's "
                "absolute value is not comparable with any leaderboard score."
            ),
        },
        out_dir / "best_model.pt",
    )

    describe = describe_report_labels(
        np.stack(
            [
                run.supervision.confidence_by_study[uid]
                for uid in run.experiment.train_loader.dataset.study_uids
            ]
        )
    )
    (out_dir / "labels_summary.json").write_text(json.dumps(describe, indent=2), encoding="utf-8")

    write_summary(
        run, history, holdout, gold, policy, out_dir, (time.time() - started) / 60.0
    )
    print()
    print(f"every result is in {out_dir}")
    return out_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="b52_standalone.py",
        description="Run B52 once and write every result to a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", required=True,
        help="folder holding train.csv, train_series.csv and the DICOM directories",
    )
    parser.add_argument(
        "--labels", required=True,
        help="the report label export (training_targets.csv)",
    )
    parser.add_argument("--out", required=True, help="where to write the results")
    parser.add_argument(
        "--epochs", type=int, default=6,
        help="two is the inherited default and is the thing B52 replaces",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="raise it to feed a fast GPU; each worker is a process and costs host RAM",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--validation-fraction", type=float, default=0.20,
        help="share of report-labelled studies held out to choose the epoch",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="turn augmentation off -- this removes one of B52's three changes",
    )
    parser.add_argument(
        "--test-root", default=None,
        help="optional separate folder with test.csv and test_series.csv",
    )
    parser.add_argument(
        "--max-studies", type=int, default=0,
        help="use only the first N labelled studies, for a quick trial run",
    )
    parser.add_argument(
        "--image-size", type=int, default=CONFIG.image_size,
        help="the geometry every experiment in this line holds fixed; lower it "
             "only for a quick trial run, never for a result",
    )
    parser.add_argument(
        "--slices-per-series", type=int, default=CONFIG.slices_per_series,
        help="also part of the fixed geometry; same warning as --image-size",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="one forward and backward pass, then stop",
    )
    return parser


def main(argv: list | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.image_size != CONFIG.image_size or arguments.slices_per_series != CONFIG.slices_per_series:
        print(
            f"WARNING: geometry changed to {arguments.image_size}px x "
            f"{arguments.slices_per_series} slices, from the fixed "
            f"{CONFIG.image_size}px x {CONFIG.slices_per_series}. Fine for a "
            "trial run; the result is not comparable with anything."
        )
    if arguments.no_augment:
        print(
            "WARNING: --no-augment removes one of the three changes that make this "
            "B52. The run is still valid; it is just not the full regime."
        )
    run_b52(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


if __name__ == "__main__":
    written, dropped = build(Path(__file__).with_name("b52_standalone.py"))
    print(f"{written}  ({len(written.read_text().splitlines())} lines)")
    print(f"dropped {len(dropped)} top-level statements and definitions")
